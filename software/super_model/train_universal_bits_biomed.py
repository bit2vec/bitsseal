#!/usr/bin/env python3
"""Universal Bits — Biomedical Cluster (Cluster A).

Four PubMed-domain tasks, shared byte vocab + task tokens. Tests positive
transfer when tasks share modality (biomedical prose) and domain (PubMed):
  1. PubMedQA      - 700 ex,   3-way Q+abstract -> yes/no/maybe
  2. PubMed MeSH   - ~20K ex,  PubMed abstract -> primary MeSH topic
  3. MedMCQA       - 50K ex,   medical MCQ
  4. med_abstracts - ~12K ex,  biomedical abstract -> 5 conditions
"""
import argparse, functools, gzip, json, os, random, re, sys, tempfile, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))

from binary_subword_sentiment import BinarySubwordConcateOnehot
from byte_subword_concate_onehot import ByteSubwordConcateOnehot

from health_classification import load_pubmedqa, load_medmcqa
from genome_biomed_classification import load_med_abstracts, load_pubmed_mesh


# ---- byte tokenization with reserved task tokens ----
# id 0 = PAD
# id 1..255 = bytes (we offset by +1 so PAD is unique)
# id 256..256+K-1 = task tokens
PAD_ID = 0
BYTE_OFFSET = 1
TASK_TOKEN_BASE = 257  # 1 (PAD) + 256 (bytes) = 257


def byte_encode(text, task_id, max_len=256):
    """Returns: [TASK=t] + UTF-8 byte ids of text, capped at max_len."""
    head = TASK_TOKEN_BASE + task_id
    if isinstance(text, list):
        text = ' '.join(str(x) for x in text)
    raw = str(text).encode('utf-8', errors='replace')[:max_len - 1]
    body = [BYTE_OFFSET + b for b in raw]
    return [head] + body


def normalize_examples(raw):
    out = []
    for ex in raw:
        if isinstance(ex, tuple):
            text, label = ex[0], ex[1]
        elif isinstance(ex, dict):
            text, label = ex.get('text', ex.get('sentence', '')), ex['label']
        else:
            continue
        out.append((str(text), int(label)))
    return out


class MultiTaskByteDataset(Dataset):
    def __init__(self, examples, task_id, max_len=256):
        self.exs = examples; self.task_id = task_id; self.max_len = max_len
    def __len__(self): return len(self.exs)
    def __getitem__(self, i):
        text, label = self.exs[i]
        ids = byte_encode(text, self.task_id, self.max_len)
        if len(ids) < 2: ids = [TASK_TOKEN_BASE + self.task_id, BYTE_OFFSET]
        return {'task_id': self.task_id,
                'ids': torch.tensor(ids, dtype=torch.long),
                'label': int(label)}


def collate(batch, pad_index=PAD_ID):
    max_len = max(b['ids'].size(0) for b in batch)
    out_ids = torch.full((len(batch), max_len), pad_index, dtype=torch.long)
    labs = torch.zeros(len(batch), dtype=torch.long)
    tids = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        out_ids[i, :b['ids'].size(0)] = b['ids']
        labs[i] = b['label']
        tids[i] = b['task_id']
    return {'ids': out_ids, 'label': labs, 'task_id': tids}


class UniversalClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, task_classes, method='binary_seb',
                 num_bits=64, interdim='128', tmpdir=None):
        super().__init__()
        self.method = method
        self.vocab_size = vocab_size
        if method == 'standard':
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        elif method == 'byte_seb':
            tmp = os.path.join(tmpdir or tempfile.mkdtemp(), 'byte_map.pt')
            self.embedding = ByteSubwordConcateOnehot(
                num_embeddings=vocab_size, embedding_dim=emb_dim,
                interdim=interdim, relu_dropout=0.1, subword_bytes_file=tmp)
        elif method == 'binary_seb':
            tmp = os.path.join(tmpdir or tempfile.mkdtemp(), 'binary_map.pt')
            self.embedding = BinarySubwordConcateOnehot(
                num_embeddings=vocab_size, embedding_dim=emb_dim,
                num_bits=num_bits, interdim=interdim,
                subword_binary_file=tmp)
        # Body D: cross-attention with learnable query + MLP residual
        self.q = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim), nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim))
        self.heads = nn.ModuleList([nn.Linear(emb_dim, c) for c in task_classes])

    def forward(self, ids, task_ids):
        e = self.embedding(ids)               # (B, L, d)
        B = e.size(0)
        q = self.q.expand(B, -1, -1)
        v, _ = self.attn(q, e, e)             # (B, 1, d)
        v = v.squeeze(1) + self.mlp(v.squeeze(1))
        per_task_logits = [h(v) for h in self.heads]
        return per_task_logits


def evaluate_per_task(model, loaders, device):
    model.eval()
    accs = {}
    with torch.no_grad():
        for name, (ti, loader) in loaders.items():
            c = t = 0
            for batch in loader:
                ids = batch['ids'].to(device)
                tids = batch['task_id'].to(device)
                ptl = model(ids, tids)
                pred = ptl[ti].argmax(-1).cpu()
                c += (pred == batch['label']).sum().item(); t += batch['label'].size(0)
            accs[name] = c / max(t, 1)
    return accs


def train(method, datasets_info, device, n_epochs, batch_size, max_len, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    K = len(datasets_info)
    vocab_size = TASK_TOKEN_BASE + K  # 257 + K
    task_classes = [info[3] for info in datasets_info]

    train_dss, test_loaders = [], {}
    coll = functools.partial(collate, pad_index=PAD_ID)
    for ti, (name, train_ex, test_ex, _) in enumerate(datasets_info):
        train_dss.append(MultiTaskByteDataset(train_ex, ti, max_len))
        test_dss = MultiTaskByteDataset(test_ex, ti, max_len)
        test_loaders[name] = (ti, DataLoader(test_dss, batch_size=batch_size, shuffle=False, collate_fn=coll))

    # Task-balanced sampling: weight each example inversely to its task size,
    # so each task gets ~equal gradient updates per epoch.
    from torch.utils.data import WeightedRandomSampler
    combined = ConcatDataset(train_dss)
    weights = []
    for ds in train_dss:
        n = len(ds)
        weights.extend([1.0 / n] * n)
    n_total = len(combined)
    # One epoch = K * (avg per-task size) draws, so all tasks see comparable counts
    avg_n = n_total // K
    epoch_samples = K * avg_n
    sampler = WeightedRandomSampler(weights, num_samples=epoch_samples, replacement=True)
    tr_loader = DataLoader(combined, batch_size=batch_size, sampler=sampler, collate_fn=coll)

    tmp = tempfile.mkdtemp()
    model = UniversalClassifier(vocab_size, 64, task_classes, method=method, tmpdir=tmp).to(device)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  vocab_size={vocab_size}  params={n_params:,} ({n_params*4/1024/1024:.2f} MB)')

    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        total_loss = 0; n_b = 0
        for batch in tr_loader:
            ids = batch['ids'].to(device)
            tids = batch['task_id'].to(device)
            labs = batch['label'].to(device)
            opt.zero_grad()
            ptl = model(ids, tids)
            loss = 0
            for ti in range(K):
                m = (tids == ti)
                if m.any():
                    loss = loss + crit(ptl[ti][m], labs[m])
            loss.backward()
            opt.step()
            total_loss += float(loss); n_b += 1
        accs = evaluate_per_task(model, test_loaders, device)
        print(f'  ep {ep+1}: loss={total_loss/max(n_b,1):.3f}  ' +
              ' '.join(f'{n}={a:.3f}' for n, a in accs.items()))
    train_time = time.time() - t0

    final = evaluate_per_task(model, test_loaders, device)
    return {
        'method': method, 'vocab_size': vocab_size, 'n_params': n_params,
        'mb_fp32': n_params * 4 / 1024 / 1024, 'train_time_s': train_time,
        'per_task_acc': final,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/universal_bits_biomed')
    p.add_argument('--n_epochs', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--max_len', type=int, default=512)
    p.add_argument('--mesh_subsample', type=int, default=20000)
    p.add_argument('--medmcqa_subsample', type=int, default=50000)
    p.add_argument('--methods', nargs='+', default=['standard', 'byte_seb', 'binary_seb'])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading biomedical cluster datasets...')

    pq_tr, _, pq_te, pq_C = load_pubmedqa()
    pq_tr_n = normalize_examples(pq_tr); pq_te_n = normalize_examples(pq_te)
    print(f'  PubMedQA: train={len(pq_tr_n)} test={len(pq_te_n)} C={pq_C}')

    mesh_tr, _, mesh_te, mesh_C = load_pubmed_mesh(subsample=args.mesh_subsample)
    mesh_tr_n = normalize_examples(mesh_tr); mesh_te_n = normalize_examples(mesh_te)
    print(f'  PubMed MeSH: train={len(mesh_tr_n)} test={len(mesh_te_n)} C={mesh_C}')

    mc_tr, mc_va, mc_te, mc_C = load_medmcqa(subsample=args.medmcqa_subsample)
    # MedMCQA's official test split is unlabeled (leaderboard-only).
    # Use validation as our held-out test.
    mc_tr_n = normalize_examples(mc_tr); mc_te_n = normalize_examples(mc_va)
    print(f'  MedMCQA: train={len(mc_tr_n)} test={len(mc_te_n)} (val-as-test) C={mc_C}')

    ma_tr, _, ma_te, ma_C = load_med_abstracts()
    ma_tr_n = normalize_examples(ma_tr); ma_te_n = normalize_examples(ma_te)
    print(f'  med_abstracts: train={len(ma_tr_n)} test={len(ma_te_n)} C={ma_C}')

    datasets_info = [
        ('pubmedqa',      pq_tr_n,   pq_te_n,   pq_C),
        ('pubmed_mesh',   mesh_tr_n, mesh_te_n, mesh_C),
        ('medmcqa',       mc_tr_n,   mc_te_n,   mc_C),
        ('med_abstracts', ma_tr_n,   ma_te_n,   ma_C),
    ]

    summary = {}
    for method in args.methods:
        print(f'\n>>> Universal-Biomed {method}')
        r = train(method, datasets_info, device, args.n_epochs, args.batch_size,
                  args.max_len, seed=42)
        summary[method] = r
        print(f'  -> {r["per_task_acc"]}')
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
