#!/usr/bin/env python3
"""Universal Bits v2: shared byte-level vocab + task tokens (multilingual MT recipe).

Design (matches NLLB / M2M):
  - One shared vocab: 256 bytes + K task tokens (K = number of tasks)
  - Each input text is UTF-8 byte encoded then prepended with [TASK=t] id
  - Same Bits encoder for every task, same body D, per-task linear heads
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

from health_classification import load_pubmedqa
from domain_classification import load_education
from new_domains import load_devign


# ---- syslog inline loader ----
SYSLOG_PATHS = ['/var/log/syslog', '/var/log/syslog.1',
                '/var/log/auth.log', '/var/log/auth.log.1',
                '/var/log/kern.log', '/var/log/kern.log.1']
SYSLOG_RE = re.compile(
    r'^(?P<ts>\S+)\s+(?P<host>\S+)\s+(?P<service>[A-Za-z][A-Za-z0-9_\-]*)(?:\[\d+\])?:\s+(?P<msg>.+)$'
)


def load_syslog_examples(seed=42, max_lines=20000, top_k=5, min_per_class=200):
    random.seed(seed)
    parsed = []
    for p in SYSLOG_PATHS:
        if not os.path.exists(p): continue
        try:
            opener = gzip.open if p.endswith('.gz') else open
            with opener(p, 'rt', encoding='utf-8', errors='replace') as f:
                for line in f:
                    m = SYSLOG_RE.match(line.rstrip('\n'))
                    if m:
                        parsed.append((m.group('service'), m.group('msg')))
                        if len(parsed) >= max_lines: break
        except Exception: pass
        if len(parsed) >= max_lines: break
    counts = Counter(s for s, _ in parsed)
    keep = [s for s, c in counts.most_common(top_k * 3) if c >= min_per_class][:top_k]
    label_map = {s: i for i, s in enumerate(keep)}
    examples = [(m, label_map[s]) for s, m in parsed if s in label_map]
    random.shuffle(examples)
    n = len(examples); n_te = max(int(0.2 * n), 200)
    train, test = examples[:n - n_te], examples[n - n_te:]
    return train, test, len(label_map)


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

    combined = ConcatDataset(train_dss)
    tr_loader = DataLoader(combined, batch_size=batch_size, shuffle=True, collate_fn=coll)

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
    p.add_argument('--out_dir', default='results/universal_bits_v2')
    p.add_argument('--n_epochs', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--max_len', type=int, default=256)
    p.add_argument('--methods', nargs='+', default=['standard', 'byte_seb', 'binary_seb'])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading datasets...')
    pq_tr, _, pq_te, pq_C = load_pubmedqa()
    pq_tr_n = normalize_examples(pq_tr); pq_te_n = normalize_examples(pq_te)
    print(f'  PubMedQA: train={len(pq_tr_n)} test={len(pq_te_n)} C={pq_C}')

    ed_tr, _, ed_te, ed_C = load_education()
    ed_tr_n = normalize_examples(ed_tr); ed_te_n = normalize_examples(ed_te)
    print(f'  Education: train={len(ed_tr_n)} test={len(ed_te_n)} C={ed_C}')

    dv_tr, _, dv_te, dv_C = load_devign(seed=42)
    dv_tr_n = normalize_examples(dv_tr); dv_te_n = normalize_examples(dv_te)
    print(f'  Devign: train={len(dv_tr_n)} test={len(dv_te_n)} C={dv_C}')

    sl_tr, sl_te, sl_C = load_syslog_examples()
    print(f'  Syslog: train={len(sl_tr)} test={len(sl_te)} C={sl_C}')

    datasets_info = [
        ('pubmedqa', pq_tr_n, pq_te_n, pq_C),
        ('education', ed_tr_n, ed_te_n, ed_C),
        ('devign', dv_tr_n, dv_te_n, dv_C),
        ('syslog', sl_tr, sl_te, sl_C),
    ]

    summary = {}
    for method in args.methods:
        print(f'\n>>> Universal-v2 {method}')
        r = train(method, datasets_info, device, args.n_epochs, args.batch_size,
                  args.max_len, seed=42)
        summary[method] = r
        print(f'  -> {r["per_task_acc"]}')
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
