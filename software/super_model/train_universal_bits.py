#!/usr/bin/env python3
"""Universal Bits: one shared encoder + body, 4 task heads, 4 language types.

Trains a single Bits encoder on PubMedQA (medical QA) + Education (topic
classification) + Devign (code vulnerability) + Syslog (log line classifier),
shares all encoder + body parameters, and tests whether multi-task training
matches single-task accuracy on each.

Layout:
  shared:   epsilon (33K) + body (cross-attn + MLP, ~50K)
  per-task: linear classification head (d -> C_t)

Vocabulary: each task has its own token id range; the combined vocab has size
V = sum_t V_t, and Bits maps every id to a 64-bit pattern. Subword baseline
also uses combined V.
"""
import argparse, functools, gzip, json, os, random, re, sys, tempfile, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'system_log'))

from binary_subword_sentiment import BinarySubwordConcateOnehot
from byte_subword_concate_onehot import ByteSubwordConcateOnehot

from health_classification import load_pubmedqa
from domain_classification import load_education
from new_domains import load_devign


# ---------- syslog loader (inline) ----------
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


def tokenize(s, max_len=96):
    toks = re.findall(r'[A-Za-z]+|\d+|[^\w\s]', s.lower())
    return toks[:max_len]


def build_task_vocab(examples, min_freq=2):
    cnt = Counter()
    for ex in examples:
        text = ex[0] if isinstance(ex, tuple) else ex
        cnt.update(tokenize(text))
    stoi = {'<pad>': 0, '<unk>': 1}
    for w, c in cnt.items():
        if c >= min_freq: stoi[w] = len(stoi)
    return stoi


def encode(s, stoi, max_len=96, offset=0):
    toks = tokenize(s, max_len)
    return [(stoi.get(t, 1) + offset) if (stoi.get(t, 1) > 1) else stoi.get(t, 1) + offset
            for t in toks]


# ----- Datasets -----
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


class MultiTaskDataset(Dataset):
    """Each item: (task_id, ids, label)."""
    def __init__(self, examples, stoi, task_id, offset, max_len=96):
        self.exs = examples; self.stoi = stoi; self.task_id = task_id
        self.offset = offset; self.max_len = max_len
    def __len__(self): return len(self.exs)
    def __getitem__(self, i):
        text, label = self.exs[i]
        ids = encode(text, self.stoi, self.max_len, offset=self.offset)
        if not ids: ids = [1 + self.offset]
        return {'task_id': self.task_id, 'ids': torch.tensor(ids, dtype=torch.long),
                'label': int(label)}


def collate(batch, pad_index=0):
    max_len = max(b['ids'].size(0) for b in batch)
    out_ids = torch.full((len(batch), max_len), pad_index, dtype=torch.long)
    labs   = torch.zeros(len(batch), dtype=torch.long)
    tids   = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        out_ids[i, :b['ids'].size(0)] = b['ids']
        labs[i] = b['label']
        tids[i] = b['task_id']
    return {'ids': out_ids, 'label': labs, 'task_id': tids}


class UniversalClassifier(nn.Module):
    """Shared Bits encoder + body D + per-task heads."""
    def __init__(self, vocab_size, emb_dim, task_classes, method='binary_seb',
                 num_bytes=8, num_bits=64, interdim='128', tmpdir=None):
        super().__init__()
        self.method = method
        self.vocab_size = vocab_size
        if method == 'standard':
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
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
        # Body D: single learnable query cross-attending to the sequence + MLP residual
        self.q = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim), nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim))
        self.heads = nn.ModuleList([nn.Linear(emb_dim, c) for c in task_classes])

    def forward(self, ids, task_ids):
        e = self.embedding(ids)  # (B, L, d)
        B = e.size(0)
        q = self.q.expand(B, -1, -1)
        v, _ = self.attn(q, e, e)
        v = v.squeeze(1) + self.mlp(v.squeeze(1))
        # Per-task heads
        out = torch.zeros(B, max(h.out_features for h in self.heads), device=v.device)
        for ti in range(len(self.heads)):
            mask = (task_ids == ti)
            if mask.any():
                logits = self.heads[ti](v[mask])
                out[mask, :logits.size(1)] = logits
        return out, [self.heads[ti](v) for ti in range(len(self.heads))]  # also return per-task logits


def evaluate_per_task(model, loaders, device):
    model.eval()
    accs = {}
    with torch.no_grad():
        for name, (ti, loader) in loaders.items():
            c = t = 0
            for batch in loader:
                ids = batch['ids'].to(device)
                tids = batch['task_id'].to(device)
                _, per_task_logits = model(ids, tids)
                pred = per_task_logits[ti].argmax(-1).cpu()
                c += (pred == batch['label']).sum().item(); t += batch['label'].size(0)
            accs[name] = c / max(t, 1)
    return accs


def train_universal(method, datasets_info, device, n_epochs, batch_size, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    # Build per-task vocabs and unified offset map
    stois = {}; offsets = {}; running_offset = 2  # leave 0=PAD, 1=UNK
    task_classes = []
    for ti, (name, train_ex, test_ex, n_classes) in enumerate(datasets_info):
        stoi = build_task_vocab(train_ex)
        stois[name] = stoi
        offsets[name] = running_offset - 2  # so token 0/1 stay as PAD/UNK
        running_offset += len(stoi)
        task_classes.append(n_classes)
    total_vocab = running_offset
    print(f'  total vocab V = {total_vocab}')

    # Build datasets
    train_dss, test_dss = [], []
    for ti, (name, train_ex, test_ex, n_classes) in enumerate(datasets_info):
        train_dss.append(MultiTaskDataset(train_ex, stois[name], ti, offsets[name]))
        test_dss.append((name, ti, MultiTaskDataset(test_ex, stois[name], ti, offsets[name])))

    coll = functools.partial(collate, pad_index=0)
    test_loaders = {name: (ti, DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=coll))
                    for name, ti, ds in test_dss}

    # Concatenate train sets; sample uniformly per epoch
    from torch.utils.data import ConcatDataset
    combined = ConcatDataset(train_dss)
    tr_loader = DataLoader(combined, batch_size=batch_size, shuffle=True, collate_fn=coll)

    tmp = tempfile.mkdtemp()
    model = UniversalClassifier(total_vocab, 64, task_classes, method=method, tmpdir=tmp).to(device)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.CrossEntropyLoss()

    print(f'  shared params: {sum(p.numel() for p in model.parameters()):,}')

    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        total_loss = 0; n_b = 0
        for batch in tr_loader:
            ids = batch['ids'].to(device)
            tids = batch['task_id'].to(device)
            labs = batch['label'].to(device)
            opt.zero_grad()
            _, per_task_logits = model(ids, tids)
            loss = 0
            for ti in range(len(task_classes)):
                mask = (tids == ti)
                if mask.any():
                    loss = loss + crit(per_task_logits[ti][mask], labs[mask])
            loss.backward()
            opt.step()
            total_loss += float(loss); n_b += 1
        accs = evaluate_per_task(model, test_loaders, device)
        print(f'  ep {ep+1}: loss={total_loss/max(n_b,1):.3f}  ' +
              ' '.join(f'{n}={a:.3f}' for n, a in accs.items()))
    train_time = time.time() - t0

    final_accs = evaluate_per_task(model, test_loaders, device)
    n_params = sum(p.numel() for p in model.parameters())
    return {
        'method': method,
        'n_params': n_params,
        'mb_fp32': n_params * 4 / 1024 / 1024,
        'train_time_s': train_time,
        'per_task_acc': final_accs,
        'total_vocab': total_vocab,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/universal_bits')
    p.add_argument('--n_epochs', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--methods', nargs='+', default=['standard', 'binary_seb'])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading datasets...')
    pq_tr, pq_va, pq_te, pq_C = load_pubmedqa()
    pq_tr_norm = normalize_examples(pq_tr); pq_te_norm = normalize_examples(pq_te)
    print(f'  PubMedQA: train={len(pq_tr_norm)} test={len(pq_te_norm)} C={pq_C}')

    ed_tr, ed_va, ed_te, ed_C = load_education()
    ed_tr_norm = normalize_examples(ed_tr); ed_te_norm = normalize_examples(ed_te)
    print(f'  Education: train={len(ed_tr_norm)} test={len(ed_te_norm)} C={ed_C}')

    dv_tr, dv_va, dv_te, dv_C = load_devign(seed=42)
    dv_tr_norm = normalize_examples(dv_tr); dv_te_norm = normalize_examples(dv_te)
    print(f'  Devign: train={len(dv_tr_norm)} test={len(dv_te_norm)} C={dv_C}')

    sl_tr, sl_te, sl_C = load_syslog_examples()
    sl_tr_norm = [(t, l) for t, l in sl_tr]; sl_te_norm = [(t, l) for t, l in sl_te]
    print(f'  Syslog: train={len(sl_tr_norm)} test={len(sl_te_norm)} C={sl_C}')

    datasets_info = [
        ('pubmedqa', pq_tr_norm, pq_te_norm, pq_C),
        ('education', ed_tr_norm, ed_te_norm, ed_C),
        ('devign', dv_tr_norm, dv_te_norm, dv_C),
        ('syslog', sl_tr_norm, sl_te_norm, sl_C),
    ]

    summary = {}
    for method in args.methods:
        print(f'\n>>> Universal {method}')
        r = train_universal(method, datasets_info, device, args.n_epochs, args.batch_size, seed=42)
        summary[method] = r
        print(f'  -> {r["per_task_acc"]}')
        print(f'  -> total_vocab={r["total_vocab"]} params={r["n_params"]:,} ({r["mb_fp32"]:.2f} MB)')
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
