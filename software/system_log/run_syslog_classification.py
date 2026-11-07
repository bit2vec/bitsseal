#!/usr/bin/env python3
"""System-log language: classify log line -> producing service.

Pulls recent lines from /var/log/syslog and /var/log/auth.log (readable as
group=adm), parses RFC5424 + traditional syslog format, extracts the service
field, runs Subword vs EmByte vs Bits with FILM/TAG/MIA.
"""
import argparse, functools, gzip, json, os, random, re, sys, time
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

# Reuse the standard byte-level model + attacks from healthcare codebase
from binary_subword_sentiment import BinarySubwordConcateOnehot
from byte_subword_concate_onehot import ByteSubwordConcateOnehot


SYSLOG_PATHS = ['/var/log/syslog', '/var/log/syslog.1',
                '/var/log/auth.log', '/var/log/auth.log.1',
                '/var/log/kern.log', '/var/log/kern.log.1']

# RFC5424-ish or traditional: "<timestamp> <hostname> <service>[<pid>]: <msg>"
LINE_RE = re.compile(
    r'^(?P<ts>\S+)\s+(?P<host>\S+)\s+(?P<service>[A-Za-z][A-Za-z0-9_\-]*)(?:\[\d+\])?:\s+(?P<msg>.+)$'
)


def load_syslog_lines(paths, max_lines=20000):
    out = []
    for p in paths:
        if not os.path.exists(p): continue
        try:
            opener = gzip.open if p.endswith('.gz') else open
            with opener(p, 'rt', encoding='utf-8', errors='replace') as f:
                for line in f:
                    m = LINE_RE.match(line.rstrip('\n'))
                    if m:
                        out.append((m.group('service'), m.group('msg')))
                        if len(out) >= max_lines: return out
        except Exception as e:
            print(f'  read err {p}: {e}')
    return out


def build_dataset(lines, top_k=5, min_per_class=200):
    counts = Counter(s for s, _ in lines)
    keep = [s for s, c in counts.most_common(top_k * 3) if c >= min_per_class][:top_k]
    if len(keep) < 2:
        raise RuntimeError(f'not enough classes: {counts.most_common(10)}')
    label_map = {s: i for i, s in enumerate(keep)}
    filtered = [(label_map[s], m) for s, m in lines if s in label_map]
    return filtered, label_map


def tokenize(s, max_len=64):
    # word-level on the message, lowercased
    toks = re.findall(r'[A-Za-z]+|\d+|[^\w\s]', s.lower())
    return toks[:max_len]


def build_vocab(examples, min_freq=2):
    cnt = Counter()
    for _, m in examples:
        cnt.update(tokenize(m))
    stoi = {'<pad>': 0, '<unk>': 1}
    for w, c in cnt.items():
        if c >= min_freq:
            stoi[w] = len(stoi)
    return stoi


def encode(s, stoi, max_len=64):
    toks = tokenize(s, max_len)
    return [stoi.get(t, 1) for t in toks]


class LogDataset(Dataset):
    def __init__(self, exs, stoi, max_len=64):
        self.exs = exs; self.stoi = stoi; self.max_len = max_len
    def __len__(self): return len(self.exs)
    def __getitem__(self, i):
        lab, msg = self.exs[i]
        ids = encode(msg, self.stoi, self.max_len)
        if not ids: ids = [1]
        return {'ids': torch.tensor(ids, dtype=torch.long), 'label': lab}


def collate(batch, pad_index=0):
    max_len = max(b['ids'].size(0) for b in batch)
    out_ids = torch.full((len(batch), max_len), pad_index, dtype=torch.long)
    labs = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        out_ids[i, :b['ids'].size(0)] = b['ids']
        labs[i] = b['label']
    return {'ids': out_ids, 'label': labs}


class TinyClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes, method='standard',
                 num_bytes=8, num_bits=64, interdim='128', pad_idx=0,
                 byte_map_path='results/system_log/byte_map.pt',
                 binary_map_path='results/system_log/binary_map.pt'):
        super().__init__()
        self.method = method
        if method == 'standard':
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        elif method == 'byte_seb':
            self.embedding = ByteSubwordConcateOnehot(
                num_embeddings=vocab_size, embedding_dim=emb_dim,
                interdim=interdim, subword_bytes_file=byte_map_path)
        elif method == 'binary_seb':
            self.embedding = BinarySubwordConcateOnehot(
                num_embeddings=vocab_size, embedding_dim=emb_dim,
                num_bits=num_bits, interdim=interdim,
                subword_binary_file=binary_map_path)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, ids):
        e = self.embedding(ids)
        return self.fc(e.mean(dim=1))


def film_attack(model, ids, label, criterion, device):
    model.zero_grad(set_to_none=False)
    logits = model(ids.to(device))
    loss = criterion(logits, label.to(device))
    loss.backward()
    if model.method == 'standard':
        gnorm = model.embedding.weight.grad.detach().norm(dim=1).cpu()
    else:
        all_grad = sum(p.grad.detach().pow(2).sum().item()
                       for p in model.embedding.parameters() if p.grad is not None)
        gnorm = torch.full((model.fc.out_features,), all_grad, dtype=torch.float)
    threshold = gnorm.max().item() * 0.05
    if threshold == 0: threshold = 1e-12
    recovered = set((gnorm > threshold).nonzero(as_tuple=True)[0].tolist())
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    if model.method != 'standard' and len(recovered) > 5 * len(actual):
        recovered = set(random.sample(list(recovered), min(10, len(recovered))))
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


def tag_attack(model, ids, label, criterion, device, vocab_size):
    model.zero_grad(set_to_none=False)
    logits = model(ids.to(device))
    loss = criterion(logits, label.to(device))
    loss.backward()
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    if model.method == 'standard':
        gnorm = model.embedding.weight.grad.detach().norm(dim=1).cpu()
        k = max(len(actual), 1)
        recovered = set(torch.topk(gnorm, k).indices.tolist())
    else:
        rows = []
        with torch.no_grad():
            for s in range(0, vocab_size, 2048):
                e = min(s + 2048, vocab_size)
                ids_v = torch.arange(s, e, device=device).unsqueeze(0)
                rows.append(model.embedding(ids_v).squeeze(0))
        emb_table = torch.cat(rows, dim=0)
        fc_grad = model.fc.weight.grad.detach()
        g = fc_grad.mean(dim=0).unsqueeze(0)
        sim = (torch.nn.functional.normalize(g, dim=-1) @
               torch.nn.functional.normalize(emb_table, dim=-1).t()).abs().squeeze(0).cpu()
        k = max(len(actual), 1)
        recovered = set(torch.topk(sim, k).indices.tolist())
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


def run_one(method, train_ds, test_ds, vocab_size, num_classes, device, n_epochs, batch_size, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    coll = functools.partial(collate, pad_index=0)
    tr   = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=coll)
    test_l = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)

    model = TinyClassifier(vocab_size, 64, num_classes, method=method).to(device)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        for batch in tr:
            opt.zero_grad()
            crit(model(batch['ids'].to(device)),
                 batch['label'].to(device)).backward()
            opt.step()
    train_time = time.time() - t0

    model.eval()
    c = t = 0
    with torch.no_grad():
        for batch in test_l:
            pred = model(batch['ids'].to(device)).argmax(-1).cpu()
            c += (pred == batch['label']).sum().item(); t += batch['label'].size(0)
    test_acc = c / max(t, 1)

    model.train()
    film_f1s, tag_f1s = [], []
    n_atk = 0
    for batch in tr:
        if n_atk >= 12: break
        try: film_f1s.append(film_attack(model, batch['ids'], batch['label'], crit, device))
        except Exception as e: print(f'  film err: {e}')
        try: tag_f1s.append(tag_attack(model, batch['ids'], batch['label'], crit, device, vocab_size))
        except Exception as e: print(f'  tag err: {e}')
        n_atk += 1

    n_params = sum(p.numel() for p in model.parameters())
    return {
        'method': method, 'test_acc': float(test_acc), 'n_params': n_params,
        'mb_fp32': n_params * 4 / 1024 / 1024, 'train_time_s': train_time,
        'film_f1': float(np.mean(film_f1s)) if film_f1s else 0.0,
        'tag_f1':  float(np.mean(tag_f1s))  if tag_f1s  else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/system_log')
    p.add_argument('--n_epochs', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--max_lines', type=int, default=20000)
    p.add_argument('--top_k', type=int, default=5)
    p.add_argument('--methods', nargs='+', default=['standard', 'byte_seb', 'binary_seb'])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading syslog lines...')
    lines = load_syslog_lines(SYSLOG_PATHS, max_lines=args.max_lines)
    print(f'  loaded {len(lines)} parsed lines')
    if len(lines) < 1000:
        raise RuntimeError(f'too few lines parsed: {len(lines)}')

    examples, label_map = build_dataset(lines, top_k=args.top_k)
    print(f'  classes: {label_map}')
    print(f'  total examples: {len(examples)}')
    counts = Counter(lab for lab, _ in examples)
    print(f'  per-class: {dict(counts)}')

    random.seed(42)
    random.shuffle(examples)
    n = len(examples); n_te = max(int(0.2 * n), 200)
    train_ex = examples[:n - n_te]
    test_ex  = examples[n - n_te:]
    print(f'  train={len(train_ex)} test={len(test_ex)}')

    stoi = build_vocab(train_ex, min_freq=2)
    vocab_size = len(stoi)
    num_classes = len(label_map)
    print(f'  V={vocab_size}  C={num_classes}')

    train_ds = LogDataset(train_ex, stoi)
    test_ds  = LogDataset(test_ex,  stoi)

    summary = {'config': vars(args), 'label_map': label_map,
               'n_train': len(train_ex), 'n_test': len(test_ex),
               'vocab_size': vocab_size, 'num_classes': num_classes,
               'results': {}}
    for method in args.methods:
        print(f'\n>>> {method}')
        r = run_one(method, train_ds, test_ds, vocab_size, num_classes, device,
                    args.n_epochs, args.batch_size, seed=42)
        print(f'  acc={r["test_acc"]:.4f}  params={r["n_params"]:>7d} ({r["mb_fp32"]:.2f} MB)  '
              f'FILM={r["film_f1"]:.4f}  TAG={r["tag_f1"]:.4f}')
        summary['results'][method] = r
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
