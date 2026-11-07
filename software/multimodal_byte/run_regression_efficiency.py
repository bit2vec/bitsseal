#!/usr/bin/env python3
"""Efficiency benchmark for the three multimodal regression tasks.

For each (dataset, method): measure
  - embedding params + size in MB (fp32)
  - total params + size in MB
  - end to end train wall clock (4 epochs, batch 16)
  - per step inference wall clock on the test set
"""
import argparse, functools, json, os, random, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))

from run_civic_extended import MultiModalClassifier
from run_civic_multimodal import PAD, UNK, SEP, tokenize
from run_pharmgkb_regression import (load_pharmgkb_regression,
    MODALITIES as PG_MOD, PGKBRegDataset, transform as pg_transform)
from run_civic_regression import (load_civic_regression,
    MODALITIES as CV_MOD, CIViCRegDataset)
from run_clinvar_regression import (PATHOGENICITY,
    MODALITIES as CL_MOD, CVRegDataset)
from clinvar_loader import load_clinvar


def reg_collate(batch, pad_index=0):
    if not batch: return None
    ids = [b[0] for b in batch]
    L = max(x.size(0) for x in ids)
    ids_p = torch.full((len(ids), L), pad_index, dtype=torch.long)
    for i, x in enumerate(ids): ids_p[i, :x.size(0)] = x
    y = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return {'ids': ids_p, 'label': y}


def count_params(model):
    emb = sum(p.numel() for p in model.embedding.parameters())
    total = sum(p.numel() for p in model.parameters())
    return emb, total


def setup_pharmgkb():
    examples = load_pharmgkb_regression(seed=42)
    n = len(examples); n_tr = int(0.7 * n); n_va = int(0.1 * n)
    train = examples[:n_tr]; val = examples[n_tr:n_tr + n_va]; test = examples[n_tr + n_va:]
    y_tr = np.array([pg_transform(e['score']) for e in train])
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    counter = Counter()
    for ex in train:
        for mod, _ in PG_MOD: counter.update(tokenize(str(ex.get(mod, ''))))
    stoi = {PAD: 0, UNK: 1}
    for mod, _ in PG_MOD: stoi[f'{SEP}_{mod}'] = len(stoi)
    for tok, c in counter.most_common():
        if c >= 2: stoi[tok] = len(stoi)
    tr = PGKBRegDataset(train, stoi, PG_MOD, y_mean, y_std)
    va = PGKBRegDataset(val,   stoi, PG_MOD, y_mean, y_std)
    te = PGKBRegDataset(test,  stoi, PG_MOD, y_mean, y_std)
    return 'PharmGKB', tr, va, te, len(stoi), stoi


def setup_civic():
    examples = load_civic_regression(seed=42)
    n = len(examples); n_tr = int(0.7 * n); n_va = int(0.1 * n)
    train = examples[:n_tr]; val = examples[n_tr:n_tr + n_va]; test = examples[n_tr + n_va:]
    y_tr = np.array([e['rating'] for e in train])
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    counter = Counter()
    for ex in train:
        for mod, _ in CV_MOD: counter.update(tokenize(str(ex.get(mod, ''))))
    stoi = {PAD: 0, UNK: 1}
    for mod, _ in CV_MOD: stoi[f'{SEP}_{mod}'] = len(stoi)
    for tok, c in counter.most_common():
        if c >= 2: stoi[tok] = len(stoi)
    tr = CIViCRegDataset(train, stoi, CV_MOD, y_mean, y_std)
    va = CIViCRegDataset(val,   stoi, CV_MOD, y_mean, y_std)
    te = CIViCRegDataset(test,  stoi, CV_MOD, y_mean, y_std)
    return 'CIViC', tr, va, te, len(stoi), stoi


def setup_clinvar():
    examples, _ = load_clinvar()
    examples = [e for e in examples if e['label'] in PATHOGENICITY]
    for e in examples: e['rating'] = PATHOGENICITY[e['label']]
    rng = random.Random(42); rng.shuffle(examples)
    n = len(examples); n_tr = int(0.7 * n); n_va = int(0.1 * n)
    train = examples[:n_tr]; val = examples[n_tr:n_tr + n_va]; test = examples[n_tr + n_va:]
    y_tr = np.array([e['rating'] for e in train])
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    counter = Counter()
    for ex in train:
        for mod, _ in CL_MOD: counter.update(tokenize(str(ex.get(mod, ''))))
    stoi = {PAD: 0, UNK: 1}
    for mod, _ in CL_MOD: stoi[f'{SEP}_{mod}'] = len(stoi)
    for tok, c in counter.most_common():
        if c >= 2: stoi[tok] = len(stoi)
    tr = CVRegDataset(train, stoi, CL_MOD, y_mean, y_std)
    va = CVRegDataset(val,   stoi, CL_MOD, y_mean, y_std)
    te = CVRegDataset(test,  stoi, CL_MOD, y_mean, y_std)
    return 'ClinVar', tr, va, te, len(stoi), stoi


def measure(name, train_ds, val_ds, test_ds, vocab_size, stoi,
            method, n_epochs, device, embedding_dim=64):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed(42)
    coll = functools.partial(reg_collate, pad_index=stoi[PAD])
    tr = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=coll)
    test_l = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=coll)
    model = MultiModalClassifier(vocab_size, embedding_dim, 1,
                                  method=method, body_kind='D',
                                  pad_idx=stoi[PAD]).to(device)
    emb, total = count_params(model)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.MSELoss()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_epochs):
        model.train()
        for batch in tr:
            if batch is None: continue
            opt.zero_grad()
            p = model(batch['ids'].to(device)).squeeze(-1)
            crit(p, batch['label'].to(device)).backward()
            opt.step()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    train_time = time.time() - t0
    model.eval()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for batch in test_l:
            if batch is None: continue
            _ = model(batch['ids'].to(device))
    if torch.cuda.is_available(): torch.cuda.synchronize()
    inf_time = time.time() - t0
    return {
        'method': method, 'vocab_size': vocab_size,
        'embed_params': emb, 'embed_mb': emb * 4 / 1024 / 1024,
        'total_params': total, 'total_mb': total * 4 / 1024 / 1024,
        'train_time_s': train_time, 'inference_time_s': inf_time,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/regression_efficiency')
    p.add_argument('--n_epochs', type=int, default=4)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setups = [setup_pharmgkb, setup_civic, setup_clinvar]
    summary = {}
    for fn in setups:
        name, tr_ds, va_ds, te_ds, vocab_size, stoi = fn()
        print(f'\n=== {name} (V={vocab_size}, train={len(tr_ds)}, test={len(te_ds)}) ===')
        summary[name] = {}
        for method in ['standard', 'byte_seb']:
            r = measure(name, tr_ds, va_ds, te_ds, vocab_size, stoi,
                        method, args.n_epochs, device)
            summary[name][method] = r
            print(f'  {method:10s}: emb {r["embed_params"]:>7d} ({r["embed_mb"]:5.2f} MB)  '
                  f'tot {r["total_params"]:>7d} ({r["total_mb"]:5.2f} MB)  '
                  f'train {r["train_time_s"]:5.2f}s  inf {r["inference_time_s"]*1000:5.2f}ms')
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
