#!/usr/bin/env python3
"""iDLG attack on Sleep-EDF EEG (5-class) and UCI Diabetes 130 (binary)
for Subword (standard) and Bits (binary_seb). Adds iDLG to the existing
new_domains FILM/TAG numbers in summary.json.
"""
import argparse, functools, json, os, random, sys
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, '..', '..'))
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))

from health_classification import (build_vocab, TextClassificationDataset,
                                    collate, PAD)
from film_lamp_classification import TinyClassifier
from phase1_attacks import idlg_attack
from new_domains import load_sleep_edf, load_diabetes


def run_one(name, loader_fn, device, methods=('standard', 'binary_seb')):
    train_ex, _, _, num_classes = loader_fn(seed=42)
    if len(train_ex) > 10000:
        rng = random.Random(42); rng.shuffle(train_ex)
        train_ex = train_ex[:10000]
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    print(f'{name}: train={len(train_ex)} V={vocab_size} C={num_classes}', flush=True)

    out = {}
    for method in methods:
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=coll)
        model = TinyClassifier(vocab_size, 64, num_classes, method=method).to(device)
        opt = optim.Adam(model.parameters(), lr=5e-4)
        crit = nn.CrossEntropyLoss()
        for _ in range(2):
            model.train()
            for batch in loader:
                if batch is None: continue
                opt.zero_grad()
                crit(model(batch['ids'].to(device)),
                     batch['label'].to(device)).backward()
                opt.step()
        model.train()
        eval_loader = DataLoader(train_ds, batch_size=16, shuffle=False, collate_fn=coll)
        f1s = []
        for i, batch in enumerate(eval_loader):
            if batch is None or i >= 12: break
            try:
                r = idlg_attack(model, batch['ids'], batch['label'], crit, device,
                                 vocab_size, embedding_dim=64, num_iters=100, lr=0.1)
                f1s.append(r['f1'] if isinstance(r, dict) else r)
            except Exception as e:
                print(f'  iDLG err: {e}')
        v = float(np.mean(f1s)) if f1s else None
        out[method] = {'idlg_f1': v}
        print(f'  {method} idlg={v}', flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results/new_domains/idlg_summary.json')
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)

    summary = {}
    for name, fn in [('sleep_edf', load_sleep_edf), ('diabetes', load_diabetes)]:
        try:
            summary[name] = run_one(name, fn, device)
        except Exception as e:
            print(f'{name} failed: {e}', flush=True)
            summary[name] = {'error': str(e)}
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2)
    print(f'Wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
