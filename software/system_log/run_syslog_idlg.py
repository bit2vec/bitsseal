#!/usr/bin/env python3
"""iDLG attack on Linux syslog (Subword and Bits). Adds iDLG to the existing
FILM/TAG numbers in system_log/summary.json.
"""
import argparse, functools, json, os, random, sys, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, THIS)

from health_classification import (build_vocab, TextClassificationDataset,
                                    collate, PAD)
from film_lamp_classification import TinyClassifier
from phase1_attacks import idlg_attack
from run_syslog_classification import load_syslog_lines, SYSLOG_PATHS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results/system_log/idlg_summary.json')
    p.add_argument('--max_lines', type=int, default=20000)
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw = load_syslog_lines(SYSLOG_PATHS, max_lines=args.max_lines)
    if not raw:
        print('No syslog lines'); return
    # Build label set from top-K services
    from collections import Counter
    services = Counter([s for s, _ in raw]).most_common(5)
    keep = {s: i for i, (s, _) in enumerate(services)}
    examples = [(msg, keep[s]) for s, msg in raw if s in keep]
    rng = random.Random(42); rng.shuffle(examples)
    n_tr = int(0.8 * len(examples))
    train_ex = examples[:n_tr]
    num_classes = len(keep)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    print(f'syslog: train={len(train_ex)} V={vocab_size} C={num_classes}', flush=True)

    summary = {}
    for method in ['standard', 'binary_seb']:
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
        idlg_f1s = []
        for i, batch in enumerate(eval_loader):
            if batch is None or i >= 12: break
            try:
                r = idlg_attack(model, batch['ids'], batch['label'], crit, device,
                                 vocab_size, embedding_dim=64, num_iters=100, lr=0.1)
                idlg_f1s.append(r['f1'] if isinstance(r, dict) else r)
            except Exception as e:
                print(f'  iDLG err: {e}')
        v = float(np.mean(idlg_f1s)) if idlg_f1s else None
        summary[method] = {'idlg_f1': v}
        print(f'  {method} idlg={v}', flush=True)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
