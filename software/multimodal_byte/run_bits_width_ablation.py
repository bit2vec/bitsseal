#!/usr/bin/env python3
"""Bits FFN width ablation on CIViC body D.

Sweep interdim in {128, 256, 512, 1024} and report acc, params, FILM, TAG.
"""
import argparse, csv, functools, json, os, random, sys, time
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
from run_civic_multimodal import (
    PAD, UNK, SEP, CIVIC_TSV, MODALITIES, tokenize, load_civic, build_vocab,
    CIViCMultiModalDataset, collate, film_attack,
)


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/bits_width')
    p.add_argument('--n_epochs', type=int, default=6)
    p.add_argument('--n_attack', type=int, default=12)
    p.add_argument('--widths', nargs='+', type=int, default=[128, 256, 512, 1024])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading CIViC...')
    train_ex, val_ex, test_ex = load_civic(seed=42)
    stoi, vocab_meta = build_vocab(train_ex, MODALITIES, boundary='separator')
    vocab_size = len(stoi)
    num_classes = 5
    print(f'  V={vocab_size}  train={len(train_ex)}/val={len(val_ex)}/test={len(test_ex)}')

    coll = functools.partial(collate, pad_index=stoi[PAD])
    train_ds = CIViCMultiModalDataset(train_ex, stoi, vocab_meta, MODALITIES, ablation='all')
    val_ds   = CIViCMultiModalDataset(val_ex,   stoi, vocab_meta, MODALITIES, ablation='all')
    test_ds  = CIViCMultiModalDataset(test_ex,  stoi, vocab_meta, MODALITIES, ablation='all')
    tr   = DataLoader(train_ds, batch_size=16, shuffle=True,  collate_fn=coll)
    val_l= DataLoader(val_ds,   batch_size=16, shuffle=False, collate_fn=coll)
    test_l= DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=coll)

    summary = {}
    for width in args.widths:
        print(f'\n>>> Bits interdim={width}')
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        if torch.cuda.is_available(): torch.cuda.manual_seed(42)
        model = MultiModalClassifier(vocab_size, 64, num_classes,
                                      method='binary_seb', body_kind='D',
                                      pad_idx=stoi[PAD], interdim=str(width)).to(device)
        opt = optim.Adam(model.parameters(), lr=5e-4)
        crit = nn.CrossEntropyLoss()

        t0 = time.time()
        best_val, best_state = 0.0, None
        for ep in range(args.n_epochs):
            model.train()
            for batch in tr:
                if batch is None: continue
                opt.zero_grad()
                crit(model(batch['ids'].to(device)),
                     batch['label'].to(device)).backward()
                opt.step()
            model.eval()
            c = t = 0
            with torch.no_grad():
                for batch in val_l:
                    if batch is None: continue
                    pred = model(batch['ids'].to(device)).argmax(-1).cpu()
                    c += (pred == batch['label']).sum().item(); t += batch['label'].size(0)
            v = c / max(t, 1)
            print(f'  ep {ep+1}: val={v:.4f}')
            if v > best_val:
                best_val = v
                best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
        if best_state is not None: model.load_state_dict(best_state)
        train_time = time.time() - t0

        model.eval()
        c = t = 0
        with torch.no_grad():
            for batch in test_l:
                if batch is None: continue
                pred = model(batch['ids'].to(device)).argmax(-1).cpu()
                c += (pred == batch['label']).sum().item(); t += batch['label'].size(0)
        test_acc = c / max(t, 1)

        # Privacy attacks
        model.train()
        film_f1s, tag_f1s = [], []
        n_atk = 0
        for batch in tr:
            if batch is None or n_atk >= args.n_attack: break
            try: film_f1s.append(film_attack(model, batch['ids'], batch['label'],
                                              crit, device))
            except Exception as e: print(f'  film err: {e}')
            try: tag_f1s.append(tag_attack(model, batch['ids'], batch['label'],
                                             crit, device, vocab_size))
            except Exception as e: print(f'  tag err: {e}')
            n_atk += 1

        n_params = sum(p.numel() for p in model.parameters())
        summary[width] = {
            'width': width,
            'val_acc': float(best_val),
            'test_acc': float(test_acc),
            'n_params': n_params,
            'mb': n_params * 4 / 1024 / 1024,
            'train_time_s': train_time,
            'film_f1': float(np.mean(film_f1s)) if film_f1s else 0.0,
            'tag_f1':  float(np.mean(tag_f1s))  if tag_f1s  else 0.0,
        }
        print(f'  test_acc={test_acc:.4f}  params={n_params:>7d} ({n_params*4/1024/1024:.2f} MB)  '
              f'FILM={summary[width]["film_f1"]:.4f}  TAG={summary[width]["tag_f1"]:.4f}')

        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
