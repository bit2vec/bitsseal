#!/usr/bin/env python3
"""Sweep over the bit-sequence length n in {16, 32, 64, 96, 128} on SST-2.
Reports test accuracy + FILM token F1 + TAG token F1 + parameter count for
each n. Subword baseline (n undefined) is included for reference.

Output: results/nbits_sweep/summary.json
"""
import argparse, functools, json, os, random, sys, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))

from health_classification import (build_vocab, TextClassificationDataset,
                                    collate, PAD)
from sentiment_privacy import load_sst2
from film_lamp_classification import TinyClassifier, get_token_grad_norms
from extra_attacks import tag_attack


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            ids = batch['ids'].to(device); lbl = batch['label'].to(device)
            logits = model(ids)
            correct += (logits.argmax(-1) == lbl).sum().item()
            total += lbl.size(0)
    return correct / total if total else 0.0


def film_attack(model, ids, label, criterion, device):
    """Set-recovery FILM: read embedding-grad sparsity, recover token set."""
    norms = get_token_grad_norms(model, ids, label, criterion, device)
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    k = max(len(actual), 1)
    recovered = set(torch.topk(norms, k).indices.cpu().tolist())
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


def run_attacks(model, loader, crit, device, vocab_size, n_batches=12):
    model.train()
    film_f1s, tag_f1s = [], []
    for i, batch in enumerate(loader):
        if batch is None or i >= n_batches: break
        try:
            film_f1s.append(film_attack(model, batch['ids'], batch['label'], crit, device))
        except Exception as e:
            print(f'  FILM err: {e}', flush=True)
        try:
            r = tag_attack(model, batch['ids'], batch['label'], crit, device,
                           vocab_size, embedding_dim=64)
            tag_f1s.append(r['f1'] if isinstance(r, dict) else r)
        except Exception as e:
            print(f'  TAG err: {e}', flush=True)
    return (float(np.mean(film_f1s)) if film_f1s else None,
            float(np.mean(tag_f1s)) if tag_f1s else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/nbits_sweep')
    p.add_argument('--n_epochs', type=int, default=2)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, val_ex, test_ex, num_classes = load_sst2(seed=42)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    test_ds = TextClassificationDataset(test_ex, stoi, max_length=128)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    print(f'SST-2: train={len(train_ex)} test={len(test_ex)} V={vocab_size} C={num_classes}',
          flush=True)

    summary = {}
    n_values = [16, 32, 64, 96, 128]
    for n_bits in n_values:
        t0 = time.time()
        print(f'\n>>> Bits n_bits={n_bits}', flush=True)
        try:
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            model = TinyClassifier(vocab_size, 64, num_classes,
                                   method='binary_seb', num_bits=n_bits).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                                       collate_fn=coll)
            test_loader = DataLoader(test_ds, batch_size=16, shuffle=False,
                                      collate_fn=coll)
            opt = optim.Adam(model.parameters(), lr=5e-4)
            crit = nn.CrossEntropyLoss()
            for ep in range(args.n_epochs):
                model.train()
                for batch in train_loader:
                    if batch is None: continue
                    opt.zero_grad()
                    crit(model(batch['ids'].to(device)),
                         batch['label'].to(device)).backward()
                    opt.step()
            test_acc = evaluate(model, test_loader, device)
            attack_loader = DataLoader(train_ds, batch_size=16, shuffle=False,
                                        collate_fn=coll)
            film, tag = run_attacks(model, attack_loader, crit, device,
                                     vocab_size)
            summary[str(n_bits)] = {
                'n_bits': n_bits,
                'test_acc': test_acc,
                'n_params': n_params,
                'mb': n_params * 4 / (1024 * 1024),
                'film_f1': film,
                'tag_f1': tag,
                'time_s': time.time() - t0,
            }
            print(f'  acc={test_acc:.4f} params={n_params} '
                  f'film={film} tag={tag} ({summary[str(n_bits)]["time_s"]:.1f}s)',
                  flush=True)
        except Exception as e:
            summary[str(n_bits)] = {'error': str(e)}
            print(f'  ERROR: {e}', flush=True)
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
