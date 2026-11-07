#!/usr/bin/env python3
"""Recompute Bits LLG ASR on SST-2 with multiple seeds and more batches
for a stable error bar. Output: results/v8_panel_a/bits_llg_recompute/summary.json
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
from film_lamp_classification import TinyClassifier
from llg_healthcare import extract_labels_llg, asr


def run_seed(seed, n_batches, n_epochs, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    train_ex, _, _, num_classes = load_sst2(seed=seed)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=coll)
    model = TinyClassifier(vocab_size, 64, num_classes, method='binary_seb').to(device)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        model.train()
        for batch in loader:
            if batch is None: continue
            opt.zero_grad()
            crit(model(batch['ids'].to(device)),
                 batch['label'].to(device)).backward()
            opt.step()
    model.train()
    asrs = []
    eval_loader = DataLoader(train_ds, batch_size=16, shuffle=False, collate_fn=coll)
    for i, batch in enumerate(eval_loader):
        if batch is None or i >= n_batches: break
        ids = batch['ids'].to(device); lbl = batch['label'].to(device)
        try:
            model.zero_grad(set_to_none=False)
            crit(model(ids), lbl).backward()
            g = model.fc.weight.grad.clone()
            pred = extract_labels_llg(g, ids.size(0), num_classes)
            asrs.append(asr(pred, lbl.cpu().tolist()))
        except Exception as e:
            print(f'  err: {e}', flush=True)
    return float(np.mean(asrs)) if asrs else None, len(asrs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/v8_panel_a/bits_llg_recompute')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    p.add_argument('--n_batches', type=int, default=50)
    p.add_argument('--n_epochs', type=int, default=2)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}, seeds={args.seeds}, n_batches={args.n_batches}', flush=True)

    per_seed = {}
    for s in args.seeds:
        t0 = time.time()
        a, n = run_seed(s, args.n_batches, args.n_epochs, device)
        per_seed[str(s)] = {'asr': a, 'n_batches_evaluated': n,
                            'time_s': time.time() - t0}
        print(f'  seed={s}: ASR={a:.4f}  ({n} batches, {time.time()-t0:.1f}s)',
              flush=True)
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump({'per_seed': per_seed}, f, indent=2)

    asrs = [v['asr'] for v in per_seed.values() if v['asr'] is not None]
    summary = {
        'per_seed': per_seed,
        'mean_asr': float(np.mean(asrs)) if asrs else None,
        'std_asr': float(np.std(asrs, ddof=1)) if len(asrs) > 1 else None,
        'min_asr': float(np.min(asrs)) if asrs else None,
        'max_asr': float(np.max(asrs)) if asrs else None,
        'distance_from_chance': float(np.mean([abs(a - 0.5) for a in asrs])) if asrs else None,
        'n_seeds': len(asrs),
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nMean ASR: {summary["mean_asr"]:.4f}  (std {summary["std_asr"]:.4f})',
          flush=True)
    print(f'|ASR - 0.5| = {summary["distance_from_chance"]:.4f}', flush=True)
    print(f'Wrote {args.out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
