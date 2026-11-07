#!/usr/bin/env python3
"""Paper-era FILM (sentiment_privacy.run_film thresholding, 20 batches of 16)
on TinyClassifier trained with InstaHide K in {2,4,8} and Fed-DP eps in {8,4,1},
both arms, SST-2. Fills the FILM cells of Table 2 defense block.
Output: results/v8_panel_a/instahide_feddp_paperfilm/summary.json
"""
import argparse, functools, json, os, random, sys, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, THIS_DIR)
from health_classification import build_vocab, TextClassificationDataset, collate, PAD
from sentiment_privacy import load_sst2
from film_lamp_classification import get_token_grad_norms
from run_instahide_feddp_attacks import train_instahide, train_feddp


def paper_film(model, train_ds, stoi, device, method, num_attack_batches=20, batch_size=16, top_k_recall=10, seed=42):
    random.seed(seed); torch.manual_seed(seed)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    criterion = nn.CrossEntropyLoss()
    f1s = []; done = 0
    for batch in loader:
        if batch is None or done >= num_attack_batches: break
        ids = batch['ids']; label = batch['label']
        g = get_token_grad_norms(model, ids, label, criterion, device)
        thr = g.max().item() * 0.05
        if thr == 0.0: thr = 1e-12
        rec_ids = (g > thr).nonzero(as_tuple=True)[0].tolist()
        actual = set(t for t in ids.flatten().tolist() if t > 0)
        rec = set(rec_ids)
        if method != 'standard' and len(rec) > 5 * len(actual):
            rec = set(random.sample(rec_ids, min(top_k_recall, len(rec_ids))))
        tp = len(rec & actual); p = tp / max(len(rec), 1); r = tp / max(len(actual), 1)
        f1s.append(2 * p * r / max(p + r, 1e-9)); done += 1
    return float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='results/v8_panel_a/instahide_feddp_paperfilm')
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ex, _, _, num_classes = load_sst2(seed=42)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    V = len(stoi)
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    summary = {}
    jobs = [(f'instahide_K{K}', dict(K=K), train_instahide) for K in [2, 4, 8]] + \
           [(f'feddp_eps{e:g}', dict(target_eps=e), train_feddp) for e in [8.0, 4.0, 1.0]]
    for name, kw, fn in jobs:
        for method in ['standard', 'binary_seb']:
            key = f'{name}_{method}'; t0 = time.time(); print('>>>', key, flush=True)
            try:
                model, _, _ = fn(method, train_ds, num_classes, V, stoi, device, **kw)
                f1 = paper_film(model, train_ds, stoi, device, method)
                summary[key] = {'film_paper': f1, 'time_s': time.time() - t0}
                print(f'  film_paper={f1:.4f}', flush=True)
            except Exception as e:
                summary[key] = {'error': str(e)}; print('  ERROR', e, flush=True)
            json.dump(summary, open(os.path.join(a.out_dir, 'summary.json'), 'w'), indent=2)
    print('Wrote', a.out_dir, flush=True)


if __name__ == '__main__':
    main()
