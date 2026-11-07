#!/usr/bin/env python3
"""Proper LLG (Wainakh '22 / Aono '17 fc.bias-grad-sign) on SST-2 across all
defenses for both Subword and Bits. Returns ASR (chance = 0.5 for binary).

Reuses training setups from run_proper_lamp_sst2.py.
Output: results/v8_panel_a/llg_sst2_defenses/summary.json
"""
import argparse, functools, json, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, THIS_DIR)

from health_classification import (build_vocab, TextClassificationDataset,
                                    collate, PAD)
from sentiment_privacy import load_sst2
from film_lamp_classification import TinyClassifier
from llg_healthcare import extract_labels_llg, asr

# Reuse training functions from the proper-LAMP runner
from run_proper_lamp_sst2 import (
    train_baseline, train_dpsgd, train_topk, train_instahide, train_feddp,
)


def run_llg_eval(model, loader, crit, device, num_classes, n_batches=20):
    model.train()
    asrs = []
    for i, batch in enumerate(loader):
        if batch is None or i >= n_batches: break
        ids = batch['ids'].to(device); lbl = batch['label'].to(device)
        try:
            model.zero_grad(set_to_none=False)
            logits = model(ids)
            crit(logits, lbl).backward()
            g = model.fc.weight.grad.clone()
            pred = extract_labels_llg(g, ids.size(0), num_classes)
            asrs.append(asr(pred, lbl.cpu().tolist()))
        except Exception as e:
            print(f'  LLG err: {e}', flush=True)
    return float(np.mean(asrs)) if asrs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/v8_panel_a/llg_sst2_defenses')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, _, _, num_classes = load_sst2(seed=42)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    print(f'SST-2 train={len(train_ex)} V={vocab_size} C={num_classes}', flush=True)

    methods = ['standard', 'binary_seb']
    summary = {}

    configs = []
    configs.append(('baseline', None))
    for eps in [None, 16.0, 8.0, 4.0, 1.0]:
        configs.append(('dpsgd', eps))
    for K in [1.0, 0.1, 0.01]:
        configs.append(('topk', K))
    for K in [2, 4, 8]:
        configs.append(('instahide', K))
    for eps in [8.0, 4.0, 1.0]:
        configs.append(('feddp', eps))

    for method in methods:
        summary[method] = {}
        for kind, param in configs:
            key = f'{kind}' + (f'_{param}' if param is not None else '')
            t0 = time.time()
            print(f'>>> {method} / {key}', flush=True)
            try:
                if kind == 'baseline':
                    model, loader, crit = train_baseline(method, train_ds, num_classes, vocab_size, stoi, device)
                elif kind == 'dpsgd':
                    model, loader, crit = train_dpsgd(method, train_ds, num_classes, vocab_size, stoi, device, param)
                elif kind == 'topk':
                    model, loader, crit = train_topk(method, train_ds, num_classes, vocab_size, stoi, device, param)
                elif kind == 'instahide':
                    model, loader, crit = train_instahide(method, train_ds, num_classes, vocab_size, stoi, device, param)
                elif kind == 'feddp':
                    model, loader, crit = train_feddp(method, train_ds, num_classes, vocab_size, stoi, device, param)
                a = run_llg_eval(model, loader, crit, device, num_classes)
                summary[method][key] = {'llg_asr': a, 'time_s': time.time() - t0}
                print(f'  llg_asr={a} ({summary[method][key]["time_s"]:.1f}s)', flush=True)
            except Exception as e:
                summary[method][key] = {'error': str(e)}
                print(f'  ERROR: {e}', flush=True)
            with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
