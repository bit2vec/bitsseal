#!/usr/bin/env python3
"""Fill the small NLP gaps in the privacy table.

Cells to populate:
- TAG on SST-2 (Subword vs EmByte)
- FL-LLG token recovery on MedMCQA and PubMedQA (Subword vs EmByte)
- DP-SGD-on-Subword + FILM on SST-2

Reuses existing code in extra_attacks.py and phase1_full.py.
"""
import argparse, functools, json, os, random, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, '..', 'Senti_analysis')))
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, '..', 'federated')))

from health_classification import (
    build_vocab, TextClassificationDataset, collate, PAD,
    load_pubmedqa, load_medmcqa,
)
from film_lamp_classification import TinyClassifier
from extra_attacks import tag_attack
from phase1_full import dp_sgd_train
from sentiment_privacy import load_sst2, load_yelp, load_imdb
from fl_mt_timing import llg_attack as fl_llg_attack

DATASETS = {
    'medmcqa':  lambda seed: load_medmcqa(seed=seed, subsample=5000),
    'pubmedqa': load_pubmedqa,
    'sst2':     load_sst2,
    'yelp':     load_yelp,
    'imdb':     load_imdb,
}


def train_target(method, dataset_name, n_epochs=2, batch_size=16, embedding_dim=64, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ex, _, _, num_classes = DATASETS[dataset_name](seed=seed)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    model = TinyClassifier(vocab_size, embedding_dim, num_classes, method=method).to(device)
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
    return model, loader, stoi, vocab_size, crit, device


def run_tag(method, ds, n_attack=12):
    model, loader, stoi, V, crit, device = train_target(method, ds)
    f1s = []; n = 0
    for batch in loader:
        if n >= n_attack: break
        if batch is None: continue
        try:
            r = tag_attack(model, batch['ids'], batch['label'], crit, device, V)
            f1s.append(r['f1'])
        except Exception as e:
            print(f'    tag err: {e}')
        n += 1
    return float(np.mean(f1s)) if f1s else 0.0


def run_fl_llg(method, ds, n_attack=12):
    model, loader, stoi, V, crit, device = train_target(method, ds)
    f1s = []; n = 0
    for batch in loader:
        if n >= n_attack: break
        if batch is None: continue
        try:
            r = fl_llg_attack(model, batch['ids'], batch['label'], crit, device, V)
            f1s.append(r['token_f1'])
        except Exception as e:
            print(f'    llg err: {e}')
        n += 1
    return float(np.mean(f1s)) if f1s else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/nlp_gaps')
    p.add_argument('--skip', nargs='+', default=[])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}

    # 1) TAG on SST-2 for both methods
    if 'tag' not in args.skip:
        print('===== TAG on SST-2 =====')
        summary['tag_sst2'] = {}
        for m in ['standard', 'byte_seb']:
            t0 = time.time()
            try:
                f1 = run_tag(m, 'sst2')
                summary['tag_sst2'][m] = float(f1)
                print(f'  {m}: F1={f1:.3f} ({time.time()-t0:.1f}s)')
            except Exception as e:
                summary['tag_sst2'][m] = {'error': str(e)}
                print(f'  {m} ERR: {e}')
            with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)

    # 2) FL-LLG token recovery on medmcqa and pubmedqa
    if 'llg' not in args.skip:
        print('\n===== FL-LLG token recovery on MedMCQA, PubMedQA =====')
        for ds in ['medmcqa', 'pubmedqa']:
            summary[f'fl_llg_{ds}'] = {}
            for m in ['standard', 'byte_seb']:
                t0 = time.time()
                try:
                    f1 = run_fl_llg(m, ds)
                    summary[f'fl_llg_{ds}'][m] = float(f1)
                    print(f'  {ds}/{m}: F1={f1:.3f} ({time.time()-t0:.1f}s)')
                except Exception as e:
                    summary[f'fl_llg_{ds}'][m] = {'error': str(e)}
                    print(f'  {ds}/{m} ERR: {e}')
                with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                    json.dump(summary, f, indent=2)

    # 3) DP-SGD on SST-2 (Subword + ε=8) and EmByte FILM at no DP for ref
    if 'dpsgd' not in args.skip:
        print('\n===== DP-SGD on SST-2 (Subword) =====')
        summary['dpsgd_sst2'] = {}
        for eps in [8.0]:
            try:
                t0 = time.time()
                r = dp_sgd_train('standard', 'sst2', target_eps=eps,
                                  batch_size=16, embedding_dim=64, n_epochs=4)
                summary['dpsgd_sst2'][f'eps_{eps}'] = r
                print(f'  Subword DP ε={eps}: acc={r["test_acc"]:.3f} '
                      f'FILM={r["film_f1"]:.3f} ({time.time()-t0:.1f}s)')
            except Exception as e:
                summary['dpsgd_sst2'][f'eps_{eps}'] = {'error': str(e)}
                print(f'  Subword DP ε={eps} ERR: {e}')
            with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)

    print(f'\nDone. Wrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
