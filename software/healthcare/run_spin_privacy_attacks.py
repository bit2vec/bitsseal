#!/usr/bin/env python3
"""FILM + TAG with SpinHash codes, on the NeurIPS Bits protocol.

Reuses run_all_attacks_extended.run_attacks unchanged (same TinyClassifier,
same training, same FILM/TAG attack code). Adds one method:

  spin_simhash : the Bits binary_seb encoder, but its random code table is
                 replaced by the SpinHash byte-trigram SimHash table
                 (same simhash_table as software/multimodal_byte/run_table7_repro.py).

standard and binary_seb run alongside as matched baselines.

Output: results/spin_privacy_attacks/summary.json
"""
import argparse, functools, hashlib, json, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import run_all_attacks_extended as rae
from run_all_attacks_extended import DATASETS, build_vocab, TextClassificationDataset, collate, PAD
from film_lamp_classification import TinyClassifier
import film_lamp_classification as flc

# sentiment sets live in the FILM script's registry, not the extended runner's
for _k in ('yelp', 'imdb', 'sst2'):
    DATASETS.setdefault(_k, flc.DATASETS[_k])


def simhash_table(itos, num_bits=64, F_=2048, seed=0):
    """Verbatim copy of run_table7_repro.simhash_table."""
    rng = np.random.RandomState(seed)
    R = rng.randn(F_, num_bits).astype('float32')
    tab = torch.zeros(len(itos), num_bits, dtype=torch.long)
    for i, tok in enumerate(itos):
        s = '  ' + str(tok) + '  '
        feat = np.zeros(F_, dtype='float32')
        for j in range(len(s) - 2):
            tg = s[j:j+3]
            feat[int(hashlib.md5(tg.encode('utf-8')).hexdigest(), 16) % F_] += 1.0
        proj = feat @ R
        tab[i] = torch.tensor((proj > 0).astype('int64'))
    return tab


def train_target(method, dataset_name, n_epochs=2, batch_size=16, embedding_dim=64, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ex, _, _, num_classes = DATASETS[dataset_name](seed=seed)
    stoi, itos = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    base = 'binary_seb' if method.startswith('spin_') else method
    model = TinyClassifier(vocab_size, embedding_dim, num_classes, method=base)
    if method == 'spin_simhash':
        tab = simhash_table(itos, num_bits=model.embedding.num_bits)
        assert tab.shape == model.embedding.subword2binary.shape
        model.embedding.subword2binary.copy_(tab)
        print(f'  spin_simhash: V={vocab_size}, unique codes={len(set(map(tuple, tab.tolist())))}', flush=True)
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        model.train()
        for batch in loader:
            if batch is None: continue
            opt.zero_grad()
            logits = model(batch['ids'].to(device))
            crit(logits, batch['label'].to(device)).backward()
            opt.step()
    return model, loader, crit, device, vocab_size


rae.train_target = train_target


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+',
                   default=['ade', 'pubmedqa', 'medmcqa', 'finance', 'education',
                            'legal', 'mental', 'yelp', 'imdb', 'sst2'])
    p.add_argument('--methods', nargs='+', default=['standard', 'binary_seb', 'spin_simhash'])
    p.add_argument('--out_dir', default='results/spin_privacy_attacks')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, 'summary.json')
    summary = json.load(open(out)) if os.path.exists(out) else {}
    for ds in args.datasets:
        summary.setdefault(ds, {})
        for m in args.methods:
            if m in summary[ds] and 'film' in summary[ds][m]:
                continue
            t0 = time.time()
            summary[ds][m] = rae.run_attacks(ds, m)
            summary[ds][m]['time_s'] = time.time() - t0
            with open(out, 'w') as f:
                json.dump(summary, f, indent=2)
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
