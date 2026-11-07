#!/usr/bin/env python3
"""Significance test for Table 1: Subword vs SpinHash (simhash_seb), body D,
on the 3 multimodal clinical tasks (CIViC, ClinVar, PharmGKB). Trains each
(method, seed) over 5 seeds, dumps per-example test correctness, then a
paired bootstrap (10k) on seed-averaged per-example correctness gives a
two-sided p-value per task. Markers: dagger p<0.05, ddagger p<0.01.
"""
import argparse, functools, json, os, random, sys, time
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader

THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, THIS)
from run_civic_multimodal import (load_civic, build_vocab, tokenize,
                                  PAD, UNK, SEP, collate, CIViCMultiModalDataset,
                                  LABEL_SET as CIVIC_LABELS)
from run_civic_multimodal import MODALITIES as CIVIC_MOD
from run_civic_extended import MultiModalClassifier
from clinvar_loader import load_clinvar
from run_clinvar_multimodal import (MODALITIES as CV_MOD, CVMultiModalDataset,
                                    extract_gene_from_variant)
from pharmgkb_loader import load_pharmgkb
from run_pharmgkb_multimodal import MODALITIES as PG_MOD, PGKBDataset

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def budgets(examples, modalities, getter):
    out = []
    for mod, default in modalities:
        lens = [len(tokenize(getter(ex, mod))) for ex in examples]
        m = max(lens) if lens else 1
        p95 = max(int(np.percentile(lens, 95)) if lens else 1, 1)
        b = m if m <= 1.5 * p95 else p95
        out.append((mod, max(min(b, default) if default else b, 1)))
    return out


def build_stoi(train_ex, mods, getter):
    counter = Counter()
    for ex in train_ex:
        for mod, _ in mods:
            counter.update(tokenize(getter(ex, mod)))
    stoi = {PAD: 0, UNK: 1}
    for mod, _ in mods: stoi[f'{SEP}_{mod}'] = len(stoi)
    for tok, c in counter.most_common():
        if c >= 2: stoi[tok] = len(stoi)
    return stoi


def setup_civic():
    tr, va, te = load_civic()
    iss = build_vocab(tr, CIVIC_MOD, boundary='separator')
    stoi = iss if isinstance(iss, dict) else iss[0]
    vmeta = {'mode': 'separator'}
    coll = functools.partial(collate, pad_index=stoi[PAD])
    mk = lambda ex, sh: DataLoader(CIViCMultiModalDataset(ex, stoi, vmeta, CIVIC_MOD, 'all'),
                                   batch_size=16, shuffle=sh, collate_fn=coll)
    return mk(tr, True), mk(te, False), len(stoi), len(CIVIC_LABELS), stoi[PAD]


def setup_clinvar():
    ex, _ = load_clinvar(n_target=3000)
    for e in ex:
        if not e.get('gene'): e['gene'] = extract_gene_from_variant(e.get('variant', ''))
    top3 = ['Pathogenic', 'Likely benign', 'Uncertain significance']
    ex = [e for e in ex if e['label'] in top3]
    l2i = {l: i for i, l in enumerate(top3)}
    for e in ex: e['label_idx'] = l2i[e['label']]
    random.Random(42).shuffle(ex)
    g = lambda e, m: str(e.get(m, ''))
    mods = budgets(ex, CV_MOD, g)
    n = len(ex); ntr = int(.7*n); nva = int(.1*n)
    tr, te = ex[:ntr], ex[ntr+nva:]
    stoi = build_stoi(tr, mods, g); vmeta = {'mode': 'separator'}
    coll = functools.partial(collate, pad_index=stoi[PAD])
    mk = lambda d, sh: DataLoader(CVMultiModalDataset(d, stoi, vmeta, mods, 'all'),
                                  batch_size=16, shuffle=sh, collate_fn=coll)
    return mk(tr, True), mk(te, False), len(stoi), len(top3), stoi[PAD]


def setup_pharmgkb():
    ex, label_set = load_pharmgkb(seed=42)
    g = lambda e, m: str(e.get(m, ''))
    mods = budgets(ex, PG_MOD, g)
    n = len(ex); ntr = int(.7*n); nva = int(.1*n)
    tr, te = ex[:ntr], ex[ntr+nva:]
    stoi = build_stoi(tr, mods, g)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    mk = lambda d, sh: DataLoader(PGKBDataset(d, stoi, mods),
                                  batch_size=16, shuffle=sh, collate_fn=coll)
    return mk(tr, True), mk(te, False), len(stoi), len(label_set), stoi[PAD]


def train_predict(method, V, nc, pad, trL, teL, seed, epochs=6):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model = MultiModalClassifier(V, 64, nc, method=method, body_kind='D', pad_idx=pad).to(DEV)
    opt = optim.Adam(model.parameters(), lr=5e-4); crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for b in trL:
            if b is None: continue
            opt.zero_grad(); crit(model(b['ids'].to(DEV)), b['label'].to(DEV)).backward(); opt.step()
    model.eval(); correct, labels = [], []
    with torch.no_grad():
        for b in teL:
            if b is None: continue
            p = model(b['ids'].to(DEV)).argmax(-1).cpu()
            correct.append((p == b['label']).int().numpy()); labels.append(b['label'].numpy())
    return np.concatenate(correct)


def bootstrap_p(c_a, c_b, n_boot=10000, seed=0):
    rng = np.random.RandomState(seed); n = len(c_a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        diffs.append(c_b[idx].mean() - c_a[idx].mean())
    diffs = np.array(diffs)
    frac_pos = (diffs > 0).mean()
    p = 2 * min(frac_pos, 1 - frac_pos)
    return float(p), float(c_b.mean() - c_a.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='results/multimodal_byte/significance')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--tasks', nargs='+', default=['civic', 'clinvar', 'pharmgkb'])
    args = ap.parse_args(); os.makedirs(args.out_dir, exist_ok=True)
    setups = {'civic': setup_civic, 'clinvar': setup_clinvar, 'pharmgkb': setup_pharmgkb}
    out = {}
    for task in args.tasks:
        print(f'\n===== {task} =====', flush=True)
        trL, teL, V, nc, pad = setups[task]()
        print(f'  vocab={V} classes={nc} test={len(teL.dataset)}', flush=True)
        per = {'standard': [], 'simhash_seb': []}
        for m in ('standard', 'simhash_seb'):
            for s in args.seeds:
                c = train_predict(m, V, nc, pad, trL, teL, s, args.epochs)
                per[m].append(c); print(f'  {task} {m:11s} s{s}: acc={c.mean():.4f}', flush=True)
        ca = np.mean(per['standard'], 0); cb = np.mean(per['simhash_seb'], 0)
        p, delta = bootstrap_p(ca, cb)
        mark = '\\ddagger' if p < 0.01 else ('\\dagger' if p < 0.05 else '')
        sw_seed = [round(float(c.mean()), 4) for c in per['standard']]
        sh_seed = [round(float(c.mean()), 4) for c in per['simhash_seb']]
        out[task] = {'subword_acc': round(float(ca.mean()), 4),
                     'spinhash_acc': round(float(cb.mean()), 4),
                     'subword_std': round(float(np.std([c.mean() for c in per['standard']])), 4),
                     'spinhash_std': round(float(np.std([c.mean() for c in per['simhash_seb']])), 4),
                     'subword_seed_accs': sw_seed, 'spinhash_seed_accs': sh_seed,
                     'delta': round(delta, 4), 'p_value': round(p, 4), 'marker': mark,
                     'n_test': int(len(ca)), 'seeds': args.seeds}
        json.dump(out, open(os.path.join(args.out_dir, 'significance.json'), 'w'), indent=2)
        print(f'  -> {task}: subword={ca.mean():.4f} spinhash={cb.mean():.4f} '
              f'delta={delta:+.4f} p={p:.4f} {mark}', flush=True)
    print('\nWrote', os.path.join(args.out_dir, 'significance.json'))


if __name__ == '__main__':
    main()
