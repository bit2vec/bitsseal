#!/usr/bin/env python3
"""Settle clinical Table 9: Subword vs SpinHash-SimHash (simhash-spin), body D,
d=64, interdim=512, on CIViC / ClinVar / PharmGKB.

Matches the paper's *recommended* Spin-SimHash variant (bit map = SimHash of
token byte-trigrams, reusing run_clinical_spinhash.simhash_table), NOT binary_seb,
and the Table-9 25-epoch config (not the 6ep-undertrained significance default).

Protocol: train `epochs` epochs, then take per-example test correctness at the
FINAL epoch (no best-test-epoch peeking), over N seeds; paired bootstrap (10k)
on seed-averaged per-example correctness -> two-sided p per task. Also logs the
best-over-epochs test acc for reference/comparison with the paper.
"""
import argparse, functools, json, os, random, sys, time
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(THIS, 'paper_modules'))
sys.path.insert(0, os.path.join(THIS, '..', 'Senti_analysis'))
from run_civic_multimodal import (load_civic, build_vocab, tokenize, PAD, UNK, SEP,
                                  collate, CIViCMultiModalDataset,
                                  LABEL_SET as CIVIC_LABELS, MODALITIES as CIVIC_MOD)
from run_civic_extended import MultiModalClassifier
from clinvar_loader import load_clinvar
from run_clinvar_multimodal import (MODALITIES as CV_MOD, CVMultiModalDataset,
                                    extract_gene_from_variant)
from pharmgkb_loader import load_pharmgkb
from run_pharmgkb_multimodal import MODALITIES as PG_MOD, PGKBDataset
from run_significance import budgets, build_stoi, bootstrap_p
from run_clinical_spinhash import simhash_table

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def itos_from(stoi):
    return [t for t, _ in sorted(stoi.items(), key=lambda kv: kv[1])]


def setup_civic():
    tr, va, te = load_civic()
    iss = build_vocab(tr, CIVIC_MOD, boundary='separator')
    stoi = iss if isinstance(iss, dict) else iss[0]
    vmeta = {'mode': 'separator'}; coll = functools.partial(collate, pad_index=stoi[PAD])
    mk = lambda ex, sh: DataLoader(CIViCMultiModalDataset(ex, stoi, vmeta, CIVIC_MOD, 'all'),
                                   batch_size=16, shuffle=sh, collate_fn=coll)
    return mk(tr, True), mk(te, False), len(stoi), len(CIVIC_LABELS), stoi[PAD], itos_from(stoi)


def setup_clinvar():
    ex, _ = load_clinvar(n_target=3000)
    for e in ex:
        if not e.get('gene'): e['gene'] = extract_gene_from_variant(e.get('variant', ''))
    top3 = ['Pathogenic', 'Likely benign', 'Uncertain significance']
    ex = [e for e in ex if e['label'] in top3]
    l2i = {l: i for i, l in enumerate(top3)}
    for e in ex: e['label_idx'] = l2i[e['label']]
    random.Random(42).shuffle(ex)
    g = lambda e, m: str(e.get(m, '')); mods = budgets(ex, CV_MOD, g)
    n = len(ex); ntr = int(.7*n); nva = int(.1*n); tr, te = ex[:ntr], ex[ntr+nva:]
    stoi = build_stoi(tr, mods, g); vmeta = {'mode': 'separator'}
    coll = functools.partial(collate, pad_index=stoi[PAD])
    mk = lambda d, sh: DataLoader(CVMultiModalDataset(d, stoi, vmeta, mods, 'all'),
                                  batch_size=16, shuffle=sh, collate_fn=coll)
    return mk(tr, True), mk(te, False), len(stoi), len(top3), stoi[PAD], itos_from(stoi)


def setup_pharmgkb():
    ex, label_set = load_pharmgkb(seed=42)
    g = lambda e, m: str(e.get(m, '')); mods = budgets(ex, PG_MOD, g)
    n = len(ex); ntr = int(.7*n); nva = int(.1*n); tr, te = ex[:ntr], ex[ntr+nva:]
    stoi = build_stoi(tr, mods, g)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    mk = lambda d, sh: DataLoader(PGKBDataset(d, stoi, mods), batch_size=16, shuffle=sh, collate_fn=coll)
    return mk(tr, True), mk(te, False), len(stoi), len(label_set), stoi[PAD], itos_from(stoi)


def train_predict(method, V, nc, pad, trL, teL, seed, epochs, sh_tab=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    base = 'binary_seb' if method == 'simhash-spin' else 'standard'
    model = MultiModalClassifier(V, 64, nc, method=base, body_kind='D',
                                 num_bits=64, interdim='512', pad_idx=pad).to(DEV)
    if method == 'simhash-spin':
        with torch.no_grad():
            model.embedding.subword2binary.copy_(sh_tab.to(model.embedding.subword2binary.device))
    opt = optim.Adam(model.parameters(), lr=5e-4); crit = nn.CrossEntropyLoss()

    def per_example():
        model.eval(); cs = []
        with torch.no_grad():
            for b in teL:
                if b is None: continue
                p = model(b['ids'].to(DEV)).argmax(-1).cpu()
                cs.append((p == b['label']).int().numpy())
        return np.concatenate(cs)

    best_acc = -1.0
    final_c = None
    for _ in range(epochs):
        model.train()
        for b in trL:
            if b is None: continue
            opt.zero_grad(); crit(model(b['ids'].to(DEV)), b['label'].to(DEV)).backward(); opt.step()
        final_c = per_example()
        best_acc = max(best_acc, float(final_c.mean()))
    return final_c, best_acc   # final-epoch per-example correctness, best-over-epochs acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='results/multimodal_byte/clinical_sig_simhash_25ep')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--tasks', nargs='+', default=['civic', 'clinvar', 'pharmgkb'])
    args = ap.parse_args(); os.makedirs(args.out_dir, exist_ok=True)
    setups = {'civic': setup_civic, 'clinvar': setup_clinvar, 'pharmgkb': setup_pharmgkb}
    out = {'config': {'epochs': args.epochs, 'seeds': args.seeds, 'variant': 'simhash-spin',
                      'd': 64, 'interdim': 512, 'body': 'D', 'eval': 'final-epoch'}}
    for task in args.tasks:
        t0 = time.time()
        print(f'\n===== {task} =====', flush=True)
        trL, teL, V, nc, pad, itos = setups[task]()
        sh_tab = simhash_table(itos)
        print(f'  vocab={V} classes={nc} test={len(teL.dataset)}', flush=True)
        per = {'standard': [], 'simhash-spin': []}
        best = {'standard': [], 'simhash-spin': []}
        for m in ('standard', 'simhash-spin'):
            for s in args.seeds:
                c, ba = train_predict(m, V, nc, pad, trL, teL, s, args.epochs, sh_tab=sh_tab)
                per[m].append(c); best[m].append(ba)
                print(f'  {task} {m:13s} s{s}: final_acc={c.mean():.4f} best_acc={ba:.4f}', flush=True)
        ca = np.mean(per['standard'], 0); cb = np.mean(per['simhash-spin'], 0)
        p, delta = bootstrap_p(ca, cb)
        mark = '\\ddagger' if p < 0.01 else ('\\dagger' if p < 0.05 else '')
        out[task] = {
            'subword_final': round(float(ca.mean()), 4),
            'simhash_final': round(float(cb.mean()), 4),
            'delta_final': round(delta, 4), 'p_value': round(p, 4), 'marker': mark,
            'subword_best_mean': round(float(np.mean(best['standard'])), 4),
            'simhash_best_mean': round(float(np.mean(best['simhash-spin'])), 4),
            'subword_best_std': round(float(np.std(best['standard'])), 4),
            'simhash_best_std': round(float(np.std(best['simhash-spin'])), 4),
            'n_test': int(len(ca)), 'seeds': args.seeds, 'minutes': round((time.time()-t0)/60, 1)}
        print(f'  -> {task}: subword {out[task]["subword_final"]} vs simhash {out[task]["simhash_final"]} '
              f'(delta {out[task]["delta_final"]:+}, p={out[task]["p_value"]}{" "+mark if mark else " n.s."})', flush=True)
        json.dump(out, open(os.path.join(args.out_dir, 'results.json'), 'w'), indent=2)
    print(f'\nWrote {args.out_dir}/results.json', flush=True)


if __name__ == '__main__':
    main()
