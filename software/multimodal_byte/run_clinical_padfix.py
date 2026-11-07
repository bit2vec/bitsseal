#!/usr/bin/env python3
"""Padding-bug investigation + fix for clinical Table 9.

BUG: the cross-attention body (D) has no key_padding_mask, and the spin encoder
gives PAD a real bit code (subword's nn.Embedding(padding_idx) gives PAD a zero
vector). So padding leaks non-zero noise into the pooled representation for spin
only, worst where padding is highest (CIViC, K=8 modalities).

FIX: zero the spin embedding at PAD positions to match subword's padding_idx.
--fix {0,1,both}: run without fix, with fix, or both (A/B) for comparison.
Reuses setups + simhash_table + bootstrap_p from run_clinical_sig_simhash.
"""
import argparse, json, os, random, sys, time
import numpy as np, torch, torch.nn as nn, torch.optim as optim
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(THIS, 'paper_modules'))
sys.path.insert(0, os.path.join(THIS, '..', 'Senti_analysis'))
from run_civic_extended import MultiModalClassifier
from run_clinical_sig_simhash import (setup_civic, setup_clinvar, setup_pharmgkb,
                                      simhash_table, bootstrap_p)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SETUPS = {'civic': setup_civic, 'clinvar': setup_clinvar, 'pharmgkb': setup_pharmgkb}


class FixedClf(MultiModalClassifier):
    """Same as MultiModalClassifier but zeros non-standard embeddings at PAD."""
    def set_pad(self, p): self._pad = p; return self
    def forward(self, ids):
        emb = self.embedding(ids)
        if self.method != 'standard':
            emb = emb * (ids != self._pad).unsqueeze(-1).to(emb.dtype)
        return self.fc(self.body(emb))


def train_predict(method, V, nc, pad, trL, teL, seed, epochs, sh_tab, fix):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    base = 'binary_seb' if method == 'simhash-spin' else 'standard'
    Cls = FixedClf if fix else MultiModalClassifier
    model = Cls(V, 64, nc, method=base, body_kind='D', num_bits=64,
                interdim='512', pad_idx=pad).to(DEV)
    if fix: model.set_pad(pad)
    if method == 'simhash-spin':
        with torch.no_grad():
            model.embedding.subword2binary.copy_(sh_tab.to(model.embedding.subword2binary.device))
    opt = optim.Adam(model.parameters(), lr=5e-4); crit = nn.CrossEntropyLoss()
    best = -1.0; final = None
    for _ in range(epochs):
        model.train()
        for b in trL:
            if b is None: continue
            opt.zero_grad(); crit(model(b['ids'].to(DEV)), b['label'].to(DEV)).backward(); opt.step()
        model.eval(); cs = []
        with torch.no_grad():
            for b in teL:
                if b is None: continue
                p = model(b['ids'].to(DEV)).argmax(-1).cpu(); cs.append((p == b['label']).int().numpy())
        c = np.concatenate(cs); best = max(best, float(c.mean())); final = c
    return final, best


def run_setting(task, fix, seeds, epochs):
    trL, teL, V, nc, pad, itos = SETUPS[task]()
    sh = simhash_table(itos)
    per = {'standard': [], 'simhash-spin': []}; bst = {'standard': [], 'simhash-spin': []}
    for m in ('standard', 'simhash-spin'):
        for s in seeds:
            c, ba = train_predict(m, V, nc, pad, trL, teL, s, epochs, sh, fix)
            per[m].append(c); bst[m].append(ba)
            print(f'  [fix={int(fix)}] {task} {m:13s} s{s}: final={c.mean():.4f} best={ba:.4f}', flush=True)
    ca = np.mean(per['standard'], 0); cb = np.mean(per['simhash-spin'], 0)
    p, delta = bootstrap_p(ca, cb)
    return {'fix': int(fix), 'subword_final': round(float(ca.mean()), 4),
            'simhash_final': round(float(cb.mean()), 4), 'delta_final': round(delta, 4),
            'p_value': round(p, 4),
            'subword_best': round(float(np.mean(bst['standard'])), 4),
            'simhash_best': round(float(np.mean(bst['simhash-spin'])), 4),
            'simhash_best_std': round(float(np.std(bst['simhash-spin'])), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=['civic'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--fix', default='both', choices=['0', '1', 'both'])
    ap.add_argument('--out_dir', default='results/multimodal_byte/clinical_padfix')
    args = ap.parse_args(); os.makedirs(args.out_dir, exist_ok=True)
    fixes = [False, True] if args.fix == 'both' else [args.fix == '1']
    out = {'config': {'epochs': args.epochs, 'seeds': args.seeds}}
    for task in args.tasks:
        out[task] = {}
        for fix in fixes:
            r = run_setting(task, fix, args.seeds, args.epochs)
            out[task][f'fix{int(fix)}'] = r
            print(f'  => {task} fix={int(fix)}: subword {r["subword_final"]} vs simhash '
                  f'{r["simhash_final"]} (delta {r["delta_final"]:+}, p={r["p_value"]})', flush=True)
            json.dump(out, open(os.path.join(args.out_dir, 'padfix.json'), 'w'), indent=2)
    print(f'\nWrote {args.out_dir}/padfix.json', flush=True)


if __name__ == '__main__':
    main()
