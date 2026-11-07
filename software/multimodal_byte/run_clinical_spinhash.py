#!/usr/bin/env python3
"""Clinical multimodal re-run with the corrected SpinHash encoder (chunk 1).
Re-evaluates CIViC (and, same pattern, ClinVar/PharmGKB) classification with
body D (cross-attn+MLP), comparing:
  subword        : learned V x d table
  random-spin    : SpinHash, random bit map (= prior 001 / binary_seb)
  simhash-spin   : SpinHash, similarity-preserving SimHash bit map (best on IMDB)
The SimHash variant reuses MultiModalClassifier's binary_seb encoder and simply
overwrites its random subword2binary buffer with a SimHash code table.
"""
import argparse, functools, hashlib, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(THIS, '..', 'Senti_analysis'))
from run_civic_multimodal import (load_civic, MODALITIES, tokenize, build_vocab,
                                  PAD, UNK, SEP, collate, CIViCMultiModalDataset, LABEL_SET)
from run_civic_extended import MultiModalClassifier


def simhash_table(itos, num_bits=64, F_=2048, seed=0):
    rng = np.random.RandomState(seed); R = rng.randn(F_, num_bits).astype('float32')
    tab = torch.zeros(len(itos), num_bits, dtype=torch.long)
    for i, tok in enumerate(itos):
        s = '  ' + str(tok) + '  '; feat = np.zeros(F_, 'float32')
        for j in range(len(s) - 2):
            feat[int(hashlib.md5(s[j:j+3].encode()).hexdigest(), 16) % F_] += 1.0
        tab[i] = torch.tensor(((feat @ R) > 0).astype('int64'))
    return tab


def evaluate(model, loader, dev):
    model.eval(); c = t = 0
    with torch.no_grad():
        for b in loader:
            p = model(b['ids'].to(dev)).argmax(-1).cpu()
            c += (p == b['label']).sum().item(); t += b['label'].size(0)
    return c / max(t, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/multimodal_byte/clinical_spinhash')
    p.add_argument('--methods', nargs='+', default=['standard', 'random-spin', 'simhash-spin'])
    p.add_argument('--body', default='D'); p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--d', type=int, default=64); p.add_argument('--bs', type=int, default=16)
    args = p.parse_args(); os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tr_ex, va_ex, te_ex = load_civic()
    itos_stoi = build_vocab(tr_ex, MODALITIES, boundary='separator')
    stoi = itos_stoi if isinstance(itos_stoi, dict) else itos_stoi[0]
    vmeta = {'mode': 'separator'}
    itos = [None] * len(stoi)
    for k, v in stoi.items():
        itos[v] = k
    V = len(stoi); nc = len(LABEL_SET)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    mk = lambda ex: DataLoader(CIViCMultiModalDataset(ex, stoi, vmeta, MODALITIES, 'all'),
                               batch_size=args.bs, shuffle=False, collate_fn=coll)
    trL = DataLoader(CIViCMultiModalDataset(tr_ex, stoi, vmeta, MODALITIES, 'all'),
                     batch_size=args.bs, shuffle=True, collate_fn=coll)
    teL = mk(te_ex)
    print(f'CIViC vocab={V} classes={nc} tr/te={len(tr_ex)}/{len(te_ex)}')
    sh_tab = simhash_table(itos)

    summary = {}
    for m in args.methods:
        accs = []
        for seed in args.seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            base = 'binary_seb' if m in ('random-spin', 'simhash-spin') else 'standard'
            model = MultiModalClassifier(V, args.d, nc, method=base, body_kind=args.body,
                                         num_bits=64, interdim='512', pad_idx=stoi[PAD]).to(dev)
            if m == 'simhash-spin':   # swap random map -> SimHash map
                with torch.no_grad():
                    model.embedding.subword2binary.copy_(sh_tab.to(model.embedding.subword2binary.device))
            opt = optim.Adam(model.parameters(), lr=5e-4); crit = nn.CrossEntropyLoss()
            best = 0.0
            for _ in range(args.epochs):
                model.train()
                for b in trL:
                    opt.zero_grad(); crit(model(b['ids'].to(dev)), b['label'].to(dev)).backward(); opt.step()
                best = max(best, evaluate(model, teL, dev))
            accs.append(best)
            print(f'  {m:13s} s{seed}: acc={best:.4f}')
        summary[m] = {'mean_acc': round(float(np.mean(accs)), 4),
                      'std_acc': round(float(np.std(accs)), 4), 'accs': accs}
        json.dump(summary, open(os.path.join(args.out_dir, 'civic.json'), 'w'), indent=2)
        print(f'  -> {m}: {np.mean(accs):.4f} +/- {np.std(accs):.4f}')
    print(f'Wrote {args.out_dir}/civic.json')


if __name__ == '__main__':
    main()
