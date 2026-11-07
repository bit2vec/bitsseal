#!/usr/bin/env python3
"""SpinHash n-gram sweep. Clones the canonical harnesses and changes ONLY the
byte n-gram set used by SimHash (paper: n=3). Each n-gram is padded with n-1
spaces per side (paper: two spaces for n=3) and bucketed with md5 into F=2048.

  --harness imdb     run_table7_repro.py, method bits_hash_mlp512 (paper Table 12 text setup)
  --harness massive  run_bits_variants.py, method simhash_bits (MASSIVE, 60 intents)
  --ns 3 | 2,3,4     single n or mixed set
  --weight none|pern|idf  how mixed orders are combined
"""
import argparse, hashlib, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def grams(tok, n):
    s = ' ' * (n - 1) + str(tok) + ' ' * (n - 1)
    return [s[j:j + n] for j in range(len(s) - n + 1)]


def bucket(g, F_):
    return int(hashlib.md5(g.encode('utf-8')).hexdigest(), 16) % F_


def ngram_simhash(itos, ns, num_bits=64, F_=2048, seed=0, weight='none'):
    """weight: none = raw counts summed over n (fastText style);
    pern = each n's count vector L2-normalised before summing (equal weight per order);
    idf  = each n-gram weighted by log(V / number of vocab tokens containing it)."""
    rng = np.random.RandomState(seed)
    R = rng.randn(F_, num_bits).astype('float32')      # same hyperplanes as canonical
    if weight == 'idf':
        df = {}
        for tok in itos:
            for g in {g for n in ns for g in grams(tok, n)}:
                df[g] = df.get(g, 0) + 1
        V = len(itos)
    tab = torch.zeros(len(itos), num_bits, dtype=torch.long)
    for i, tok in enumerate(itos):
        feat = np.zeros(F_, dtype='float32')
        for n in ns:
            fn = np.zeros(F_, dtype='float32')
            for g in grams(tok, n):
                fn[bucket(g, F_)] += np.log(V / df[g]) if weight == 'idf' else 1.0
            if weight == 'pern' and fn.any():
                fn /= np.linalg.norm(fn)
            feat += fn
        tab[i] = torch.tensor(((feat @ R) > 0).astype('int64'))
    return tab


if __name__ == '__main__':
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--harness', required=True, choices=['imdb', 'massive'])
    pre.add_argument('--ns', required=True)
    pre.add_argument('--weight', default='none', choices=['none', 'pern', 'idf'])
    a, rest = pre.parse_known_args()
    ns = [int(x) for x in a.ns.split(',')]
    print(f'ngram set {ns} weight {a.weight}', flush=True)
    if a.harness == 'imdb':
        import run_table7_repro as H
        H.simhash_table = lambda itos, num_bits=64, F_=2048, seed=0: ngram_simhash(itos, ns, num_bits, F_, seed, a.weight)
        sys.argv = [sys.argv[0], '--methods', 'bits_hash_mlp512'] + rest
    else:
        import run_bits_variants as H
        H.t_simhash_bits = lambda itos, F_=2048, seed=0: ngram_simhash(itos, ns, H.NB, F_, seed, a.weight)
        sys.argv = [sys.argv[0], '--methods', 'simhash_bits'] + rest
    H.main()
