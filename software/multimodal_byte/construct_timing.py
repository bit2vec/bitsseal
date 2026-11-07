#!/usr/bin/env python3
"""Table 1: one-time construction cost by hash assignment rule (CPU).
Times Spin-SimHash (O(V), byte-trigram hash), Spin-Random (O(V)), and
Spin-Repulse (O(V^2): dense V x V co-occurrence + antiferromagnetic Ising)
across vocabulary sizes. Repulse is skipped where the V x V matrix is infeasible.
Reuses repulse_table() from run_table7_repro.py (the real paper construction).
"""
import sys, os, time, hashlib, json, argparse
import numpy as np, torch
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(THIS, 'paper_modules'))
from run_table7_repro import repulse_table


def simhash_build(itos, num_bits=64, F_=2048, seed=0):
    rng = np.random.RandomState(seed); R = rng.randn(F_, num_bits).astype('float32')
    tab = torch.zeros(len(itos), num_bits, dtype=torch.long)
    for i, tok in enumerate(itos):
        s = '  ' + str(tok) + '  '; feat = np.zeros(F_, 'float32')
        for j in range(len(s) - 2):
            feat[int(hashlib.md5(s[j:j + 3].encode()).hexdigest(), 16) % F_] += 1.0
        tab[i] = torch.tensor(((feat @ R) > 0).astype('int64'))
    return tab


def random_build(V, num_bits=64, seed=0):
    # matches BinarySubwordConcateOnehot._create_unique_mapping (the paper's Random)
    torch.manual_seed(seed)
    mapping = torch.randint(0, 2, (V, num_bits), dtype=torch.long)
    seen = set()
    for i in range(V):
        seq = tuple(mapping[i].tolist()); attempts = 0
        while seq in seen and attempts < 1000:
            mapping[i] = torch.randint(0, 2, (num_bits,), dtype=torch.long)
            seq = tuple(mapping[i].tolist()); attempts += 1
        seen.add(seq)
    return mapping


def synth_corpus(V, n_docs=2000, doc_len=256, seed=0):
    rng = np.random.RandomState(seed)
    return [list(rng.randint(0, V, doc_len)) for _ in range(n_docs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocabs', nargs='+', type=int, default=[1000, 10000, 30000, 100000])
    ap.add_argument('--repulse_max_v', type=int, default=30000, help='skip Repulse above this (V^2 memory)')
    ap.add_argument('--out_dir', default='results/multimodal_byte/construct_timing')
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    out = {}
    for V in a.vocabs:
        itos = [f'tok{i}_{i%97}' for i in range(V)]  # varied strings for SimHash trigrams
        t = {}
        t0 = time.time(); simhash_build(itos); t['SimHash'] = round(time.time() - t0, 3)
        t0 = time.time(); random_build(V); t['Random'] = round(time.time() - t0, 3)
        if V <= a.repulse_max_v:
            toks = synth_corpus(V)
            t0 = time.time(); repulse_table(itos, toks, num_bits=64, beta=0.4, sweeps=12, seed=0)
            t['Repulse'] = round(time.time() - t0, 3)
        else:
            gb = (V * V * 4) / 1e9
            t['Repulse'] = f'~{gb:.0f} GB V x V - infeasible'
        out[V] = t
        print(f'V={V:7d}  SimHash={t["SimHash"]}s  Random={t["Random"]}s  Repulse={t["Repulse"]}', flush=True)
        json.dump(out, open(os.path.join(a.out_dir, 'construct_timing.json'), 'w'), indent=2)
    print(f'Wrote {a.out_dir}/construct_timing.json')


if __name__ == '__main__':
    main()
