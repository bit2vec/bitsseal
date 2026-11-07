"""Codebook-aware adaptive attack, Random codes vs SpinHash SimHash codes.

Same attacker as adaptive_codebook_attack.py (knows the code table C, the FFN, the head and the
label; DLG gradient inversion through the shared FFN). Only the code ASSIGNMENT RULE changes:
  bits_random  : i.i.d. uniform codes, as in the NeurIPS Bits paper
  bits_simhash : SpinHash byte-trigram SimHash codes of the V most frequent real SST-2 tokens
Everything else is matched to the canonical run: V=1000, d=64, nbits=128, W=256, nc=10, 600 iters.

Metrics per sequence length L:
  recovery  : token-SET recovery |pred ∩ true| / L  (exact tokens, as before)
  hamming   : mean over true tokens of min Hamming distance (fraction of bits) from the recovered
              codes to the true code. 0.5 = unrelated. Small = the attacker landed in the neighborhood
              even when the exact token is wrong.
  hamming_chance : same statistic for a random predicted set of size L.
Usage: python adaptive_spin_attack.py --arm bits_random|bits_simhash --seqlens 1,4,8,16 --trials 30"""
import argparse, hashlib, json, os, sys
import numpy as np, torch
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(THIS, '..', 'healthcare'))
sys.path.insert(0, os.path.join(THIS, '..', 'Senti_analysis'))
from adaptive_codebook_attack import Model, real_grad, dlg_attack, DEV


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


def sst2_top_tokens(V):
    from film_lamp_classification import DATASETS
    from health_classification import build_vocab
    train_ex, _, _, _ = DATASETS['sst2'](seed=42)
    _, itos = build_vocab([t for t, _ in train_ex], min_freq=2)
    return itos[2:2 + V]          # drop PAD, UNK; most frequent first


def hamming_stat(C, pred, true):
    """mean over true tokens of min fraction-of-bits distance to any predicted token."""
    P = C[list(pred)]; T = C[list(true)]          # ±1 codes
    d = (P.unsqueeze(0) != T.unsqueeze(1)).float().mean(-1)   # (|T|, |P|)
    return d.min(1).values.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['bits_random', 'bits_simhash'])
    ap.add_argument('--seqlens', default='1,4,8,16'); ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--V', type=int, default=1000); ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--nbits', type=int, default=128); ap.add_argument('--W', type=int, default=256)
    ap.add_argument('--nc', type=int, default=10); ap.add_argument('--iters', type=int, default=600)
    ap.add_argument('--out', default='adaptive_spin_results.jsonl'); a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    m = Model('bits', a.V, a.d, a.nbits, a.W, a.nc, seed=12345).to(DEV)
    if a.arm == 'bits_simhash':
        itos = sst2_top_tokens(a.V)
        tab = simhash_table(itos, num_bits=a.nbits)
        print(f'simhash codes: V={len(itos)} unique={len(set(map(tuple, tab.tolist())))}', flush=True)
        m.C.copy_((tab.float() * 2 - 1).to(DEV))
    fout = open(a.out, 'a')
    for L in [int(x) for x in a.seqlens.split(',')]:
        recs, hams, hams_ch = [], [], []
        for t in range(a.trials):
            ids = torch.randint(0, a.V, (1, L), device=DEV)
            label = torch.randint(0, a.nc, (1,), device=DEV)
            gs = real_grad(m, ids, label)
            A = dlg_attack(m, gs, label, L, None, iters=a.iters)
            pred = set(A.argmax(-1).tolist()); true = set(ids.view(-1).tolist())
            recs.append(len(pred & true) / len(true))
            hams.append(hamming_stat(m.C, pred, true))
            hams_ch.append(hamming_stat(m.C, set(torch.randint(0, a.V, (L,)).tolist()), true))
        rec = dict(arm=a.arm, seqlen=L, recovery=round(float(np.mean(recs)), 4), chance=round(L / a.V, 5),
                   hamming=round(float(np.mean(hams)), 4), hamming_chance=round(float(np.mean(hams_ch)), 4),
                   nbits=a.nbits, trials=a.trials)
        fout.write(json.dumps(rec) + '\n'); fout.flush()
        print(f"[{a.arm}] L={L:3d}: recovery={rec['recovery']:.4f} (chance {rec['chance']:.4f})  "
              f"hamming={rec['hamming']:.3f} (chance {rec['hamming_chance']:.3f})", flush=True)
    fout.close()


if __name__ == '__main__':
    main()
