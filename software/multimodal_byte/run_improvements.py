#!/usr/bin/env python3
"""SpinHash improvements + byte-level baseline testbed (IMDB + SST-2, 3 seeds).

Reuses the EXACT Model/DS/training loop and code builders from run_table7_repro
(one-axis discipline: only the code mechanism changes). Adds:
  #1 multi-channel : M independent code tables concatenated (variance reduction).
  #2 semantic-SimHash : SimHash over hashed co-occurrence (context) features,
       instead of byte trigrams.
  #3 spin_contrast : LearnedHash + structured code loss
       (attract surface-similar pairs, repel context-confusable pairs,
        + bit balance + decorrelation).
  + SST-2 loader (for the CANINE/ByT5 head-to-head).
"""
import argparse, os, json, time, hashlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from collections import Counter, defaultdict
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_table7_repro import (basic_english, simhash_table, repulse_table,
                              LearnedHash, make_embed, Model, DS, coll,
                              BinarySubwordConcateOnehot)

DATASETS = {  # (hf, cfg, text_field, label_field, n_classes, test_split, max_len)
    'imdb': ('stanfordnlp/imdb', None, 'text', 'label', 2, 'test', 256),
    'sst2': ('stanfordnlp/sst2', None, 'sentence', 'label', 2, 'validation', 64),
    'agnews': ('fancyzhx/ag_news', None, 'text', 'label', 4, 'test', 128),
    'dbpedia': ('fancyzhx/dbpedia_14', None, 'content', 'label', 14, 'test', 256),
    'yelp': ('fancyzhx/yelp_polarity', None, 'text', 'label', 2, 'test', 256),
    'amazon': ('fancyzhx/amazon_polarity', None, 'content', 'label', 2, 'test', 256),
}


# ---------- #1 multi-channel ----------
def random_table(itos, num_bits, seed):
    rng = np.random.RandomState(10_000 + seed)
    return torch.tensor(rng.randint(0, 2, size=(len(itos), num_bits)), dtype=torch.long)


def mc_table(base, itos, train_toks, M, num_bits=64, seed0=0, repulse_beta=0.4):
    cols = []
    for m in range(M):
        if base == 'random':
            cols.append(random_table(itos, num_bits, seed0 + m))
        elif base == 'simhash':
            cols.append(simhash_table(itos, num_bits=num_bits, seed=seed0 + m))
        elif base == 'repulse':
            cols.append(repulse_table(itos, train_toks, num_bits=num_bits,
                                      beta=repulse_beta, seed=seed0 + m))
    return torch.cat(cols, dim=1)  # [V, M*num_bits]


# ---------- #2 semantic SimHash ----------
def simhash_semantic_table(itos, train_toks, num_bits=64, F_=2048, window=5, seed=0,
                           cache=None):
    if cache is not None and os.path.isfile(cache):
        feat = np.load(cache)
    else:
        V = len(itos); stoi = {w: i for i, w in enumerate(itos)}
        feat = np.zeros((V, F_), dtype=np.float32)
        for toks in train_toks:
            ids = [stoi.get(t, 1) for t in toks]
            n = len(ids)
            for i, wi in enumerate(ids):
                lo, hi = max(0, i - window), min(n, i + window + 1)
                for j in range(lo, hi):
                    if j == i:
                        continue
                    feat[wi, (ids[j] * 2654435761) % F_] += 1.0
        feat = np.log1p(feat)
        nrm = np.linalg.norm(feat, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
        feat /= nrm
        if cache is not None:
            np.save(cache, feat)
    R = np.random.RandomState(seed).randn(F_, num_bits).astype('float32')
    return torch.tensor((feat @ R > 0).astype('int64'), dtype=torch.long)


# ---------- #3 Spin-Contrast pairs ----------
def build_pairs(itos, train_toks, n_pos=20000, n_neg=20000, seed=0, vmax=8000):
    """P+ : tokens sharing many byte-trigrams (surface). P- : top signed-PMI
    context neighbours (interchangeable) -- same signal Spin-Repulse uses."""
    from scipy import sparse
    rng = np.random.RandomState(seed)
    V = len(itos); stoi = {t: i for i, t in enumerate(itos)}
    use = min(vmax, V)  # mine pairs over the most frequent tokens only
    # --- P+ surface: trigram inverted index ---
    tg2tok = defaultdict(list)
    for i in range(use):
        s = '  ' + str(itos[i]) + '  '
        for j in range(len(s) - 2):
            tg2tok[s[j:j + 3]].append(i)
    pos = []
    keys = [k for k, v in tg2tok.items() if 2 <= len(v) <= 200]
    rng.shuffle(keys)
    for k in keys:
        toks = tg2tok[k]
        for _ in range(min(3, len(toks))):
            a, b = rng.choice(toks, 2, replace=False)
            if a != b:
                pos.append((a, b))
        if len(pos) >= n_pos:
            break
    # --- P- context: signed-PMI top-k (doc co-occurrence over a doc sample) ---
    docs = train_toks[:6000]
    rows, cols = [], []
    for d, toks in enumerate(docs):
        for tid in {stoi.get(x, 1) for x in toks}:
            if tid < use:
                rows.append(d); cols.append(tid)
    X = sparse.csr_matrix((np.ones(len(rows), 'float32'), (rows, cols)),
                          shape=(len(docs), use))
    C = (X.T @ X).toarray().astype('float32')
    tot = C.sum() + 1e-8; pi = C.sum(1, keepdims=True) / tot; Pij = C / tot
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log((Pij + 1e-12) / (pi @ pi.T + 1e-12))
    np.fill_diagonal(pmi, -1e9)
    k = 8
    nbr = np.argpartition(-pmi, k, axis=1)[:, :k]
    neg = []
    for i in range(use):
        for j in nbr[i]:
            if pmi[i, j] > 0:
                neg.append((i, int(j)))
    rng.shuffle(neg); neg = neg[:n_neg]
    if not pos:
        pos = [(0, 1)]
    if not neg:
        neg = [(0, 1)]
    return np.array(pos[:n_pos], dtype=np.int64), np.array(neg, dtype=np.int64)


class SpinContrastHash(LearnedHash):
    def __init__(self, itos, emb_dim, train_toks, num_bits=64, interdim=512,
                 margin=0.30, tau=0.5, lam_pos=1.0, lam_neg=1.0, lam_bal=0.1,
                 lam_dec=0.05, sample=4096, seed=0):
        super().__init__(itos, emb_dim, num_bits=num_bits, interdim=interdim)
        self.tau = tau; self.margin = margin; self.sample = sample
        self.lam = (lam_pos, lam_neg, lam_bal, lam_dec)
        pos, neg = build_pairs(itos, train_toks, seed=seed)
        self.register_buffer('pos', torch.tensor(pos, dtype=torch.long))
        self.register_buffer('neg', torch.tensor(neg, dtype=torch.long))

    def _soft(self, ids):  # ids: 1D unique token ids -> [n, B] soft spins
        b = self.btab[ids]
        x = self.byte(b).view(ids.size(0), -1)
        return torch.tanh(self.hashnet(x) / self.tau)

    def aux_loss(self):
        dev = self.hashnet.weight.device
        def samp(pairs):
            idx = torch.randint(0, pairs.size(0), (min(self.sample, pairs.size(0)),), device=dev)
            return pairs[idx]
        pp, nn_ = samp(self.pos), samp(self.neg)
        uniq, inv = torch.unique(torch.cat([pp.reshape(-1), nn_.reshape(-1)]), return_inverse=True)
        s = self._soft(uniq)                       # [U, B]
        B = s.size(1)
        ip = inv[:pp.numel()].view(-1, 2); ineg = inv[pp.numel():].view(-1, 2)
        def dist(pidx):
            a = s[pidx[:, 0]]; b = s[pidx[:, 1]]
            return 0.5 * (1 - (a * b).sum(1) / B)
        lpos = dist(ip).mean()
        lneg = F.relu(self.margin - dist(ineg)).mean()
        sbar = s.mean(0); lbal = (sbar ** 2).mean()
        sc = s - s.mean(0, keepdim=True)
        cov = (sc.t() @ sc) / s.size(0)
        ldec = (cov - torch.diag(torch.diag(cov))).pow(2).mean()
        lp, ln, lb, ld = self.lam
        return lp * lpos + ln * lneg + lb * lbal + ld * ldec


# ---------- embedding dispatch ----------
def build_embed(method, V, d, itos, train_toks, tmpdir, seed, sem_cache):
    t = lambda n: os.path.join(tmpdir, n)
    wrap = lambda tab, nb: (torch.save(tab, t('tab.pt')),
                            BinarySubwordConcateOnehot(num_embeddings=V, embedding_dim=d,
                                                       num_bits=nb, interdim='512',
                                                       subword_binary_file=t('tab.pt')))[1]
    if method in ('standard', 'spin_random', 'spin_simhash', 'spin_repulse'):
        m = {'standard': 'standard', 'spin_random': 'bits_mlp512',
             'spin_simhash': 'bits_hash_mlp512', 'spin_repulse': 'bits_repulse_mlp512'}[method]
        return make_embed(m, V, d, tmpdir, itos=itos, train_toks=train_toks, seed=seed)
    if method == 'random_mc2': return wrap(mc_table('random', itos, train_toks, 2, seed0=seed), 128)
    if method == 'random_mc4': return wrap(mc_table('random', itos, train_toks, 4, seed0=seed), 256)
    if method == 'repulse_mc2': return wrap(mc_table('repulse', itos, train_toks, 2, seed0=seed), 128)
    if method == 'simhash_sem':
        return wrap(simhash_semantic_table(itos, train_toks, seed=seed, cache=sem_cache), 64)
    if method == 'spin_contrast':
        return SpinContrastHash(itos, d, train_toks, seed=seed)
    raise ValueError(method)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--datasets', nargs='+', default=['imdb', 'sst2'])
    p.add_argument('--methods', nargs='+', default=[
        'standard', 'spin_random', 'spin_simhash', 'spin_repulse',
        'random_mc2', 'random_mc4', 'repulse_mc2', 'simhash_sem', 'spin_contrast'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--d', type=int, default=256)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--min_freq', type=int, default=5)
    p.add_argument('--n_train', type=int, default=25000)
    p.add_argument('--aux_w', type=float, default=0.3)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args(); os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from datasets import load_dataset

    for ds in args.datasets:
        hf, cfg, tf, lf, NC, tesp, ml = DATASETS[ds]
        if args.smoke:
            ml = 64
        tr_raw = load_dataset(hf, cfg, split='train').shuffle(seed=args.seeds[0]) if cfg \
            else load_dataset(hf, split='train').shuffle(seed=args.seeds[0])
        te_raw = load_dataset(hf, cfg, split=tesp) if cfg else load_dataset(hf, split=tesp)
        ntr = 1500 if args.smoke else args.n_train
        nte = 800 if args.smoke else 25000
        tr = [(basic_english(r[tf]), int(r[lf])) for r in tr_raw.select(range(min(ntr, len(tr_raw))))]
        te = [(basic_english(r[tf]), int(r[lf])) for r in te_raw.select(range(min(nte, len(te_raw))))]
        cnt = Counter(w for toks, _ in tr for w in toks)
        itos = ['<pad>', '<unk>'] + [w for w, c in cnt.most_common() if c >= args.min_freq]
        stoi = {w: i for i, w in enumerate(itos)}; V = len(itos)
        train_toks = [x[0] for x in tr]
        trL = DataLoader(DS(tr, stoi, ml), batch_size=args.bs, shuffle=True, collate_fn=coll)
        teL = DataLoader(DS(te, stoi, ml), batch_size=args.bs, collate_fn=coll)
        print(f'[{ds}] train={len(tr)} test={len(te)} vocab={V} classes={NC} max_len={ml}')
        out = os.path.join(args.out_dir, f'{ds}.json')
        summary = json.load(open(out)) if os.path.isfile(out) else {}
        tmproot = os.path.join(args.out_dir, f'maps_{ds}'); os.makedirs(tmproot, exist_ok=True)
        sem_cache = os.path.join(tmproot, 'sem_feat.npy')

        for m in (args.methods if not args.smoke else ['standard', 'spin_random', 'spin_contrast']):
            accs, embp = [], None
            for seed in args.seeds:
                torch.manual_seed(seed); np.random.seed(seed)
                stmp = os.path.join(tmproot, f'{m}_s{seed}'); os.makedirs(stmp, exist_ok=True)
                try:
                    emb = build_embed(m, V, args.d, itos, train_toks, stmp, seed, sem_cache)
                except Exception as e:
                    print(f'  {m} s{seed}: SKIP ({type(e).__name__}: {str(e)[:80]})'); continue
                model = Model(emb, d=args.d, nc=NC).to(dev)
                embp = sum(pp.numel() for pp in model.emb.parameters())
                opt = optim.Adam(model.parameters(), lr=args.lr); crit = nn.CrossEntropyLoss()
                t0 = time.time(); best = 0.0
                has_aux = hasattr(model.emb, 'aux_loss')
                ep_range = 1 if args.smoke else args.epochs
                for ep in range(ep_range):
                    model.train()
                    for b in trL:
                        opt.zero_grad()
                        loss = crit(model(b['ids'].to(dev)), b['y'].to(dev))
                        if has_aux:
                            loss = loss + args.aux_w * model.emb.aux_loss()
                        loss.backward(); opt.step()
                    model.eval(); c = n = 0
                    with torch.no_grad():
                        for b in teL:
                            pr = model(b['ids'].to(dev)).argmax(-1).cpu()
                            c += (pr == b['y']).sum().item(); n += b['y'].size(0)
                    best = max(best, c / n)
                accs.append(best)
                print(f'  {m:16s} s{seed}: acc={best:.4f} emb={embp} [{time.time()-t0:.0f}s]', flush=True)
            if accs:
                summary[m] = {'mean': round(float(np.mean(accs)), 4),
                              'std': round(float(np.std(accs)), 4),
                              'accs': [round(a, 4) for a in accs], 'emb_params': embp}
                json.dump(summary, open(out, 'w'), indent=2)
                print(f'  -> {m:16s} {np.mean(accs):.4f}+/-{np.std(accs):.4f} (emb {embp})', flush=True)
    print('DONE')


if __name__ == '__main__':
    main()
