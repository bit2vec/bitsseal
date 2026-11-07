#!/usr/bin/env python3
"""P1 — reproduce Table 7 (EmByte) on IMDB and run the SAME compressor sweep for
Bits, using the paper's exact modules (paper_modules/). Fidelity gate: EmByte
TF-Enc-H4-FFN512 should reach ~0.841.

Locked spec (from imdb.py + both papers):
  basic_english word tokenizer + build_vocab; embedding_dim=256; max_len=256;
  LSTM(300, 2 layers) body; 10 epochs; lr=5e-4; bs=64.
  EmByte: random 8-byte map -> one-hot(256) -> compressor.   (ByteSubwordConcateOnehot)
  Bits  : random 64-bit map -> one-hot(2)   -> compressor.   (BinarySubword*; EnsembleBinaryEmbedding)
  Compressors (Table 7 menu): MLP-HD{256,512,768}, TF-Enc-H4-FFN512, TF-Enc-H8-FFN1024,
                              (Bits also: Sum, Attention, Ensemble).
"""
import argparse, hashlib, json, os, re, sys, time
from collections import Counter
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_modules'))
from byte_subword_concate_onehot import ByteSubwordConcateOnehot
from binary_subword_sentiment import (BinarySubwordConcateOnehot, BinarySubwordSum,
                                      BinarySubwordAttention)
try:
    from ensemble_binary_seb import EnsembleBinaryEmbedding
except Exception:
    EnsembleBinaryEmbedding = None

# ---- torchtext basic_english tokenizer (replicated exactly) ----
_pat = [r"\'", r"\"", r"\.", r"<br \/>", r",", r"\(", r"\)", r"\!", r"\?", r"\;", r"\:", r"\s+"]
_rep = [" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "]
_pat = [re.compile(p) for p in _pat]
def basic_english(line):
    line = line.lower()
    for p, r in zip(_pat, _rep): line = p.sub(r, line)
    return line.split()


# ---- TF-Enc compressor (Table 7), reusing the paper's random map verbatim ----
class _TFEncByte(nn.Module):
    def __init__(self, V, d, nhead, ffn, tmp):
        super().__init__()
        base = ByteSubwordConcateOnehot(V, d, interdim='256', subword_bytes_file=tmp)
        self.register_buffer('map', base.subword2bytes.clone())     # random 8-byte map
        self.L = base.max_byte_seq_len
        oh = F.one_hot(torch.arange(256)).float()
        self.sym = nn.Embedding(256, 256, _weight=oh); self.sym.requires_grad_(False)
        self.proj = nn.Linear(256, d)
        self.pos = nn.Parameter(torch.randn(self.L, d) * 0.02)
        self.tf = nn.TransformerEncoderLayer(d, nhead, ffn, dropout=0.1, batch_first=True)
    def forward(self, ids):
        c = self.map[ids]; B, T = c.shape[:2]
        x = self.proj(self.sym(c)) + self.pos
        return self.tf(x.reshape(B * T, self.L, -1)).mean(1).reshape(B, T, -1)


class _TFEncBits(nn.Module):
    def __init__(self, V, d, nhead, ffn, num_bits, tmp):
        super().__init__()
        base = BinarySubwordConcateOnehot(num_embeddings=V, embedding_dim=d,
                                          num_bits=num_bits, subword_binary_file=tmp)
        self.register_buffer('map', base.subword2binary.clone())    # random 64-bit map
        self.L = num_bits
        oh = F.one_hot(torch.arange(2)).float()
        self.sym = nn.Embedding(2, 2, _weight=oh); self.sym.requires_grad_(False)
        self.proj = nn.Linear(2, d)
        self.pos = nn.Parameter(torch.randn(self.L, d) * 0.02)
        self.tf = nn.TransformerEncoderLayer(d, nhead, ffn, dropout=0.1, batch_first=True)
    def forward(self, ids):
        c = self.map[ids]; B, T = c.shape[:2]
        x = self.proj(self.sym(c)) + self.pos
        return self.tf(x.reshape(B * T, self.L, -1)).mean(1).reshape(B, T, -1)


def simhash_table(itos, num_bits=64, F_=2048, seed=0):
    """NEW method (vs 001's random map): SimHash of token byte-trigrams ->
    num_bits bits. Deterministic, vocabulary-independent, and SIMILARITY-
    PRESERVING (similar byte strings -> small Hamming distance), unlike the
    random Stochastic Mapping of EmByte/001."""
    import numpy as np, hashlib
    rng = np.random.RandomState(seed)
    R = rng.randn(F_, num_bits).astype('float32')      # random hyperplanes (fixed)
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


def repulse_table(itos, train_token_lists, num_bits=64, beta=0.4, sweeps=12,
                  lam=0.02, seed=0, device=None):
    """NEW method (context-REPULSIVE stochastic coding). Inverse of SimHash:
    tokens that co-occur in similar contexts are STOCHASTICALLY pushed to LARGER
    Hamming distance, by sampling a finite-temperature antiferromagnetic Ising
    model on the context-similarity graph (one independent spin system per bit).
    beta=0 -> uniform random codes (= Bits/001); beta>0 -> repulsion proportional
    to context similarity, never deterministic (collisions remain possible)."""
    import numpy as np
    from scipy import sparse
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    V = len(itos); stoi = {t: i for i, t in enumerate(itos)}
    # 1. document-level co-occurrence -> SIGNED PMI (can be negative => real
    #    frustration; sparse top-k neighbours so it is not a complete graph)
    rows, cols = [], []
    for d, toks in enumerate(train_token_lists):
        for tid in {stoi.get(x, 1) for x in toks}:
            rows.append(d); cols.append(tid)
    X = sparse.csr_matrix((np.ones(len(rows), 'float32'), (rows, cols)),
                          shape=(len(train_token_lists), V))
    C = (X.T @ X).toarray().astype('float32')        # [V,V] co-document counts
    tot = C.sum() + 1e-8
    pi = C.sum(1, keepdims=True) / tot               # marginal
    Pij = C / tot
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log((Pij + 1e-12) / (pi @ pi.T + 1e-12))   # signed PMI
    np.fill_diagonal(pmi, 0.0)
    # keep only top-k |PMI| neighbours per row (sparse, signed)
    k = min(30, V - 1)
    idx = np.argpartition(-np.abs(pmi), k, axis=1)[:, :k]
    Ssp = np.zeros_like(pmi)
    np.put_along_axis(Ssp, idx, np.take_along_axis(pmi, idx, axis=1), axis=1)
    Ssp = 0.5 * (Ssp + Ssp.T)                          # symmetrise
    S = torch.tensor(Ssp, dtype=torch.float32, device=dev)
    # 2. antiferromagnetic Ising, ASYNCHRONOUS (random-subset) Glauber at temp 1/beta
    g = torch.Generator(device=dev).manual_seed(seed)
    code = torch.zeros(V, num_bits, dtype=torch.long, device=dev)
    for b in range(num_bits):
        spins = torch.randint(0, 2, (V,), generator=g, device=dev).float() * 2 - 1
        for _ in range(sweeps):
            h = S @ spins
            M = spins.sum()
            dU = -2.0 * spins * h + 4.0 * lam * (1.0 - spins * M)   # repulsion + uniformity
            p = torch.exp(torch.clamp(-beta * dU, max=0.0))        # Metropolis accept prob
            wants = torch.rand(V, generator=g, device=dev) < p
            subset = torch.rand(V, generator=g, device=dev) < 0.3   # async: flip only ~30%
            flip = wants & subset
            spins = torch.where(flip, -spins, spins)
        code[:, b] = (spins > 0).long()
    code = code.cpu()
    distinct = torch.unique(code, dim=0).shape[0]
    print(f'    [repulse beta={beta}] distinct codes {distinct}/{V} '
          f'({100*distinct/V:.1f}%)')
    return code


class LearnedHash(nn.Module):
    """#5: LEARNED task-adaptive hash. Token's real UTF-8 bytes -> frozen
    one-hot(256) -> small net -> num_bits logits -> straight-through binarize
    -> one-hot(2) -> FFN. Vocabulary-independent (a function of the bytes), but
    the bit code is learned end-to-end, unlike fixed random/SimHash/Ising maps."""
    def __init__(self, itos, emb_dim, num_bits=64, nbytes=8, interdim=512):
        super().__init__()
        tab = torch.zeros(len(itos), nbytes, dtype=torch.long)
        for i, tok in enumerate(itos):
            for j, by in enumerate(str(tok).encode('utf-8')[:nbytes]):
                tab[i, j] = by
        self.register_buffer('btab', tab)
        self.byte = nn.Embedding(256, 256, _weight=F.one_hot(torch.arange(256)).float())
        self.byte.requires_grad_(False)
        self.bit = nn.Embedding(2, 2, _weight=F.one_hot(torch.arange(2)).float())
        self.bit.requires_grad_(False)
        self.num_bits = num_bits
        self.hashnet = nn.Linear(256 * nbytes, num_bits)
        self.ff1 = nn.Linear(2 * num_bits, interdim)
        self.ff2 = nn.Linear(interdim, emb_dim)

    def forward(self, ids):
        b = self.btab[ids]                                  # [B,T,nbytes]
        x = self.byte(b).view(*ids.shape, -1)               # [B,T,nbytes*256]
        logits = self.hashnet(x)                            # [B,T,num_bits]
        hard = (logits > 0).float()
        st = hard + (torch.sigmoid(logits) - torch.sigmoid(logits).detach())  # straight-through
        oh = torch.stack([1.0 - st, st], dim=-1)            # [B,T,num_bits,2]
        return self.ff2(F.relu(self.ff1(oh.view(*ids.shape, -1))))


def t_tfree_bits(itos, num_bits=64):
    """CodeHash/T-FREE rule as a BIT MAP: each token's character trigrams set
    bits (hash%num_bits) -> multi-hot code. Shared trigrams -> shared set bits
    (similarity-preserving), vocabulary-independent. Fed to the same EmCode FFN."""
    tab = torch.zeros(len(itos), num_bits, dtype=torch.long)
    for i, tok in enumerate(itos):
        s = '##' + str(tok) + '##'
        for j in range(len(s) - 2):
            tab[i, int(hashlib.md5(s[j:j+3].encode()).hexdigest(), 16) % num_bits] = 1
    return tab


def t_multihash_bits(itos, num_bits=64, k=8):
    """multihash rule as a BIT MAP: k independent hashes of the token each set
    one bit -> sparse code. Fed to the same EmCode FFN."""
    tab = torch.zeros(len(itos), num_bits, dtype=torch.long)
    for i, tok in enumerate(itos):
        for j in range(k):
            tab[i, int(hashlib.md5(f'{j}:{tok}'.encode()).hexdigest(), 16) % num_bits] = 1
    return tab


def t_hashformers_bits(itos, num_bits=64, k=2):
    """HashFormers (EMNLP'22) rule as a BIT MAP: hash the token id into k bit
    positions (its bucket signature). Fed to the shared EmCode FFN."""
    tab = torch.zeros(len(itos), num_bits, dtype=torch.long)
    for i in range(len(itos)):
        for j in range(k):
            tab[i, int(hashlib.md5(f'hf{j}:{i}'.encode()).hexdigest(), 16) % num_bits] = 1
    return tab


def make_embed(method, V, d, tmpdir, itos=None, train_toks=None, repulse_beta=0.4, seed=0, num_bits=64):
    t = lambda n: os.path.join(tmpdir, n)
    if method == 'hashformers_codehash':
        f = t('hf.pt'); torch.save(t_hashformers_bits(itos), f)
        return BinarySubwordConcateOnehot(num_embeddings=V, embedding_dim=d, num_bits=64,
                                          interdim='512', subword_binary_file=f)
    # code-mapping rules ON TOP OF the EmCode/Bits MLP-512 encoder (like SimHash)
    if method == 'tfree_codehash':
        f = t('tfree.pt'); torch.save(t_tfree_bits(itos), f)
        return BinarySubwordConcateOnehot(num_embeddings=V, embedding_dim=d, num_bits=64,
                                          interdim='512', subword_binary_file=f)
    if method == 'multihash_codehash':
        f = t('mh.pt'); torch.save(t_multihash_bits(itos), f)
        return BinarySubwordConcateOnehot(num_embeddings=V, embedding_dim=d, num_bits=64,
                                          interdim='512', subword_binary_file=f)
    t = lambda n: os.path.join(tmpdir, n)
    # NEW: hash-based mapping (similarity-preserving) through the working MLP path
    if method == 'bits_hash_mlp512':
        hf = t('hash.pt')
        torch.save(simhash_table(itos, num_bits=64), hf)
        return BinarySubwordConcateOnehot(num_embeddings=V, embedding_dim=d, num_bits=64,
                                          interdim='512', subword_binary_file=hf)
    # NEW (#5): learned/task-adaptive hash (straight-through)
    if method == 'bits_learned_hash':
        return LearnedHash(itos, d, num_bits=64, interdim=512)
    # NEW: context-repulsive stochastic mapping (antiferromagnetic Ising)
    if method == 'bits_repulse_mlp512':
        rf = t(f'repulse_b{repulse_beta}.pt')
        if not os.path.isfile(rf):
            torch.save(repulse_table(itos, train_toks, num_bits=64, beta=repulse_beta, seed=seed), rf)
        return BinarySubwordConcateOnehot(num_embeddings=V, embedding_dim=d, num_bits=64,
                                          interdim='512', subword_binary_file=rf)
    if method == 'standard':         return nn.Embedding(V, d, padding_idx=0)
    # EmByte (byte)
    if method == 'embyte_mlp256':    return ByteSubwordConcateOnehot(V, d, interdim='256', subword_bytes_file=t('eb.pt'))
    if method == 'embyte_mlp512':    return ByteSubwordConcateOnehot(V, d, interdim='512', subword_bytes_file=t('eb.pt'))
    if method == 'embyte_mlp768':    return ByteSubwordConcateOnehot(V, d, interdim='768', subword_bytes_file=t('eb.pt'))
    if method == 'embyte_tfenc_h4_512':  return _TFEncByte(V, d, 4, 512, t('eb.pt'))
    if method == 'embyte_tfenc_h8_1024': return _TFEncByte(V, d, 8, 1024, t('eb.pt'))
    # Bits (binary)
    if method == 'bits_mlp256':      return BinarySubwordConcateOnehot(num_embeddings=V, embedding_dim=d, num_bits=64, interdim='256', subword_binary_file=t('bi.pt'))
    if method == 'bits_mlp512':      return BinarySubwordConcateOnehot(num_embeddings=V, embedding_dim=d, num_bits=num_bits, interdim='512', subword_binary_file=t('bi.pt'))
    if method == 'bits_mlp768':      return BinarySubwordConcateOnehot(num_embeddings=V, embedding_dim=d, num_bits=64, interdim='768', subword_binary_file=t('bi.pt'))
    if method == 'bits_sum':         return BinarySubwordSum(num_embeddings=V, embedding_dim=d, num_bits=64, subword_binary_file=t('bi.pt'))
    if method == 'bits_attention':   return BinarySubwordAttention(num_embeddings=V, embedding_dim=d, num_bits=64, subword_binary_file=t('bi.pt'))
    if method == 'bits_tfenc_h4_512':    return _TFEncBits(V, d, 4, 512, 64, t('bi.pt'))
    if method == 'bits_tfenc_h8_1024':   return _TFEncBits(V, d, 8, 1024, 64, t('bi.pt'))
    if method == 'bits_ensemble' and EnsembleBinaryEmbedding is not None:
        return EnsembleBinaryEmbedding(num_embeddings=V, embedding_dim=d, num_bits=64)
    raise ValueError(method)


class Model(nn.Module):
    def __init__(self, emb, d=256, hidden=300, layers=2, nc=2):
        super().__init__()
        self.emb = emb
        self.lstm = nn.LSTM(d, hidden, layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(2 * hidden, nc)
    def forward(self, ids):
        e = self.emb(ids)
        out, _ = self.lstm(e)
        return self.fc(out.mean(1))


class DS(Dataset):
    def __init__(self, rows, stoi, ml=256):
        self.rows = rows; self.stoi = stoi; self.ml = ml
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        toks = self.rows[i][0][:self.ml]
        ids = [self.stoi.get(t, 1) for t in toks] + [0] * (self.ml - len(toks))
        return torch.tensor(ids[:self.ml]), self.rows[i][1]


def coll(b): return {'ids': torch.stack([x[0] for x in b]), 'y': torch.tensor([x[1] for x in b])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--methods', nargs='+', default=[
        'standard', 'embyte_mlp256', 'embyte_mlp512', 'embyte_mlp768',
        'embyte_tfenc_h4_512', 'embyte_tfenc_h8_1024',
        'bits_mlp256', 'bits_mlp512', 'bits_mlp768', 'bits_sum', 'bits_attention',
        'bits_tfenc_h4_512', 'bits_tfenc_h8_1024', 'bits_ensemble'])
    p.add_argument('--d', type=int, default=256); p.add_argument('--max_len', type=int, default=256)
    p.add_argument('--epochs', type=int, default=10); p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--bs', type=int, default=64); p.add_argument('--min_freq', type=int, default=5)
    p.add_argument('--n_train', type=int, default=25000)
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--dataset', default='imdb')
    p.add_argument('--num_bits', type=int, default=64)
    p.add_argument('--dump_preds', action='store_true')
    args = p.parse_args(); os.makedirs(args.out_dir, exist_ok=True)
    tmpdir = os.path.join(args.out_dir, 'maps'); os.makedirs(tmpdir, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # (hf_name, config, text_field, label_field, n_classes, test_split)
    DATASETS = {
        'imdb':    ('stanfordnlp/imdb', None, 'text', 'label', 2, 'test'),
        'agnews':  ('fancyzhx/ag_news', None, 'text', 'label', 4, 'test'),
        'yelp':    ('fancyzhx/yelp_polarity', None, 'text', 'label', 2, 'test'),
        'dbpedia': ('fancyzhx/dbpedia_14', None, 'content', 'label', 14, 'test'),
        'amazon':  ('fancyzhx/amazon_polarity', None, 'content', 'label', 2, 'test'),
    }
    hf, cfg, tf, lf, NC, tesp = DATASETS[args.dataset]
    from datasets import load_dataset
    tr_raw = (load_dataset(hf, cfg, split='train') if cfg else load_dataset(hf, split='train')).shuffle(seed=args.seeds[0])
    te_raw = load_dataset(hf, cfg, split=tesp) if cfg else load_dataset(hf, split=tesp)
    tr = [(basic_english(r[tf]), int(r[lf])) for r in tr_raw.select(range(min(args.n_train, len(tr_raw))))]
    te = [(basic_english(r[tf]), int(r[lf])) for r in te_raw.select(range(min(25000, len(te_raw))))]
    cnt = Counter(t for toks, _ in tr for t in toks)
    itos = ['<pad>', '<unk>'] + [w for w, c in cnt.most_common() if c >= args.min_freq]
    stoi = {w: i for i, w in enumerate(itos)}; V = len(itos)
    print(f'{args.dataset} train={len(tr)} test={len(te)} vocab={V} classes={NC}')
    trL = DataLoader(DS(tr, stoi, args.max_len), batch_size=args.bs, shuffle=True, collate_fn=coll)
    teL = DataLoader(DS(te, stoi, args.max_len), batch_size=args.bs, collate_fn=coll)

    summary = {}
    pred_store = {}
    for m in args.methods:
        accs, emb_params = [], None
        for seed in args.seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            seed_tmp = os.path.join(tmpdir, f'seed{seed}'); os.makedirs(seed_tmp, exist_ok=True)
            try:
                emb = make_embed(m, V, args.d, seed_tmp, itos=itos,
                                 train_toks=[x[0] for x in tr], seed=seed, num_bits=args.num_bits)
            except Exception as e:
                print(f'  {m} s{seed}: SKIP ({str(e)[:50]})'); continue
            model = Model(emb, d=args.d, nc=NC).to(dev)
            emb_params = sum(p.numel() for p in model.emb.parameters())
            opt = optim.Adam(model.parameters(), lr=args.lr); crit = nn.CrossEntropyLoss()
            t0 = time.time(); best = 0.0; best_preds = None; te_labels = None
            for ep in range(args.epochs):
                model.train()
                for b in trL:
                    opt.zero_grad(); crit(model(b['ids'].to(dev)), b['y'].to(dev)).backward(); opt.step()
                model.eval(); preds = []; labels = []
                with torch.no_grad():
                    for b in teL:
                        pr = model(b['ids'].to(dev)).argmax(-1).cpu()
                        preds.extend(pr.tolist()); labels.extend(b['y'].tolist())
                acc = sum(int(p == l) for p, l in zip(preds, labels)) / len(labels)
                if acc > best:
                    best = acc; best_preds = preds; te_labels = labels
            accs.append(best)
            if args.dump_preds and best_preds is not None:
                pred_store.setdefault(m, {})[str(seed)] = best_preds
                pred_store['_labels'] = te_labels
            print(f'  {m:22s} s{seed}: acc={best:.4f} [{time.time()-t0:.0f}s]')
        if accs:
            summary[m] = {'mean_acc': round(float(np.mean(accs)), 4),
                          'std_acc': round(float(np.std(accs)), 4),
                          'accs': [round(a, 4) for a in accs], 'emb_params': emb_params}
            print(f'  -> {m:22s} {np.mean(accs):.4f} +/- {np.std(accs):.4f} (emb {emb_params})')
            json.dump(summary, open(os.path.join(args.out_dir, 'summary.json'), 'w'), indent=2)
    if args.dump_preds:
        json.dump(pred_store, open(os.path.join(args.out_dir, 'preds.json'), 'w'))
        print(f'Wrote {args.out_dir}/preds.json')
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
