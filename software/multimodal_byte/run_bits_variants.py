#!/usr/bin/env python3
"""Bits-improvement bake-off. All encoders are token-level and (except the two
baselines) vocabulary-independent: each token's code is a function of its bytes,
so OOV/new tokens still get a code. Same data, same body (D=Transformer), same
seeds; ONLY the encoder changes.

Variants:
  standard       learned |V|x d table (baseline; grows with vocab)
  byte           EmByte: token's real UTF-8 bytes -> one-hot(256) -> FFN
  rand_bits      Bits baseline: random {0,1}^B per vocab-ID (no structure)
  realbit_bits   (#2) the REAL bits of the token's UTF-8 bytes (structure)
  simhash_bits   (#1) SimHash of byte-trigram bag -> B bits (similarity-preserving)
  multihash      (#3) k hashes -> shared bucket table, summed (Hash Embeddings)
  freq_hybrid    (#4) learned table for top-K tokens + simhash for the tail
  learned_hash   (#5) byte->bits via straight-through (task-adaptive code)
  byte_xor_bits  (#8) concat real-byte code (structure) + random bits (capacity)

All constant-size w.r.t. |V| except standard and freq_hybrid's small top-K table.
"""
import argparse, hashlib, json, os, random, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_massive_v2 import tokenize, all_locales, load_locales, PAD, UNK  # unicode tokenizer + MASSIVE

NB = 64  # bits


# ---------------- code tables (precomputed per vocab token string) ----------------
def t_rand_bits(itos):
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 2, (len(itos), NB), generator=g, dtype=torch.long)


def t_realbit_bits(itos):
    nbytes = (NB + 7) // 8
    tab = torch.zeros(len(itos), NB, dtype=torch.long)
    for i, tok in enumerate(itos):
        b = tok.encode('utf-8')[:nbytes]
        bits = []
        for by in b:
            bits += [(by >> (7 - k)) & 1 for k in range(8)]
        bits = (bits + [0] * NB)[:NB]
        tab[i] = torch.tensor(bits)
    return tab


def t_simhash_bits(itos, F_=2048, seed=0):
    rng = np.random.RandomState(seed)
    R = rng.randn(F_, NB).astype('float32')          # random hyperplanes
    tab = torch.zeros(len(itos), NB, dtype=torch.long)
    for i, tok in enumerate(itos):
        s = ('  ' + tok + '  ')
        feat = np.zeros(F_, dtype='float32')
        for j in range(len(s) - 2):                  # byte trigrams
            tg = s[j:j+3]
            feat[hash(tg) % F_] += 1.0
        proj = feat @ R
        tab[i] = torch.tensor((proj > 0).astype('int64'))
    return tab


class CodeEmb(nn.Module):
    """frozen one-hot(2) over a [V,B] bit table -> trainable FFN. const params."""
    def __init__(self, table, emb_dim, interdim=128):
        super().__init__()
        self.register_buffer('table', table)
        oh = F.one_hot(torch.arange(2)).float()
        self.sym = nn.Embedding(2, 2, _weight=oh); self.sym.requires_grad_(False)
        self.ff1 = nn.Linear(2 * table.size(1), interdim); self.ff2 = nn.Linear(interdim, emb_dim)
    def forward(self, ids):
        c = self.table[ids]
        return self.ff2(F.relu(self.ff1(self.sym(c).view(*c.shape[:2], -1))))


class ByteEmb(nn.Module):
    def __init__(self, itos, emb_dim, nbytes=8, interdim=128):
        super().__init__()
        tab = torch.zeros(len(itos), nbytes, dtype=torch.long)
        for i, tok in enumerate(itos):
            b = tok.encode('utf-8')[:nbytes]
            for j, by in enumerate(b): tab[i, j] = by
        self.register_buffer('table', tab)
        oh = F.one_hot(torch.arange(256)).float()
        self.sym = nn.Embedding(256, 256, _weight=oh); self.sym.requires_grad_(False)
        self.ff1 = nn.Linear(256 * nbytes, interdim); self.ff2 = nn.Linear(interdim, emb_dim)
    def forward(self, ids):
        c = self.table[ids]
        return self.ff2(F.relu(self.ff1(self.sym(c).view(*c.shape[:2], -1))))


class MultiHash(nn.Module):
    """Hash Embeddings: k hashes -> shared B-bucket table, summed. const params."""
    def __init__(self, itos, emb_dim, k=4, buckets=4096):
        super().__init__()
        tab = torch.zeros(len(itos), k, dtype=torch.long)
        for i, tok in enumerate(itos):
            for j in range(k):
                tab[i, j] = int(hashlib.md5(f'{j}:{tok}'.encode()).hexdigest(), 16) % buckets
        self.register_buffer('table', tab)
        self.bucket = nn.Embedding(buckets, emb_dim)
    def forward(self, ids):
        return self.bucket(self.table[ids]).sum(2)


class FreqHybrid(nn.Module):
    """learned table for top-K tokens; simhash-Bits for the tail. small table."""
    def __init__(self, itos, freqs, emb_dim, topk=2000):
        super().__init__()
        order = sorted(range(len(itos)), key=lambda i: -freqs[i])
        self.register_buffer('is_top', torch.zeros(len(itos), dtype=torch.long))
        slot = torch.zeros(len(itos), dtype=torch.long)
        for s, i in enumerate(order[:topk]):
            self.is_top[i] = 1; slot[i] = s + 1
        self.register_buffer('slot', slot)
        self.table = nn.Embedding(topk + 1, emb_dim, padding_idx=0)
        self.tail = CodeEmb(t_simhash_bits(itos), emb_dim)
    def forward(self, ids):
        top = self.table(self.slot[ids])
        tail = self.tail(ids)
        m = self.is_top[ids].unsqueeze(-1).float()
        return m * top + (1 - m) * tail


class LearnedHash(nn.Module):
    """byte one-hot -> small MLP -> B logits -> straight-through sign -> onehot -> FFN."""
    def __init__(self, itos, emb_dim, nbytes=8, interdim=128):
        super().__init__()
        tab = torch.zeros(len(itos), nbytes, dtype=torch.long)
        for i, tok in enumerate(itos):
            for j, by in enumerate(tok.encode('utf-8')[:nbytes]): tab[i, j] = by
        self.register_buffer('table', tab)
        oh = F.one_hot(torch.arange(256)).float()
        self.sym = nn.Embedding(256, 256, _weight=oh); self.sym.requires_grad_(False)
        self.hashnet = nn.Linear(256 * nbytes, NB)
        self.ff1 = nn.Linear(2 * NB, interdim); self.ff2 = nn.Linear(interdim, emb_dim)
    def forward(self, ids):
        c = self.table[ids]
        logits = self.hashnet(self.sym(c).view(*c.shape[:2], -1))
        hard = (logits > 0).long()
        bits = hard + (torch.sigmoid(logits) - torch.sigmoid(logits).detach())  # STE
        oh = torch.stack([1 - bits, bits], dim=-1)
        return self.ff2(F.relu(self.ff1(oh.view(*c.shape[:2], -1))))


class ByteXorBits(nn.Module):
    """#8: concat real-byte code (structure) + random bits (capacity) -> FFN."""
    def __init__(self, itos, emb_dim, nbytes=8, interdim=128):
        super().__init__()
        self.byte = ByteEmb(itos, emb_dim // 2, nbytes, interdim)
        self.bits = CodeEmb(t_rand_bits(itos), emb_dim - emb_dim // 2, interdim)
    def forward(self, ids):
        return torch.cat([self.byte(ids), self.bits(ids)], dim=-1)


def make_emb(method, itos, freqs, emb_dim):
    if method == 'standard':      return nn.Embedding(len(itos), emb_dim, padding_idx=0)
    if method == 'byte':          return ByteEmb(itos, emb_dim)
    if method == 'rand_bits':     return CodeEmb(t_rand_bits(itos), emb_dim)
    if method == 'realbit_bits':  return CodeEmb(t_realbit_bits(itos), emb_dim)
    if method == 'simhash_bits':  return CodeEmb(t_simhash_bits(itos), emb_dim)
    if method == 'multihash':     return MultiHash(itos, emb_dim)
    if method == 'freq_hybrid':   return FreqHybrid(itos, freqs, emb_dim)
    if method == 'learned_hash':  return LearnedHash(itos, emb_dim)
    if method == 'byte_xor_bits': return ByteXorBits(itos, emb_dim)
    raise ValueError(method)


class Net(nn.Module):
    def __init__(self, method, itos, freqs, emb_dim, n_classes):
        super().__init__()
        self.emb = make_emb(method, itos, freqs, emb_dim)
        layer = nn.TransformerEncoderLayer(emb_dim, 4, emb_dim * 2, dropout=0.1, batch_first=True)
        self.body = nn.TransformerEncoder(layer, 2)
        self.fc = nn.Linear(emb_dim, n_classes)
    def forward(self, ids):
        return self.fc(self.body(self.emb(ids)).mean(1))


class DS(Dataset):
    def __init__(self, rows, stoi, b=48):
        self.rows = rows; self.stoi = stoi; self.b = b
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        toks = tokenize(self.rows[i]['text'])
        ids = [self.stoi.get(t, 1) for t in toks][:self.b]
        ids += [0] * (self.b - len(ids))
        return torch.tensor(ids), self.rows[i]['label_idx']


def coll(b):
    return {'ids': torch.stack([x[0] for x in b]),
            'label': torch.tensor([x[1] for x in b])}


def run(rows, method, emb_dim, n_classes, device, epochs, lr, bs, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed); rng.shuffle(rows)
    n = len(rows); ntr = int(0.8 * n); nva = int(0.1 * n)
    tr, va, te = rows[:ntr], rows[ntr:ntr+nva], rows[ntr+nva:]
    cnt = Counter()
    for r in tr: cnt.update(tokenize(r['text']))
    itos = [PAD, UNK] + [t for t, c in cnt.most_common() if c >= 2]
    stoi = {t: i for i, t in enumerate(itos)}
    freqs = [0, 0] + [c for t, c in cnt.most_common() if c >= 2]
    trL = DataLoader(DS(tr, stoi), batch_size=bs, shuffle=True, collate_fn=coll)
    vaL = DataLoader(DS(va, stoi), batch_size=bs, collate_fn=coll)
    teL = DataLoader(DS(te, stoi), batch_size=bs, collate_fn=coll)
    model = Net(method, itos, freqs, emb_dim, n_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=lr); crit = nn.CrossEntropyLoss()
    def ev(L):
        model.eval(); c = t = 0
        with torch.no_grad():
            for b in L:
                p = model(b['ids'].to(device)).argmax(-1).cpu()
                c += (p == b['label']).sum().item(); t += b['label'].size(0)
        return c / max(t, 1)
    best, st = 0.0, None
    for _ in range(epochs):
        model.train()
        for b in trL:
            opt.zero_grad(); crit(model(b['ids'].to(device)), b['label'].to(device)).backward(); opt.step()
        v = ev(vaL)
        if v > best: best = v; st = {k: vv.clone() for k, vv in model.state_dict().items()}
    if st: model.load_state_dict(st)
    return {'test_acc': ev(teL), 'val_acc': best,
            'emb_params': sum(p.numel() for p in model.emb.parameters()),
            'vocab': len(itos)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--n_langs', type=int, default=16)
    p.add_argument('--methods', nargs='+', default=['standard', 'byte', 'rand_bits',
        'realbit_bits', 'simhash_bits', 'multihash', 'freq_hybrid', 'learned_hash', 'byte_xor_bits'])
    p.add_argument('--emb_dim', type=int, default=64)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--n_per', type=int, default=4000)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--bs', type=int, default=64)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = load_locales(all_locales()[:args.n_langs], args.n_per)
    print(f'{args.n_langs} langs, {len(rows)} utts')
    summary = {}
    for m in args.methods:
        accs = []; ep = None
        for s in args.seeds:
            t0 = time.time()
            r = run(list(rows), m, args.emb_dim, 60, device, args.epochs, args.lr, args.bs, s)
            accs.append(r['test_acc']); ep = r['emb_params']
            print(f'  {m:14s} s{s}: acc={r["test_acc"]:.4f} emb={r["emb_params"]} [{time.time()-t0:.0f}s]')
        summary[m] = {'mean_acc': float(np.mean(accs)), 'std_acc': float(np.std(accs)),
                      'accs': accs, 'emb_params': ep}
        json.dump(summary, open(os.path.join(args.out_dir, 'summary.json'), 'w'), indent=2)
        print(f'  -> {m}: {np.mean(accs):.4f} +/- {np.std(accs):.4f} (emb {ep})')
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
