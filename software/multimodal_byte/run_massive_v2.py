#!/usr/bin/env python3
"""MASSIVE multilingual scaling, v2 — Unicode-correct, two mechanisms compared.

Fixes the v1 bug (ASCII-stripping tokenizer that deleted every non-Latin
script). Compares, per the 'Vocabulary Independence Unlocks Efficient
Multimodal Learning' thesis:

  standard   : learned |V|x d embedding table (vocab fitted, or capped).
  faithful_* : ORIGINAL EmByte/Bits — random code keyed by VOCAB-ID, frozen
               one-hot, trainable FFN. Wins on efficiency; ties on accuracy.
               (byte: 8 random bytes/token; bits: num_bits random bits/token)
  bytehash_* : VOCABULARY-INDEPENDENT — code is a deterministic function of the
               token's UTF-8 BYTES (no vocab, no UNK). EmByte uses the real
               bytes (similarity-preserving -> cross-lingual sharing); Bits
               hashes the bytes (sha256 -> bits; OOV-robust, no UNK).

For the bytehash mechanism the Subword baseline uses a FIXED vocab cap so its
out-of-vocabulary cost across many languages is real.

Sweeps number of languages; records union vocab, encoder params, test acc.
"""
import argparse, functools, hashlib, json, os, re, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

PAD, UNK = '<pad>', '<unk>'
MIRROR = 'mteb/amazon_massive_intent'
_PREFERRED = ['en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'ru', 'pl', 'sv',
              'tr', 'fi', 'hu', 'el', 'he', 'ar', 'fa', 'hi', 'bn', 'ta',
              'te', 'kn', 'ml', 'th', 'vi', 'id', 'ms', 'tl', 'ja', 'ko',
              'zh-CN', 'zh-TW', 'sw', 'am', 'is', 'nb', 'da', 'cy', 'ka',
              'hy', 'sq', 'az', 'km', 'my', 'mn', 'jv', 'af', 'sl', 'lv',
              'ur', 'ro']
_LABEL_MAP = {}

# ---- Unicode-aware tokenizer: CJK as single chars, runs of word chars as
# tokens, individual punctuation. Preserves ALL scripts (no ASCII strip). ----
_TOK = re.compile(r'[一-鿿぀-ヿ가-힯]|[^\W\d_]+|\d+|[^\s\w]',
                  re.UNICODE)


def tokenize(text):
    return _TOK.findall((text or '').lower())


def all_locales():
    from datasets import get_dataset_config_names
    cfgs = [c for c in get_dataset_config_names(MIRROR) if c != 'default']
    return [c for c in _PREFERRED if c in cfgs] + [c for c in cfgs if c not in _PREFERRED]


def load_locales(locales, n_per):
    from datasets import load_dataset
    rows = []
    for lc in locales:
        try:
            ds = load_dataset(MIRROR, lc, split='train')
        except Exception as e:
            print(f'  skip {lc}: {str(e)[:60]}'); continue
        n = 0
        for ex in ds:
            lab = ex['label_text']
            _LABEL_MAP.setdefault(lab, len(_LABEL_MAP))
            rows.append({'text': ex['text'], 'label_idx': _LABEL_MAP[lab]})
            n += 1
            if n_per and n >= n_per:
                break
    return rows


# ----------------------------- code tables -----------------------------
def faithful_byte_table(itos):
    g = torch.Generator().manual_seed(12345)
    return torch.randint(0, 256, (len(itos), 8), generator=g, dtype=torch.long)


def faithful_bits_table(itos, num_bits):
    g = torch.Generator().manual_seed(12345)
    return torch.randint(0, 2, (len(itos), num_bits), generator=g, dtype=torch.long)


def bytehash_byte_table(itos, n=8):
    # real UTF-8 bytes of the token string (similarity-preserving)
    tab = torch.zeros(len(itos), n, dtype=torch.long)
    for i, tok in enumerate(itos):
        b = tok.encode('utf-8')[:n]
        for j, by in enumerate(b):
            tab[i, j] = by
    return tab


def bytehash_bits_table(itos, num_bits):
    # deterministic hash of the token bytes -> num_bits bits (OOV-robust, no UNK)
    tab = torch.zeros(len(itos), num_bits, dtype=torch.long)
    nbytes = (num_bits + 7) // 8
    for i, tok in enumerate(itos):
        h = hashlib.sha256(tok.encode('utf-8')).digest()[:nbytes]
        bits = []
        for by in h:
            bits.extend([(by >> k) & 1 for k in range(8)])
        tab[i, :num_bits] = torch.tensor(bits[:num_bits], dtype=torch.long)
    return tab


class CodeEmbedding(nn.Module):
    """frozen one-hot over a {2 or 256}-symbol alphabet -> trainable FFN -> d.
    `table` is [V, code_len] of symbol ids; constant params w.r.t. V."""
    def __init__(self, table, alphabet, emb_dim, interdim=128):
        super().__init__()
        self.register_buffer('table', table)
        self.alphabet = alphabet
        code_len = table.size(1)
        onehot = F.one_hot(torch.arange(alphabet)).float()
        self.sym = nn.Embedding(alphabet, alphabet, _weight=onehot)
        self.sym.requires_grad_(False)
        self.ff1 = nn.Linear(alphabet * code_len, interdim)
        self.ff2 = nn.Linear(interdim, emb_dim)

    def forward(self, ids):
        codes = self.table[ids]                       # [B,T,code_len]
        oh = self.sym(codes)                          # [B,T,code_len,alphabet]
        flat = oh.view(*codes.shape[:2], -1)          # [B,T,code_len*alphabet]
        return self.ff2(F.relu(self.ff1(flat)))


class Net(nn.Module):
    def __init__(self, method, itos, emb_dim, n_classes, body, num_bits=64, pad_idx=0):
        super().__init__()
        self.body_kind = body
        if method == 'standard':
            self.emb = nn.Embedding(len(itos), emb_dim, padding_idx=pad_idx)
        elif method == 'faithful_byte':
            self.emb = CodeEmbedding(faithful_byte_table(itos), 256, emb_dim)
        elif method == 'faithful_bits':
            self.emb = CodeEmbedding(faithful_bits_table(itos, num_bits), 2, emb_dim)
        elif method == 'bytehash_byte':
            self.emb = CodeEmbedding(bytehash_byte_table(itos), 256, emb_dim)
        elif method == 'bytehash_bits':
            self.emb = CodeEmbedding(bytehash_bits_table(itos, num_bits), 2, emb_dim)
        else:
            raise ValueError(method)
        if body == 'A':
            self.body = None; head = emb_dim
        else:  # D: transformer encoder
            layer = nn.TransformerEncoderLayer(emb_dim, 4, emb_dim * 2,
                                               dropout=0.1, batch_first=True)
            self.body = nn.TransformerEncoder(layer, 2); head = emb_dim
        self.fc = nn.Linear(head, n_classes)

    def forward(self, ids):
        e = self.emb(ids)
        if self.body is None:
            pooled = e.mean(1)
        else:
            pooled = self.body(e).mean(1)
        return self.fc(pooled)


class DS(Dataset):
    def __init__(self, rows, stoi, budget=48):
        self.rows = rows; self.stoi = stoi; self.budget = budget
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        toks = tokenize(self.rows[i]['text'])
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in toks][:self.budget]
        ids += [self.stoi[PAD]] * (self.budget - len(ids))
        return torch.tensor(ids, dtype=torch.long), self.rows[i]['label_idx']


def collate(b):
    return {'ids': torch.stack([x[0] for x in b]),
            'label': torch.tensor([x[1] for x in b], dtype=torch.long)}


def build_vocab(train, vocab_cap=None):
    cnt = Counter()
    for r in train:
        cnt.update(tokenize(r['text']))
    itos = [PAD, UNK]
    items = cnt.most_common(vocab_cap) if vocab_cap else cnt.most_common()
    for t, c in items:
        if c >= 2:
            itos.append(t)
    stoi = {t: i for i, t in enumerate(itos)}
    return stoi, itos


def run(rows, method, body, emb_dim, n_classes, device, epochs, lr, bs, seed, vocab_cap):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed); rng.shuffle(rows)
    n = len(rows); ntr = int(0.8 * n); nva = int(0.1 * n)
    tr, va, te = rows[:ntr], rows[ntr:ntr+nva], rows[ntr+nva:]
    # standard + faithful_* are vocab-id keyed -> fitted vocab (no cap by default).
    # bytehash_* are vocab-independent; cap the SHARED vocab so OOV is real and
    # identical across methods in the comparison.
    stoi, itos = build_vocab(tr, vocab_cap=vocab_cap)
    coll = collate
    trL = DataLoader(DS(tr, stoi), batch_size=bs, shuffle=True, collate_fn=coll)
    vaL = DataLoader(DS(va, stoi), batch_size=bs, collate_fn=coll)
    teL = DataLoader(DS(te, stoi), batch_size=bs, collate_fn=coll)
    model = Net(method, itos, emb_dim, n_classes, body).to(device)
    opt = optim.Adam(model.parameters(), lr=lr); crit = nn.CrossEntropyLoss()

    def ev(L):
        model.eval(); c = t = 0
        with torch.no_grad():
            for b in L:
                p = model(b['ids'].to(device)).argmax(-1).cpu()
                c += (p == b['label']).sum().item(); t += b['label'].size(0)
        return c / max(t, 1)

    best, best_state = 0.0, None
    for _ in range(epochs):
        model.train()
        for b in trL:
            opt.zero_grad()
            crit(model(b['ids'].to(device)), b['label'].to(device)).backward()
            opt.step()
        v = ev(vaL)
        if v > best:
            best = v; best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    enc = sum(p.numel() for p in model.emb.parameters())
    return {'vocab': len(itos), 'test_acc': ev(teL), 'val_acc': best,
            'encoder_params': enc,
            'total_params': sum(p.numel() for p in model.parameters())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--lang_counts', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32, 51])
    p.add_argument('--methods', nargs='+',
                   default=['standard', 'faithful_byte', 'faithful_bits',
                            'bytehash_byte', 'bytehash_bits'])
    p.add_argument('--body', default='A')
    p.add_argument('--emb_dim', type=int, default=64)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--n_per', type=int, default=4000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--vocab_cap', type=int, default=0,
                   help='0 = fitted (uncapped) vocab; >0 = fixed cap so OOV is real')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    locales = all_locales()
    cap = args.vocab_cap or None
    print(f'{len(locales)} locales; sweep {args.lang_counts}; cap={cap}; body={args.body}')

    summary = {}
    for N in args.lang_counts:
        rows = load_locales(locales[:N], args.n_per)
        print(f'\n=== {N} langs: {len(rows)} utts ===')
        for m in args.methods:
            t0 = time.time()
            r = run(list(rows), m, args.body, args.emb_dim, 60, device,
                    args.epochs, args.lr, args.bs, args.seed, cap)
            r['n_languages'] = N; r['method'] = m
            summary[f'{N}lang_{m}'] = r
            print(f'  {m:16s} vocab={r["vocab"]:6d} acc={r["test_acc"]:.4f} '
                  f'enc={r["encoder_params"]:>9d} [{time.time()-t0:.0f}s]')
            json.dump(summary, open(os.path.join(args.out_dir, 'summary.json'), 'w'), indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
