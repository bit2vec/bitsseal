#!/usr/bin/env python3
"""MASSIVE multilingual scaling, v3 — FAITHFUL to the paper's LANGUAGE method.

Reuses the authoritative byte-level encoders from generalizability_coling.py:
  standard   : BPE subword (vocab~=10k) -> learned embedding table.
  byte_seb   : byte-level; sliding window of real UTF-8 bytes -> one-hot(256)
               -> FFN. Vocabulary-INDEPENDENT (256 fixed symbols), no UNK.
  binary_seb : byte-level; REAL bits of the actual UTF-8 bytes -> one-hot(2)
               -> FFN. Vocabulary-INDEPENDENT (2 symbols), no UNK.

Across 51 languages the byte/binary encoders are constant-size and handle every
script with no UNK, so they should win on BOTH efficiency (vs the growing BPE
table) and accuracy (vs BPE fragmentation/OOV). Sentiment-style random-per-ID
codes are NOT used here — that regime is monolingual in the paper.
"""
import argparse, json, os, random, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

MIRROR = 'mteb/amazon_massive_intent'
_PREFERRED = ['en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'ru', 'pl', 'sv',
              'tr', 'fi', 'hu', 'el', 'he', 'ar', 'fa', 'hi', 'bn', 'ta',
              'te', 'kn', 'ml', 'th', 'vi', 'id', 'ms', 'tl', 'ja', 'ko',
              'zh-CN', 'zh-TW', 'sw', 'am', 'is', 'nb', 'da', 'cy', 'ka',
              'hy', 'sq', 'az', 'km', 'my', 'mn', 'jv', 'af', 'sl', 'lv',
              'ur', 'ro']
_LABEL_MAP = {}


# ===== authoritative encoders (verbatim from generalizability_coling.py) =====
class ByteSEBEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_bytes=8, interdim='128', dropout=0.1):
        super().__init__()
        self.num_bytes = num_bytes
        onehot = F.one_hot(torch.arange(256)).float()
        self.byte_embed = nn.Embedding(256, 256, _weight=onehot); self.byte_embed.requires_grad_(False)
        layers, cur = [], 256 * num_bytes
        for h in [int(x) for x in interdim.split('_') if x]:
            layers += [nn.Linear(cur, h), nn.ReLU(), nn.Dropout(dropout)]; cur = h
        layers.append(nn.Linear(cur, embedding_dim)); self.ffn = nn.Sequential(*layers)

    def forward(self, b):
        B, T = b.shape
        padded = F.pad(b, (0, self.num_bytes - 1), value=0)
        w = padded.unfold(1, self.num_bytes, 1)
        return self.ffn(self.byte_embed(w).view(B, T, -1))


class BinarySEBEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_bits=64, interdim='128', dropout=0.1):
        super().__init__()
        self.num_bits = num_bits
        onehot = F.one_hot(torch.arange(2)).float()
        self.bit_embed = nn.Embedding(2, 2, _weight=onehot); self.bit_embed.requires_grad_(False)
        layers, cur = [], 2 * num_bits
        for h in [int(x) for x in interdim.split('_') if x]:
            layers += [nn.Linear(cur, h), nn.ReLU(), nn.Dropout(dropout)]; cur = h
        layers.append(nn.Linear(cur, embedding_dim)); self.ffn = nn.Sequential(*layers)

    def byte_to_bits(self, b):
        B, T = b.shape
        nbytes = (self.num_bits + 7) // 8
        if nbytes > 1:
            padded = F.pad(b, (0, nbytes - 1), value=0)
            w = padded.unfold(1, nbytes, 1)
            bits = torch.stack([(w >> (7 - i)) & 1 for i in range(8)], dim=-1).view(B, T, -1)
            return bits[:, :, :self.num_bits]
        bits = torch.stack([(b >> (7 - i)) & 1 for i in range(8)], dim=-1)
        if self.num_bits < 8: return bits[:, :, :self.num_bits]
        if self.num_bits > 8: return F.pad(bits, (0, self.num_bits - 8), value=0)
        return bits

    def forward(self, b):
        B, T = b.shape
        return self.ffn(self.bit_embed(self.byte_to_bits(b)).view(B, T, -1))


# ============================ data ============================
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
            _LABEL_MAP.setdefault(ex['label_text'], len(_LABEL_MAP))
            rows.append({'text': ex['text'], 'label_idx': _LABEL_MAP[ex['label_text']]})
            n += 1
            if n_per and n >= n_per: break
    return rows


class ByteDS(Dataset):
    def __init__(self, rows, max_len=180):
        self.rows = rows; self.max_len = max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        b = list(self.rows[i]['text'].encode('utf-8'))[:self.max_len]
        b += [0] * (self.max_len - len(b))
        return torch.tensor(b, dtype=torch.long), self.rows[i]['label_idx']


class BPEDS(Dataset):
    def __init__(self, rows, tok, max_len=64):
        self.rows = rows; self.tok = tok; self.max_len = max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        ids = self.tok.encode(self.rows[i]['text']).ids[:self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), self.rows[i]['label_idx']


def collate(b):
    return {'ids': torch.stack([x[0] for x in b]),
            'label': torch.tensor([x[1] for x in b], dtype=torch.long)}


def train_bpe(texts, vocab_size=10000):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tr = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=['<pad>', '<unk>'])
    tok.train_from_iterator(texts, tr)
    return tok


class Net(nn.Module):
    def __init__(self, method, emb_dim, n_classes, body, bpe_vocab=0, num_bits=64):
        super().__init__()
        self.method = method
        if method == 'standard':
            self.emb = nn.Embedding(bpe_vocab, emb_dim, padding_idx=0)
        elif method == 'byte_seb':
            self.emb = ByteSEBEmbedding(emb_dim)
        elif method == 'binary_seb':
            self.emb = BinarySEBEmbedding(emb_dim, num_bits=num_bits)
        if body == 'A':
            self.body = None
        else:
            layer = nn.TransformerEncoderLayer(emb_dim, 4, emb_dim * 2, dropout=0.1, batch_first=True)
            self.body = nn.TransformerEncoder(layer, 2)
        self.fc = nn.Linear(emb_dim, n_classes)

    def forward(self, ids):
        e = self.emb(ids)
        pooled = e.mean(1) if self.body is None else self.body(e).mean(1)
        return self.fc(pooled)


def run(rows, method, body, emb_dim, n_classes, device, epochs, lr, bs, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed); rng.shuffle(rows)
    n = len(rows); ntr = int(0.8 * n); nva = int(0.1 * n)
    tr, va, te = rows[:ntr], rows[ntr:ntr+nva], rows[ntr+nva:]
    bpe_vocab = 0; tok = None
    if method == 'standard':
        tok = train_bpe([r['text'] for r in tr], vocab_size=10000)
        bpe_vocab = tok.get_vocab_size()
        mk = lambda d: DataLoader(BPEDS(d, tok), batch_size=bs, shuffle=False, collate_fn=collate)
        trL = DataLoader(BPEDS(tr, tok), batch_size=bs, shuffle=True, collate_fn=collate)
    else:
        mk = lambda d: DataLoader(ByteDS(d), batch_size=bs, shuffle=False, collate_fn=collate)
        trL = DataLoader(ByteDS(tr), batch_size=bs, shuffle=True, collate_fn=collate)
    vaL, teL = mk(va), mk(te)
    model = Net(method, emb_dim, n_classes, body, bpe_vocab=bpe_vocab).to(device)
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
        if v > best: best = v; best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    return {'method': method, 'bpe_vocab': bpe_vocab, 'test_acc': ev(teL), 'val_acc': best,
            'encoder_params': sum(p.numel() for p in model.emb.parameters()),
            'total_params': sum(p.numel() for p in model.parameters())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--lang_counts', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32, 51])
    p.add_argument('--methods', nargs='+', default=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--body', default='A')
    p.add_argument('--emb_dim', type=int, default=64)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--n_per', type=int, default=4000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    locales = all_locales()
    print(f'{len(locales)} locales; sweep {args.lang_counts}; body={args.body}')
    summary = {}
    for N in args.lang_counts:
        rows = load_locales(locales[:N], args.n_per)
        print(f'\n=== {N} langs: {len(rows)} utts ===')
        for m in args.methods:
            t0 = time.time()
            r = run(list(rows), m, args.body, args.emb_dim, 60, device,
                    args.epochs, args.lr, args.bs, args.seed)
            r['n_languages'] = N
            summary[f'{N}lang_{m}'] = r
            print(f'  {m:11s} vocab={r["bpe_vocab"]:6d} acc={r["test_acc"]:.4f} '
                  f'enc={r["encoder_params"]:>9d} [{time.time()-t0:.0f}s]')
            json.dump(summary, open(os.path.join(args.out_dir, 'summary.json'), 'w'), indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
