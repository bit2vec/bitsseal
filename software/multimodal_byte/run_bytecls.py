#!/usr/bin/env python3
"""Generic byte-level classification harness (faithful EmByte/Bits language
method). Reuses the authoritative byte-level encoders:
  standard   : BPE subword (~10k) -> learned table.
  byte_seb   : real UTF-8 bytes (sliding window) -> one-hot(256) -> FFN.
  binary_seb : real bits of the bytes -> one-hot(2) -> FFN.
All vocabulary-independent (byte/binary). Single-field or multi-field fusion
(fields joined by a separator byte). Pluggable loaders in bytecls_loaders.py.
"""
import argparse, json, os, random, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_massive_v3 import ByteSEBEmbedding, BinarySEBEmbedding   # authoritative
from bytecls_loaders import LOADERS

SEP_BYTE = 10  # '\n' separates fused fields in the byte stream


class ByteDS(Dataset):
    def __init__(self, rows, fields, max_len=256):
        self.rows = rows; self.fields = fields; self.max_len = max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        parts = []
        for fi, f in enumerate(self.fields):
            if fi: parts.append(SEP_BYTE)
            parts.extend(list(str(self.rows[i].get(f, '')).encode('utf-8')))
        b = parts[:self.max_len]; b += [0] * (self.max_len - len(b))
        return torch.tensor(b, dtype=torch.long), self.rows[i]['label_idx']


class BPEDS(Dataset):
    def __init__(self, rows, fields, tok, max_len=96):
        self.rows = rows; self.fields = fields; self.tok = tok; self.max_len = max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        text = ' \n '.join(str(self.rows[i].get(f, '')) for f in self.fields)
        ids = self.tok.encode(text).ids[:self.max_len]
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
    def __init__(self, method, emb_dim, n_classes, body, bpe_vocab=0, num_bits=64, max_byte=256):
        super().__init__()
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


def run(rows, fields, n_classes, method, body, emb_dim, device, epochs, lr, bs, seed, max_len):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed); rng.shuffle(rows)
    n = len(rows); ntr = int(0.8 * n); nva = int(0.1 * n)
    tr, va, te = rows[:ntr], rows[ntr:ntr+nva], rows[ntr+nva:]
    bpe_vocab = 0
    if method == 'standard':
        tok = train_bpe([' \n '.join(str(r.get(f, '')) for f in fields) for r in tr])
        bpe_vocab = tok.get_vocab_size()
        ds = lambda d: BPEDS(d, fields, tok, max_len=max(64, max_len // 3))
    else:
        ds = lambda d: ByteDS(d, fields, max_len=max_len)
    trL = DataLoader(ds(tr), batch_size=bs, shuffle=True, collate_fn=collate)
    vaL = DataLoader(ds(va), batch_size=bs, collate_fn=collate)
    teL = DataLoader(ds(te), batch_size=bs, collate_fn=collate)
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
    return {'method': method, 'body': body, 'bpe_vocab': bpe_vocab,
            'test_acc': ev(teL), 'val_acc': best,
            'encoder_params': sum(p.numel() for p in model.emb.parameters()),
            'total_params': sum(p.numel() for p in model.parameters())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=list(LOADERS.keys()))
    p.add_argument('--out_dir', required=True)
    p.add_argument('--methods', nargs='+', default=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--bodies', nargs='+', default=['A', 'D'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--emb_dim', type=int, default=64)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--n_target', type=int, default=20000)
    p.add_argument('--max_len', type=int, default=256)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows, fields, label_names = LOADERS[args.dataset](n_target=args.n_target)
    print(f'{args.dataset}: {len(rows)} rows, fields={fields}, classes={len(label_names)}')
    summary = {}
    for method in args.methods:
        for body in args.bodies:
            accs = []; params = None
            for seed in args.seeds:
                r = run(list(rows), fields, len(label_names), method, body,
                        args.emb_dim, device, args.epochs, args.lr, args.bs, seed, args.max_len)
                accs.append(r['test_acc']); params = r['encoder_params']
                print(f'  {method}_{body}_s{seed}: acc={r["test_acc"]:.4f} '
                      f'enc={r["encoder_params"]} vocab={r["bpe_vocab"]}')
            summary[f'{method}_{body}'] = {
                'method': method, 'body': body, 'test_accs': accs,
                'mean_acc': float(np.mean(accs)), 'std_acc': float(np.std(accs)),
                'encoder_params': params}
            json.dump(summary, open(os.path.join(args.out_dir, 'summary.json'), 'w'), indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
