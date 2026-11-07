#!/usr/bin/env python3
"""Encoding-scheme ablation on real CIViC / body D.

Compares the binary code-assignment scheme at fixed W and n:
  standard (full table) | random (ours) | hash (SimHash) | Gray | hierarchical
Everything else identical to run_bits_width_ablation.py (seed 42, Adam 5e-4,
6 epochs, body D, d=64). Structured tables are injected through the
simhash_seb code-load path (SPINHASH_CODES)."""
import argparse, functools, json, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))

import run_civic_extended as RCE
from run_civic_extended import MultiModalClassifier
from run_civic_multimodal import (
    PAD, MODALITIES, load_civic, build_vocab, CIViCMultiModalDataset, collate, film_attack,
)
import hashlib


def tag_attack(model, ids, label, criterion, device, vocab_size):
    """TAG gradient-inversion token recovery F1 (lower = more private)."""
    model.zero_grad(set_to_none=False)
    loss = criterion(model(ids.to(device)), label.to(device))
    loss.backward()
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    if model.method == 'standard':
        gnorm = model.embedding.weight.grad.detach().norm(dim=1).cpu()
        recovered = set(torch.topk(gnorm, max(len(actual), 1)).indices.tolist())
    else:
        rows = []
        with torch.no_grad():
            for s in range(0, vocab_size, 2048):
                e = min(s + 2048, vocab_size)
                rows.append(model.embedding(torch.arange(s, e, device=device).unsqueeze(0)).squeeze(0))
        emb_table = torch.cat(rows, dim=0)
        g = model.fc.weight.grad.detach().mean(dim=0).unsqueeze(0)
        sim = (torch.nn.functional.normalize(g, dim=-1) @
               torch.nn.functional.normalize(emb_table, dim=-1).t()).abs().squeeze(0).cpu()
        recovered = set(torch.topk(sim, max(len(actual), 1)).indices.tolist())
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


def simple_hash_table(V, nbits, seed=0):
    """Simplest possible deterministic hash: MD5(seed:id) -> take n bits.
    No similarity preservation, no stored codebook, attributable to no paper."""
    tab = torch.zeros(V, nbits, dtype=torch.long)
    for i in range(V):
        h = hashlib.md5(f"{seed}:{i}".encode()).digest()   # 16 bytes = 128 bits
        bits = [(byte >> b) & 1 for byte in h for b in range(8)]
        while len(bits) < nbits:                            # extend for n>128 by re-hashing
            h = hashlib.md5(h).digest()
            bits += [(byte >> b) & 1 for byte in h for b in range(8)]
        tab[i] = torch.tensor(bits[:nbits], dtype=torch.long)
    return tab


def _svd_sign_codes(feat, nbits):
    """feat: (V, F) -> top-nbits SVD directions -> per-dim median-threshold sign bits (balanced, similarity-preserving)."""
    U, S, _ = np.linalg.svd(feat, full_matrices=False)
    k = min(nbits, U.shape[1])
    emb = U[:, :k] * S[:k]
    codes = (emb > np.median(emb, axis=0, keepdims=True)).astype(np.int64)
    if codes.shape[1] < nbits:                       # pad if fewer SVD dims than nbits
        codes = np.concatenate([codes, np.zeros((codes.shape[0], nbits - codes.shape[1]), dtype=np.int64)], axis=1)
    return torch.tensor(codes, dtype=torch.long)


def semantic_code(V, nbits, id_lists):
    """Semantically-aware (attribution-free): PPMI co-occurrence -> SVD -> sign bits.
    Similar-context tokens get similar codes. No hashing paper involved."""
    C = np.zeros((V, V), dtype=np.float32)
    for ids in id_lists:
        u = np.unique(np.array([i for i in ids if 0 < i < V], dtype=np.int64))
        if len(u):
            C[np.ix_(u, u)] += 1.0
    tot = C.sum() + 1e-9
    r = C.sum(1, keepdims=True); c = C.sum(0, keepdims=True)
    pmi = np.log((C * tot) / (r * c + 1e-9) + 1e-9); pmi[pmi < 0] = 0.0
    return _svd_sign_codes(pmi, nbits)


def surface_code(itos, nbits, F=512):
    """Surface-similarity (attribution-free): char-trigram features -> SVD -> sign bits.
    Similar-spelling tokens get similar codes."""
    feat = np.zeros((len(itos), F), dtype=np.float32)
    for i, tok in enumerate(itos):
        s = '  ' + str(tok) + '  '
        for j in range(len(s) - 2):
            h = int(hashlib.md5(s[j:j+3].encode()).hexdigest(), 16) % F
            feat[i, h] += 1.0
    return _svd_sign_codes(feat, nbits)


class LearnedCodeEncoder(nn.Module):
    """Learned codes: a TRAINABLE (V x n) code table -> shared FFN (vs fixed random)."""
    def __init__(self, V, n, W, d):
        super().__init__()
        self.codes = nn.Parameter(torch.randn(V, n) * 0.1)
        self.ffn = nn.Sequential(nn.Linear(n, W), nn.GELU(), nn.Linear(W, d))
    def forward(self, ids, dropout=None):
        return self.ffn(self.codes[ids])


def gray_table(V, nbits):
    """Reflected Gray code: consecutive ids differ by exactly 1 bit."""
    tab = torch.zeros(V, nbits, dtype=torch.long)
    for i in range(V):
        g = i ^ (i >> 1)
        for b in range(nbits):
            tab[i, b] = (g >> b) & 1
    return tab


def hierarchical_table(V, nbits):
    """Prefix bits = coarse cluster (id>>6), suffix bits = fine position (id&63).
    Nearby ids share a prefix -> taxonomic/structured codes."""
    half = nbits // 2
    shift = 6
    tab = torch.zeros(V, nbits, dtype=torch.long)
    for i in range(V):
        coarse = i >> shift
        fine = i & ((1 << shift) - 1)
        for b in range(half):
            tab[i, b] = (coarse >> b) & 1
        for b in range(half):
            tab[i, half + b] = (fine >> b) & 1
    return tab


def evaluate(model, loader, device):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            pred = model(batch['ids'].to(device)).argmax(-1).cpu()
            c += (pred == batch['label']).sum().item()
            t += batch['label'].size(0)
    return c / max(t, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='results/bits_encoding')
    ap.add_argument('--n_epochs', type=int, default=6)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--nbits', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_attack', type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, val_ex, test_ex = load_civic(seed=42)
    stoi, vocab_meta = build_vocab(train_ex, MODALITIES, boundary='separator')
    V = len(stoi)
    num_classes = 5
    itos = [None] * V
    for tok, i in stoi.items():
        itos[i] = tok
    print(f'V={V} train={len(train_ex)}/val={len(val_ex)}/test={len(test_ex)} '
          f'W={a.width} n={a.nbits}')

    coll = functools.partial(collate, pad_index=stoi[PAD])
    train_ds = CIViCMultiModalDataset(train_ex, stoi, vocab_meta, MODALITIES, ablation='all')
    val_ds = CIViCMultiModalDataset(val_ex, stoi, vocab_meta, MODALITIES, ablation='all')
    test_ds = CIViCMultiModalDataset(test_ex, stoi, vocab_meta, MODALITIES, ablation='all')
    tr = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=coll)
    val_l = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=coll)
    test_l = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=coll)

    codedir = os.path.join(a.out_dir, 'codes')
    os.makedirs(codedir, exist_ok=True)
    # per-example token-id lists for the co-occurrence (semantic) code
    id_lists = []
    for batch in tr:
        if batch is None:
            continue
        for row in batch['ids']:
            id_lists.append([int(x) for x in row.tolist() if x > 0])
    print(f'  collected {len(id_lists)} training sequences for co-occurrence')
    tables = {
        'hash': simple_hash_table(V, a.nbits),
        'semantic': semantic_code(V, a.nbits, id_lists),
        'surface': surface_code(itos, a.nbits),
        'gray': gray_table(V, a.nbits),
        'hierarchical': hierarchical_table(V, a.nbits),
    }
    paths = {}
    for name, tab in tables.items():
        p = os.path.join(codedir, f'{name}.pt')
        torch.save(tab, p)
        paths[name] = p

    plan = [
        ('standard',     'standard',    None),
        ('random(ours)', 'binary_seb',  None),
        ('hash(md5)',    'simhash_seb', paths['hash']),
        ('learned',      'learned',     None),
        ('semantic',     'simhash_seb', paths['semantic']),
        ('surface',      'simhash_seb', paths['surface']),
        ('gray',         'simhash_seb', paths['gray']),
        ('hierarchical', 'simhash_seb', paths['hierarchical']),
    ]

    summary = {}
    for enc, method, cpath in plan:
        print(f'\n>>> encoding={enc} method={method}')
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(a.seed)
        if method == 'learned':
            model = MultiModalClassifier(V, 64, num_classes, method='binary_seb', body_kind='D',
                                         pad_idx=stoi[PAD], num_bits=a.nbits,
                                         interdim=str(a.width)).to(device)
            model.embedding = LearnedCodeEncoder(V, a.nbits, a.width, 64).to(device)
            model.method = 'learned'
        else:
            if cpath is not None:
                RCE.SPINHASH_CODES[method] = cpath
            model = MultiModalClassifier(V, 64, num_classes, method=method, body_kind='D',
                                         pad_idx=stoi[PAD], num_bits=a.nbits,
                                         interdim=str(a.width)).to(device)
        opt = optim.Adam(model.parameters(), lr=5e-4)
        crit = nn.CrossEntropyLoss()
        best_val, best_state = 0.0, None
        for ep in range(a.n_epochs):
            model.train()
            for batch in tr:
                if batch is None:
                    continue
                opt.zero_grad()
                crit(model(batch['ids'].to(device)), batch['label'].to(device)).backward()
                opt.step()
            v = evaluate(model, val_l, device)
            print(f'  ep{ep+1} val={v:.4f}')
            if v > best_val:
                best_val = v
                best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        test_acc = evaluate(model, test_l, device)
        emb_params = sum(p.numel() for p in model.embedding.parameters())
        n_params = sum(p.numel() for p in model.parameters())
        # privacy: FILM + TAG gradient-inversion token-recovery F1 (lower = more private)
        model.train()
        film_f1s, tag_f1s, n_atk = [], [], 0
        for batch in tr:
            if batch is None or n_atk >= a.n_attack:
                break
            try: film_f1s.append(film_attack(model, batch['ids'], batch['label'], crit, device))
            except Exception as e: print(f'  film err: {e}')
            try: tag_f1s.append(tag_attack(model, batch['ids'], batch['label'], crit, device, V))
            except Exception as e: print(f'  tag err: {e}')
            n_atk += 1
        film_f1 = float(np.mean(film_f1s)) if film_f1s else 0.0
        tag_f1 = float(np.mean(tag_f1s)) if tag_f1s else 0.0
        summary[enc] = {
            'encoding': enc, 'method': method,
            'val_acc': float(best_val), 'test_acc': float(test_acc),
            'emb_params': int(emb_params), 'total_params': int(n_params),
            'film_f1': film_f1, 'tag_f1': tag_f1,
        }
        print(f'  test={test_acc:.4f}  emb_params={emb_params}  FILM={film_f1:.4f}  TAG={tag_f1:.4f}')
        with open(os.path.join(a.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    print(f'\nWrote {a.out_dir}/summary.json')


if __name__ == '__main__':
    main()
