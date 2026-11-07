#!/usr/bin/env python3
"""Extended privacy attacks on the four multimodal genomics regression tasks.

Adds Robbing the Fed, Decepticons, FL LLG token, LiRA, APRIL on top of the
FILM/TAG numbers already in run_*_regression.py.

Outputs results/regression_extended/summary.json with per dataset, per attack,
per method scores.
"""
import argparse, functools, json, os, random, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))

from run_civic_extended import MultiModalClassifier
from run_civic_multimodal import PAD, UNK, SEP, tokenize

from run_pharmgkb_regression import (
    load_pharmgkb_regression, MODALITIES as PG_MOD,
    PGKBRegDataset, transform as pg_transform,
)
from run_civic_regression import (
    load_civic_regression, MODALITIES as CV_MOD, CIViCRegDataset,
)
from run_clinvar_regression import (
    PATHOGENICITY, MODALITIES as CL_MOD, CVRegDataset,
)
from run_tcga_regression import (
    load_tcga_regression, MODALITIES as TC_MOD, TCGARegDataset,
)
from clinvar_loader import load_clinvar


def reg_collate(batch, pad_index=0):
    if not batch: return None
    ids = [b[0] for b in batch]
    L = max(x.size(0) for x in ids)
    ids_p = torch.full((len(ids), L), pad_index, dtype=torch.long)
    for i, x in enumerate(ids): ids_p[i, :x.size(0)] = x
    y = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return {'ids': ids_p, 'label': y}


def materialize_embeddings(model, vocab_size, device, chunk=2048):
    rows = []; model.eval()
    with torch.no_grad():
        for s in range(0, vocab_size, chunk):
            e = min(s + chunk, vocab_size)
            ids = torch.arange(s, e, device=device).unsqueeze(0)
            rows.append(model.embedding(ids).squeeze(0))
    return torch.cat(rows, dim=0)


# =====================================================================
# RtF
# =====================================================================
class ImprintWrappedReg(nn.Module):
    def __init__(self, base, imprint_dim=512):
        super().__init__()
        self.embedding = base.embedding
        self.body = base.body
        self.method = base.method
        d_in = base.fc.in_features
        self.imprint = nn.Linear(d_in, imprint_dim, bias=True)
        with torch.no_grad():
            W = torch.randn(imprint_dim, d_in)
            W = W / W.norm(dim=1, keepdim=True)
            self.imprint.weight.copy_(W); self.imprint.bias.zero_()
        self.fc = nn.Linear(imprint_dim, 1)

    def forward(self, ids):
        emb = self.embedding(ids)
        h = self.body(emb)
        z = torch.relu(self.imprint(h))
        return self.fc(z).squeeze(-1)


def rtf_reg(model, ids, target_y, criterion, device, vocab_size):
    wrapped = ImprintWrappedReg(model, imprint_dim=512).to(device)
    wrapped.train(); wrapped.zero_grad(set_to_none=False)
    pred = wrapped(ids.to(device))
    loss = criterion(pred, target_y.to(device))
    loss.backward()
    g = wrapped.imprint.weight.grad.detach()
    row_norms = g.norm(dim=1)
    threshold = row_norms.max().item() * 0.10
    if threshold == 0: threshold = 1e-12
    active = (row_norms > threshold).nonzero(as_tuple=True)[0]
    if len(active) == 0:
        recovered = set()
    else:
        with torch.no_grad():
            tab = materialize_embeddings(model, vocab_size, device)
            sim = F.normalize(g[active], dim=-1) @ F.normalize(tab, dim=-1).t()
            recovered = set(sim.argmax(dim=-1).cpu().tolist())
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    if model.method != 'standard' and len(recovered) > 5 * len(actual):
        recovered = set(random.sample(list(recovered), min(20, len(recovered))))
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


# =====================================================================
# Decepticons
# =====================================================================
class AttnWrappedReg(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.embedding = base.embedding
        self.body = base.body
        self.fc = base.fc
        self.method = base.method
        d = base.fc.in_features
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            for L in (self.q, self.k, self.v):
                L.weight.copy_(torch.eye(d) + 1e-3 * torch.randn(d, d))


def decepticons_reg(model, ids, target_y, criterion, device, vocab_size):
    wrapped = AttnWrappedReg(model).to(device)
    wrapped.train(); wrapped.zero_grad(set_to_none=False)
    ids_d = ids.to(device)
    emb = wrapped.embedding(ids_d).detach().clone().requires_grad_(True)
    Q = wrapped.q(emb); K = wrapped.k(emb); V = wrapped.v(emb)
    scale = (Q.size(-1)) ** -0.5
    attn = F.softmax(Q @ K.transpose(-1, -2) * scale, dim=-1)
    ctx = attn @ V
    h = wrapped.body(ctx)
    pred = wrapped.fc(h).squeeze(-1)
    loss = criterion(pred, target_y.to(device))
    loss.backward()
    g = emb.grad.detach()
    B, T, d = g.shape
    flat = g.reshape(B * T, d)
    with torch.no_grad():
        tab = materialize_embeddings(model, vocab_size, device)
        sim = F.normalize(flat, dim=-1) @ F.normalize(tab, dim=-1).t()
        recovered = set(sim.argmax(dim=-1).cpu().tolist())
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


# =====================================================================
# FL LLG token recovery (loss agnostic — embedding gradient sparsity)
# =====================================================================
def llg_token_reg(model, ids, target_y, criterion, device, vocab_size):
    model.zero_grad(set_to_none=False)
    pred = model(ids.to(device)).squeeze(-1)
    loss = criterion(pred, target_y.to(device))
    loss.backward()
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    if model.method == 'standard':
        gnorm = model.embedding.weight.grad.detach().norm(dim=1).cpu()
        # LLG variant: top-k by norm with k = expected count
        B, T = ids.shape
        k = max(int(B * T * 0.5), len(actual))
        recovered = set(torch.topk(gnorm, min(k, vocab_size)).indices.tolist())
    else:
        recovered = set(range(vocab_size))
        if len(recovered) > 5 * len(actual):
            recovered = set(random.sample(list(recovered), min(20, len(recovered))))
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


# =====================================================================
# APRIL multi-round trajectory matching (regression variant)
# =====================================================================
def april_reg(model, batches, criterion, device, vocab_size,
              embedding_dim=64, num_iters=200, n_rounds=3):
    init_W = model.fc.weight.detach().clone()
    init_b = model.fc.bias.detach().clone()
    snap_Ws, snap_bs = [], []
    real_Ws, real_bs = [], []
    real_ids = []
    for r in range(n_rounds):
        pW = (init_W + 0.005 * torch.randn_like(init_W)).requires_grad_(True)
        pb = (init_b + 0.005 * torch.randn_like(init_b)).requires_grad_(True)
        ids, target_y = batches[r]
        with torch.no_grad():
            emb = model.embedding(ids.to(device))
            h = model.body(emb).detach()
        pred = (h @ pW.t() + pb).squeeze(-1)
        loss = criterion(pred, target_y.to(device))
        gW, gb = torch.autograd.grad(loss, [pW, pb])
        snap_Ws.append(pW.detach()); snap_bs.append(pb.detach())
        real_Ws.append(gW.detach()); real_bs.append(gb.detach())
        real_ids.append(ids)

    B, T = real_ids[0].shape
    dummies = [torch.randn(B, T, embedding_dim, device=device, requires_grad=True)
               for _ in range(n_rounds)]
    dummy_y = torch.zeros(B, device=device, requires_grad=True)
    optim_ = torch.optim.Adam(dummies + [dummy_y], lr=0.05)

    for it in range(num_iters):
        optim_.zero_grad()
        total = 0.0
        for r in range(n_rounds):
            W_r = snap_Ws[r].detach().requires_grad_(True)
            b_r = snap_bs[r].detach().requires_grad_(True)
            d_h = dummies[r].mean(dim=1)
            d_pred = (d_h @ W_r.t() + b_r).squeeze(-1)
            d_loss = ((d_pred - dummy_y) ** 2).mean()
            d_gW, d_gb = torch.autograd.grad(d_loss, [W_r, b_r],
                                              create_graph=True, retain_graph=True)
            cos_w = -(d_gW.flatten() * real_Ws[r].flatten()).sum() / (d_gW.norm() * real_Ws[r].norm() + 1e-9)
            cos_b = -(d_gb.flatten() * real_bs[r].flatten()).sum() / (d_gb.norm() * real_bs[r].norm() + 1e-9)
            total = total + cos_w + cos_b
        total.backward()
        optim_.step()

    actual_all, recovered_all = set(), set()
    with torch.no_grad():
        tab = materialize_embeddings(model, vocab_size, device)
        for r in range(n_rounds):
            flat = dummies[r].detach().reshape(B * T, embedding_dim)
            sim = F.normalize(flat, dim=-1) @ F.normalize(tab, dim=-1).t()
            recovered_all.update(sim.argmax(dim=-1).cpu().tolist())
            actual_all.update(int(t) for t in real_ids[r].flatten().tolist() if t > 0)
    tp = len(recovered_all & actual_all)
    p = tp / max(len(recovered_all), 1); r = tp / max(len(actual_all), 1)
    return 2 * p * r / max(p + r, 1e-9)


# =====================================================================
# LiRA-lite (regression: per-example MSE losses as the score)
# =====================================================================
def lira_reg(method, build_train_fn, vocab_size, stoi, embedding_dim, n_epochs,
             seed, device, n_shadow=4):
    from sklearn.metrics import roc_auc_score
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_ds, _, _, _ = build_train_fn()
    n = len(train_ds)
    n_audit = min(n, 600)
    rng = np.random.default_rng(seed)
    inclusion = np.zeros((n_shadow, n_audit), dtype=bool)
    coll = functools.partial(reg_collate, pad_index=stoi[PAD])
    losses_matrix = np.zeros((n_shadow, n_audit))

    for s in range(n_shadow):
        mask = rng.random(n_audit) < 0.5
        inclusion[s] = mask
        sub_ids = [i for i in range(n_audit) if mask[i]]
        if not sub_ids: sub_ids = list(range(n_audit // 2))
        sub_ds = Subset(train_ds, sub_ids)
        loader = DataLoader(sub_ds, batch_size=16, shuffle=True, collate_fn=coll)
        torch.manual_seed(seed + s + 1)
        model = MultiModalClassifier(vocab_size, embedding_dim, 1,
                                      method=method, body_kind='D',
                                      pad_idx=stoi[PAD]).to(device)
        opt = optim.Adam(model.parameters(), lr=5e-4)
        crit = nn.MSELoss()
        for _ in range(n_epochs):
            model.train()
            for batch in loader:
                if batch is None: continue
                opt.zero_grad()
                p = model(batch['ids'].to(device)).squeeze(-1)
                crit(p, batch['label'].to(device)).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            for j in range(n_audit):
                ids, y = train_ds[j]
                ids_b = ids.unsqueeze(0).to(device)
                p = model(ids_b).squeeze(-1).item()
                losses_matrix[s, j] = (p - y) ** 2

    # LiRA score: per-example membership = lower loss when included
    scores, labels = [], []
    for j in range(n_audit):
        in_loss = losses_matrix[inclusion[:, j], j]
        out_loss = losses_matrix[~inclusion[:, j], j]
        if len(in_loss) == 0 or len(out_loss) == 0: continue
        # Score: difference (out - in); membership label is 1 = in
        for s in range(n_shadow):
            scores.append(losses_matrix[s, j])
            labels.append(0 if inclusion[s, j] else 1)  # 1 when 'out' (high loss)
    if not scores:
        return 0.5
    auc = roc_auc_score(labels, scores)
    return float(auc)


# =====================================================================
# Per dataset setup
# =====================================================================
def setup_pharmgkb():
    examples = load_pharmgkb_regression(seed=42)
    n = len(examples); n_tr = int(0.7 * n); n_va = int(0.1 * n)
    train = examples[:n_tr]; val = examples[n_tr:n_tr + n_va]; test = examples[n_tr + n_va:]
    y_tr = np.array([pg_transform(e['score']) for e in train])
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    counter = Counter()
    for ex in train:
        for mod, _ in PG_MOD: counter.update(tokenize(str(ex.get(mod, ''))))
    stoi = {PAD: 0, UNK: 1}
    for mod, _ in PG_MOD: stoi[f'{SEP}_{mod}'] = len(stoi)
    for tok, c in counter.most_common():
        if c >= 2: stoi[tok] = len(stoi)

    def build_train_fn():
        tr = PGKBRegDataset(train, stoi, PG_MOD, y_mean, y_std)
        va = PGKBRegDataset(val,   stoi, PG_MOD, y_mean, y_std)
        te = PGKBRegDataset(test,  stoi, PG_MOD, y_mean, y_std)
        return tr, va, te, None

    return 'PharmGKB', build_train_fn, len(stoi), stoi


def setup_civic():
    examples = load_civic_regression(seed=42)
    n = len(examples); n_tr = int(0.7 * n); n_va = int(0.1 * n)
    train = examples[:n_tr]; val = examples[n_tr:n_tr + n_va]; test = examples[n_tr + n_va:]
    y_tr = np.array([e['rating'] for e in train])
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    counter = Counter()
    for ex in train:
        for mod, _ in CV_MOD: counter.update(tokenize(str(ex.get(mod, ''))))
    stoi = {PAD: 0, UNK: 1}
    for mod, _ in CV_MOD: stoi[f'{SEP}_{mod}'] = len(stoi)
    for tok, c in counter.most_common():
        if c >= 2: stoi[tok] = len(stoi)

    def build_train_fn():
        tr = CIViCRegDataset(train, stoi, CV_MOD, y_mean, y_std)
        va = CIViCRegDataset(val,   stoi, CV_MOD, y_mean, y_std)
        te = CIViCRegDataset(test,  stoi, CV_MOD, y_mean, y_std)
        return tr, va, te, None

    return 'CIViC', build_train_fn, len(stoi), stoi


def setup_clinvar():
    examples, _ = load_clinvar()
    examples = [e for e in examples if e['label'] in PATHOGENICITY]
    for e in examples: e['rating'] = PATHOGENICITY[e['label']]
    rng = random.Random(42); rng.shuffle(examples)
    n = len(examples); n_tr = int(0.7 * n); n_va = int(0.1 * n)
    train = examples[:n_tr]; val = examples[n_tr:n_tr + n_va]; test = examples[n_tr + n_va:]
    y_tr = np.array([e['rating'] for e in train])
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    counter = Counter()
    for ex in train:
        for mod, _ in CL_MOD: counter.update(tokenize(str(ex.get(mod, ''))))
    stoi = {PAD: 0, UNK: 1}
    for mod, _ in CL_MOD: stoi[f'{SEP}_{mod}'] = len(stoi)
    for tok, c in counter.most_common():
        if c >= 2: stoi[tok] = len(stoi)

    def build_train_fn():
        tr = CVRegDataset(train, stoi, CL_MOD, y_mean, y_std)
        va = CVRegDataset(val,   stoi, CL_MOD, y_mean, y_std)
        te = CVRegDataset(test,  stoi, CL_MOD, y_mean, y_std)
        return tr, va, te, None

    return 'ClinVar', build_train_fn, len(stoi), stoi


def setup_tcga():
    examples = load_tcga_regression()
    n = len(examples); n_tr = int(0.7 * n); n_va = int(0.1 * n)
    train = examples[:n_tr]; val = examples[n_tr:n_tr + n_va]; test = examples[n_tr + n_va:]
    y_tr = np.array([e['age'] for e in train])
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    counter = Counter()
    for ex in train:
        for mod, _ in TC_MOD: counter.update(tokenize(str(ex.get(mod, ''))))
    stoi = {PAD: 0, UNK: 1}
    for mod, _ in TC_MOD: stoi[f'{SEP}_{mod}'] = len(stoi)
    for tok, c in counter.most_common():
        if c >= 2: stoi[tok] = len(stoi)

    def build_train_fn():
        tr = TCGARegDataset(train, stoi, TC_MOD, y_mean, y_std)
        va = TCGARegDataset(val,   stoi, TC_MOD, y_mean, y_std)
        te = TCGARegDataset(test,  stoi, TC_MOD, y_mean, y_std)
        return tr, va, te, None

    return 'TCGA', build_train_fn, len(stoi), stoi


# =====================================================================
# Main
# =====================================================================
def run_dataset_attacks(name, build_train_fn, vocab_size, stoi, methods,
                        n_epochs, n_attack, device, embedding_dim=64):
    out = {m: {} for m in methods}
    coll = functools.partial(reg_collate, pad_index=stoi[PAD])
    for method in methods:
        print(f'\n[{name} | {method}]')
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        if torch.cuda.is_available(): torch.cuda.manual_seed(42)
        train_ds, _, _, _ = build_train_fn()
        loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=coll)
        model = MultiModalClassifier(vocab_size, embedding_dim, 1,
                                      method=method, body_kind='D',
                                      pad_idx=stoi[PAD]).to(device)
        opt = optim.Adam(model.parameters(), lr=5e-4)
        crit = nn.MSELoss()
        for _ in range(n_epochs):
            model.train()
            for batch in loader:
                if batch is None: continue
                opt.zero_grad()
                p = model(batch['ids'].to(device)).squeeze(-1)
                crit(p, batch['label'].to(device)).backward()
                opt.step()

        model.train()
        rtf_f1, dec_f1, llg_f1 = [], [], []
        atk_batches = []
        n_atk = 0
        for batch in loader:
            if batch is None or n_atk >= n_attack: break
            atk_batches.append((batch['ids'], batch['label']))
            try: rtf_f1.append(rtf_reg(model, batch['ids'], batch['label'], crit, device, vocab_size))
            except Exception as e: print(f'  rtf err: {e}')
            try: dec_f1.append(decepticons_reg(model, batch['ids'], batch['label'], crit, device, vocab_size))
            except Exception as e: print(f'  dec err: {e}')
            try: llg_f1.append(llg_token_reg(model, batch['ids'], batch['label'], crit, device, vocab_size))
            except Exception as e: print(f'  llg err: {e}')
            n_atk += 1

        try:
            april_score = april_reg(model, atk_batches[:3], crit, device,
                                     vocab_size, embedding_dim=embedding_dim)
        except Exception as e:
            print(f'  april err: {e}'); april_score = 0.0

        try:
            lira_auc = lira_reg(method, build_train_fn, vocab_size, stoi,
                                embedding_dim, n_epochs, seed=42, device=device,
                                n_shadow=4)
        except Exception as e:
            print(f'  lira err: {e}'); lira_auc = 0.5

        out[method] = {
            'rtf':         float(np.mean(rtf_f1))  if rtf_f1  else 0.0,
            'decepticons': float(np.mean(dec_f1))  if dec_f1  else 0.0,
            'fl_llg':      float(np.mean(llg_f1))  if llg_f1  else 0.0,
            'april':       float(april_score),
            'lira_auc':    float(lira_auc),
            'vocab_size':  vocab_size,
        }
        print(f'  RtF={out[method]["rtf"]:.3f} Dec={out[method]["decepticons"]:.3f} '
              f'LLG={out[method]["fl_llg"]:.3f} APRIL={out[method]["april"]:.3f} '
              f'LiRA={out[method]["lira_auc"]:.3f}')
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/regression_extended')
    p.add_argument('--n_epochs', type=int, default=4)
    p.add_argument('--n_attack', type=int, default=8)
    p.add_argument('--datasets', nargs='+',
                   default=['pharmgkb', 'civic', 'clinvar', 'tcga'])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setups = {
        'pharmgkb': setup_pharmgkb,
        'civic':    setup_civic,
        'clinvar':  setup_clinvar,
        'tcga':     setup_tcga,
    }
    summary = {}
    for d in args.datasets:
        print(f'\n{"="*60}\n{d}\n{"="*60}')
        name, build_train_fn, vocab_size, stoi = setups[d]()
        out = run_dataset_attacks(name, build_train_fn, vocab_size, stoi,
                                  ['standard', 'byte_seb'],
                                  args.n_epochs, args.n_attack, device)
        summary[name] = out
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
