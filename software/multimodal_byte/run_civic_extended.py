#!/usr/bin/env python3
"""Extended CIViC multi-modal grid:
  bodies: A=meanpool, B=BiLSTM, C=cross-attention, D=cross-attn+MLP, E=transformer
  attacks per cell: FILM, TAG, LLG, LAMP, MIA, DLG, EmbInv

Same dataset/loader/budget rule as run_civic_multimodal.py (see README.md).
"""
import argparse, csv, functools, json, math, os, random, re, sys, time
from collections import Counter

import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))

from binary_subword_sentiment import BinarySubwordConcateOnehot
from byte_subword_concate_onehot import ByteSubwordConcateOnehot

# Reuse loader + collator from the v1 runner
sys.path.insert(0, THIS_DIR)
from run_civic_multimodal import (
    PAD, UNK, SEP, CIVIC_TSV, LABEL_SET, LABEL2IDX, MODALITIES,
    tokenize, load_civic, build_vocab, encode_example,
    CIViCMultiModalDataset, collate, film_attack,
)


# ============== Body modules ==============
class MeanPoolBody(nn.Module):
    def forward(self, emb): return emb.mean(dim=1)


class BiLSTMBody(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(d, hidden, batch_first=True, bidirectional=True)
        self.out_dim = hidden * 2
    def forward(self, emb):
        out, (hN, _) = self.lstm(emb)
        return torch.cat([hN[0], hN[1]], dim=-1)


class CrossAttnBody(nn.Module):
    """Single learnable query token cross-attends over the sequence."""
    def __init__(self, d, n_heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d))
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.out_dim = d
    def forward(self, emb):
        B = emb.size(0)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, emb, emb, need_weights=False)
        return out.squeeze(1)


class CrossAttnMLPBody(nn.Module):
    """Cross-attention + MLP refinement."""
    def __init__(self, d, n_heads=4, mlp_dim=256):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d))
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_dim), nn.ReLU(),
            nn.Linear(mlp_dim, d), nn.LayerNorm(d),
        )
        self.out_dim = d
    def forward(self, emb):
        B = emb.size(0)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, emb, emb, need_weights=False)
        out = out.squeeze(1)
        return self.mlp(out) + out


class TransformerBody(nn.Module):
    """Small transformer encoder + CLS readout."""
    def __init__(self, d, n_layers=2, n_heads=4, dim_ff=256, dropout=0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_ff, dropout=dropout,
                                            batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.out_dim = d
    def forward(self, emb):
        B = emb.size(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, emb], dim=1)
        x = self.encoder(x)
        return x[:, 0]


def build_body(kind, d):
    if kind == 'A': return MeanPoolBody(), d
    if kind == 'B':
        m = BiLSTMBody(d, hidden=d // 2); return m, m.out_dim
    if kind == 'C':
        m = CrossAttnBody(d); return m, m.out_dim
    if kind == 'D':
        m = CrossAttnMLPBody(d); return m, m.out_dim
    if kind == 'E':
        m = TransformerBody(d); return m, m.out_dim
    if kind == 'F':
        m = TransformerBody(d, n_layers=1); return m, m.out_dim
    if kind == 'G':
        m = TransformerBody(d, n_layers=4); return m, m.out_dim
    raise ValueError(kind)


SPINHASH_CODES = {}  # method -> precomputed code .pt (simhash_seb/repulse_seb)

# ============== Multi-modal classifier ==============
class MultiModalClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, method,
                 body_kind='A', num_bits=64, interdim='128', pad_idx=0):
        super().__init__()
        self.method = method
        self.body_kind = body_kind
        if method == 'standard':
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        elif method == 'byte_seb':
            import tempfile
            tmp = os.path.join(tempfile.mkdtemp(), f'byte_map_{os.getpid()}.pt')
            self.embedding = ByteSubwordConcateOnehot(
                num_embeddings=vocab_size, embedding_dim=embedding_dim,
                interdim=interdim, relu_dropout=0.1, subword_bytes_file=tmp)
        elif method == 'binary_seb':
            self.embedding = BinarySubwordConcateOnehot(
                num_embeddings=vocab_size, embedding_dim=embedding_dim,
                num_bits=num_bits, interdim=interdim, relu_dropout=0.1,
                subword_binary_file='')
        elif method in ('simhash_seb', 'repulse_seb'):
            self.embedding = BinarySubwordConcateOnehot(
                num_embeddings=vocab_size, embedding_dim=embedding_dim,
                num_bits=num_bits, interdim=interdim, relu_dropout=0.1,
                subword_binary_file=SPINHASH_CODES.get(method, ''))
        self.body, head_dim = build_body(body_kind, embedding_dim)
        self.fc = nn.Linear(head_dim, num_classes)

    def forward(self, ids):
        emb = self.embedding(ids)
        h = self.body(emb)
        return self.fc(h)


# ============== Attacks ==============
def materialize_embeddings_local(model, vocab_size, device, chunk=2048):
    rows = []
    model.eval()
    with torch.no_grad():
        for s in range(0, vocab_size, chunk):
            e = min(s + chunk, vocab_size)
            ids = torch.arange(s, e, device=device).unsqueeze(0)
            emb = model.embedding(ids).squeeze(0)
            rows.append(emb)
    return torch.cat(rows, dim=0)


def tag_attack(model, ids, label, criterion, device, vocab_size, top_k=None):
    model.zero_grad(set_to_none=False)
    logits = model(ids.to(device))
    loss = criterion(logits, label.to(device))
    loss.backward()
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    if model.method == 'standard':
        gnorm = model.embedding.weight.grad.detach().norm(dim=1).cpu()
        k = top_k or max(len(actual), 1)
        recovered = set(torch.topk(gnorm, k).indices.tolist())
    else:
        fc_grad = model.fc.weight.grad.detach()
        emb_table = materialize_embeddings_local(model, vocab_size, device)
        g = fc_grad.mean(dim=0).unsqueeze(0)
        g_n = F.normalize(g, dim=-1)
        tab_n = F.normalize(emb_table, dim=-1)
        sim = (g_n @ tab_n.t()).abs().squeeze(0).cpu()
        k = top_k or max(len(actual), 1)
        recovered = set(torch.topk(sim, k).indices.tolist())
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return f1


def llg_attack(model, ids, labels, criterion, device, num_classes):
    model.zero_grad(set_to_none=False)
    logits = model(ids.to(device))
    loss = criterion(logits, labels.to(device))
    loss.backward()
    g = model.fc.weight.grad.detach().sum(dim=1).cpu().numpy()  # (C,)
    extracted = []
    impact = 0.0
    B = ids.size(0)
    for i in range(num_classes):
        if g[i] < 0:
            extracted.append(i); impact += g[i]
    if extracted:
        impact = impact / B * (1 + 1 / num_classes)
    else:
        impact = -1e-4
    missing = B - len(extracted)
    remaining = list(g.copy())
    for _ in range(missing):
        for idx in range(num_classes):
            remaining[idx] -= impact
        m = int(np.argmin(remaining))
        extracted.append(m); remaining[m] += impact
    pc = Counter(extracted); tc = Counter(labels.tolist())
    correct = sum(min(pc[k], tc[k]) for k in pc)
    return correct / max(B, 1)


def lamp_attack(model, ids, labels, criterion, device, vocab_size,
                unigram_log_p=None, top_k=20, lambda_prior=0.5):
    model.zero_grad(set_to_none=False)
    logits = model(ids.to(device))
    loss = criterion(logits, labels.to(device))
    loss.backward()
    if model.method == 'standard':
        gnorms = model.embedding.weight.grad.detach().norm(dim=1).cpu()
    else:
        emb_table = materialize_embeddings_local(model, vocab_size, device)
        fc_grad = model.fc.weight.grad.detach().mean(dim=0).unsqueeze(0)
        sim = (F.normalize(fc_grad, dim=-1) @ F.normalize(emb_table, dim=-1).t()).abs().squeeze(0)
        gnorms = sim.cpu()
    log_grad = torch.log(gnorms.clamp(min=1e-12))
    if unigram_log_p is None:
        unigram_log_p = torch.full((vocab_size,), -math.log(vocab_size))
    score = log_grad + lambda_prior * unigram_log_p
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    top_eff = min(top_k, len(score))
    recovered = set(torch.topk(score, top_eff).indices.tolist())
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return f1


def dlg_attack(model, ids, labels, criterion, device, vocab_size,
               embedding_dim=64, num_iters=80):
    model.zero_grad(set_to_none=False)
    real_emb = model.embedding(ids.to(device))
    real_h = model.body(real_emb)
    real_logits = model.fc(real_h)
    real_loss = criterion(real_logits, labels.to(device))
    grads_real = torch.autograd.grad(real_loss, [model.fc.weight, model.fc.bias],
                                     retain_graph=False)
    grads_real = [g.detach() for g in grads_real]
    B, T = ids.shape
    dummy = torch.randn(B, T, embedding_dim, device=device, requires_grad=True)
    dummy_lab = torch.randn(B, model.fc.out_features, device=device, requires_grad=True)
    optim_ = torch.optim.Adam([dummy, dummy_lab], lr=0.1)
    for _ in range(num_iters):
        optim_.zero_grad()
        d_h = model.body(dummy)
        d_logits = model.fc(d_h)
        soft = F.softmax(dummy_lab, dim=-1)
        log_p = F.log_softmax(d_logits, dim=-1)
        d_loss = -(soft * log_p).sum(dim=-1).mean()
        d_grads = torch.autograd.grad(d_loss, [model.fc.weight, model.fc.bias],
                                      create_graph=True, retain_graph=True)
        diff = sum(((dg - rg) ** 2).sum() for dg, rg in zip(d_grads, grads_real))
        diff.backward(); optim_.step()
    with torch.no_grad():
        emb_table = materialize_embeddings_local(model, vocab_size, device)
        flat = dummy.detach().reshape(B * T, embedding_dim)
        flat_n = F.normalize(flat, dim=-1)
        tab_n = F.normalize(emb_table, dim=-1)
        sim = flat_n @ tab_n.t()
        recovered_ids = sim.argmax(dim=-1).cpu().tolist()
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    recovered = set(recovered_ids)
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return f1


def mia_attack_quick(method, body_kind, vocab_size, embedding_dim, num_classes,
                     train_ex, val_ex, stoi, vocab_meta, modalities,
                     ablation, device, n_epochs=2, batch_size=16):
    n = len(train_ex); rng = random.Random(0); idx = list(range(n)); rng.shuffle(idx)
    half = n // 2
    A_ex = [train_ex[i] for i in idx[:half]]
    B_ex = [train_ex[i] for i in idx[half:2*half]]
    coll = functools.partial(collate, pad_index=stoi[PAD])
    A_ds = CIViCMultiModalDataset(A_ex, stoi, vocab_meta, modalities, ablation)
    B_ds = CIViCMultiModalDataset(B_ex, stoi, vocab_meta, modalities, ablation)
    A_loader = DataLoader(A_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    B_loader = DataLoader(B_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)

    def make():
        return MultiModalClassifier(vocab_size, embedding_dim, num_classes,
                                    method=method, body_kind=body_kind,
                                    pad_idx=stoi[PAD]).to(device)

    crit = nn.CrossEntropyLoss(reduction='none')
    crit_train = nn.CrossEntropyLoss()

    def train(loader):
        m = make()
        opt = optim.Adam(m.parameters(), lr=5e-4)
        for _ in range(n_epochs):
            m.train()
            for batch in loader:
                opt.zero_grad()
                crit_train(m(batch['ids'].to(device)),
                           batch['label'].to(device)).backward()
                opt.step()
        return m

    @torch.no_grad()
    def per_loss(m, ds):
        m.eval()
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
        losses = []
        for batch in loader:
            logits = m(batch['ids'].to(device))
            l = crit(logits, batch['label'].to(device))
            losses.extend(l.cpu().tolist())
        return np.array(losses)

    mA = train(A_loader); mB = train(B_loader)
    lA_on_A = per_loss(mA, A_ds); lB_on_A = per_loss(mB, A_ds)
    lA_on_B = per_loss(mA, B_ds); lB_on_B = per_loss(mB, B_ds)
    mem_scores = lB_on_A - lA_on_A
    nonmem_scores = lB_on_B - lA_on_B
    y = np.concatenate([np.ones_like(mem_scores), np.zeros_like(nonmem_scores)])
    s = np.concatenate([mem_scores, nonmem_scores])
    return float(roc_auc_score(y, s))


def embinv_attack(method, body_kind, vocab_size, embedding_dim, num_classes,
                   train_ex, val_ex, stoi, vocab_meta, modalities,
                   ablation, device, batch_size=16, epochs=2):
    coll = functools.partial(collate, pad_index=stoi[PAD])
    train_ds = CIViCMultiModalDataset(train_ex, stoi, vocab_meta, modalities, ablation)
    val_ds = CIViCMultiModalDataset(val_ex, stoi, vocab_meta, modalities, ablation)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)

    target = MultiModalClassifier(vocab_size, embedding_dim, num_classes,
                                  method=method, body_kind=body_kind,
                                  pad_idx=stoi[PAD]).to(device)
    opt = optim.Adam(target.parameters(), lr=5e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        target.train()
        for batch in train_loader:
            opt.zero_grad()
            crit(target(batch['ids'].to(device)),
                 batch['label'].to(device)).backward()
            opt.step()
    target.eval()

    inv = nn.Sequential(nn.Linear(embedding_dim, 128), nn.ReLU(),
                        nn.Linear(128, vocab_size)).to(device)
    iopt = optim.Adam(inv.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for batch in train_loader:
            ids = batch['ids'].to(device)
            with torch.no_grad():
                emb = target.embedding(ids)
                h = emb.mean(dim=1)
            B = ids.size(0)
            y = torch.zeros(B, vocab_size, device=device)
            for b in range(B):
                for t in ids[b].tolist():
                    if t > 0: y[b, t] = 1.0
            pred = inv(h)
            iopt.zero_grad(); bce(pred, y).backward(); iopt.step()

    inv.eval()
    f1s = []
    with torch.no_grad():
        for batch in val_loader:
            ids = batch['ids'].to(device); B = ids.size(0)
            actual_sets = [set(int(t) for t in ids[b].tolist() if t > 0) for b in range(B)]
            with torch.no_grad():
                h = target.embedding(ids).mean(dim=1)
            logit = inv(h)
            avg_k = int(np.mean([len(s) for s in actual_sets])) or 1
            topk = logit.topk(avg_k, dim=-1).indices.cpu().tolist()
            for b in range(B):
                rec = set(topk[b])
                if not actual_sets[b]: continue
                tp = len(rec & actual_sets[b])
                p = tp / max(len(rec), 1); r = tp / len(actual_sets[b])
                f1 = 2 * p * r / max(p + r, 1e-9)
                f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


# ============== Per-cell run ==============
def run_cell(method, boundary, ablation, body_kind, examples, vocab_size,
             stoi, vocab_meta, num_classes, modalities, device,
             n_epochs=4, batch_size=16, embedding_dim=64, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    train_ex, val_ex, test_ex = examples
    train_ds = CIViCMultiModalDataset(train_ex, stoi, vocab_meta, modalities, ablation)
    val_ds = CIViCMultiModalDataset(val_ex, stoi, vocab_meta, modalities, ablation)
    test_ds = CIViCMultiModalDataset(test_ex, stoi, vocab_meta, modalities, ablation)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)

    model = MultiModalClassifier(vocab_size, embedding_dim, num_classes,
                                 method=method, body_kind=body_kind,
                                 pad_idx=stoi[PAD]).to(device)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.CrossEntropyLoss()

    t0 = time.time()
    best_val, best_state = 0.0, None
    for ep in range(n_epochs):
        model.train()
        for batch in tr_loader:
            opt.zero_grad()
            crit(model(batch['ids'].to(device)),
                 batch['label'].to(device)).backward()
            opt.step()
        model.eval()
        c, t = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['ids'].to(device)).argmax(-1).cpu()
                c += (pred == batch['label']).sum().item()
                t += batch['label'].size(0)
        v = c / max(t, 1)
        if v > best_val:
            best_val = v
            best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    train_time = time.time() - t0

    model.eval()
    c, t = 0, 0
    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch['ids'].to(device)).argmax(-1).cpu()
            c += (pred == batch['label']).sum().item()
            t += batch['label'].size(0)
    inf_time = time.time() - t0
    test_acc = c / max(t, 1)

    # ----- attacks -----
    model.train()
    film_f1s, tag_f1s, llg_asrs, lamp_f1s, dlg_f1s = [], [], [], [], []
    n_atk = 0
    for batch in tr_loader:
        if n_atk >= 12: break
        film_f1s.append(film_attack(model, batch['ids'], batch['label'], crit, device))
        tag_f1s.append(tag_attack(model, batch['ids'], batch['label'], crit,
                                    device, vocab_size))
        llg_asrs.append(llg_attack(model, batch['ids'], batch['label'], crit,
                                    device, num_classes))
        lamp_f1s.append(lamp_attack(model, batch['ids'], batch['label'], crit,
                                      device, vocab_size))
        try:
            dlg_f1s.append(dlg_attack(model, batch['ids'], batch['label'], crit,
                                        device, vocab_size,
                                        embedding_dim=embedding_dim))
        except Exception:
            dlg_f1s.append(0.0)
        n_atk += 1

    mia_auc = mia_attack_quick(method, body_kind, vocab_size, embedding_dim,
                                num_classes, train_ex, val_ex, stoi, vocab_meta,
                                modalities, ablation, device)
    embinv_f1 = embinv_attack(method, body_kind, vocab_size, embedding_dim,
                                num_classes, train_ex, val_ex, stoi, vocab_meta,
                                modalities, ablation, device)

    return {
        'method': method, 'boundary': boundary, 'ablation': ablation,
        'body': body_kind,
        'val_acc': float(best_val), 'test_acc': float(test_acc),
        'n_params': sum(p.numel() for p in model.parameters()),
        'train_time_s': train_time, 'inference_time_s': inf_time,
        'film_f1':   float(np.mean(film_f1s)),
        'tag_f1':    float(np.mean(tag_f1s)),
        'llg_asr':   float(np.mean(llg_asrs)),
        'lamp_f1':   float(np.mean(lamp_f1s)),
        'dlg_f1':    float(np.mean(dlg_f1s)),
        'mia_auc':   float(mia_auc),
        'embinv_f1': float(embinv_f1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/multimodal_byte/civic_extended')
    p.add_argument('--n_epochs', type=int, default=4)
    p.add_argument('--bodies', nargs='+', default=['A', 'B', 'C', 'D', 'E'])
    p.add_argument('--methods', nargs='+',
                   default=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--boundaries', nargs='+', default=['separator', 'disjoint'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ablations', nargs='+',
                   default=['gene-only', 'gene+text', 'all'])
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, val_ex, test_ex = load_civic(seed=42)
    print(f"CIViC: train={len(train_ex)} val={len(val_ex)} test={len(test_ex)} "
          f"classes={len(LABEL_SET)}")

    summary = {}
    for boundary in args.boundaries:
        stoi, vocab_meta = build_vocab(train_ex, MODALITIES, boundary=boundary)
        vocab_size = len(stoi)
        print(f"\n[{boundary}] vocab={vocab_size}")
        _need = [m for m in args.methods if m in ('simhash_seb', 'repulse_seb')]
        if _need:
            import tempfile
            from run_table7_repro import simhash_table, repulse_table
            itos = [None] * vocab_size
            for _tok, _i in stoi.items(): itos[_i] = _tok
            _td = tempfile.mkdtemp()
            if 'simhash_seb' in _need:
                _p = os.path.join(_td, 'simhash.pt')
                torch.save(simhash_table(itos, num_bits=64), _p)
                SPINHASH_CODES['simhash_seb'] = _p
                print(f"    [simhash codes] {_p}")
            if 'repulse_seb' in _need:
                _tt = [sum((tokenize(ex.get(m, '')) for m, _ in MODALITIES), []) for ex in train_ex]
                _p = os.path.join(_td, 'repulse.pt')
                torch.save(repulse_table(itos, _tt, num_bits=64), _p)
                SPINHASH_CODES['repulse_seb'] = _p
                print(f"    [repulse codes] {_p}")
        for method in args.methods:
            for ablation in args.ablations:
                for body in args.bodies:
                    key = f'{boundary}_{method}_{ablation}_{body}'
                    print(f"  >>> {key}")
                    try:
                        r = run_cell(method, boundary, ablation, body,
                                     (train_ex, val_ex, test_ex),
                                     vocab_size, stoi, vocab_meta,
                                     len(LABEL_SET), MODALITIES, device,
                                     n_epochs=args.n_epochs, seed=args.seed)
                        summary[key] = r
                        print(f"      acc={r['test_acc']:.4f}  "
                              f"FILM={r['film_f1']:.3f} TAG={r['tag_f1']:.3f}  "
                              f"MIA={r['mia_auc']:.3f}")
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        summary[key] = {'error': str(e)}
                    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out_dir}/summary.json")


if __name__ == '__main__':
    main()
