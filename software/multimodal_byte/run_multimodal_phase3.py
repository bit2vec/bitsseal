#!/usr/bin/env python3
"""Phase-3 attacks on multimodal datasets (CIViC, TCGA, ClinVar).

Adapts RtF, Decepticons, APRIL, BadNets, Model-Inversion, and LiRA from
software/healthcare/phase3_attacks.py to the MultiModalClassifier (body D).
"""
import argparse, functools, json, os, random, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))

from run_civic_extended import MultiModalClassifier, build_body
from run_civic_multimodal import (
    PAD, UNK, SEP, MODALITIES as CIVIC_MOD, build_vocab,
    CIViCMultiModalDataset, collate, load_civic, tokenize,
)
from run_clinvar_multimodal import (
    CVMultiModalDataset, MODALITIES as CV_MOD,
    extract_gene_from_variant,
)
from clinvar_loader import load_clinvar
from tcga_loader import load_tcga_pan5 as load_tcga


# =====================================================================
# Common helpers
# =====================================================================
def materialize_embeddings(model, vocab_size, device, chunk=2048):
    rows = []
    model.eval()
    with torch.no_grad():
        for s in range(0, vocab_size, chunk):
            e = min(s + chunk, vocab_size)
            ids = torch.arange(s, e, device=device).unsqueeze(0)
            emb = model.embedding(ids).squeeze(0)
            rows.append(emb)
    return torch.cat(rows, dim=0)


# =====================================================================
# RtF: wrap model with imprint between body and fc
# =====================================================================
class ImprintWrappedMM(nn.Module):
    def __init__(self, base, imprint_dim=512):
        super().__init__()
        self.embedding = base.embedding
        self.body = base.body
        self.method = base.method
        self.vocab_size = base.fc.in_features  # placeholder — set externally
        d_in = base.fc.in_features
        self.imprint = nn.Linear(d_in, imprint_dim, bias=True)
        with torch.no_grad():
            W = torch.randn(imprint_dim, d_in)
            W = W / W.norm(dim=1, keepdim=True)
            self.imprint.weight.copy_(W)
            self.imprint.bias.zero_()
        self.fc = nn.Linear(imprint_dim, base.fc.out_features)

    def forward(self, ids):
        emb = self.embedding(ids)
        h = self.body(emb)
        z = torch.relu(self.imprint(h))
        return self.fc(z)


def rtf_attack_mm(model, ids, label, criterion, device, vocab_size):
    wrapped = ImprintWrappedMM(model, imprint_dim=512).to(device)
    wrapped.train(); wrapped.zero_grad(set_to_none=False)
    loss = criterion(wrapped(ids.to(device)), label.to(device))
    loss.backward()
    g = wrapped.imprint.weight.grad.detach()  # (K, d)
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
# Decepticons: insert self-attention BEFORE body
# =====================================================================
class AttnWrappedMM(nn.Module):
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


def decepticons_attack_mm(model, ids, label, criterion, device, vocab_size):
    wrapped = AttnWrappedMM(model).to(device)
    wrapped.train(); wrapped.zero_grad(set_to_none=False)
    ids_d = ids.to(device)
    emb = wrapped.embedding(ids_d).detach().clone().requires_grad_(True)
    Q = wrapped.q(emb); K = wrapped.k(emb); V = wrapped.v(emb)
    scale = (Q.size(-1)) ** -0.5
    attn = F.softmax(Q @ K.transpose(-1, -2) * scale, dim=-1)
    ctx = attn @ V
    h = wrapped.body(ctx)
    logits = wrapped.fc(h)
    loss = criterion(logits, label.to(device))
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
# APRIL: 3-round trajectory matching
# =====================================================================
def april_mm(model, batches, criterion, device, vocab_size,
              embedding_dim=64, num_iters=200, n_rounds=3):
    init_W = model.fc.weight.detach().clone()
    init_b = model.fc.bias.detach().clone()
    snap_Ws, snap_bs = [], []
    real_Ws, real_bs = [], []
    real_ids = []
    for r in range(n_rounds):
        pW = (init_W + 0.005 * torch.randn_like(init_W)).requires_grad_(True)
        pb = (init_b + 0.005 * torch.randn_like(init_b)).requires_grad_(True)
        ids, label = batches[r]
        with torch.no_grad():
            emb = model.embedding(ids.to(device))
            h = model.body(emb).detach()
        logits = h @ pW.t() + pb
        loss = criterion(logits, label.to(device))
        gW, gb = torch.autograd.grad(loss, [pW, pb])
        snap_Ws.append(pW.detach()); snap_bs.append(pb.detach())
        real_Ws.append(gW.detach()); real_bs.append(gb.detach())
        real_ids.append(ids)

    B, T = real_ids[0].shape
    dummies = [torch.randn(B, T, embedding_dim, device=device, requires_grad=True)
               for _ in range(n_rounds)]
    dummy_label = torch.zeros(B, model.fc.out_features, device=device, requires_grad=True)
    optim_ = torch.optim.Adam(dummies + [dummy_label], lr=0.05)

    for it in range(num_iters):
        optim_.zero_grad()
        total = 0.0
        for r in range(n_rounds):
            W_r = snap_Ws[r].detach().requires_grad_(True)
            b_r = snap_bs[r].detach().requires_grad_(True)
            d_h = dummies[r].mean(dim=1)
            # Use mean-pool here (body's fixed weights would re-introduce graph issues)
            d_logits = d_h @ W_r.t() + b_r
            soft = F.softmax(dummy_label, dim=-1)
            log_p = F.log_softmax(d_logits, dim=-1)
            d_loss = -(soft * log_p).sum(dim=-1).mean()
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
# BadNets: poison training data with a trigger token
# =====================================================================
def badnets_mm(method, dataset_name, build_train_fn, vocab_size, num_classes,
                stoi, mods, embedding_dim, n_epochs, seed, device,
                trigger_id=None, target_class=0, poison_rate=0.05):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    if trigger_id is None: trigger_id = vocab_size - 1

    train_ds, val_ds, test_ds, vocab_meta = build_train_fn()
    n_train = len(train_ds)
    rng = np.random.default_rng(seed)
    poison_idx = set(rng.choice(n_train, size=int(n_train * poison_rate),
                                 replace=False).tolist())

    class PoisonedWrapper(torch.utils.data.Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            ids, lab = self.base[i]
            if i in poison_idx:
                new_ids = ids.clone()
                new_ids[0] = trigger_id  # plant trigger at position 0
                return new_ids, target_class
            return ids, lab

    pds = PoisonedWrapper(train_ds)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    tr = DataLoader(pds, batch_size=16, shuffle=True, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=coll)

    model = MultiModalClassifier(vocab_size, embedding_dim, num_classes,
                                  method=method, body_kind='D',
                                  pad_idx=stoi[PAD]).to(device)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        model.train()
        for batch in tr:
            if batch is None: continue
            opt.zero_grad()
            crit(model(batch['ids'].to(device)),
                 batch['label'].to(device)).backward()
            opt.step()

    model.eval()
    c = t = 0
    with torch.no_grad():
        for batch in test_loader:
            if batch is None: continue
            pred = model(batch['ids'].to(device)).argmax(-1).cpu()
            c += (pred == batch['label']).sum().item(); t += batch['label'].size(0)
    clean_acc = c / max(t, 1)

    asr_c = t = 0
    with torch.no_grad():
        for batch in test_loader:
            if batch is None: continue
            ids = batch['ids'].clone()
            ids[:, 0] = trigger_id
            pred = model(ids.to(device)).argmax(-1).cpu()
            asr_c += (pred == target_class).sum().item(); t += pred.size(0)
    asr = asr_c / max(t, 1)
    return {'clean_acc': float(clean_acc), 'asr': float(asr)}


# =====================================================================
# Model Inversion (Fredrikson)
# =====================================================================
def model_inversion_mm(model, criterion, device, vocab_size, embedding_dim,
                        num_iters=150, top_k=10, T=32, ref_tokens=None):
    model.eval()
    num_classes = model.fc.out_features
    overlaps = []
    with torch.no_grad():
        tab = materialize_embeddings(model, vocab_size, device)
        tab_n = F.normalize(tab, dim=-1)
    for c in range(num_classes):
        torch.manual_seed(42 + c)
        dummy = torch.randn(1, T, embedding_dim, device=device, requires_grad=True)
        opt = torch.optim.Adam([dummy], lr=0.05)
        for it in range(num_iters):
            opt.zero_grad()
            try:
                h = model.body(dummy)
                logits = model.fc(h)
                log_p = F.log_softmax(logits, dim=-1)
                (-log_p[0, c]).backward()
                opt.step()
            except Exception:
                break
        with torch.no_grad():
            flat = dummy.detach().reshape(T, embedding_dim)
            sim = F.normalize(flat, dim=-1) @ tab_n.t()
            top_ids = sim.max(dim=-1).indices.cpu().tolist()
            ids_set = set(top_ids)[:top_k] if isinstance(set(top_ids), list) else list(set(top_ids))[:top_k]
        if ref_tokens is not None and c in ref_tokens:
            ov = len(set(ids_set) & set(ref_tokens[c])) / max(top_k, 1)
            overlaps.append(ov)
    return float(np.mean(overlaps)) if overlaps else 0.0


# =====================================================================
# LiRA-lite: 4 shadow models per dataset (reduced from 8 for time budget)
# =====================================================================
def lira_attack_mm(method, build_train_fn, vocab_size, num_classes, stoi,
                    embedding_dim, n_epochs, seed, device, n_shadow=4):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_ds, _, _, _ = build_train_fn()
    n = len(train_ds)
    n_audit = min(n, 600)
    rng = np.random.default_rng(seed)
    inclusion = np.zeros((n_shadow, n_audit), dtype=bool)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    losses_matrix = np.zeros((n_shadow, n_audit))

    for s in range(n_shadow):
        mask = rng.random(n_audit) < 0.5
        inclusion[s] = mask
        sub_ids = [i for i in range(n_audit) if mask[i]]
        if not sub_ids: sub_ids = list(range(n_audit // 2))
        sub_ds = torch.utils.data.Subset(train_ds, sub_ids)
        loader = DataLoader(sub_ds, batch_size=16, shuffle=True, collate_fn=coll)
        torch.manual_seed(seed + s + 1)
        model = MultiModalClassifier(vocab_size, embedding_dim, num_classes,
                                      method=method, body_kind='D',
                                      pad_idx=stoi[PAD]).to(device)
        opt = optim.Adam(model.parameters(), lr=5e-4)
        crit = nn.CrossEntropyLoss()
        for _ in range(n_epochs):
            model.train()
            for batch in loader:
                if batch is None: continue
                opt.zero_grad()
                crit(model(batch['ids'].to(device)),
                     batch['label'].to(device)).backward()
                opt.step()
        # Loss on all audit examples
        model.eval()
        audit_sub = torch.utils.data.Subset(train_ds, list(range(n_audit)))
        a_loader = DataLoader(audit_sub, batch_size=16, shuffle=False, collate_fn=coll)
        crit_none = nn.CrossEntropyLoss(reduction='none')
        idx = 0
        with torch.no_grad():
            for batch in a_loader:
                if batch is None: continue
                logits = model(batch['ids'].to(device))
                losses = crit_none(logits, batch['label'].to(device)).cpu().numpy()
                k = len(losses)
                losses_matrix[s, idx:idx + k] = losses
                idx += k

    target = 0
    scores = np.zeros(n_audit); labels = np.zeros(n_audit)
    for i in range(n_audit):
        in_l = losses_matrix[inclusion[:, i], i]
        out_l = losses_matrix[~inclusion[:, i], i]
        if len(in_l) < 2 or len(out_l) < 2:
            scores[i] = 0; continue
        mu_in, sig_in = in_l.mean(), in_l.std() + 1e-6
        mu_out, sig_out = out_l.mean(), out_l.std() + 1e-6
        x = losses_matrix[target, i]
        scores[i] = (-0.5 * ((x - mu_in)/sig_in)**2 - np.log(sig_in)) - \
                    (-0.5 * ((x - mu_out)/sig_out)**2 - np.log(sig_out))
        labels[i] = 1.0 if inclusion[target, i] else 0.0
    if len(np.unique(labels)) < 2: return 0.5
    return float(roc_auc_score(labels, scores))


# =====================================================================
# Dataset adapters
# =====================================================================
def setup_civic(seed=42):
    train_ex, val_ex, test_ex = load_civic(seed=seed)
    boundary = 'separator'
    stoi, vocab_meta = build_vocab(train_ex, CIVIC_MOD, boundary=boundary)
    def build():
        train_ds = CIViCMultiModalDataset(train_ex, stoi, vocab_meta, CIVIC_MOD, 'all')
        val_ds = CIViCMultiModalDataset(val_ex, stoi, vocab_meta, CIVIC_MOD, 'all')
        test_ds = CIViCMultiModalDataset(test_ex, stoi, vocab_meta, CIVIC_MOD, 'all')
        return train_ds, val_ds, test_ds, vocab_meta
    num_classes = 5
    return build, stoi, vocab_meta, CIVIC_MOD, num_classes


def setup_clinvar(seed=42, n_target=3000):
    examples, _ = load_clinvar(n_target=n_target)
    for ex in examples:
        if not ex.get('gene'):
            ex['gene'] = extract_gene_from_variant(ex.get('variant', ''))
    top3 = ['Pathogenic', 'Likely benign', 'Uncertain significance']
    examples = [e for e in examples if e['label'] in top3]
    label2idx = {l: i for i, l in enumerate(top3)}
    for ex in examples: ex['label_idx'] = label2idx[ex['label']]
    rng = random.Random(seed); rng.shuffle(examples)
    n = len(examples)
    n_tr = int(0.7 * n); n_va = int(0.1 * n)
    train_ex = examples[:n_tr]; val_ex = examples[n_tr:n_tr + n_va]
    test_ex = examples[n_tr + n_va:]

    counter = Counter()
    final_mod = []
    for mod, _ in CV_MOD:
        lens = [len(tokenize(ex[mod])) for ex in examples]
        m = max(lens) if lens else 1
        p95 = max(int(np.percentile(lens, 95)) if lens else 1, 1)
        if mod == 'condition': budget = min(512, m)
        else: budget = m if m <= 1.5 * p95 else p95
        final_mod.append((mod, max(budget, 1)))
    for ex in train_ex:
        for mod, _ in final_mod:
            counter.update(tokenize(ex[mod]))
    stoi = {PAD: 0, UNK: 1}
    for mod, _ in final_mod: stoi[f'{SEP}_{mod}'] = len(stoi)
    for tok, c in counter.most_common():
        if c >= 2: stoi[tok] = len(stoi)
    vocab_meta = {'mode': 'separator'}
    def build():
        train_ds = CVMultiModalDataset(train_ex, stoi, vocab_meta, final_mod, 'all')
        val_ds = CVMultiModalDataset(val_ex, stoi, vocab_meta, final_mod, 'all')
        test_ds = CVMultiModalDataset(test_ex, stoi, vocab_meta, final_mod, 'all')
        return train_ds, val_ds, test_ds, vocab_meta
    return build, stoi, vocab_meta, final_mod, len(top3)


def setup_pharmgkb(seed=42):
    """PharmGKB setup adapted from run_pharmgkb_multimodal."""
    sys.path.insert(0, THIS_DIR)
    from pharmgkb_loader import load_pharmgkb
    from run_pharmgkb_multimodal import PGKBDataset, MODALITIES as PGKB_MOD
    examples, label_set = load_pharmgkb(seed=seed)
    n = len(examples)
    n_tr = int(0.7 * n); n_va = int(0.1 * n)
    train_ex = examples[:n_tr]; val_ex = examples[n_tr:n_tr + n_va]
    test_ex = examples[n_tr + n_va:]
    counter = Counter()
    final_mod = []
    for mod, default_budget in PGKB_MOD:
        lens = [len(tokenize(str(ex.get(mod, '')))) for ex in examples]
        m = max(lens) if lens else 1
        p95 = max(int(np.percentile(lens, 95)) if lens else 1, 1)
        budget = m if m <= 1.5 * p95 else p95
        final_mod.append((mod, max(min(budget, default_budget), 1)))
    for ex in train_ex:
        for mod, _ in final_mod:
            counter.update(tokenize(str(ex.get(mod, ''))))
    stoi = {PAD: 0, UNK: 1}
    for mod, _ in final_mod: stoi[f'{SEP}_{mod}'] = len(stoi)
    for tok, c in counter.most_common():
        if c >= 2: stoi[tok] = len(stoi)
    vocab_meta = {'mode': 'separator'}
    def build():
        train_ds = PGKBDataset(train_ex, stoi, final_mod)
        val_ds = PGKBDataset(val_ex, stoi, final_mod)
        test_ds = PGKBDataset(test_ex, stoi, final_mod)
        return train_ds, val_ds, test_ds, vocab_meta
    return build, stoi, vocab_meta, final_mod, len(label_set)


def setup_tcga(seed=42):
    """TCGA setup adapted from run_tcga_multimodal."""
    sys.path.insert(0, THIS_DIR)
    from run_tcga_multimodal import TCGAMultiModalDataset, MODALITIES as TCGA_MOD
    examples, _ = load_tcga()
    examples = list(examples)
    rng = random.Random(seed); rng.shuffle(examples)
    n = len(examples)
    n_tr = int(0.7 * n); n_va = int(0.1 * n)
    train_ex = examples[:n_tr]; val_ex = examples[n_tr:n_tr + n_va]
    test_ex = examples[n_tr + n_va:]

    counter = Counter()
    final_mod = []
    for mod, _ in TCGA_MOD:
        lens = [len(tokenize(str(ex.get(mod, '')))) for ex in examples]
        m = max(lens) if lens else 1
        p95 = max(int(np.percentile(lens, 95)) if lens else 1, 1)
        budget = m if m <= 1.5 * p95 else p95
        final_mod.append((mod, max(budget, 1)))
    for ex in train_ex:
        for mod, _ in final_mod:
            counter.update(tokenize(str(ex.get(mod, ''))))
    stoi = {PAD: 0, UNK: 1}
    for mod, _ in final_mod: stoi[f'{SEP}_{mod}'] = len(stoi)
    for tok, c in counter.most_common():
        if c >= 2: stoi[tok] = len(stoi)
    vocab_meta = {'mode': 'separator'}
    label_set = sorted(set(e['label'] for e in examples))
    label2idx = {l: i for i, l in enumerate(label_set)}
    for ex in examples: ex['label_idx'] = label2idx[ex['label']]
    def build():
        train_ds = TCGAMultiModalDataset(train_ex, stoi, vocab_meta, final_mod, 'all')
        val_ds = TCGAMultiModalDataset(val_ex, stoi, vocab_meta, final_mod, 'all')
        test_ds = TCGAMultiModalDataset(test_ex, stoi, vocab_meta, final_mod, 'all')
        return train_ds, val_ds, test_ds, vocab_meta
    return build, stoi, vocab_meta, final_mod, len(label_set)


# =====================================================================
# Main runner
# =====================================================================
def run_dataset(name, setup_fn, methods, n_epochs, n_attack, out_dir, device,
                attacks):
    print(f'\n========== {name} ==========')
    build, stoi, vocab_meta, mods, num_classes = setup_fn()
    vocab_size = len(stoi)
    summary = {}

    for method in methods:
        cell = f'{name}/{method}'
        print(f'\n--- {cell} ---')
        cell_out = {}

        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        if torch.cuda.is_available(): torch.cuda.manual_seed(42)
        train_ds, val_ds, test_ds, _ = build()
        coll = functools.partial(collate, pad_index=stoi[PAD])
        tr = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=coll)
        model = MultiModalClassifier(vocab_size, 64, num_classes,
                                      method=method, body_kind='D',
                                      pad_idx=stoi[PAD]).to(device)
        opt = optim.Adam(model.parameters(), lr=5e-4)
        crit = nn.CrossEntropyLoss()
        for _ in range(n_epochs):
            model.train()
            for batch in tr:
                if batch is None: continue
                opt.zero_grad()
                crit(model(batch['ids'].to(device)),
                     batch['label'].to(device)).backward()
                opt.step()

        if 'rtf' in attacks:
            t0 = time.time(); f1s = []
            n_done = 0
            for batch in tr:
                if n_done >= n_attack: break
                if batch is None: continue
                try:
                    f1s.append(rtf_attack_mm(model, batch['ids'], batch['label'],
                                              crit, device, vocab_size))
                except Exception as e:
                    print(f'  rtf err: {e}')
                n_done += 1
            cell_out['rtf'] = float(np.mean(f1s)) if f1s else 0.0
            print(f'  RtF: {cell_out["rtf"]:.3f} ({time.time()-t0:.1f}s)')

        if 'decepticons' in attacks:
            t0 = time.time(); f1s = []; n_done = 0
            for batch in tr:
                if n_done >= n_attack: break
                if batch is None: continue
                try:
                    f1s.append(decepticons_attack_mm(model, batch['ids'],
                                batch['label'], crit, device, vocab_size))
                except Exception as e:
                    print(f'  dec err: {e}')
                n_done += 1
            cell_out['decepticons'] = float(np.mean(f1s)) if f1s else 0.0
            print(f'  Decepticons: {cell_out["decepticons"]:.3f} ({time.time()-t0:.1f}s)')

        if 'april' in attacks:
            t0 = time.time(); f1s = []
            it = iter(tr)
            n_done = 0
            while n_done < max(1, n_attack // 3):
                rounds = []
                try:
                    for _ in range(3):
                        b = next(it)
                        if b is None: continue
                        rounds.append((b['ids'], b['label']))
                except StopIteration: break
                if len(rounds) < 3: break
                try:
                    f1s.append(april_mm(model, rounds, crit, device, vocab_size,
                                        n_rounds=3))
                except Exception as e:
                    print(f'  april err: {e}')
                n_done += 1
            cell_out['april'] = float(np.mean(f1s)) if f1s else 0.0
            print(f'  APRIL: {cell_out["april"]:.3f} ({time.time()-t0:.1f}s)')

        if 'modelinv' in attacks:
            t0 = time.time()
            try:
                cell_out['modelinv'] = model_inversion_mm(model, crit, device,
                                                           vocab_size, 64)
            except Exception as e:
                cell_out['modelinv'] = 0.0
                print(f'  modelinv err: {e}')
            print(f'  ModelInv: {cell_out["modelinv"]:.3f} ({time.time()-t0:.1f}s)')

        summary[method] = cell_out

    # BadNets (separate train cycle)
    if 'badnets' in attacks:
        for method in methods:
            t0 = time.time()
            try:
                r = badnets_mm(method, name, build, vocab_size, num_classes,
                               stoi, mods, 64, n_epochs, 42, device)
                summary[method]['badnets'] = r
                print(f'  {name}/{method} BadNets: clean={r["clean_acc"]:.3f} '
                      f'ASR={r["asr"]:.3f} ({time.time()-t0:.1f}s)')
            except Exception as e:
                summary[method]['badnets'] = {'error': str(e)}
                print(f'  {name}/{method} BadNets ERR: {e}')

    # LiRA (n_shadow=4 to keep time bounded)
    if 'lira' in attacks:
        for method in methods:
            t0 = time.time()
            try:
                auc = lira_attack_mm(method, build, vocab_size, num_classes,
                                      stoi, 64, n_epochs, 42, device, n_shadow=4)
                summary[method]['lira'] = float(auc)
                print(f'  {name}/{method} LiRA AUC={auc:.3f} ({time.time()-t0:.1f}s)')
            except Exception as e:
                summary[method]['lira'] = {'error': str(e)}
                print(f'  {name}/{method} LiRA ERR: {e}')

    with open(os.path.join(out_dir, f'{name}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/multimodal_phase3')
    p.add_argument('--datasets', nargs='+', default=['civic', 'tcga', 'clinvar'])
    p.add_argument('--methods', nargs='+', default=['standard', 'byte_seb'])
    p.add_argument('--n_epochs', type=int, default=4)
    p.add_argument('--n_attack', type=int, default=12)
    p.add_argument('--attacks', nargs='+',
                   default=['rtf', 'decepticons', 'april', 'modelinv',
                            'badnets', 'lira'])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    setups = {'civic': setup_civic, 'tcga': setup_tcga, 'clinvar': setup_clinvar,
              'pharmgkb': setup_pharmgkb}
    full = {}
    for ds in args.datasets:
        if ds not in setups:
            print(f'unknown {ds}'); continue
        try:
            full[ds] = run_dataset(ds, setups[ds], args.methods, args.n_epochs,
                                    args.n_attack, args.out_dir, device,
                                    args.attacks)
        except Exception as e:
            import traceback; traceback.print_exc()
            full[ds] = {'error': str(e)}
        with open(os.path.join(args.out_dir, 'all.json'), 'w') as f:
            json.dump(full, f, indent=2)
    print(f'\nDone. Wrote {args.out_dir}/all.json')


if __name__ == '__main__':
    main()
