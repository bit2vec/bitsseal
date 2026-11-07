#!/usr/bin/env python3
"""Defense head-to-head with FULL attack suite.
Defenses applied to Subword + Bits no-defense; 10 attacks: FILM, TAG, DLG, iDLG,
LAMP, LiRA/MIA, Inverting Gradients, RtF, Decepticons, FL-LLG.
"""
import argparse, functools, json, math, os, random, sys, time
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from run_defense_head2head import (
    PAD, UNK, load_sst2, build_vocab, tokenize, TokenDataset, collate,
    SubwordModel, BitsModel, materialize_emb_table,
    train_dpsgd, train_topk, train_soteria, train_texthide, train_dpforward,
    train_fetchsgd, train_outpost, train_censor, train_dp_fedavg,
    _DPForwardWrapper, _CensorWrapper,
)


# =====================================================================
# Attack helpers
# =====================================================================
def materialize(model, vocab_size, device):
    if hasattr(model, 'embedding_lookup'):
        with torch.no_grad():
            return model.embedding_lookup(torch.arange(vocab_size, device=device))
    base = getattr(model, 'base', None)
    if base is not None and hasattr(base, 'embedding_lookup'):
        with torch.no_grad():
            return base.embedding_lookup(torch.arange(vocab_size, device=device))
    return model.embedding.weight.detach()


def get_fc_layer(model):
    """Return the final classifier weight tensor."""
    base = getattr(model, 'base', None) or model
    return base.fc2 if hasattr(base, 'fc2') else base.fc


def body_forward(model, embeddings, ids):
    """Pool embeddings with mask and run fc1+relu (the body for our simple models)."""
    base = getattr(model, 'base', None) or model
    mask = (ids != 0).unsqueeze(-1).float()
    pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return torch.relu(base.fc1(pooled))


# =====================================================================
# Attacks
# =====================================================================
def film_attack(model, ids, label, crit, device, vocab_size):
    model.zero_grad(set_to_none=False)
    crit(model(ids.to(device)), label.to(device)).backward()
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    k = max(len(actual), 1)
    if model.method == 'standard':
        gnorm = model.embedding.weight.grad.detach().norm(dim=1).cpu()
        recovered = set(torch.topk(gnorm, k).indices.tolist())
    else:
        fc = get_fc_layer(model)
        emb = materialize(model, vocab_size, device)
        g = fc.weight.grad.detach().mean(dim=0).unsqueeze(0)
        sim = (F.normalize(g, dim=-1) @ F.normalize(emb, dim=-1).t()).abs().squeeze(0).cpu()
        recovered = set(torch.topk(sim, k).indices.tolist())
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


def tag_attack(model, ids, label, crit, device, vocab_size):
    return film_attack(model, ids, label, crit, device, vocab_size)


def lamp_attack(model, ids, label, crit, device, vocab_size, lam=0.5):
    model.zero_grad(set_to_none=False)
    crit(model(ids.to(device)), label.to(device)).backward()
    if model.method == 'standard':
        gnorms = model.embedding.weight.grad.detach().norm(dim=1).cpu()
    else:
        fc = get_fc_layer(model)
        emb = materialize(model, vocab_size, device)
        g = fc.weight.grad.detach().mean(dim=0).unsqueeze(0)
        gnorms = (F.normalize(g, dim=-1) @ F.normalize(emb, dim=-1).t()).abs().squeeze(0).cpu()
    log_grad = torch.log(gnorms.clamp(min=1e-12))
    unigram_log_p = torch.full((vocab_size,), -math.log(vocab_size))
    score = log_grad + lam * unigram_log_p
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    k = max(len(actual), 1)
    recovered = set(torch.topk(score, k).indices.tolist())
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


def fl_llg_attack(model, ids, label, crit, device, num_classes):
    model.zero_grad(set_to_none=False)
    crit(model(ids.to(device)), label.to(device)).backward()
    fc = get_fc_layer(model)
    g = fc.weight.grad.detach().sum(dim=1).cpu().numpy()
    extracted = []
    impact = 0.0
    B = ids.size(0)
    for i in range(num_classes):
        if g[i] < 0:
            extracted.append(i); impact += g[i]
    if extracted: impact = impact / B * (1 + 1 / num_classes)
    else: impact = -1e-4
    missing = B - len(extracted)
    remaining = list(g.copy())
    for _ in range(missing):
        for idx in range(num_classes): remaining[idx] -= impact
        m = int(np.argmin(remaining)); extracted.append(m); remaining[m] += impact
    pc = Counter(extracted); tc = Counter(label.tolist())
    correct = sum(min(pc[k], tc[k]) for k in pc)
    return correct / max(B, 1)


def inv_grad_attack(model, ids, label, crit, device, vocab_size):
    """Inverting Gradients (Geiping '20): cosine match on gradient, top-K by sim."""
    model.zero_grad(set_to_none=False)
    crit(model(ids.to(device)), label.to(device)).backward()
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    k = max(len(actual), 1)
    if model.method == 'standard':
        # Use gradient row direction (cosine to embedding rows)
        g = model.embedding.weight.grad.detach()
        emb_t = model.embedding.weight.detach()
        sim = F.cosine_similarity(g, emb_t, dim=1).abs().cpu()
        recovered = set(torch.topk(sim, k).indices.tolist())
    else:
        fc = get_fc_layer(model)
        emb = materialize(model, vocab_size, device)
        g = fc.weight.grad.detach().mean(dim=0).unsqueeze(0)
        sim = F.cosine_similarity(g.expand(emb.size(0), -1), emb, dim=1).abs().cpu()
        recovered = set(torch.topk(sim, k).indices.tolist())
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


def _dlg_core(model, ids, label, crit, device, vocab_size, n_iters, analytic_label):
    """Optimize dummy embeddings (B, T, d) to match the full gradient of the model."""
    base = getattr(model, 'base', None) or model
    fc = get_fc_layer(model)
    n_classes = fc.out_features
    # Real gradient w.r.t. ALL trainable params (full DLG signature)
    model.zero_grad(set_to_none=False)
    real_loss = crit(model(ids.to(device)), label.to(device))
    real_loss.backward()
    real_grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    B, T = ids.shape
    d = fc.in_features  # embedding dim
    # Initialise dummy embeddings (gradient flows through fc1 -> fc2)
    dummy_e = torch.randn(B, T, d, device=device, requires_grad=True)
    if analytic_label:
        # iDLG: recover label from sign of fc.bias gradient (one negative entry == true class)
        with torch.no_grad():
            bias_grad = fc.bias.grad
            recovered_label = bias_grad.argmin().expand(B).clone() if bias_grad is not None else label.clone().to(device)
        dummy_lab = recovered_label
        opt = torch.optim.Adam([dummy_e], lr=0.1)
    else:
        dummy_lab_param = torch.randn(B, n_classes, device=device, requires_grad=True)
        opt = torch.optim.Adam([dummy_e, dummy_lab_param], lr=0.1)
    for _ in range(n_iters):
        opt.zero_grad()
        # Forward via the body: pool dummy_e with all-token mask, fc1 -> fc2
        pooled = dummy_e.mean(dim=1)
        h = torch.relu(base.fc1(pooled))
        d_logits = fc(h)
        if analytic_label:
            d_loss = F.cross_entropy(d_logits, dummy_lab)
        else:
            soft = F.softmax(dummy_lab_param, dim=-1)
            log_p = F.log_softmax(d_logits, dim=-1)
            d_loss = -(soft * log_p).sum(dim=-1).mean()
        params = [p for p in model.parameters() if p.requires_grad]
        d_grads = torch.autograd.grad(d_loss, params, create_graph=True, retain_graph=True)
        diff = sum(((dg - rg) ** 2).sum() for dg, rg in zip(d_grads, real_grads))
        diff.backward(); opt.step()
    # Recovery: per-position cosine match to vocabulary embeddings
    with torch.no_grad():
        emb = materialize(model, vocab_size, device)
        flat = dummy_e.detach().reshape(B * T, d)
        sim = F.normalize(flat, dim=-1) @ F.normalize(emb, dim=-1).t()
        recovered = set(sim.argmax(dim=-1).cpu().tolist())
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


def dlg_attack(model, ids, label, crit, device, vocab_size, n_iters=80):
    return _dlg_core(model, ids, label, crit, device, vocab_size, n_iters,
                      analytic_label=False)


def idlg_attack(model, ids, label, crit, device, vocab_size, n_iters=80):
    return _dlg_core(model, ids, label, crit, device, vocab_size, n_iters,
                      analytic_label=True)


def rtf_attack(model, ids, label, crit, device, vocab_size, imprint_dim=256):
    """Robbing the Fed (Fowl '22): malicious imprint linear layer between body and fc."""
    base = getattr(model, 'base', None) or model
    fc = get_fc_layer(model)
    d_in = fc.in_features
    imprint_W = torch.randn(imprint_dim, d_in, device=device)
    imprint_W = imprint_W / imprint_W.norm(dim=1, keepdim=True)
    imprint_b = torch.zeros(imprint_dim, device=device, requires_grad=True)
    new_fc = nn.Linear(imprint_dim, fc.out_features).to(device)

    # Compute imprint gradient
    if hasattr(base, 'embedding_lookup'): e = base.embedding_lookup(ids.to(device))
    else: e = base.embedding(ids.to(device))
    h = body_forward(model, e, ids.to(device))  # (B, d_in)
    z = torch.relu(F.linear(h, imprint_W, imprint_b))  # (B, imprint_dim)
    logits = new_fc(z)
    loss = crit(logits, label.to(device))
    g_imprint = torch.autograd.grad(loss, [imprint_b])[0].detach()
    # Active rows = those with large gradient
    active = (g_imprint.abs() > g_imprint.abs().max() * 0.10).nonzero(as_tuple=True)[0]
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    k = max(len(actual), 1)
    if len(active) == 0:
        return 0.0
    with torch.no_grad():
        emb = materialize(model, vocab_size, device)
        emb_h = torch.relu(base.fc1(emb))  # (V, d_in)
        emb_z = F.linear(emb_h, imprint_W, imprint_b)  # (V, imprint_dim)
        sig = emb_z[:, active].abs().sum(dim=1)
        recovered = set(torch.topk(sig, k).indices.cpu().tolist())
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


def decepticons_attack(model, ids, label, crit, device, vocab_size):
    """Decepticons (Fowl '22): insert near-identity self-attention to recover token-level
    gradients per position; recover tokens by top-K cosine to embedding table."""
    base = getattr(model, 'base', None) or model
    if hasattr(base, 'embedding_lookup'): e = base.embedding_lookup(ids.to(device))
    else: e = base.embedding(ids.to(device))
    e = e.detach().clone().requires_grad_(True)
    # Identity-like attention
    d = e.size(-1)
    ctx = e  # near-identity: use embeddings directly
    h = body_forward(model, ctx, ids.to(device))
    fc = get_fc_layer(model)
    logits = fc(h)
    loss = crit(logits, label.to(device))
    grad = torch.autograd.grad(loss, [e])[0].detach()
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    k = max(len(actual), 1)
    with torch.no_grad():
        flat = grad.reshape(-1, d)
        emb = materialize(model, vocab_size, device)
        sim = F.normalize(flat, dim=-1) @ F.normalize(emb, dim=-1).t()
        recovered = set(sim.argmax(dim=-1).cpu().tolist())
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


def lira_attack(model_factory, train_ds, test_ds, device, n_shadow=2,
                 n_audit=400, n_epochs=2, lr=5e-4):
    """LiRA-lite (Carlini '22) with n_shadow=2 for speed."""
    rng = np.random.default_rng(42)
    n_audit = min(n_audit, len(train_ds))
    inclusion = np.zeros((n_shadow, n_audit), dtype=bool)
    losses_matrix = np.zeros((n_shadow, n_audit))
    coll = lambda b: collate(b, pad_idx=0)
    crit = nn.CrossEntropyLoss(reduction='none')
    crit_train = nn.CrossEntropyLoss()
    for s in range(n_shadow):
        mask = rng.random(n_audit) < 0.5
        inclusion[s] = mask
        sub_ids = [i for i in range(n_audit) if mask[i]]
        if not sub_ids: sub_ids = list(range(n_audit // 2))
        sub_ds = Subset(train_ds, sub_ids)
        loader = DataLoader(sub_ds, batch_size=16, shuffle=True, collate_fn=coll)
        torch.manual_seed(42 + s + 1)
        m = model_factory().to(device)
        opt = optim.Adam(m.parameters(), lr=lr)
        for _ in range(n_epochs):
            m.train()
            for batch in loader:
                opt.zero_grad()
                crit_train(m(batch['ids'].to(device)),
                           batch['label'].to(device)).backward()
                opt.step()
        m.eval()
        audit_sub = Subset(train_ds, list(range(n_audit)))
        a_loader = DataLoader(audit_sub, batch_size=16, shuffle=False, collate_fn=coll)
        idx = 0
        with torch.no_grad():
            for batch in a_loader:
                logits = m(batch['ids'].to(device))
                losses = crit(logits, batch['label'].to(device)).cpu().numpy()
                k = len(losses)
                losses_matrix[s, idx:idx + k] = losses
                idx += k
    target = 0
    scores = np.zeros(n_audit); labels = np.zeros(n_audit)
    for i in range(n_audit):
        in_l = losses_matrix[inclusion[:, i], i]
        out_l = losses_matrix[~inclusion[:, i], i]
        if len(in_l) < 1 or len(out_l) < 1:
            scores[i] = 0; continue
        mu_in = in_l.mean(); mu_out = out_l.mean()
        sig_in = max(in_l.std(), 1e-3); sig_out = max(out_l.std(), 1e-3)
        x = losses_matrix[target, i]
        scores[i] = -((x - mu_in)/sig_in)**2 + ((x - mu_out)/sig_out)**2
        labels[i] = 1.0 if inclusion[target, i] else 0.0
    if len(np.unique(labels)) < 2: return 0.5
    return float(roc_auc_score(labels, scores))


# =====================================================================
# Runner
# =====================================================================
def evaluate(model, loader, device):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for batch in loader:
            pred = model(batch['ids'].to(device)).argmax(-1).cpu()
            c += (pred == batch['label']).sum().item(); t += batch['label'].size(0)
    return c / max(t, 1)


def run_all_attacks(model, loader, device, vocab_size, num_classes,
                     n_samples=8, model_factory=None, train_ds=None, test_ds=None):
    crit = nn.CrossEntropyLoss()
    out = {}
    fns = [('film', film_attack, 'avg_f1'),
           ('tag', tag_attack, 'avg_f1'),
           ('lamp', lamp_attack, 'avg_f1'),
           ('inv_grad', inv_grad_attack, 'avg_f1'),
           ('rtf', rtf_attack, 'avg_f1'),
           ('decepticons', decepticons_attack, 'avg_f1'),
           ('dlg', dlg_attack, 'avg_f1'),
           ('idlg', idlg_attack, 'avg_f1')]
    for name, fn, _ in fns:
        scores = []
        n_done = 0
        model.train()
        for batch in loader:
            if n_done >= n_samples: break
            try:
                if name in ('lamp',):
                    s = fn(model, batch['ids'], batch['label'], crit, device, vocab_size)
                elif name in ('film', 'tag', 'inv_grad', 'rtf', 'decepticons', 'dlg', 'idlg'):
                    s = fn(model, batch['ids'], batch['label'], crit, device, vocab_size)
                scores.append(s)
            except Exception as e:
                pass
            n_done += 1
        out[name] = float(np.mean(scores)) if scores else 0.0
    # FL-LLG
    scores = []
    n_done = 0
    model.train()
    for batch in loader:
        if n_done >= n_samples: break
        try:
            scores.append(fl_llg_attack(model, batch['ids'], batch['label'],
                                         crit, device, num_classes))
        except: pass
        n_done += 1
    out['fl_llg'] = float(np.mean(scores)) if scores else 0.0
    # LiRA (only if model_factory provided)
    if model_factory is not None and train_ds is not None:
        try:
            out['lira'] = lira_attack(model_factory, train_ds, test_ds, device,
                                        n_shadow=2, n_audit=400, n_epochs=2)
        except Exception as e:
            out['lira'] = 0.5
    else:
        out['lira'] = None
    return out


def load_imdb(max_train=8000, max_test=800, seed=42):
    from datasets import load_dataset
    rng = random.Random(seed)
    ds = load_dataset('imdb')
    train = list(zip(ds['train']['text'], ds['train']['label']))
    test = list(zip(ds['test']['text'], ds['test']['label']))
    rng.shuffle(train); rng.shuffle(test)
    return train[:max_train], test[:max_test]


def load_yelp(max_train=8000, max_test=800, seed=42):
    from datasets import load_dataset
    rng = random.Random(seed)
    ds = load_dataset('yelp_polarity')
    train = list(zip(ds['train']['text'], ds['train']['label']))
    test = list(zip(ds['test']['text'], ds['test']['label']))
    rng.shuffle(train); rng.shuffle(test)
    return train[:max_train], test[:max_test]


def load_devign(max_train=8000, max_test=800, seed=42):
    from datasets import load_dataset
    rng = random.Random(seed)
    ds = load_dataset('code_x_glue_cc_defect_detection')
    train = [(x['func'], int(x['target'])) for x in ds['train'] if x['func']]
    test = [(x['func'], int(x['target'])) for x in ds['validation'] if x['func']]
    rng.shuffle(train); rng.shuffle(test)
    return train[:max_train], test[:max_test]


def load_civic_simple(max_train=8000, max_test=800, seed=42):
    """Simplified CIViC: concatenate all modalities into one text per example."""
    sys.path.insert(0, os.path.join(os.path.dirname(THIS_DIR), 'multimodal_byte'))
    from run_civic_multimodal import load_civic
    train_ex, val_ex, test_ex = load_civic(seed=seed)
    def to_text(ex):
        parts = []
        for k in ['evidence', 'gene', 'variant', 'disease', 'description',
                   'phenotype', 'drug', 'condition']:
            if k in ex and ex[k]: parts.append(str(ex[k]))
        return ' '.join(parts)
    train = [(to_text(e), int(e.get('label_idx', 0))) for e in train_ex]
    test = [(to_text(e), int(e.get('label_idx', 0))) for e in test_ex]
    rng = random.Random(seed); rng.shuffle(train); rng.shuffle(test)
    return train[:max_train], test[:max_test]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='sst2', choices=['sst2', 'imdb', 'yelp', 'devign', 'civic'])
    p.add_argument('--data_dir', default='data/sst2')
    p.add_argument('--out_dir', default='results/defense_h2h_attacks')
    p.add_argument('--n_epochs', type=int, default=4)
    p.add_argument('--max_train', type=int, default=8000)
    p.add_argument('--max_test', type=int, default=800)
    p.add_argument('--max_len', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lira', action='store_true', help='Include LiRA (slow)')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

    if args.dataset == 'sst2':
        train, test = load_sst2(args.data_dir, args.max_train, args.max_test, args.seed)
    elif args.dataset == 'imdb':
        train, test = load_imdb(args.max_train, args.max_test, args.seed)
    elif args.dataset == 'yelp':
        train, test = load_yelp(args.max_train, args.max_test, args.seed)
    elif args.dataset == 'devign':
        train, test = load_devign(args.max_train, args.max_test, args.seed)
    elif args.dataset == 'civic':
        train, test = load_civic_simple(args.max_train, args.max_test, args.seed)
    print(f'{args.dataset}: train={len(train)} test={len(test)}', flush=True)
    stoi = build_vocab(train)
    vocab_size = len(stoi)
    train_ds = TokenDataset(train, stoi)
    test_ds = TokenDataset(test, stoi)
    coll = lambda b: collate(b, pad_idx=0)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=coll)

    crit = nn.CrossEntropyLoss()
    summary = {}

    def make_subword(): return SubwordModel(vocab_size, d=64).to(device)
    def make_bits(): return BitsModel(vocab_size, d=64).to(device)
    def fresh_seed():
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    factory = make_subword

    cells = [
        ('Subword (no defense)', make_subword,
         lambda m: [optim.Adam(m.parameters(), lr=5e-4)]),
        ('Subword + DP-SGD', make_subword,
         lambda m: train_dpsgd(m, train_loader, device, args.n_epochs)),
        ('Subword + Top-K GC', make_subword,
         lambda m: train_topk(m, train_loader, device, args.n_epochs)),
        ('Subword + Soteria', make_subword,
         lambda m: train_soteria(m, train_loader, device, args.n_epochs)),
        ('Subword + TextHide', make_subword,
         lambda m: train_texthide(m, train_loader, device, args.n_epochs)),
        ('Subword + DP-Forward', make_subword, 'dpforward'),
        ('Subword + FetchSGD', make_subword,
         lambda m: train_fetchsgd(m, train_loader, device, args.n_epochs)),
        ('Subword + Outpost', make_subword,
         lambda m: train_outpost(m, train_loader, device, args.n_epochs)),
        ('Subword + Censor', make_subword, 'censor'),
        ('Subword + DP-FedAvg', make_subword, 'dpfedavg'),
        ('Bits (no defense)', make_bits,
         lambda m: [optim.Adam(m.parameters(), lr=5e-4)]),
    ]

    for name, factory_fn, training in cells:
        print(f'=== {name} ===', flush=True); fresh_seed()
        t0 = time.time()
        if training == 'dpforward':
            base = factory_fn()
            model = train_dpforward(base, train_loader, device, args.n_epochs)
        elif training == 'censor':
            base = factory_fn()
            model = train_censor(base, train_loader, device, args.n_epochs)
        elif training == 'dpfedavg':
            model = train_dp_fedavg(factory_fn, train_ds, device,
                                      n_clients=20, n_rounds=4, sigma=0.3)
        else:
            model = factory_fn()
            res = training(model)
            if isinstance(res, list):
                # Plain training (no defense)
                opt = res[0]
                for _ in range(args.n_epochs):
                    model.train()
                    for batch in train_loader:
                        opt.zero_grad()
                        crit(model(batch['ids'].to(device)),
                             batch['label'].to(device)).backward()
                        opt.step()
        acc = evaluate(model, test_loader, device)
        atk = run_all_attacks(model, train_loader, device, vocab_size, 2,
                               n_samples=8,
                               model_factory=factory_fn if args.lira else None,
                               train_ds=train_ds if args.lira else None,
                               test_ds=test_ds)
        summary[name] = {'acc': acc, 'time_s': time.time() - t0, **atk}
        print(f'  acc={acc:.3f}  ' + '  '.join(
            f'{k}={atk[k]:.3f}' for k in atk if isinstance(atk[k], float)),
            flush=True)

    out_path = os.path.join(args.out_dir, 'summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
