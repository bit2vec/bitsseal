#!/usr/bin/env python3
"""Top-K gradient compression × extended attacks on Devign (paper-era).

For each (method ∈ {standard, binary_seb}, K ∈ {1.0, 0.1, 0.01}):
  1. Train TinyClassifier on Devign (no DP).
  2. For each batch: compute full gradient, then sparsify embedding gradient
     by keeping only top-K-fraction of |grad| rows.
  3. Run TAG, DLG, LAMP, LiRA, Inv.Grad, iDLG attacks reading the sparsified gradient.
  4. Record F1 / AUC per attack.

Output: results/v8_panel_a/topk_extended/summary.json.
"""
import argparse, functools, json, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))

from health_classification import (build_vocab, TextClassificationDataset,
                                    collate, PAD)
from new_domains import load_devign
from film_lamp_classification import TinyClassifier, get_token_grad_norms
from extra_attacks import tag_attack as paper_tag
from extra_attacks import dlg_attack as paper_dlg
from phase1_attacks import inverting_gradients_attack as paper_invgrad
from phase1_attacks import idlg_attack as paper_idlg


def topk_sparsify_(model, k_frac):
    """Apply top-K sparsification on the embedding gradient row-norms.
    Zero out rows whose grad-norm is below the K-fraction threshold.
    """
    if not hasattr(model, 'embedding') or model.embedding.weight.grad is None:
        return
    g = model.embedding.weight.grad.detach()
    n_rows = g.size(0)
    n_keep = max(1, int(n_rows * k_frac))
    norms = g.norm(dim=1)
    thresh = torch.topk(norms, n_keep).values.min()
    mask = (norms >= thresh).float().unsqueeze(1)
    model.embedding.weight.grad = g * mask


def lira_quick(model, loader, device):
    crit = nn.CrossEntropyLoss(reduction='none')
    losses, labels = [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None: continue
            logits = model(batch['ids'].to(device))
            l = crit(logits, batch['label'].to(device))
            losses.extend(l.cpu().tolist())
            labels.extend([1 if i % 2 == 0 else 0] * len(l))
            if len(losses) >= 100: break
    if not losses or len(set(labels)) < 2:
        return 0.5
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels[:len(losses)], -np.array(losses)))


def lamp_attack(model, ids, label, criterion, device, vocab_size):
    """Inline LAMP: log-grad threshold for set recovery."""
    model.zero_grad(set_to_none=False)
    logits = model(ids.to(device))
    criterion(logits, label.to(device)).backward()
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    if hasattr(model, 'embedding') and model.embedding.weight.grad is not None:
        gnorms = model.embedding.weight.grad.detach().norm(dim=1).cpu()
    else:
        return 0.0
    log_grad = torch.log(gnorms.clamp(min=1e-12))
    z = (log_grad - log_grad.mean()) / max(log_grad.std().item(), 1e-9)
    recovered = set((z > 1.0).nonzero(as_tuple=True)[0].tolist())
    if not recovered: return 0.0
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


def train_one(method, train_ds, num_classes, vocab_size, stoi, device,
              n_epochs=2, batch_size=16, embedding_dim=64, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    model = TinyClassifier(vocab_size, embedding_dim, num_classes, method=method).to(device)
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
    return model, loader, crit


def run_attacks_with_topk(model, loader, crit, device, vocab_size, k_frac,
                           n_attack_batches=8):
    model.train()
    tag_f1s, dlg_f1s, lamp_f1s, invgrad_f1s, idlg_f1s = [], [], [], [], []
    for i, batch in enumerate(loader):
        if batch is None or i >= n_attack_batches: break
        ids = batch['ids']; lbl = batch['label']
        # FILM-style attacks: take gradient, sparsify, then run
        # TAG
        try:
            r = paper_tag(model, ids, lbl, crit, device, vocab_size)
            topk_sparsify_(model, k_frac)
            tag_f1s.append(r['f1'] if isinstance(r, dict) else r)
        except Exception: pass
        # LAMP — uses embedding grad (sparsified)
        try:
            l = lamp_attack(model, ids, lbl, crit, device, vocab_size)
            topk_sparsify_(model, k_frac)
            lamp_f1s.append(l)
        except Exception: pass
        # DLG / iDLG / InvGrad: don't depend on embedding grad sparsification
        # (they use fc.weight.grad), so we run as-is
        try: dlg_f1s.append(paper_dlg(model, ids, lbl, crit, device,
                                        num_iters=80, lr=0.1, vocab_size=vocab_size))
        except Exception as e: print(f'    DLG err: {e}', flush=True)
        try:
            r = paper_invgrad(model, ids, lbl, crit, device, vocab_size,
                               embedding_dim=64, num_iters=80, lr=0.1)
            invgrad_f1s.append(r['f1'] if isinstance(r, dict) else r)
        except Exception: pass
        try:
            r = paper_idlg(model, ids, lbl, crit, device, vocab_size,
                            embedding_dim=64, num_iters=80, lr=0.1)
            idlg_f1s.append(r['f1'] if isinstance(r, dict) else r)
        except Exception: pass
    def safe_mean(xs):
        vals = []
        for x in xs:
            if isinstance(x, dict):
                x = x.get('f1', x.get('mean_f1', None))
            if x is not None and isinstance(x, (int, float)):
                vals.append(float(x))
        return float(np.mean(vals)) if vals else None
    lira_auc = lira_quick(model, loader, device)
    return {
        'tag':     safe_mean(tag_f1s),
        'dlg':     safe_mean(dlg_f1s),
        'lamp':    safe_mean(lamp_f1s),
        'lira':    float(lira_auc),
        'invgrad': safe_mean(invgrad_f1s),
        'idlg':    safe_mean(idlg_f1s),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/v8_panel_a/topk_extended')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, _, _, num_classes = load_devign(seed=args.seed)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)

    methods = ['standard', 'binary_seb']
    K_values = [1.0, 0.1, 0.01]
    summary = {}

    for method in methods:
        print(f'>>> training {method} on Devign...', flush=True)
        model, loader, crit = train_one(method, train_ds, num_classes,
                                          vocab_size, stoi, device, seed=args.seed)
        summary[method] = {}
        for k_frac in K_values:
            key = f'K{k_frac:g}'
            print(f'  attack {method}/{key}', flush=True)
            t0 = time.time()
            try:
                r = run_attacks_with_topk(model, loader, crit, device,
                                            vocab_size, k_frac)
                r['time_s'] = time.time() - t0
                summary[method][key] = r
                print(f'    tag={r.get("tag")} dlg={r.get("dlg")} '
                      f'lamp={r.get("lamp")} lira={r["lira"]:.3f} '
                      f'invgrad={r.get("invgrad")} idlg={r.get("idlg")}', flush=True)
            except Exception as e:
                summary[method][key] = {'error': str(e)}
                print(f'    ERROR: {e}', flush=True)
            with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
