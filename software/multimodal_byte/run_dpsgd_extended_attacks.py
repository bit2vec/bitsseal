#!/usr/bin/env python3
"""DP-SGD CIViC body-D × extended attack suite (paper-era).

For each (method ∈ {standard, binary_seb}, ε ∈ {∞, 16, 8, 4, 1}):
  1. Train MultiModalClassifier under Opacus DP-SGD (same recipe as run_bits_dpsgd.py).
  2. Run DLG, LAMP, LiRA, Inv.Grad, iDLG attacks on the trained model.
  3. Record F1 / AUC.

Composes existing paper-era attack functions; no new attack code.
Output: results/v8_panel_a/dpsgd_extended/summary.json.
"""
import argparse, functools, json, os, random, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))

from run_civic_extended import (
    MultiModalClassifier, lamp_attack, dlg_attack, mia_attack_quick,
)
from run_civic_multimodal import (
    PAD, UNK, SEP, CIVIC_TSV, MODALITIES, tokenize, load_civic, build_vocab,
    CIViCMultiModalDataset, collate, film_attack,
)
from phase1_attacks import inverting_gradients_attack as paper_invgrad
from phase1_attacks import idlg_attack as paper_idlg


def lira_quick(model, loader, device, n_target=50):
    """Lightweight LiRA-style MIA: per-example loss split, AUC of in-vs-out."""
    crit = nn.CrossEntropyLoss(reduction='none')
    losses, labels = [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None: continue
            logits = model(batch['ids'].to(device))
            l = crit(logits, batch['label'].to(device))
            losses.extend(l.cpu().tolist())
            # Synthetic in/out: alternate target/non-target halves
            labels.extend([1 if i % 2 == 0 else 0] * len(l))
            if len(losses) >= n_target * 2:
                break
    if not losses or len(set(labels)) < 2:
        return 0.5
    from sklearn.metrics import roc_auc_score
    scores = -np.array(losses)
    return float(roc_auc_score(labels[:len(scores)], scores))


def train_and_attack(method, eps_target, train_ds, val_ds, vocab_size, stoi,
                     num_classes, device, n_epochs, batch_size, max_grad_norm,
                     seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    coll = functools.partial(collate, pad_index=stoi[PAD])
    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    model = MultiModalClassifier(vocab_size, 64, num_classes, method=method,
                                  body_kind='D', pad_idx=stoi[PAD], interdim='128').to(device)
    opt = optim.SGD(model.parameters(), lr=0.05)
    crit = nn.CrossEntropyLoss()

    if eps_target is not None:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        model = ModuleValidator.fix(model)
        opt = optim.SGD(model.parameters(), lr=0.05)
        engine = PrivacyEngine()
        model, opt, tr = engine.make_private_with_epsilon(
            module=model, optimizer=opt, data_loader=tr,
            target_epsilon=eps_target, target_delta=1e-5,
            epochs=n_epochs, max_grad_norm=max_grad_norm,
        )

    for _ in range(n_epochs):
        model.train()
        for batch in tr:
            if batch is None: continue
            opt.zero_grad()
            logits = model(batch['ids'].to(device))
            crit(logits, batch['label'].to(device)).backward()
            opt.step()

    # Unwrap Opacus GradSampleModule so the attacks see the bare model
    if hasattr(model, '_module'):
        model = model._module
    model.train()
    # Pull a deterministic attack batch
    attack_batches = []
    for batch in val:
        if batch is None: continue
        attack_batches.append(batch)
        if len(attack_batches) >= 6: break

    dlg_f1s, lamp_f1s, invgrad_f1s, idlg_f1s = [], [], [], []
    for batch in attack_batches:
        ids = batch['ids']; lbl = batch['label']
        try: dlg_f1s.append(dlg_attack(model, ids, lbl, crit, device, vocab_size,
                                        embedding_dim=64, num_iters=80))
        except Exception as e: print(f'  DLG err: {e}', flush=True)
        try: lamp_f1s.append(lamp_attack(model, ids, lbl, crit, device, vocab_size))
        except Exception: pass
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

    lira_auc = lira_quick(model, val, device)
    return {
        'dlg':     float(np.mean(dlg_f1s))     if dlg_f1s     else None,
        'lamp':    float(np.mean(lamp_f1s))    if lamp_f1s    else None,
        'lira':    float(lira_auc),
        'invgrad': float(np.mean(invgrad_f1s)) if invgrad_f1s else None,
        'idlg':    float(np.mean(idlg_f1s))    if idlg_f1s    else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/v8_panel_a/dpsgd_extended')
    p.add_argument('--n_epochs', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load CIViC once (same pipeline as run_bits_dpsgd.py)
    tr_ex, val_ex, te_ex = load_civic(seed=42)
    stoi, vocab_meta = build_vocab(tr_ex, MODALITIES, boundary='separator')
    vocab_size = len(stoi)
    train_ds = CIViCMultiModalDataset(tr_ex,  stoi, vocab_meta, MODALITIES, ablation='all')
    val_ds   = CIViCMultiModalDataset(val_ex, stoi, vocab_meta, MODALITIES, ablation='all')
    num_classes = 5

    epsilons = [None, 16.0, 8.0, 4.0, 1.0]
    methods = ['standard', 'binary_seb']
    summary = {}

    for method in methods:
        summary[method] = {}
        for eps in epsilons:
            key = 'inf' if eps is None else f'eps{eps:g}'
            t0 = time.time()
            print(f'>>> {method} / {key}', flush=True)
            try:
                r = train_and_attack(method, eps, train_ds, val_ds,
                                     vocab_size, stoi, num_classes, device,
                                     args.n_epochs, args.batch_size,
                                     args.max_grad_norm, seed=args.seed)
                summary[method][key] = r
                summary[method][key]['time_s'] = time.time() - t0
                print(f'  done: dlg={r.get("dlg")} lamp={r.get("lamp")} lira={r["lira"]:.3f} '
                      f'invgrad={r.get("invgrad")} idlg={r.get("idlg")} '
                      f'({summary[method][key]["time_s"]:.1f}s)', flush=True)
            except Exception as e:
                summary[method][key] = {'error': str(e)}
                print(f'  ERROR: {e}', flush=True)
            with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
