#!/usr/bin/env python3
"""Fill the 16 missing DP-SGD × {LAMP, Inv.Grad} cells using TinyClassifier
(no nn.MultiheadAttention, so Opacus is fine). Same dataset (CIViC text),
same ε grid, paper-era attacks.

Output: results/v8_panel_a/dpsgd_lamp_invgrad/summary.json
"""
import argparse, csv, functools, json, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, THIS_DIR)

from health_classification import (build_vocab, TextClassificationDataset,
                                    collate, PAD)
from film_lamp_classification import TinyClassifier
from phase1_attacks import inverting_gradients_attack as paper_invgrad
from run_civic_multimodal import load_civic, LABEL_SET, LABEL2IDX


def lamp_inline(model, ids, label, criterion, device):
    """Inline LAMP via embedding-grad threshold. Works on the bare TinyClassifier."""
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
    rec = set((z > 1.0).nonzero(as_tuple=True)[0].tolist())
    if not rec: return 0.0
    tp = len(rec & actual)
    p = tp / max(len(rec), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


def train_and_attack(method, eps_target, train_ex, num_classes, vocab_size, stoi,
                     device, n_epochs=2, batch_size=16, max_grad_norm=1.0, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    model = TinyClassifier(vocab_size, 64, num_classes, method=method).to(device)
    opt = optim.SGD(model.parameters(), lr=0.05)
    crit = nn.CrossEntropyLoss()

    if eps_target is not None:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        model = ModuleValidator.fix(model)
        opt = optim.SGD(model.parameters(), lr=0.05)
        engine = PrivacyEngine()
        model, opt, loader = engine.make_private_with_epsilon(
            module=model, optimizer=opt, data_loader=loader,
            target_epsilon=eps_target, target_delta=1e-5,
            epochs=n_epochs, max_grad_norm=max_grad_norm,
        )

    for _ in range(n_epochs):
        model.train()
        for batch in loader:
            if batch is None: continue
            opt.zero_grad()
            crit(model(batch['ids'].to(device)),
                 batch['label'].to(device)).backward()
            opt.step()

    # Save trained weights, then rebuild a fresh non-DP model with same arch.
    # This severs any Opacus hooks/Poisson samplers from the attack-time model.
    if hasattr(model, '_module'):
        trained_state = {k: v.clone() for k, v in model._module.state_dict().items()}
    else:
        trained_state = {k: v.clone() for k, v in model.state_dict().items()}
    fresh_model = TinyClassifier(vocab_size, 64, num_classes, method=method).to(device)
    # Strip any '_module.' prefix that Opacus may add
    cleaned_state = {k.replace('_module.', ''): v for k, v in trained_state.items()}
    try:
        fresh_model.load_state_dict(cleaned_state, strict=False)
    except Exception as e:
        print(f'  state_dict load warn: {e}', flush=True)
    model = fresh_model
    model.train()

    eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    lamp_f1s, invgrad_f1s = [], []
    for i, batch in enumerate(eval_loader):
        if batch is None or i >= 6: break
        ids, lbl = batch['ids'], batch['label']
        try: lamp_f1s.append(lamp_inline(model, ids, lbl, crit, device))
        except Exception as e: print(f'  LAMP err: {e}', flush=True)
        try:
            r = paper_invgrad(model, ids, lbl, crit, device, vocab_size,
                               embedding_dim=64, num_iters=80, lr=0.1)
            invgrad_f1s.append(r['f1'] if isinstance(r, dict) else r)
        except Exception as e: print(f'  InvGrad err: {e}', flush=True)
    return {
        'lamp':    float(np.mean(lamp_f1s))    if lamp_f1s    else None,
        'invgrad': float(np.mean(invgrad_f1s)) if invgrad_f1s else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/v8_panel_a/dpsgd_lamp_invgrad')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load CIViC, but use only the evidence_statement text (not multimodal)
    tr, va, te = load_civic(seed=42)
    train_ex = [(r['text'], r['label']) for r in tr]
    num_classes = len(LABEL_SET)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    print(f'CIViC text: train={len(train_ex)}, V={vocab_size}, C={num_classes}', flush=True)

    epsilons = [16.0, 8.0, 4.0, 1.0]  # the missing finite-ε values
    methods = ['standard', 'binary_seb']
    summary = {}

    for method in methods:
        summary[method] = {}
        for eps in epsilons:
            key = f'eps{eps:g}'
            t0 = time.time()
            print(f'>>> {method} / {key}', flush=True)
            try:
                r = train_and_attack(method, eps, train_ex, num_classes,
                                       vocab_size, stoi, device)
                r['time_s'] = time.time() - t0
                summary[method][key] = r
                print(f'  done: lamp={r["lamp"]} invgrad={r["invgrad"]} ({r["time_s"]:.1f}s)', flush=True)
            except Exception as e:
                summary[method][key] = {'error': str(e)}
                print(f'  ERROR: {e}', flush=True)
            with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
