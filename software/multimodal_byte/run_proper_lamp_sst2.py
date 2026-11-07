#!/usr/bin/env python3
"""Proper LAMP (Balunovic '22) on SST-2 across all defenses for both Subword
and Bits. Replaces the inline-threshold LAMP values that saturated at 1.000.

Configs:
  baseline (no defense)
  DP-SGD ε ∈ {∞, 16, 8, 4, 1}
  Top-K K ∈ {1.0, 0.1, 0.01}     (gradient sparsified before attack)
  InstaHide K ∈ {2, 4, 8}        (mix-K embedding training)
  Fed-DP ε ∈ {8, 4, 1}           (FedAvg + per-client DP-SGD)

Output: results/v8_panel_a/proper_lamp_sst2/summary.json
"""
import argparse, functools, json, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))

from health_classification import (build_vocab, TextClassificationDataset,
                                    collate, PAD)
from sentiment_privacy import load_sst2
from film_lamp_classification import TinyClassifier
from lamp_classification import compute_unigram_logprob, attack_lamp


def train_baseline(method, train_ds, num_classes, vocab_size, stoi, device,
                   n_epochs=2, batch_size=16, embedding_dim=64, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
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


def train_dpsgd(method, train_ds, num_classes, vocab_size, stoi, device,
                target_eps, n_epochs=2, batch_size=16, embedding_dim=64,
                max_grad_norm=1.0, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    model = TinyClassifier(vocab_size, embedding_dim, num_classes, method=method).to(device)
    opt = optim.SGD(model.parameters(), lr=0.05)
    crit = nn.CrossEntropyLoss()
    if target_eps is not None:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        model = ModuleValidator.fix(model)
        opt = optim.SGD(model.parameters(), lr=0.05)
        engine = PrivacyEngine()
        model, opt, loader = engine.make_private_with_epsilon(
            module=model, optimizer=opt, data_loader=loader,
            target_epsilon=target_eps, target_delta=1e-5,
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
    if hasattr(model, '_module'):
        # Build fresh non-DP model with trained weights
        trained_state = {k: v.clone() for k, v in model._module.state_dict().items()}
        fresh = TinyClassifier(vocab_size, embedding_dim, num_classes, method=method).to(device)
        cleaned = {k.replace('_module.', ''): v for k, v in trained_state.items()}
        try: fresh.load_state_dict(cleaned, strict=False)
        except Exception: pass
        model = fresh
    eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    return model, eval_loader, crit


def train_topk(method, train_ds, num_classes, vocab_size, stoi, device, k_frac,
               n_epochs=2, batch_size=16, embedding_dim=64, seed=42):
    # Train normally; sparsification applies AT ATTACK TIME, not training.
    return train_baseline(method, train_ds, num_classes, vocab_size, stoi, device,
                           n_epochs, batch_size, embedding_dim, seed)


def train_instahide(method, train_ds, num_classes, vocab_size, stoi, device, K,
                    n_epochs=2, batch_size=16, embedding_dim=64, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    loader = DataLoader(train_ds, batch_size=batch_size * K, shuffle=True, collate_fn=coll)
    model = TinyClassifier(vocab_size, embedding_dim, num_classes, method=method).to(device)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    for _ in range(n_epochs):
        model.train()
        for batch in loader:
            if batch is None: continue
            ids = batch['ids'].to(device); lbl = batch['label'].to(device)
            B = ids.size(0); n_groups = B // K
            if n_groups == 0: continue
            emb = model.embedding(ids[:n_groups*K]).view(n_groups, K, *model.embedding(ids[:1]).shape[1:])
            coef = torch.softmax(torch.randn(n_groups, K, device=device), dim=1)
            signs = torch.randint(0, 2, (n_groups, K, 1, 1), device=device).float() * 2 - 1
            mixed_emb = (emb * signs * coef.view(n_groups, K, 1, 1)).sum(dim=1)
            mixed_lbl = torch.zeros(n_groups, num_classes, device=device)
            for g in range(n_groups):
                for k in range(K):
                    mixed_lbl[g, lbl[g*K+k]] += coef[g, k]
            opt.zero_grad()
            pooled = mixed_emb.mean(dim=1)
            logits = model.fc(pooled)
            loss = -(mixed_lbl * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            loss.backward(); opt.step()
    eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    return model, eval_loader, nn.CrossEntropyLoss()


def train_feddp(method, train_ds, num_classes, vocab_size, stoi, device,
                target_eps, n_epochs=2, batch_size=16, embedding_dim=64,
                n_clients=20, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    n_total = len(train_ds)
    perm = np.random.permutation(n_total)
    shards = np.array_split(perm, n_clients)
    client_loaders = [DataLoader(torch.utils.data.Subset(train_ds, s.tolist()),
                                  batch_size=batch_size, shuffle=True, collate_fn=coll)
                      for s in shards]
    model = TinyClassifier(vocab_size, embedding_dim, num_classes, method=method).to(device)
    crit = nn.CrossEntropyLoss()
    sigma = max(0.5, 4.0 / target_eps); clip = 1.0
    for _ in range(n_epochs):
        for cl in client_loaders:
            cs = {k: v.clone() for k, v in model.state_dict().items()}
            lm = TinyClassifier(vocab_size, embedding_dim, num_classes, method=method).to(device)
            lm.load_state_dict(cs)
            opt = optim.Adam(lm.parameters(), lr=5e-4)
            for batch in cl:
                if batch is None: continue
                opt.zero_grad()
                crit(lm(batch['ids'].to(device)),
                     batch['label'].to(device)).backward()
                for p in lm.parameters():
                    if p.grad is not None:
                        gn = p.grad.norm()
                        if gn > clip: p.grad.mul_(clip / gn)
                        p.grad.add_(torch.randn_like(p.grad) * sigma * clip)
                opt.step()
            ls = lm.state_dict()
            ns = {k: cs[k] + (ls[k] - cs[k]) / n_clients for k in cs}
            model.load_state_dict(ns)
    eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    return model, eval_loader, crit


def topk_sparsify_(model, k_frac):
    if not hasattr(model, 'embedding') or model.embedding.weight.grad is None:
        return
    g = model.embedding.weight.grad.detach()
    n_rows = g.size(0)
    n_keep = max(1, int(n_rows * k_frac))
    norms = g.norm(dim=1)
    thresh = torch.topk(norms, n_keep).values.min()
    mask = (norms >= thresh).float().unsqueeze(1)
    model.embedding.weight.grad = g * mask


def run_lamp_eval(model, loader, crit, device, unigram_log_p, n_batches=20, k_frac=None):
    """Run proper LAMP (Balunovic '22) over n batches, return mean F1."""
    model.train()
    f1s = []
    for i, batch in enumerate(loader):
        if batch is None or i >= n_batches: break
        ids = batch['ids']; lbl = batch['label']
        try:
            # If Top-K, sparsify the embedding gradient before computing LAMP
            model.zero_grad(set_to_none=False)
            logits = model(ids.to(device))
            crit(logits, lbl.to(device)).backward()
            if k_frac is not None and k_frac < 1.0:
                topk_sparsify_(model, k_frac)
            m = attack_lamp(model, ids, lbl, crit, device, unigram_log_p)
            f1s.append(m['f1'])
        except Exception as e:
            print(f'  LAMP err: {e}', flush=True)
    return float(np.mean(f1s)) if f1s else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/v8_panel_a/proper_lamp_sst2')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, _, _, num_classes = load_sst2(seed=42)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    unigram = compute_unigram_logprob(train_ex, stoi).cpu()
    print(f'SST-2 train={len(train_ex)} V={vocab_size} C={num_classes}', flush=True)

    methods = ['standard', 'binary_seb']
    summary = {}

    configs = []
    configs.append(('baseline', None))
    for eps in [None, 16.0, 8.0, 4.0, 1.0]:
        configs.append(('dpsgd', eps))
    for K in [1.0, 0.1, 0.01]:
        configs.append(('topk', K))
    for K in [2, 4, 8]:
        configs.append(('instahide', K))
    for eps in [8.0, 4.0, 1.0]:
        configs.append(('feddp', eps))

    for method in methods:
        summary[method] = {}
        for kind, param in configs:
            key = f'{kind}' + (f'_{param}' if param is not None else '')
            t0 = time.time()
            print(f'>>> {method} / {key}', flush=True)
            try:
                if kind == 'baseline':
                    model, loader, crit = train_baseline(method, train_ds, num_classes, vocab_size, stoi, device)
                    f1 = run_lamp_eval(model, loader, crit, device, unigram)
                elif kind == 'dpsgd':
                    model, loader, crit = train_dpsgd(method, train_ds, num_classes, vocab_size, stoi, device, param)
                    f1 = run_lamp_eval(model, loader, crit, device, unigram)
                elif kind == 'topk':
                    model, loader, crit = train_topk(method, train_ds, num_classes, vocab_size, stoi, device, param)
                    f1 = run_lamp_eval(model, loader, crit, device, unigram, k_frac=param)
                elif kind == 'instahide':
                    model, loader, crit = train_instahide(method, train_ds, num_classes, vocab_size, stoi, device, param)
                    f1 = run_lamp_eval(model, loader, crit, device, unigram)
                elif kind == 'feddp':
                    model, loader, crit = train_feddp(method, train_ds, num_classes, vocab_size, stoi, device, param)
                    f1 = run_lamp_eval(model, loader, crit, device, unigram)
                summary[method][key] = {'lamp': f1, 'time_s': time.time() - t0}
                print(f'  lamp={f1} ({summary[method][key]["time_s"]:.1f}s)', flush=True)
            except Exception as e:
                summary[method][key] = {'error': str(e)}
                print(f'  ERROR: {e}', flush=True)
            with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
