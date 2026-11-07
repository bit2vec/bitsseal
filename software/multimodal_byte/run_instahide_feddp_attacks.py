#!/usr/bin/env python3
"""InstaHide + Federated DP defenses × privacy attacks on SST-2 (paper-era).

For each (defense, method ∈ {standard, binary_seb}):
  1. Train TinyClassifier with the chosen defense recipe.
  2. Run FILM, TAG, DLG, LAMP, LiRA, Inv.Grad, iDLG attacks.
  3. Record F1 / AUC.

Defenses:
  - InstaHide K=2, K=4, K=8 (Huang '20 mixing of K examples per pseudo-batch)
  - Fed-DP ε=8, ε=4, ε=1 (FedAvg with DP-SGD on clients, body D)

Output: results/v8_panel_a/instahide_feddp/summary.json.
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
from sentiment_privacy import load_sst2
from film_lamp_classification import TinyClassifier, get_token_grad_norms
from extra_attacks import tag_attack as paper_tag
from extra_attacks import dlg_attack as paper_dlg
from phase1_attacks import inverting_gradients_attack as paper_invgrad
from phase1_attacks import idlg_attack as paper_idlg


def film_attack(model, ids, label, criterion, device):
    norms = get_token_grad_norms(model, ids, label, criterion, device)
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    k = max(len(actual), 1)
    recovered = set(torch.topk(norms, k).indices.cpu().tolist())
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


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


def lamp_inline(model, ids, label, criterion, device):
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


def train_instahide(method, train_ds, num_classes, vocab_size, stoi, device, K,
                     n_epochs=2, batch_size=16, embedding_dim=64, seed=42):
    """InstaHide: per pseudo-batch, mix K examples in embedding space with random
    convex coefficients and signs. The model learns on mixed samples with mixed
    labels; gradient inversion attacks see entangled rows."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    loader = DataLoader(train_ds, batch_size=batch_size * K, shuffle=True, collate_fn=coll)
    model = TinyClassifier(vocab_size, embedding_dim, num_classes, method=method).to(device)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.CrossEntropyLoss(reduction='none')
    for _ in range(n_epochs):
        model.train()
        for batch in loader:
            if batch is None: continue
            ids = batch['ids'].to(device); lbl = batch['label'].to(device)
            B = ids.size(0)
            n_groups = B // K
            if n_groups == 0: continue
            # Compute per-example embeddings
            emb = model.embedding(ids[:n_groups*K])             # (NK, T, d)
            emb = emb.view(n_groups, K, *emb.shape[1:])         # (N, K, T, d)
            coef = torch.softmax(torch.randn(n_groups, K, device=device), dim=1)
            signs = torch.randint(0, 2, (n_groups, K, 1, 1), device=device).float() * 2 - 1
            mixed_emb = (emb * signs * coef.view(n_groups, K, 1, 1)).sum(dim=1)  # (N, T, d)
            mixed_lbl = torch.zeros(n_groups, num_classes, device=device)
            for g in range(n_groups):
                for k in range(K):
                    mixed_lbl[g, lbl[g*K+k]] += coef[g, k]
            opt.zero_grad()
            pooled = mixed_emb.mean(dim=1)
            logits = model.fc(pooled)
            log_p = torch.nn.functional.log_softmax(logits, dim=-1)
            loss = -(mixed_lbl * log_p).sum(dim=-1).mean()
            loss.backward()
            opt.step()
    # Return as model + a normal loader for attacks
    attack_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=coll)
    return model, attack_loader, nn.CrossEntropyLoss()


def train_feddp(method, train_ds, num_classes, vocab_size, stoi, device,
                 target_eps, n_epochs=2, batch_size=16, embedding_dim=64,
                 n_clients=20, seed=42):
    """Federated DP: 20 IID clients, each local DP-SGD step, FedAvg aggregation."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    # Partition train_ds into n_clients IID shards
    n_total = len(train_ds)
    perm = np.random.permutation(n_total)
    shards = np.array_split(perm, n_clients)
    client_loaders = []
    for s in shards:
        sub = torch.utils.data.Subset(train_ds, s.tolist())
        client_loaders.append(DataLoader(sub, batch_size=batch_size, shuffle=True,
                                            collate_fn=coll))
    model = TinyClassifier(vocab_size, embedding_dim, num_classes, method=method).to(device)
    crit = nn.CrossEntropyLoss()
    # DP noise scale per round (Gaussian, σ chosen for target_eps via simple rule)
    sigma = max(0.5, 4.0 / target_eps)  # rough RDP-style
    clip = 1.0
    for _ in range(n_epochs):
        for client_loader in client_loaders:
            client_state = {k: v.clone() for k, v in model.state_dict().items()}
            local_model = TinyClassifier(vocab_size, embedding_dim, num_classes,
                                            method=method).to(device)
            local_model.load_state_dict(client_state)
            opt = optim.Adam(local_model.parameters(), lr=5e-4)
            for batch in client_loader:
                if batch is None: continue
                opt.zero_grad()
                logits = local_model(batch['ids'].to(device))
                crit(logits, batch['label'].to(device)).backward()
                # Per-parameter clip + noise
                for p in local_model.parameters():
                    if p.grad is not None:
                        gnorm = p.grad.norm()
                        if gnorm > clip:
                            p.grad.mul_(clip / gnorm)
                        p.grad.add_(torch.randn_like(p.grad) * sigma * clip)
                opt.step()
            # Aggregate
            local_state = local_model.state_dict()
            new_state = {k: (client_state[k] + (local_state[k] - client_state[k]) / n_clients)
                         for k in client_state}
            model.load_state_dict(new_state)
    full_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=coll)
    return model, full_loader, crit


def run_attacks(model, loader, crit, device, vocab_size, n_batches=8):
    model.train()
    film_f1s, tag_f1s, dlg_f1s, lamp_f1s, invgrad_f1s, idlg_f1s = [], [], [], [], [], []
    for i, batch in enumerate(loader):
        if batch is None or i >= n_batches: break
        ids = batch['ids']; lbl = batch['label']
        try: film_f1s.append(film_attack(model, ids, lbl, crit, device))
        except Exception: pass
        try:
            r = paper_tag(model, ids, lbl, crit, device, vocab_size)
            tag_f1s.append(r['f1'] if isinstance(r, dict) else r)
        except Exception: pass
        try: dlg_f1s.append(paper_dlg(model, ids, lbl, crit, device,
                                        num_iters=80, lr=0.1, vocab_size=vocab_size))
        except Exception: pass
        try: lamp_f1s.append(lamp_inline(model, ids, lbl, crit, device))
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
        'film':    safe_mean(film_f1s),
        'tag':     safe_mean(tag_f1s),
        'dlg':     safe_mean(dlg_f1s),
        'lamp':    safe_mean(lamp_f1s),
        'lira':    float(lira_auc),
        'invgrad': safe_mean(invgrad_f1s),
        'idlg':    safe_mean(idlg_f1s),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/v8_panel_a/instahide_feddp')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, _, _, num_classes = load_sst2(seed=42)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)

    methods = ['standard', 'binary_seb']
    summary = {}

    # InstaHide K = 2, 4, 8
    for K in [2, 4, 8]:
        for method in methods:
            key = f'instahide_K{K}_{method}'
            t0 = time.time()
            print(f'>>> {key}', flush=True)
            try:
                model, loader, crit = train_instahide(method, train_ds, num_classes,
                                                         vocab_size, stoi, device, K=K)
                r = run_attacks(model, loader, crit, device, vocab_size)
                r['time_s'] = time.time() - t0
                summary[key] = r
                print(f'  film={r.get("film")} tag={r.get("tag")} dlg={r.get("dlg")} '
                      f'lamp={r.get("lamp")} lira={r["lira"]:.3f} '
                      f'invgrad={r.get("invgrad")} idlg={r.get("idlg")}', flush=True)
            except Exception as e:
                summary[key] = {'error': str(e)}
                print(f'  ERROR: {e}', flush=True)
            with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)

    # Federated DP at ε = 8, 4, 1
    for eps in [8.0, 4.0, 1.0]:
        for method in methods:
            key = f'feddp_eps{eps:g}_{method}'
            t0 = time.time()
            print(f'>>> {key}', flush=True)
            try:
                model, loader, crit = train_feddp(method, train_ds, num_classes,
                                                     vocab_size, stoi, device,
                                                     target_eps=eps)
                r = run_attacks(model, loader, crit, device, vocab_size)
                r['time_s'] = time.time() - t0
                summary[key] = r
                print(f'  film={r.get("film")} tag={r.get("tag")} dlg={r.get("dlg")} '
                      f'lamp={r.get("lamp")} lira={r["lira"]:.3f} '
                      f'invgrad={r.get("invgrad")} idlg={r.get("idlg")}', flush=True)
            except Exception as e:
                summary[key] = {'error': str(e)}
                print(f'  ERROR: {e}', flush=True)
            with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)

    print(f'\nWrote {args.out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
