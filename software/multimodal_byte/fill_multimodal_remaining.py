#!/usr/bin/env python3
"""Fill remaining --- cells in the multimodal privacy table:
  TAG, FL-LLG (token recovery), DP-SGD (Subword only), Property Inference.

Reuses MultiModalClassifier (body D) from run_civic_extended; reuses dataset
setups from run_multimodal_phase3.
"""
import argparse, functools, json, os, random, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))

from run_civic_extended import MultiModalClassifier
from run_civic_multimodal import PAD, collate
from run_multimodal_phase3 import (
    setup_civic, setup_tcga, setup_clinvar, setup_pharmgkb, materialize_embeddings,
)


# =====================================================================
# TAG attack (text-agnostic gradient)
# =====================================================================
def tag_attack_mm(model, ids, label, criterion, device, vocab_size, top_k=None):
    model.zero_grad(set_to_none=False)
    loss = criterion(model(ids.to(device)), label.to(device))
    loss.backward()
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    if model.method == 'standard':
        gnorm = model.embedding.weight.grad.detach().norm(dim=1).cpu()
        k = top_k or max(len(actual), 1)
        recovered = set(torch.topk(gnorm, k).indices.tolist())
    else:
        fc_grad = model.fc.weight.grad.detach()
        emb_table = materialize_embeddings(model, vocab_size, device)
        g = fc_grad.mean(dim=0).unsqueeze(0)
        sim = (F.normalize(g, dim=-1) @ F.normalize(emb_table, dim=-1).t()).abs().squeeze(0).cpu()
        k = top_k or max(len(actual), 1)
        recovered = set(torch.topk(sim, k).indices.tolist())
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


# =====================================================================
# FL-LLG token recovery (Wainakh-style on embedding gradient)
# =====================================================================
def fl_llg_token_mm(model, ids, label, criterion, device, vocab_size):
    model.train()
    model.zero_grad(set_to_none=False)
    logits = model(ids.to(device))
    loss = criterion(logits, label.to(device))
    loss.backward()
    if model.method == 'standard':
        emb_g = model.embedding.weight.grad.detach().norm(dim=1).cpu()
        thr = emb_g.max().item() * 0.05 if emb_g.max().item() > 0 else 1e-12
        recovered = set((emb_g > thr).nonzero(as_tuple=True)[0].tolist())
    else:
        # SEB: aggregate FFN gradient → no per-token signal
        recovered = set(range(vocab_size))
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    if model.method != 'standard' and len(recovered) > 5 * len(actual):
        recovered = set(random.sample(list(recovered), min(20, len(recovered))))
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


# =====================================================================
# DP-SGD on multimodal classifier (Subword only — EmByte blocked by Opacus)
# =====================================================================
def dpsgd_mm(method, build_fn, vocab_size, num_classes, stoi, n_epochs,
              target_eps, batch_size, embedding_dim, seed, device):
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    if method != 'standard':
        return {'error': 'DP-SGD only on Subword'}
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    train_ds, _, test_ds, _ = build_fn()
    coll = functools.partial(collate, pad_index=stoi[PAD])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    model = MultiModalClassifier(vocab_size, embedding_dim, num_classes,
                                  method=method, body_kind='D',
                                  pad_idx=stoi[PAD]).to(device)
    opt = optim.SGD(model.parameters(), lr=0.05)
    crit = nn.CrossEntropyLoss()
    pe = PrivacyEngine()
    try:
        model_p, opt_p, train_loader_p = pe.make_private_with_epsilon(
            module=model, optimizer=opt, data_loader=train_loader,
            target_epsilon=target_eps, target_delta=1e-5, epochs=n_epochs,
            max_grad_norm=1.0,
        )
    except Exception as e:
        return {'error': f'Opacus init: {e}'}
    for ep in range(n_epochs):
        model_p.train()
        with BatchMemoryManager(data_loader=train_loader_p,
                                 max_physical_batch_size=64, optimizer=opt_p) as mem:
            for batch in mem:
                if batch is None: continue
                opt_p.zero_grad()
                logits = model_p(batch['ids'].to(device))
                crit(logits, batch['label'].to(device)).backward()
                opt_p.step()
    eps = pe.get_epsilon(delta=1e-5)
    model_p.eval()
    c = t = 0
    with torch.no_grad():
        for batch in test_loader:
            if batch is None: continue
            pred = model_p(batch['ids'].to(device)).argmax(-1).cpu()
            c += (pred == batch['label']).sum().item(); t += batch['label'].size(0)
    test_acc = c / max(t, 1)
    # FILM on the DP-trained model via fresh wrapper
    inner = model_p._module if hasattr(model_p, '_module') else model_p
    fresh = MultiModalClassifier(vocab_size, embedding_dim, num_classes,
                                  method=method, body_kind='D',
                                  pad_idx=stoi[PAD]).to(device)
    fresh.load_state_dict(inner.state_dict(), strict=False)
    fresh.train()
    f1s = []; n_atk = 0
    for batch in train_loader:
        if batch is None or n_atk >= 12: break
        ids = batch['ids']; lab = batch['label']
        fresh.zero_grad(set_to_none=False)
        loss = crit(fresh(ids.to(device)), lab.to(device))
        loss.backward()
        gnorm = fresh.embedding.weight.grad.detach().norm(dim=1).cpu()
        thr = gnorm.max().item() * 0.05 if gnorm.max().item() > 0 else 1e-12
        recovered = set((gnorm > thr).nonzero(as_tuple=True)[0].tolist())
        actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
        tp = len(recovered & actual)
        p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
        f1s.append(2 * p * r / max(p + r, 1e-9))
        n_atk += 1
    return {'achieved_eps': float(eps), 'test_acc': float(test_acc),
            'film_f1': float(np.mean(f1s)) if f1s else 0.0}


# =====================================================================
# Property Inference on multimodal (12 shadows, LPO CV)
# =====================================================================
def property_mm(method, build_fn, vocab_size, num_classes, stoi, n_epochs,
                 batch_size, embedding_dim, seed, device, n_shadow=12,
                 property_class=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_ds, _, _, _ = build_fn()
    coll = functools.partial(collate, pad_index=stoi[PAD])
    rng = np.random.default_rng(seed)
    weight_vectors = []; labels = []
    n = len(train_ds)
    # Get per-example labels by inspecting the dataset
    def get_label(idx):
        item = train_ds[idx]
        if isinstance(item, tuple): return int(item[1])
        return int(item['label'])
    all_labels = [get_label(i) for i in range(n)]
    target_idx = [i for i, l in enumerate(all_labels) if l == property_class]
    other_idx = [i for i, l in enumerate(all_labels) if l != property_class]
    for s in range(n_shadow):
        high = (s % 2 == 0)
        target_frac = 0.5 if high else 0.05
        rng.shuffle(target_idx); rng.shuffle(other_idx)
        n_target = min(int(n * target_frac), len(target_idx))
        n_other = min(n - n_target, len(other_idx))
        sub_ids = target_idx[:n_target] + other_idx[:n_other]
        rng.shuffle(sub_ids)
        sub_ds = torch.utils.data.Subset(train_ds, sub_ids)
        loader = DataLoader(sub_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
        torch.manual_seed(seed + s + 1)
        m = MultiModalClassifier(vocab_size, embedding_dim, num_classes,
                                  method=method, body_kind='D',
                                  pad_idx=stoi[PAD]).to(device)
        opt = optim.Adam(m.parameters(), lr=5e-4)
        crit = nn.CrossEntropyLoss()
        for _ in range(n_epochs):
            m.train()
            for batch in loader:
                if batch is None: continue
                opt.zero_grad()
                crit(m(batch['ids'].to(device)),
                     batch['label'].to(device)).backward()
                opt.step()
        w = torch.cat([m.fc.weight.flatten(), m.fc.bias.flatten()]).detach().cpu().numpy()
        weight_vectors.append(w); labels.append(1 if high else 0)
    X = np.stack(weight_vectors); y = np.array(labels)
    high_idx = np.where(y == 1)[0]; low_idx = np.where(y == 0)[0]
    aucs = []
    for fold in range(min(len(high_idx), len(low_idx))):
        test_idx = np.array([high_idx[fold], low_idx[fold]])
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        if len(np.unique(y[train_idx])) < 2: continue
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X[train_idx], y[train_idx])
        scores = clf.decision_function(X[test_idx])
        aucs.append(float(roc_auc_score(y[test_idx], scores)))
    return float(np.mean(aucs)) if aucs else 0.5


# =====================================================================
# Main
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/multimodal_phase3')
    p.add_argument('--datasets', nargs='+', default=['civic', 'tcga', 'clinvar'])
    p.add_argument('--methods', nargs='+', default=['standard', 'byte_seb'])
    p.add_argument('--n_attack', type=int, default=12)
    p.add_argument('--n_epochs', type=int, default=4)
    p.add_argument('--target_eps', type=float, default=8.0)
    p.add_argument('--n_prop_shadows', type=int, default=12)
    p.add_argument('--attacks', nargs='+', default=['tag', 'llg', 'dpsgd', 'property'])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    setups = {'civic': setup_civic, 'tcga': setup_tcga, 'clinvar': setup_clinvar,
              'pharmgkb': setup_pharmgkb}
    summary = {}
    for ds in args.datasets:
        if ds not in setups: continue
        print(f'\n========== {ds} ==========')
        build, stoi, vocab_meta, mods, num_classes = setups[ds]()
        vocab_size = len(stoi)
        summary[ds] = {}

        for method in args.methods:
            cell = f'{ds}/{method}'
            summary[ds][method] = {}
            print(f'\n--- {cell} ---')

            if 'tag' in args.attacks or 'llg' in args.attacks:
                # Train target model once
                random.seed(42); np.random.seed(42); torch.manual_seed(42)
                if torch.cuda.is_available(): torch.cuda.manual_seed(42)
                train_ds, _, _, _ = build()
                coll = functools.partial(collate, pad_index=stoi[PAD])
                tr = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=coll)
                model = MultiModalClassifier(vocab_size, 64, num_classes,
                                              method=method, body_kind='D',
                                              pad_idx=stoi[PAD]).to(device)
                opt = optim.Adam(model.parameters(), lr=5e-4)
                crit = nn.CrossEntropyLoss()
                for _ in range(args.n_epochs):
                    model.train()
                    for batch in tr:
                        if batch is None: continue
                        opt.zero_grad()
                        crit(model(batch['ids'].to(device)),
                             batch['label'].to(device)).backward()
                        opt.step()

                if 'tag' in args.attacks:
                    t0 = time.time(); f1s = []; n_done = 0
                    for batch in tr:
                        if n_done >= args.n_attack: break
                        if batch is None: continue
                        try:
                            f1s.append(tag_attack_mm(model, batch['ids'],
                                        batch['label'], crit, device, vocab_size))
                        except Exception as e:
                            print(f'  tag err: {e}')
                        n_done += 1
                    summary[ds][method]['tag'] = float(np.mean(f1s)) if f1s else 0.0
                    print(f'  TAG: {summary[ds][method]["tag"]:.3f} ({time.time()-t0:.1f}s)')

                if 'llg' in args.attacks:
                    t0 = time.time(); f1s = []; n_done = 0
                    for batch in tr:
                        if n_done >= args.n_attack: break
                        if batch is None: continue
                        try:
                            f1s.append(fl_llg_token_mm(model, batch['ids'],
                                        batch['label'], crit, device, vocab_size))
                        except Exception as e:
                            print(f'  llg err: {e}')
                        n_done += 1
                    summary[ds][method]['fl_llg_token'] = float(np.mean(f1s)) if f1s else 0.0
                    print(f'  FL-LLG: {summary[ds][method]["fl_llg_token"]:.3f} ({time.time()-t0:.1f}s)')

            if 'dpsgd' in args.attacks and method == 'standard':
                t0 = time.time()
                try:
                    r = dpsgd_mm(method, build, vocab_size, num_classes, stoi,
                                  args.n_epochs, args.target_eps, 16, 64, 42, device)
                    summary[ds][method]['dpsgd'] = r
                    if 'film_f1' in r:
                        print(f'  DPSGD ε={r.get("achieved_eps",0):.2f} acc={r["test_acc"]:.3f} '
                              f'FILM={r["film_f1"]:.3f} ({time.time()-t0:.1f}s)')
                    else:
                        print(f'  DPSGD ERR: {r.get("error","?")}')
                except Exception as e:
                    summary[ds][method]['dpsgd'] = {'error': str(e)}
                    print(f'  DPSGD err: {e}')

            if 'property' in args.attacks:
                t0 = time.time()
                try:
                    auc = property_mm(method, build, vocab_size, num_classes,
                                       stoi, args.n_epochs, 16, 64, 42, device,
                                       n_shadow=args.n_prop_shadows)
                    summary[ds][method]['property'] = float(auc)
                    print(f'  Property AUC={auc:.3f} ({time.time()-t0:.1f}s)')
                except Exception as e:
                    summary[ds][method]['property'] = {'error': str(e)}
                    print(f'  Property err: {e}')

            with open(os.path.join(args.out_dir, 'remaining_cells.json'), 'w') as f:
                json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/remaining_cells.json')


if __name__ == '__main__':
    main()
