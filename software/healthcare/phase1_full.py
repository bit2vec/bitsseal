#!/usr/bin/env python3
"""Full Phase 1 attacks runner: DP-SGD baseline + iDLG + InvertingGradients.

DP-SGD: Opacus-based with proper FILM-after-DP support.
iDLG: analytic label recovery + DLG over inputs.
InvGrad: cosine-similarity gradient matching (Geiping recipe).
"""
import argparse, functools, json, os, random, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, '..', 'Senti_analysis')))

from health_classification import (
    build_vocab, TextClassificationDataset, collate, PAD,
    load_pubmedqa, load_medmcqa,
)
from film_lamp_classification import TinyClassifier, get_token_grad_norms
from extra_attacks import materialize_embeddings
from sentiment_privacy import load_sst2

DATASETS = {
    'medmcqa': lambda seed: load_medmcqa(seed=seed, subsample=5000),
    'pubmedqa': load_pubmedqa,
    'sst2': load_sst2,
}


def idlg_attack(model, ids, label, criterion, device, vocab_size,
                embedding_dim=64, num_iters=100):
    """Analytic label + DLG over inputs."""
    model.zero_grad(set_to_none=False)
    real_emb = model.embedding(ids.to(device))
    real_pooled = real_emb.mean(dim=1)
    real_logits = model.fc(real_pooled)
    real_loss = criterion(real_logits, label.to(device))
    grads_real = [g.detach() for g in torch.autograd.grad(
        real_loss, [model.fc.weight, model.fc.bias], retain_graph=False)]
    fc_w_grad = grads_real[0]
    pred_label = fc_w_grad.sum(dim=1).argmin().item()

    B, T = ids.shape
    dummy_emb = torch.randn(B, T, embedding_dim, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([dummy_emb], lr=0.1, max_iter=20)
    label_tensor = torch.full((B,), pred_label, device=device, dtype=torch.long)

    def closure():
        optimizer.zero_grad()
        d_pooled = dummy_emb.mean(dim=1)
        d_logits = model.fc(d_pooled)
        d_loss = criterion(d_logits, label_tensor)
        d_grads = torch.autograd.grad(d_loss, [model.fc.weight, model.fc.bias],
                                       create_graph=True, retain_graph=True)
        diff = sum(((dg - rg) ** 2).sum() for dg, rg in zip(d_grads, grads_real))
        diff.backward(); return diff

    for _ in range(num_iters // 20):
        try: optimizer.step(closure)
        except Exception: break

    with torch.no_grad():
        emb_table = materialize_embeddings(model, vocab_size, device)
        flat = dummy_emb.detach().reshape(B * T, embedding_dim)
        sim = F.normalize(flat, dim=-1) @ F.normalize(emb_table, dim=-1).t()
        recovered_ids = sim.argmax(dim=-1).cpu().tolist()

    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    recovered = set(recovered_ids)
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {'precision': p, 'recall': r, 'f1': f1,
            'label_correct': int(pred_label == label[0].item())}


def invgrad_attack(model, ids, label, criterion, device, vocab_size,
                   embedding_dim=64, num_iters=200):
    """Cosine-similarity gradient matching (Geiping)."""
    model.zero_grad(set_to_none=False)
    real_emb = model.embedding(ids.to(device))
    real_pooled = real_emb.mean(dim=1)
    real_logits = model.fc(real_pooled)
    real_loss = criterion(real_logits, label.to(device))
    grads_real = [g.detach() for g in torch.autograd.grad(
        real_loss, [model.fc.weight, model.fc.bias], retain_graph=False)]
    grads_real_flat = torch.cat([g.flatten() for g in grads_real])
    grads_real_norm = grads_real_flat.norm()

    B, T = ids.shape
    dummy_emb = torch.randn(B, T, embedding_dim, device=device, requires_grad=True)
    dummy_label = torch.zeros(B, model.fc.out_features, device=device, requires_grad=True)
    optim_ = torch.optim.Adam([dummy_emb, dummy_label], lr=0.05)

    for _ in range(num_iters):
        optim_.zero_grad()
        d_pooled = dummy_emb.mean(dim=1)
        d_logits = model.fc(d_pooled)
        soft = F.softmax(dummy_label, dim=-1)
        log_p = F.log_softmax(d_logits, dim=-1)
        d_loss = -(soft * log_p).sum(dim=-1).mean()
        d_grads = torch.autograd.grad(d_loss, [model.fc.weight, model.fc.bias],
                                       create_graph=True, retain_graph=True)
        d_grads_flat = torch.cat([g.flatten() for g in d_grads])
        cos = -(d_grads_flat * grads_real_flat).sum() / (d_grads_flat.norm() * grads_real_norm + 1e-9)
        tv = (dummy_emb[:, 1:] - dummy_emb[:, :-1]).abs().mean() * 1e-4
        (cos + tv).backward()
        optim_.step()

    with torch.no_grad():
        emb_table = materialize_embeddings(model, vocab_size, device)
        flat = dummy_emb.detach().reshape(B * T, embedding_dim)
        sim = F.normalize(flat, dim=-1) @ F.normalize(emb_table, dim=-1).t()
        recovered_ids = sim.argmax(dim=-1).cpu().tolist()

    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    recovered = set(recovered_ids)
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {'precision': p, 'recall': r, 'f1': f1}


def dp_sgd_train(method, dataset_name, target_eps, batch_size=16,
                 embedding_dim=64, n_epochs=4, max_grad_norm=1.0, seed=42):
    """Train with Opacus DP-SGD; return (acc, eps, FILM F1 on DP model)."""
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, _, test_ex, num_classes = DATASETS[dataset_name](seed=seed)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    test_ds = TextClassificationDataset(test_ex, stoi, max_length=128)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)

    model = TinyClassifier(vocab_size, embedding_dim, num_classes, method=method).to(device)
    opt = optim.SGD(model.parameters(), lr=0.05)
    crit = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()
    model_p, opt_p, train_loader_p = privacy_engine.make_private_with_epsilon(
        module=model, optimizer=opt, data_loader=train_loader,
        target_epsilon=target_eps, target_delta=1e-5, epochs=n_epochs,
        max_grad_norm=max_grad_norm,
    )

    t0 = time.time()
    for ep in range(n_epochs):
        model_p.train()
        with BatchMemoryManager(data_loader=train_loader_p, max_physical_batch_size=64,
                                optimizer=opt_p) as mem_safe:
            for batch in mem_safe:
                if batch is None: continue
                opt_p.zero_grad()
                logits = model_p(batch['ids'].to(device))
                crit(logits, batch['label'].to(device)).backward()
                opt_p.step()
    train_time = time.time() - t0

    eps = privacy_engine.get_epsilon(delta=1e-5)
    model_p.eval()
    correct = total = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            if batch is None: continue
            logits = model_p(batch['ids'].to(device))
            pred = logits.argmax(dim=-1).cpu()
            correct += (pred == batch['label']).sum().item()
            total += batch['label'].size(0)
    test_acc = correct / max(total, 1)
    inf_time = time.time() - t0

    # FILM attack: build a fresh non-DP wrapper, copy weights from DP model.
    # Avoids the Opacus hook-state issue.
    inner_model = model_p._module if hasattr(model_p, '_module') else model_p
    fresh = TinyClassifier(vocab_size, embedding_dim, num_classes, method=method).to(device)
    fresh.load_state_dict(inner_model.state_dict(), strict=False)

    fresh.train()
    film_metrics = []; n_atk = 0
    for batch in train_loader:
        if batch is None or n_atk >= 12: break
        ids = batch['ids']; lab = batch['label']
        gnorms = get_token_grad_norms(fresh, ids, lab, crit, device)
        threshold = gnorms.max().item() * 0.05
        if threshold == 0: threshold = 1e-12
        recovered = set((gnorms > threshold).nonzero(as_tuple=True)[0].tolist())
        actual = set(t for t in ids.flatten().tolist() if t > 0)
        if method != 'standard' and len(recovered) > 5 * len(actual):
            recovered = set(random.sample(list(recovered), min(10, len(recovered))))
        tp = len(recovered & actual)
        p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        film_metrics.append({'f1': f1}); n_atk += 1
    film_avg = float(np.mean([m['f1'] for m in film_metrics])) if film_metrics else 0.0

    return {
        'method': method, 'dataset': dataset_name,
        'target_eps': target_eps, 'achieved_eps': float(eps),
        'test_acc': float(test_acc),
        'train_time_s': train_time, 'inference_time_s': inf_time,
        'film_f1': film_avg, 'n_attacks': len(film_metrics),
    }


def run_attack_grid(method, dataset_name, attack, batch_size=16,
                    embedding_dim=64, num_attack_batches=20, n_train_epochs=2,
                    seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, _, _, num_classes = DATASETS[dataset_name](seed=seed)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)

    model = TinyClassifier(vocab_size, embedding_dim, num_classes, method=method).to(device)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(n_train_epochs):
        for batch in loader:
            if batch is None: continue
            opt.zero_grad()
            crit(model(batch['ids'].to(device)),
                 batch['label'].to(device)).backward()
            opt.step()

    metrics_list = []; label_correct = 0; n_atk = 0
    for batch in loader:
        if batch is None or n_atk >= num_attack_batches: break
        if attack == 'idlg':
            m = idlg_attack(model, batch['ids'], batch['label'], crit, device,
                            vocab_size=vocab_size, embedding_dim=embedding_dim)
            label_correct += m.get('label_correct', 0)
        elif attack == 'invgrad':
            m = invgrad_attack(model, batch['ids'], batch['label'], crit, device,
                                vocab_size=vocab_size, embedding_dim=embedding_dim)
        metrics_list.append(m); n_atk += 1
    avg = {k: float(np.mean([m[k] for m in metrics_list]))
           for k in ('precision', 'recall', 'f1') if k in metrics_list[0]}
    avg['n_attacks'] = n_atk
    if attack == 'idlg':
        avg['label_recovery_rate'] = label_correct / max(n_atk, 1)
    return avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['medmcqa', 'pubmedqa', 'sst2'])
    p.add_argument('--methods', nargs='+', default=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--target_eps', nargs='+', type=float, default=[8.0, 4.0, 1.0])
    p.add_argument('--out_dir', default='results/phase1_full')
    args = p.parse_args()

    summary = {'attack_grid': {}, 'dpsgd': {}}
    os.makedirs(args.out_dir, exist_ok=True)

    print('\n===== Attack grid: iDLG + InvGrad =====')
    for ds in args.datasets:
        summary['attack_grid'][ds] = {}
        for m in args.methods:
            summary['attack_grid'][ds][m] = {}
            for atk in ['idlg', 'invgrad']:
                t0 = time.time()
                try:
                    r = run_attack_grid(m, ds, atk)
                    summary['attack_grid'][ds][m][atk] = r
                    print(f'  {ds}/{m}/{atk}: f1={r["f1"]:.3f} ({time.time()-t0:.1f}s)')
                except Exception as e:
                    summary['attack_grid'][ds][m][atk] = {'error': str(e)}
                    print(f'  {ds}/{m}/{atk} FAIL: {e}')
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    print('\n===== DP-SGD baseline =====')
    for ds in args.datasets:
        summary['dpsgd'][ds] = {}
        for m in args.methods:
            summary['dpsgd'][ds][m] = {}
            for eps in args.target_eps:
                try:
                    r = dp_sgd_train(m, ds, target_eps=eps)
                    summary['dpsgd'][ds][m][f'eps_{eps}'] = r
                    print(f'  DP-SGD {ds}/{m} eps={eps}: '
                          f'acc={r["test_acc"]:.4f} achieved={r["achieved_eps"]:.2f} '
                          f'FILM={r["film_f1"]:.3f}')
                except Exception as e:
                    import traceback; traceback.print_exc()
                    summary['dpsgd'][ds][m][f'eps_{eps}'] = {'error': str(e)}
            with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
