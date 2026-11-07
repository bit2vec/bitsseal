#!/usr/bin/env python3
"""Phase 2 attacks:
  * LiRA (Carlini '22) -- SOTA membership inference via likelihood-ratio test
                          over N shadow models per IN/OUT split per example.
  * Property Inference (Ganju '18) -- recover global training-set property
                                       (e.g., demographic ratio) via shadow
                                       models + meta-classifier on weights.
"""
import argparse, functools, json, os, random, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, '..', 'Senti_analysis')))

from health_classification import (
    build_vocab, TextClassificationDataset, collate, PAD,
    load_pubmedqa, load_medmcqa,
)
from film_lamp_classification import TinyClassifier
from sentiment_privacy import load_sst2, load_yelp, load_imdb

DATASETS = {
    'medmcqa': lambda seed: load_medmcqa(seed=seed, subsample=5000),
    'pubmedqa': load_pubmedqa,
    'sst2': load_sst2,
    'yelp': load_yelp,
    'imdb': load_imdb,
}


def train_one_model(method, train_ex, stoi, num_classes, embedding_dim,
                    epochs, batch_size, device, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    model = TinyClassifier(len(stoi), embedding_dim, num_classes, method=method).to(device)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for batch in loader:
            if batch is None: continue
            opt.zero_grad()
            crit(model(batch['ids'].to(device)),
                 batch['label'].to(device)).backward()
            opt.step()
    return model


@torch.no_grad()
def per_example_loss(model, examples_ds, stoi, device, batch_size=32):
    crit = nn.CrossEntropyLoss(reduction='none')
    coll = functools.partial(collate, pad_index=stoi[PAD])
    loader = DataLoader(examples_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    losses = []
    model.eval()
    for batch in loader:
        if batch is None: continue
        logits = model(batch['ids'].to(device))
        l = crit(logits, batch['label'].to(device))
        losses.extend(l.cpu().tolist())
    return np.array(losses)


# ============== LiRA ==============
def lira_attack(method, dataset_name, n_shadow=8, embedding_dim=64,
                n_epochs=2, batch_size=16, seed=42):
    """LiRA: train n_shadow models, half include each example, half don't.
    For each test example, fit Gaussians over IN-loss and OUT-loss; report
    likelihood-ratio AUC.
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, _, _, num_classes = DATASETS[dataset_name](seed=seed)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    n = len(train_ex)
    # Use first half as eval pool (the "audit set"); shadow models train on subsets that include or exclude each
    audit_pool = train_ex[:min(n, 1000)]
    n_audit = len(audit_pool)

    # Shadow training: each shadow model trains on a random 50% subset
    print(f'  LiRA {method}/{dataset_name}: training {n_shadow} shadow models...')
    shadow_models = []
    inclusion_masks = np.zeros((n_shadow, n_audit), dtype=bool)
    rng = np.random.default_rng(seed)
    for s in range(n_shadow):
        mask = rng.random(n_audit) < 0.5
        inclusion_masks[s] = mask
        sub = [audit_pool[i] for i in range(n_audit) if mask[i]]
        if not sub: sub = audit_pool[:n_audit//2]
        m = train_one_model(method, sub, stoi, num_classes, embedding_dim,
                            n_epochs, batch_size, device, seed=seed + s + 1)
        shadow_models.append(m)

    # Compute per-example loss under each shadow
    audit_ds = TextClassificationDataset(audit_pool, stoi, max_length=128)
    losses_matrix = np.zeros((n_shadow, n_audit))
    for s, m in enumerate(shadow_models):
        losses_matrix[s] = per_example_loss(m, audit_ds, stoi, device)

    # For each example: fit IN-loss Gaussian + OUT-loss Gaussian, score = log(P_IN / P_OUT)
    scores = np.zeros(n_audit)
    labels = np.zeros(n_audit)  # ground truth in/out for the *first* shadow as the target
    target = 0
    for i in range(n_audit):
        in_losses = losses_matrix[inclusion_masks[:, i], i]
        out_losses = losses_matrix[~inclusion_masks[:, i], i]
        if len(in_losses) < 2 or len(out_losses) < 2:
            scores[i] = 0.0; continue
        mu_in, sig_in = in_losses.mean(), in_losses.std() + 1e-6
        mu_out, sig_out = out_losses.mean(), out_losses.std() + 1e-6
        x = losses_matrix[target, i]
        log_p_in = -0.5 * ((x - mu_in)/sig_in)**2 - np.log(sig_in)
        log_p_out = -0.5 * ((x - mu_out)/sig_out)**2 - np.log(sig_out)
        scores[i] = log_p_in - log_p_out
        labels[i] = 1.0 if inclusion_masks[target, i] else 0.0

    if len(np.unique(labels)) < 2: return {'auc': 0.5, 'n_shadow': n_shadow}
    auc = float(roc_auc_score(labels, scores))
    return {'auc': auc, 'n_shadow': n_shadow, 'n_audit': n_audit}


# ============== Property Inference ==============
def property_inference(method, dataset_name, n_shadow=12, n_epochs=2,
                       batch_size=16, embedding_dim=64, seed=42,
                       property_class=0):
    """Property inference: train n_shadow models, half with high prevalence
    of property_class label, half low. Train meta-classifier on flattened
    classifier head weights to predict property.
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, _, _, num_classes = DATASETS[dataset_name](seed=seed)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    n = len(train_ex)

    # Half shadows: inflate property_class to 50%; half: deflate to 5%.
    rng = np.random.default_rng(seed)
    weight_vectors = []
    labels = []
    for s in range(n_shadow):
        high = (s % 2 == 0)
        # Build a subset with the desired class-prevalence
        target_frac = 0.5 if high else 0.05
        with_target = [(t, l) for t, l in train_ex if l == property_class]
        without_target = [(t, l) for t, l in train_ex if l != property_class]
        rng.shuffle(with_target); rng.shuffle(without_target)
        target_count = min(int(len(train_ex) * target_frac), len(with_target))
        non_count = min(len(train_ex) - target_count, len(without_target))
        sub = with_target[:target_count] + without_target[:non_count]
        rng.shuffle(sub)

        m = train_one_model(method, sub, stoi, num_classes, embedding_dim,
                            n_epochs, batch_size, device, seed=seed + s + 1)
        # Flatten fc layer weights as the input feature
        w_flat = torch.cat([m.fc.weight.flatten(), m.fc.bias.flatten()]).detach().cpu().numpy()
        weight_vectors.append(w_flat)
        labels.append(1 if high else 0)

    X = np.stack(weight_vectors)
    y = np.array(labels)
    # Stratified leave-pair-out: take 1 high+1 low for test, rest for train.
    aucs = []
    high_idx = np.where(y == 1)[0]; low_idx = np.where(y == 0)[0]
    for fold in range(min(len(high_idx), len(low_idx))):
        test_idx = np.array([high_idx[fold], low_idx[fold]])
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        if len(np.unique(y[train_idx])) < 2: continue
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X[train_idx], y[train_idx])
        scores = clf.decision_function(X[test_idx])
        aucs.append(float(roc_auc_score(y[test_idx], scores)))
    auc = float(np.mean(aucs)) if aucs else 0.5
    return {'auc': auc, 'n_shadow': n_shadow, 'property_class': property_class}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['medmcqa', 'pubmedqa', 'sst2'])
    p.add_argument('--methods', nargs='+', default=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--out_dir', default='results/phase2_attacks')
    p.add_argument('--n_lira_shadows', type=int, default=8)
    p.add_argument('--n_prop_shadows', type=int, default=12)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    summary = {'lira': {}, 'property': {}}

    print('===== LiRA SOTA MIA =====')
    for ds in args.datasets:
        summary['lira'][ds] = {}
        for m in args.methods:
            t0 = time.time()
            try:
                r = lira_attack(m, ds, n_shadow=args.n_lira_shadows)
                summary['lira'][ds][m] = r
                print(f'  {ds}/{m}: AUC={r["auc"]:.3f} ({time.time()-t0:.1f}s)')
            except Exception as e:
                summary['lira'][ds][m] = {'error': str(e)}
                print(f'  {ds}/{m} FAIL: {e}')
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    print('\n===== Property Inference =====')
    for ds in args.datasets:
        summary['property'][ds] = {}
        for m in args.methods:
            t0 = time.time()
            try:
                r = property_inference(m, ds, n_shadow=args.n_prop_shadows)
                summary['property'][ds][m] = r
                print(f'  {ds}/{m}: AUC={r["auc"]:.3f} ({time.time()-t0:.1f}s)')
            except Exception as e:
                summary['property'][ds][m] = {'error': str(e)}
                print(f'  {ds}/{m} FAIL: {e}')
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
