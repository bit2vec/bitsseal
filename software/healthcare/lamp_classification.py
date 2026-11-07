#!/usr/bin/env python3
"""LAMP-style attack adapted to classification.

Same gradient extraction as FILM but uses a unigram language-model prior
(computed from the training corpus) to rerank recovered tokens. For
classification — where input has no fixed ordering to evaluate — LAMP's
benefit over FILM is that the recovery focuses on content tokens rather
than function words, since rare-but-informative tokens get higher prior
weight.

Reports: precision, recall, F1, plus a "content-token recall" metric that
weights recall by inverse-unigram-frequency (proxy for how much information
the recovered set contains).
"""

import argparse
import functools
import json
import math
import os
import random
import sys
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

from health_classification import (                                # noqa: E402
    build_vocab, tokenize, TextClassificationDataset, collate, PAD,
)
from film_lamp_classification import (                              # noqa: E402
    DATASETS, TinyClassifier, get_token_grad_norms,
)


def compute_unigram_logprob(train_examples, stoi):
    counts = Counter()
    for text, _ in train_examples:
        counts.update(tokenize(text))
    log_p = torch.full((len(stoi),), -1e9)
    total = sum(counts.values()) + len(stoi)
    for tok, c in counts.items():
        if tok in stoi:
            log_p[stoi[tok]] = math.log((c + 1) / total)
    return log_p


def attack_lamp(model, ids, label, criterion, device,
                unigram_log_p, lambda_prior=0.5, top_k=20):
    """LAMP-style: gradient norm + unigram-prior reranking."""
    grad_norms = get_token_grad_norms(model, ids, label, criterion, device)
    log_grad = torch.log(grad_norms.clamp(min=1e-12))
    score = log_grad + lambda_prior * unigram_log_p
    # Take top-k by score
    actual_ids = [t for t in ids.flatten().tolist() if t > 0]
    actual = set(actual_ids)
    top_k_eff = min(top_k, len(score))
    top = torch.topk(score, top_k_eff).indices.tolist()
    recovered = set(top)
    tp = len(recovered & actual)
    precision = tp / max(len(recovered), 1)
    recall = tp / max(len(actual), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    # Content-recall: weight by -log(unigram freq); rare tokens count more
    weights = (-unigram_log_p).clamp(min=0)
    if len(actual) == 0:
        content_recall = 0.0
    else:
        actual_w = weights[list(actual)].sum().item()
        recovered_w = weights[list(recovered & actual)].sum().item()
        content_recall = recovered_w / max(actual_w, 1e-9)
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'content_recall': content_recall,
            'recovered_size': len(recovered), 'actual_size': len(actual)}


def attack_dataset(method, dataset_name, output_dir, train_epochs=2,
                   num_attack_batches=20, batch_size=16, seed=42):
    print(f"\n===== LAMP :: {dataset_name} / {method} =====")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader_fn = DATASETS[dataset_name]
    kwargs = {'seed': seed}
    if dataset_name in ('medmcqa', 'legal', 'mental'):
        kwargs['subsample'] = 5000
    train_ex, val_ex, test_ex, num_classes = loader_fn(**kwargs)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    unigram = compute_unigram_logprob(train_ex, stoi).to(device)

    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)

    model = TinyClassifier(vocab_size, 64, num_classes, method=method).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    for ep in range(train_epochs):
        for batch in train_loader:
            if batch is None: continue
            optimizer.zero_grad()
            criterion(model(batch['ids'].to(device)), batch['label'].to(device)).backward()
            optimizer.step()

    metrics = []
    done = 0
    for batch in train_loader:
        if batch is None or done >= num_attack_batches: break
        m = attack_lamp(model, batch['ids'], batch['label'], criterion,
                        device, unigram.cpu())
        metrics.append(m); done += 1
    avg = {k: float(np.mean([m[k] for m in metrics]))
           for k in ('precision', 'recall', 'f1', 'content_recall',
                     'recovered_size', 'actual_size')}
    print(f"  prec={avg['precision']:.3f} rec={avg['recall']:.3f} "
          f"f1={avg['f1']:.3f} content_rec={avg['content_recall']:.3f}")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{dataset_name}_{method}_lamp.json'), 'w') as f:
        json.dump({'dataset': dataset_name, 'method': method,
                   'metrics': avg, 'per_batch': metrics}, f, indent=2)
    return avg


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=list(DATASETS.keys()))
    p.add_argument('--method', required=True, choices=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--output_dir', required=True)
    p.add_argument('--train_epochs', type=int, default=2)
    p.add_argument('--num_attack_batches', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=16)
    args = p.parse_args()
    attack_dataset(args.method, args.dataset, args.output_dir,
                   train_epochs=args.train_epochs,
                   num_attack_batches=args.num_attack_batches,
                   batch_size=args.batch_size)
