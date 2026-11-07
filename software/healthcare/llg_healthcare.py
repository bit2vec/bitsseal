#!/usr/bin/env python3
"""LLG (Label Leakage from Gradients) attack on healthcare datasets.

Based on software/Privacy_protection/llg_attack_full.py but generalised
to classification datasets with arbitrary #classes, three embedding types
(standard / byte_seb / binary_seb), and three healthcare sources.

Metric: Attack Success Rate (ASR) — what fraction of the true batch-labels
the attacker can recover from the last-layer gradient alone.
A privacy-preserving embedding should drive ASR toward 1 / num_classes
(chance level) or lower.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SENTI_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'Senti_analysis'))
sys.path.insert(0, SENTI_DIR)
sys.path.insert(0, THIS_DIR)

from binary_subword_sentiment import BinarySubwordConcateOnehot   # noqa: E402
from byte_subword_concate_onehot import ByteSubwordConcateOnehot  # noqa: E402
from health_classification import (                                # noqa: E402
    load_ade, load_pubmedqa, load_medmcqa, build_vocab, tokenize,
    TextClassificationDataset, collate, PAD,
)
from domain_classification import (                                # noqa: E402
    load_finance, load_education, load_legal, load_mental,
)


DATASET_LOADERS = {
    'ade': load_ade,
    'pubmedqa': load_pubmedqa,
    'medmcqa': load_medmcqa,
    'finance': load_finance,
    'education': load_education,
    'legal': load_legal,
    'mental': load_mental,
}


class HealthcareClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, embedding_type='standard',
                 num_bits=64, interdim='128', pad_index=0):
        super().__init__()
        if embedding_type == 'standard':
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        elif embedding_type == 'byte_seb':
            # ByteSubwordConcateOnehot misbehaves with empty file path; use a
            # non-existent temp path so it generates fresh random mapping.
            import tempfile
            tmpdir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmpdir, f'byte_mapping_{os.getpid()}.pt')
            self.embedding = ByteSubwordConcateOnehot(
                num_embeddings=vocab_size, embedding_dim=embedding_dim,
                interdim=interdim, relu_dropout=0.1,
                subword_bytes_file=tmp_path,
            )
        elif embedding_type == 'binary_seb':
            self.embedding = BinarySubwordConcateOnehot(
                num_embeddings=vocab_size, embedding_dim=embedding_dim,
                num_bits=num_bits, interdim=interdim, relu_dropout=0.1,
                subword_binary_file='',
            )
        else:
            raise ValueError(embedding_type)
        self.lstm = nn.LSTM(embedding_dim, 128, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        out, (h, _) = self.lstm(emb)
        h = torch.cat([h[-1], h[-2]], dim=-1)
        return self.fc(h)


def extract_labels_llg(gradients, batch_size, num_classes):
    """LLG algorithm (Wainakh et al. 2022) — recover labels from last-layer gradient."""
    g = gradients.sum(dim=1).detach().cpu().numpy()

    extracted = []
    impact = 0.0
    for i in range(num_classes):
        if g[i] < 0:
            extracted.append(i)
            impact += g[i]

    if len(extracted) > 0:
        impact = impact / batch_size * (1 + 1 / num_classes)
    else:
        impact = -1e-4

    missing = batch_size - len(extracted)
    remaining = list(g.copy())
    for _ in range(missing):
        for idx in range(num_classes):
            remaining[idx] -= impact
        min_idx = int(np.argmin(remaining))
        extracted.append(min_idx)
        remaining[min_idx] += impact

    return extracted


def asr(predicted, true):
    pc = Counter(predicted); tc = Counter(true)
    correct = sum(min(pc[k], tc[k]) for k in pc)
    return correct / max(len(true), 1)


def run_attack(method, dataset_name, out_dir,
               embedding_dim=64, num_bits=64, interdim='128',
               batch_size=16, num_attack_batches=100,
               epochs=5, attack_every=1, seed=42):
    print(f"\n===== LLG :: {dataset_name} / {method} =====")
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader_fn = DATASET_LOADERS[dataset_name]
    kwargs = {'seed': seed}
    if dataset_name in ('medmcqa', 'legal', 'mental'):
        kwargs['subsample'] = 10000
    train_ex, val_ex, test_ex, num_classes = loader_fn(**kwargs)
    stoi, itos = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    print(f"  train={len(train_ex)}  vocab={vocab_size}  classes={num_classes}")

    ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    import functools
    coll = functools.partial(collate, pad_index=stoi[PAD])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=coll)

    model = HealthcareClassifier(vocab_size, embedding_dim, num_classes,
                                 embedding_type=method, num_bits=num_bits,
                                 interdim=interdim, pad_index=stoi[PAD]).to(device)
    optim_ = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Untrained-attack phase first (the most-studied threat model)
    print("  phase 1: attacks against untrained model (epoch 0)")
    model.train()
    attack_records = []
    attacked = 0
    for batch in loader:
        if batch is None: continue
        if attacked >= num_attack_batches: break
        ids = batch['ids'].to(device); labels = batch['label'].to(device)
        optim_.zero_grad()
        logits = model(ids, batch['length']) if hasattr(model, 'forward_with_length') else model(ids)
        loss = criterion(logits, labels)
        loss.backward()
        g = model.fc.weight.grad.clone()
        pred = extract_labels_llg(g, ids.size(0), num_classes)
        true = labels.cpu().tolist()
        a = asr(pred, true)
        attack_records.append({'epoch': 0, 'asr': a})
        attacked += 1
        optim_.zero_grad()  # don't actually update; attacker assumption

    # Then train briefly and re-attack
    for epoch in range(1, epochs + 1):
        attacks_this_epoch = 0
        model.train()
        for batch in loader:
            if batch is None: continue
            ids = batch['ids'].to(device); labels = batch['label'].to(device)
            optim_.zero_grad()
            logits = model(ids)
            loss = criterion(logits, labels)
            loss.backward()
            if (epoch % attack_every == 0) and attacks_this_epoch < num_attack_batches:
                g = model.fc.weight.grad.clone()
                pred = extract_labels_llg(g, ids.size(0), num_classes)
                true = labels.cpu().tolist()
                attack_records.append({'epoch': epoch, 'asr': asr(pred, true)})
                attacks_this_epoch += 1
            optim_.step()
        ep_asr = np.mean([r['asr'] for r in attack_records if r['epoch'] == epoch]) if attacks_this_epoch else None
        print(f"  epoch {epoch}: {attacks_this_epoch} attacks, avg ASR={ep_asr:.3f}" if ep_asr is not None else f"  epoch {epoch}: no attacks")

    avg_per_epoch = {}
    for e in sorted(set(r['epoch'] for r in attack_records)):
        recs = [r['asr'] for r in attack_records if r['epoch'] == e]
        avg_per_epoch[int(e)] = float(np.mean(recs))
    overall_mean = float(np.mean([r['asr'] for r in attack_records]))
    chance = 1.0 / num_classes

    result = {
        'dataset': dataset_name, 'method': method,
        'num_classes': num_classes, 'chance_level': chance,
        'overall_mean_asr': overall_mean,
        'asr_per_epoch': avg_per_epoch,
        'num_attack_batches': num_attack_batches,
        'batch_size': batch_size,
        'vocab_size': vocab_size,
    }
    out_file = os.path.join(out_dir, f'{dataset_name}_{method}_llg.json')
    os.makedirs(out_dir, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  overall_ASR={overall_mean:.3f}  chance={chance:.3f}  "
          f"privacy_margin={overall_mean - chance:+.3f}")
    print(f"  wrote {out_file}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True,
                   choices=['ade', 'pubmedqa', 'medmcqa',
                            'finance', 'education', 'legal', 'mental'])
    p.add_argument('--method', required=True, choices=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--output_dir', required=True)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_attack_batches', type=int, default=100)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    run_attack(args.method, args.dataset, args.output_dir,
               batch_size=args.batch_size, num_attack_batches=args.num_attack_batches,
               epochs=args.epochs, seed=args.seed)


if __name__ == '__main__':
    main()
