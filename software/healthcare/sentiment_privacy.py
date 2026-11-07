#!/usr/bin/env python3
"""Adapt the healthcare FILM/LAMP/LLG attack scripts to sentiment datasets:
Yelp polarity, IMDB, SST-2. Same TinyClassifier (mean-pool + linear head)
and same attack code, just with sentiment dataset loaders.

Why these three:
- Yelp polarity: the cross_dataset/yelp_to_imdb_zeroshot pretrain task
- IMDB: the zero-shot target of the same task
- SST-2: the basis for the federated_learning experiment
"""
import argparse, functools, json, os, random, sys, time
from collections import Counter

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, '..', 'Senti_analysis')))

from health_classification import (
    build_vocab, tokenize, TextClassificationDataset, collate, PAD,
)
from film_lamp_classification import TinyClassifier, get_token_grad_norms
from lamp_classification import compute_unigram_logprob, attack_lamp
from llg_healthcare import HealthcareClassifier, extract_labels_llg, asr


# -------------- dataset loaders --------------
def _split(examples, val_frac, test_frac, seed):
    rng = random.Random(seed)
    examples = list(examples); rng.shuffle(examples)
    n = len(examples)
    nv = int(n * val_frac); nt = int(n * test_frac)
    return examples[:n - nv - nt], examples[n - nv - nt:n - nt], examples[n - nt:]


def load_yelp(seed=42, train_n=20000, val_n=2000, test_n=2000):
    train_raw = datasets.load_dataset('yelp_polarity', split=f'train[:{train_n}]')
    test_raw = datasets.load_dataset('yelp_polarity', split=f'test[:{val_n + test_n}]')
    train_ex = [(r['text'], int(r['label'])) for r in train_raw]
    eval_ex = [(r['text'], int(r['label'])) for r in test_raw]
    return train_ex, eval_ex[:val_n], eval_ex[val_n:val_n + test_n], 2


def load_imdb(seed=42, train_n=20000, val_n=2000, test_n=2000):
    train_raw = datasets.load_dataset('imdb', split=f'train[:{train_n}]')
    test_raw = datasets.load_dataset('imdb', split=f'test[:{val_n + test_n}]')
    train_ex = [(r['text'], int(r['label'])) for r in train_raw]
    eval_ex = [(r['text'], int(r['label'])) for r in test_raw]
    return train_ex, eval_ex[:val_n], eval_ex[val_n:val_n + test_n], 2


def load_sst2(seed=42):
    raw = datasets.load_dataset('glue', 'sst2')
    train_ex = [(r['sentence'], int(r['label'])) for r in raw['train']]
    val = list(raw['validation'])
    val_ex = [(r['sentence'], int(r['label'])) for r in val[:len(val) // 2]]
    test_ex = [(r['sentence'], int(r['label'])) for r in val[len(val) // 2:]]
    return train_ex, val_ex, test_ex, 2


DATASETS = {'yelp': load_yelp, 'imdb': load_imdb, 'sst2': load_sst2}


# -------------- attack runners --------------
def run_film(method, dataset_name, output_dir, train_epochs=2,
             num_attack_batches=20, batch_size=16, embedding_dim=64,
             num_bits=64, interdim='128', seed=42, top_k_recall=10):
    print(f"\n===== FILM :: {dataset_name} / {method} =====")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, _, _, num_classes = DATASETS[dataset_name](seed=seed)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    print(f"  vocab={vocab_size} classes={num_classes} train={len(train_ex)}")

    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)

    model = TinyClassifier(vocab_size, embedding_dim, num_classes,
                           method=method, num_bits=num_bits, interdim=interdim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    for _ in range(train_epochs):
        for batch in loader:
            if batch is None: continue
            optimizer.zero_grad()
            criterion(model(batch['ids'].to(device)), batch['label'].to(device)).backward()
            optimizer.step()

    metrics = []; done = 0
    for batch in loader:
        if batch is None or done >= num_attack_batches: break
        ids = batch['ids']; label = batch['label']
        grad_norms = get_token_grad_norms(model, ids, label, criterion, device)
        threshold = grad_norms.max().item() * 0.05
        if threshold == 0.0: threshold = 1e-12
        recovered_ids = (grad_norms > threshold).nonzero(as_tuple=True)[0].tolist()
        actual = set(t for t in ids.flatten().tolist() if t > 0)
        recovered = set(recovered_ids)
        if method != 'standard' and len(recovered) > 5 * len(actual):
            recovered = set(random.sample(recovered_ids, min(top_k_recall, len(recovered_ids))))
        tp = len(recovered & actual)
        precision = tp / max(len(recovered), 1)
        recall = tp / max(len(actual), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        metrics.append({'precision': precision, 'recall': recall, 'f1': f1,
                        'recovered_size': len(recovered), 'actual_size': len(actual)})
        done += 1

    avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    print(f"  precision={avg['precision']:.3f} recall={avg['recall']:.3f} f1={avg['f1']:.3f}")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{dataset_name}_{method}_film.json'), 'w') as f:
        json.dump({'dataset': dataset_name, 'method': method,
                   'metrics': avg, 'per_batch': metrics,
                   'num_attack_batches': done}, f, indent=2)
    return avg


def run_lamp(method, dataset_name, output_dir, train_epochs=2,
             num_attack_batches=20, batch_size=16, seed=42):
    print(f"\n===== LAMP :: {dataset_name} / {method} =====")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, _, _, num_classes = DATASETS[dataset_name](seed=seed)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    unigram = compute_unigram_logprob(train_ex, stoi).to(device)

    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)

    model = TinyClassifier(vocab_size, 64, num_classes, method=method).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    for _ in range(train_epochs):
        for batch in loader:
            if batch is None: continue
            optimizer.zero_grad()
            criterion(model(batch['ids'].to(device)), batch['label'].to(device)).backward()
            optimizer.step()

    metrics = []; done = 0
    for batch in loader:
        if batch is None or done >= num_attack_batches: break
        m = attack_lamp(model, batch['ids'], batch['label'], criterion, device, unigram.cpu())
        metrics.append(m); done += 1
    avg = {k: float(np.mean([m[k] for m in metrics]))
           for k in ('precision', 'recall', 'f1', 'content_recall',
                     'recovered_size', 'actual_size')}
    print(f"  precision={avg['precision']:.3f} recall={avg['recall']:.3f} f1={avg['f1']:.3f}")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{dataset_name}_{method}_lamp.json'), 'w') as f:
        json.dump({'dataset': dataset_name, 'method': method,
                   'metrics': avg, 'per_batch': metrics}, f, indent=2)
    return avg


def run_llg(method, dataset_name, output_dir, batch_size=16,
            num_attack_batches=100, epochs=3, seed=42):
    print(f"\n===== LLG :: {dataset_name} / {method} =====")
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, _, _, num_classes = DATASETS[dataset_name](seed=seed)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    print(f"  train={len(train_ex)} vocab={vocab_size} classes={num_classes}")

    ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=coll)

    model = HealthcareClassifier(vocab_size, 64, num_classes, embedding_type=method,
                                 pad_index=stoi[PAD]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    records = []
    # Untrained
    model.train()
    done = 0
    for batch in loader:
        if batch is None or done >= num_attack_batches: break
        ids = batch['ids'].to(device); labels = batch['label'].to(device)
        opt.zero_grad()
        criterion(model(ids), labels).backward()
        g = model.fc.weight.grad.clone()
        pred = extract_labels_llg(g, ids.size(0), num_classes)
        records.append({'epoch': 0, 'asr': asr(pred, labels.cpu().tolist())})
        done += 1
        opt.zero_grad()
    # Few epochs of training, attacks during
    for epoch in range(1, epochs + 1):
        n_atk = 0
        model.train()
        for batch in loader:
            if batch is None: continue
            ids = batch['ids'].to(device); labels = batch['label'].to(device)
            opt.zero_grad()
            criterion(model(ids), labels).backward()
            if n_atk < num_attack_batches:
                g = model.fc.weight.grad.clone()
                pred = extract_labels_llg(g, ids.size(0), num_classes)
                records.append({'epoch': epoch, 'asr': asr(pred, labels.cpu().tolist())})
                n_atk += 1
            opt.step()

    overall = float(np.mean([r['asr'] for r in records]))
    per_ep = {int(e): float(np.mean([r['asr'] for r in records if r['epoch'] == e]))
              for e in sorted(set(r['epoch'] for r in records))}
    chance = 1.0 / num_classes
    result = {'dataset': dataset_name, 'method': method,
              'num_classes': num_classes, 'chance_level': chance,
              'overall_mean_asr': overall, 'asr_per_epoch': per_ep,
              'num_attack_batches': num_attack_batches,
              'batch_size': batch_size, 'vocab_size': vocab_size}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{dataset_name}_{method}_llg.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  ASR={overall:.3f}  chance={chance:.3f}  margin={overall - chance:+.3f}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['yelp', 'imdb', 'sst2'])
    p.add_argument('--methods', nargs='+', default=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--attacks', nargs='+', default=['film', 'lamp', 'llg'])
    p.add_argument('--out_dir', default='results/sentiment_privacy')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_attack_batches', type=int, default=20)
    p.add_argument('--llg_attack_batches', type=int, default=100)
    args = p.parse_args()

    summary = {}
    for ds in args.datasets:
        summary[ds] = {}
        for m in args.methods:
            summary[ds][m] = {}
            t0 = time.time()
            if 'film' in args.attacks:
                summary[ds][m]['film'] = run_film(m, ds, args.out_dir,
                                                   num_attack_batches=args.num_attack_batches,
                                                   batch_size=args.batch_size)
            if 'lamp' in args.attacks:
                summary[ds][m]['lamp'] = run_lamp(m, ds, args.out_dir,
                                                   num_attack_batches=args.num_attack_batches,
                                                   batch_size=args.batch_size)
            if 'llg' in args.attacks:
                summary[ds][m]['llg'] = run_llg(m, ds, args.out_dir,
                                                  num_attack_batches=args.llg_attack_batches,
                                                  batch_size=args.batch_size)
            summary[ds][m]['wall_s'] = time.time() - t0
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {args.out_dir}/summary.json")


if __name__ == '__main__':
    main()
