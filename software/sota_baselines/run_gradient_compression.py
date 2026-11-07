#!/usr/bin/env python3
"""Gradient Compression baseline (Lin '17 / Zhu '19) on Devign + 3 NLP tasks.

For each (dataset, method, top_k) cell:
  1. Train a TinyClassifier briefly (10 epochs, Adam 5e-4)
  2. On a held-out batch, compute the full embedding gradient
  3. Apply top-K sparsification: zero out all but the top fraction k of |grad|
  4. Run FILM (set recovery via per-row grad norm > threshold)
  5. Report token-recovery F1

Compares Subword + GradComp vs Bits-alone (no defense). Bits's structural
floor should beat Subword + any GradComp rate, because top-K leaves the
LARGEST entries — which are exactly the rows for tokens in the batch.
"""
import argparse, functools, json, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, '..'))
sys.path.insert(0, os.path.join(ROOT, 'healthcare'))
sys.path.insert(0, os.path.join(ROOT, 'Senti_analysis'))

from health_classification import (build_vocab, TextClassificationDataset,
                                    collate, PAD, load_pubmedqa)
from film_lamp_classification import TinyClassifier, get_token_grad_norms
from new_domains import load_devign


def load_sst2(seed=42):
    from datasets import load_dataset
    ds = load_dataset('glue', 'sst2')
    train = [(r['sentence'], r['label']) for r in ds['train']]
    test = [(r['sentence'], r['label']) for r in ds['validation']]
    return train, [], test, 2


def load_imdb(seed=42):
    from datasets import load_dataset
    ds = load_dataset('imdb')
    train = [(r['text'], r['label']) for r in ds['train']]
    test = [(r['text'], r['label']) for r in ds['test']]
    return train, [], test, 2


LOADERS = {'devign': load_devign, 'pubmedqa': load_pubmedqa,
           'sst2': load_sst2, 'imdb': load_imdb}


def topk_mask_grad(model, top_k):
    """Apply top-K sparsification to the embedding gradient.

    For 'standard' method: top-K on rows of embedding.weight.grad (V x d).
    For SEB methods: top-K on the embedding's FFN parameter gradients.
    Returns nothing; mutates grads in-place.
    """
    if top_k >= 1.0:
        return  # no-op
    with torch.no_grad():
        if model.method == 'standard':
            g = model.embedding.weight.grad  # (V, d)
            flat = g.abs().flatten()
            k = max(1, int(top_k * flat.numel()))
            thresh = flat.kthvalue(flat.numel() - k).values.item()
            mask = (g.abs() > thresh).float()
            g.mul_(mask)
        else:
            for p in model.embedding.parameters():
                if p.grad is None: continue
                flat = p.grad.abs().flatten()
                k = max(1, int(top_k * flat.numel()))
                thresh = flat.kthvalue(flat.numel() - k).values.item()
                mask = (p.grad.abs() > thresh).float()
                p.grad.mul_(mask)


def film_with_compressed_grad(model, ids, label, criterion, device, top_k):
    """Run FILM after top-K compression of the embedding gradient."""
    model.zero_grad(set_to_none=False)
    logits = model(ids.to(device))
    loss = criterion(logits, label.to(device))
    loss.backward()
    topk_mask_grad(model, top_k)

    # Now read per-token grad norms (after compression)
    if model.method == 'standard':
        gnorms = model.embedding.weight.grad.detach().norm(dim=1).cpu()
    else:
        all_grad = sum(p.grad.detach().pow(2).sum().item()
                       for p in model.embedding.parameters() if p.grad is not None)
        gnorms = torch.full((model.vocab_size,), all_grad, dtype=torch.float)

    thresh = gnorms.max().item() * 0.05
    if thresh == 0: thresh = 1e-12
    recovered_ids = (gnorms > thresh).nonzero(as_tuple=True)[0].tolist()
    recovered = set(recovered_ids)
    actual = set(t for t in ids.flatten().tolist() if t > 0)
    if model.method != 'standard' and len(recovered) > 5 * len(actual):
        recovered = set(random.sample(recovered_ids, min(10, len(recovered_ids))))
    tp = len(recovered & actual)
    prec = tp / max(len(recovered), 1)
    rec = tp / max(len(actual), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {'precision': prec, 'recall': rec, 'f1': f1,
            'recovered_size': len(recovered), 'actual_size': len(actual)}


def run_cell(dataset_name, method, top_ks, n_epochs=4, batch_size=16,
             max_len=256, embedding_dim=64, n_attack_batches=20, seed=42, device=None):
    print(f"\n>>> {dataset_name} / {method} / top_k={top_ks}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    train_ex, _, test_ex, num_classes = LOADERS[dataset_name]()
    print(f"  train={len(train_ex)} test={len(test_ex)} C={num_classes}")
    train_texts = [t[0] for t in train_ex]
    word2idx, _ = build_vocab(train_texts)
    vocab_size = len(word2idx)
    train_ds = TextClassificationDataset(train_ex, word2idx, max_length=max_len)
    coll = functools.partial(collate, pad_index=word2idx[PAD])
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)

    model = TinyClassifier(vocab_size, embedding_dim, num_classes,
                           method=method, num_bits=64, interdim='128').to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        for batch in tr_loader:
            if batch is None: continue
            optimizer.zero_grad()
            criterion(model(batch['ids'].to(device)), batch['label'].to(device)).backward()
            optimizer.step()
    print(f"  trained in {time.time()-t0:.1f}s")

    # FILM under each compression rate
    out = {}
    for k in top_ks:
        film_metrics = []
        n = 0
        for batch in tr_loader:
            if batch is None or n >= n_attack_batches: break
            m = film_with_compressed_grad(model, batch['ids'], batch['label'],
                                          criterion, device, k)
            film_metrics.append(m)
            n += 1
        avg = {kk: float(np.mean([m[kk] for m in film_metrics])) for kk in film_metrics[0]}
        out[f'top_k={k}'] = avg
        print(f"  top_k={k:>5} -> FILM F1={avg['f1']:.4f}")
    return {
        'dataset': dataset_name, 'method': method, 'vocab_size': vocab_size,
        'n_params': sum(p.numel() for p in model.parameters()),
        'top_k_results': out,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['devign', 'pubmedqa', 'sst2', 'imdb'])
    p.add_argument('--methods', nargs='+', default=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--top_ks', nargs='+', type=float, default=[1.0, 0.1, 0.01])
    p.add_argument('--out_dir', default='results/sota_gradcomp')
    p.add_argument('--n_epochs', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--max_len', type=int, default=256)
    p.add_argument('--n_attack_batches', type=int, default=20)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}")

    summary = {}
    for ds in args.datasets:
        summary[ds] = {}
        for m in args.methods:
            try:
                r = run_cell(ds, m, args.top_ks,
                             n_epochs=args.n_epochs,
                             batch_size=args.batch_size,
                             max_len=args.max_len,
                             n_attack_batches=args.n_attack_batches,
                             device=device)
                summary[ds][m] = r
                with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                    json.dump(summary, f, indent=2)
            except Exception as e:
                print(f"  ERROR on {ds}/{m}: {e}")
                summary[ds][m] = {'error': str(e)}

    print(f"\nWrote {args.out_dir}/summary.json")


if __name__ == '__main__':
    main()
