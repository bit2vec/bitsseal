#!/usr/bin/env python3
"""
Tables 15-17: Gradient Pruning Defense Comparison

Compares gradient pruning defense vs Binary SEB's inherent privacy.
Tests FILM attack with different prune ratios on WikiText-103.

Paper: "We compare the defense method of gradient pruning in the FILM attack
for batch size = 8, 16, 32."

Prune ratios: 0, 0.9, 0.99, 0.999, 0.9999

Usage:
    python gradient_prune_defense.py --output_dir ./results_prune
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm


def prune_gradients(grad, prune_ratio):
    """Prune gradients by zeroing out the smallest values."""
    if prune_ratio == 0:
        return grad
    flat = grad.abs().view(-1)
    # torch.quantile fails on very large tensors; use kthvalue instead
    k = max(1, int(prune_ratio * flat.numel()))
    threshold = flat.kthvalue(k).values
    mask = grad.abs() >= threshold
    return grad * mask.float()


def compute_precision_recall(extracted_tokens, original_tokens, vocab_size):
    """Compute precision and recall for token extraction."""
    if len(extracted_tokens) == 0:
        return 0.0, 0.0
    precision = len(extracted_tokens & original_tokens) / len(extracted_tokens)
    recall = len(extracted_tokens & original_tokens) / len(original_tokens) if len(original_tokens) > 0 else 0.0
    return precision, recall


def run_prune_experiment(batch_size, prune_ratios, num_samples=100, device='cpu'):
    """Run gradient pruning defense experiment for one batch size."""

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # Load WikiText-103 test
    dataset = load_dataset('wikitext', 'wikitext-103-v1', split='test')
    samples = []
    for item in dataset:
        text = item['text'].strip()
        if not text or text.startswith('='):
            continue
        encoded = tokenizer(text, truncation=True, max_length=128, return_tensors='pt')
        if encoded['input_ids'].shape[1] >= 20:
            samples.append(encoded['input_ids'].squeeze(0))
        if len(samples) >= num_samples * 10:
            break

    results = {'subword': {}, 'binary_seb': {}}

    for embed_type in ['subword', 'binary_seb']:
        print(f"\n  Embedding: {embed_type}")

        # Create embedding
        if embed_type == 'subword':
            embedding = nn.Embedding(vocab_size, 768).to(device)
        else:
            # Binary SEB: only 2 embeddings
            onehot = F.one_hot(torch.arange(2)).float()
            binary_embed = nn.Embedding(2, 2, _weight=onehot).to(device)
            binary_embed.requires_grad_(False)
            mapping = torch.randint(0, 2, (vocab_size, 64), dtype=torch.long).to(device)
            fc1 = nn.Linear(128, 512).to(device)
            fc2 = nn.Linear(512, 768).to(device)

        for prune_ratio in prune_ratios:
            precisions = []
            recalls = []

            for sample_idx in range(min(num_samples, len(samples) // batch_size)):
                # Get batch
                batch_tokens = []
                for j in range(batch_size):
                    idx = sample_idx * batch_size + j
                    if idx >= len(samples):
                        break
                    batch_tokens.append(samples[idx])

                if len(batch_tokens) < batch_size:
                    break

                # Pad batch
                max_len = max(t.size(0) for t in batch_tokens)
                input_ids = torch.zeros(batch_size, max_len, dtype=torch.long).to(device)
                for j, t in enumerate(batch_tokens):
                    input_ids[j, :t.size(0)] = t

                original_tokens = set(input_ids.view(-1).tolist())

                if embed_type == 'subword':
                    embedding.zero_grad()
                    embeds = embedding(input_ids)
                    loss = embeds.sum()
                    loss.backward()

                    grad = embedding.weight.grad.clone()
                    pruned_grad = prune_gradients(grad, prune_ratio)

                    # Extract tokens with non-zero gradients
                    grad_norms = pruned_grad.norm(dim=1)
                    extracted = set((grad_norms > 1e-10).nonzero(as_tuple=True)[0].tolist())
                else:
                    binary_embed.weight.requires_grad_(True)
                    if binary_embed.weight.grad is not None:
                        binary_embed.weight.grad.zero_()

                    binary_seqs = mapping[input_ids]
                    bit_embeds = binary_embed(binary_seqs)
                    flat = bit_embeds.view(batch_size, max_len, -1)
                    out = F.relu(fc1(flat))
                    out = fc2(out)
                    loss = out.sum()
                    loss.backward()

                    grad = binary_embed.weight.grad.clone()
                    pruned_grad = prune_gradients(grad, prune_ratio)

                    # For binary SEB: only 2 embeddings, both always active
                    grad_norms = pruned_grad.norm(dim=1)
                    active_bits = set((grad_norms > 1e-10).nonzero(as_tuple=True)[0].tolist())

                    # ALL tokens map to these bits → massive candidate set
                    extracted = set()
                    if active_bits:
                        mapping_np = mapping.cpu().numpy()
                        for token_id in range(min(vocab_size, 7000)):
                            token_bits = set(mapping_np[token_id].tolist())
                            if token_bits & active_bits:
                                extracted.add(token_id)

                    binary_embed.weight.requires_grad_(False)

                p, r = compute_precision_recall(extracted, original_tokens, vocab_size)
                precisions.append(p)
                recalls.append(r)

            avg_p = np.mean(precisions) if precisions else 0
            avg_r = np.mean(recalls) if recalls else 0

            results[embed_type][str(prune_ratio)] = {
                'precision': avg_p,
                'recall': avg_r,
            }
            print(f"    Prune {prune_ratio}: P={avg_p:.3f}, R={avg_r:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Gradient Prune Defense (Tables 15-17)')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[8, 16, 32])
    parser.add_argument('--prune_ratios', type=float, nargs='+', default=[0, 0.9, 0.99, 0.999, 0.9999])
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='./results_prune')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # Resolve device with safe CUDA fallback
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        torch.zeros(1, device=args.device)
    except (AssertionError, RuntimeError):
        print(f"Warning: device '{args.device}' unavailable, falling back to CPU")
        args.device = 'cpu'

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for bs in args.batch_sizes:
        print(f"\n{'='*60}")
        print(f"Batch size: {bs}")
        print(f"{'='*60}")

        results = run_prune_experiment(
            batch_size=bs,
            prune_ratios=args.prune_ratios,
            num_samples=args.num_samples,
            device=args.device,
        )
        all_results[str(bs)] = results

        # Print table
        table_num = {8: 15, 16: 16, 32: 17}.get(bs, bs)
        print(f"\nTable {table_num}: Defense results for batch size {bs}")
        print(f"{'Prune ratio':<12} {'Subword P':<12} {'001 P':<12} {'Subword R':<12} {'001 R':<12}")
        print("-" * 60)
        for pr in args.prune_ratios:
            sp = results['subword'][str(pr)]['precision']
            bp = results['binary_seb'][str(pr)]['precision']
            sr = results['subword'][str(pr)]['recall']
            br = results['binary_seb'][str(pr)]['recall']
            print(f"{pr:<12} {sp:<12.3f} {bp:<12.3f} {sr:<12.3f} {br:<12.3f}")

    # Save
    with open(os.path.join(args.output_dir, 'gradient_prune_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/gradient_prune_results.json")


if __name__ == '__main__':
    main()
