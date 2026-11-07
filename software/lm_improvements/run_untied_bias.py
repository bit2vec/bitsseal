"""Same architecture as baseline 001 LM (untied, independent 25.7M
output projection) but with a learnable per-token output bias
initialized to log-unigram frequencies.

This tests idea #2 on top of the BETTER-performing untied 001 baseline
rather than the paper-faithful tied variant.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LM_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'lm'))
sys.path.insert(0, LM_DIR)

from lm_wikitext_comparison import (  # noqa: E402
    WikiTextDataset, TransformerLM, train_epoch, evaluate,
)


def log_unigram_from_tokens(tokens: np.ndarray, vocab_size: int, smoothing: float = 1.0) -> torch.Tensor:
    counts = np.bincount(tokens, minlength=vocab_size)[:vocab_size].astype(np.float64)
    counts = counts + smoothing
    freqs = counts / counts.sum()
    return torch.from_numpy(np.log(freqs)).float()


def attach_bias(model: TransformerLM, bias_init: torch.Tensor) -> None:
    """Replace output_proj (bias=False) with a biased variant; copy weight;
    initialize bias to log-unigram."""
    d = model.output_proj.weight.shape[1]
    v = model.output_proj.weight.shape[0]
    new = nn.Linear(d, v, bias=True)
    with torch.no_grad():
        new.weight.copy_(model.output_proj.weight)
        new.bias.copy_(bias_init)
    model.output_proj = new


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', required=True)
    p.add_argument('--n_epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--embedding_dim', type=int, default=512)
    p.add_argument('--hidden_dim', type=int, default=2048)
    p.add_argument('--num_layers', type=int, default=6)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--num_bits', type=int, default=64)
    p.add_argument('--interdim', type=str, default='4096')
    p.add_argument('--seq_length', type=int, default=256)
    p.add_argument('--stride', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed); torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n===== UNTIED + BIAS (idea #2 on baseline 001 architecture) =====")
    print(f"device={device}")

    tok = GPT2Tokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    vocab_size = len(tok)

    train_ds = WikiTextDataset(tok, 'train', args.seq_length, args.stride)
    valid_ds = WikiTextDataset(tok, 'validation', args.seq_length, args.stride, max_samples=10000)
    test_ds = WikiTextDataset(tok, 'test', args.seq_length, args.stride, max_samples=10000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print("  computing log-unigram bias init from training tokens...")
    bias_init = log_unigram_from_tokens(train_ds._tokens, vocab_size)
    print(f"  bias range: min={bias_init.min():.2f}  max={bias_init.max():.2f}  mean={bias_init.mean():.2f}")

    # Build baseline 001 LM (untied, independent output_proj)
    mapping_file = os.path.join(args.output_dir, 'binary_seb_mapping.pt')
    model = TransformerLM(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, num_heads=args.num_heads,
        dropout=args.dropout,
        embedding_type='binary_seb',
        num_bits=args.num_bits, interdim=args.interdim,
        subword_file=mapping_file,
        max_seq_len=args.seq_length + 10,
    )
    attach_bias(model, bias_init)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_params:,}")
    print(f"  output_proj.bias init from log-unigram applied")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
    criterion = nn.CrossEntropyLoss()

    model_path = os.path.join(args.output_dir, 'best_model.pt')
    partial_path = os.path.join(args.output_dir, 'results.json')
    train_losses, train_ppls, valid_losses, valid_ppls, epoch_times = [], [], [], [], []
    best_valid_ppl = float('inf'); start_epoch = 0
    if os.path.isfile(model_path) and os.path.isfile(partial_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        prev = json.load(open(partial_path))
        train_losses = prev.get('train_losses', []); train_ppls = prev.get('train_ppls', [])
        valid_losses = prev.get('valid_losses', []); valid_ppls = prev.get('valid_ppls', [])
        epoch_times = prev.get('epoch_times', [0.0] * len(valid_ppls))
        best_valid_ppl = min(valid_ppls) if valid_ppls else float('inf')
        start_epoch = len(valid_ppls)
        for _ in range(start_epoch): scheduler.step()
        print(f"  resuming epoch {start_epoch+1}/{args.n_epochs}  (best {best_valid_ppl:.2f})")

    for epoch in range(start_epoch, args.n_epochs):
        t0 = time.time()
        tr_l, tr_ppl = train_epoch(train_loader, model, criterion, optimizer, device)
        va_l, va_ppl = evaluate(valid_loader, model, criterion, device)
        scheduler.step()
        dt = time.time() - t0
        epoch_times.append(dt)
        train_losses.append(tr_l); train_ppls.append(tr_ppl)
        valid_losses.append(va_l); valid_ppls.append(va_ppl)
        if va_ppl < best_valid_ppl:
            best_valid_ppl = va_ppl
            torch.save(model.state_dict(), model_path)
        print(f"Epoch {epoch+1}/{args.n_epochs} ({dt:.1f}s)  "
              f"Train PPL {tr_ppl:.2f}  Valid PPL {va_ppl:.2f}")
        json.dump({
            'variant': 'untied_bias',
            'train_losses': train_losses, 'train_ppls': train_ppls,
            'valid_losses': valid_losses, 'valid_ppls': valid_ppls,
            'epoch_times': epoch_times, 'best_valid_ppl': best_valid_ppl,
        }, open(partial_path, 'w'), indent=2)

    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    te_l, te_ppl = evaluate(test_loader, model, criterion, device)
    summary = {
        'variant': 'untied_bias',
        'vocab_size': vocab_size, 'best_valid_ppl': best_valid_ppl, 'test_ppl': te_ppl,
        'n_params': n_params,
        'train_ppls': train_ppls, 'valid_ppls': valid_ppls, 'epoch_times': epoch_times,
    }
    json.dump(summary, open(os.path.join(args.output_dir, 'summary.json'), 'w'), indent=2)
    print(f"\n[untied_bias] Best valid PPL {best_valid_ppl:.2f}   Test PPL {te_ppl:.2f}")


if __name__ == '__main__':
    main()
