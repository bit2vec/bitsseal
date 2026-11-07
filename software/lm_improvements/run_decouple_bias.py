"""Train the paper-faithful tied-via-FFN 001 variants on WikiText-103.

Variants (select via --variant):
  tied     : single FFN used for both input path and output projection rows
  decouple : separate FFN_in / FFN_out (idea #1)
  bias     : tied FFN + learnable per-token output bias (idea #2)
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
sys.path.insert(0, THIS_DIR)

from lm_wikitext_comparison import WikiTextDataset, train_epoch, evaluate  # noqa: E402
from decouple_bias_model import DecoupledBiasLM  # noqa: E402


def log_unigram_from_tokens(tokens: np.ndarray, vocab_size: int, smoothing: float = 1.0) -> torch.Tensor:
    counts = np.bincount(tokens, minlength=vocab_size)[:vocab_size].astype(np.float64)
    counts = counts + smoothing
    freqs = counts / counts.sum()
    return torch.from_numpy(np.log(freqs)).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['tied', 'decouple', 'bias'], required=True)
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
    p.add_argument('--max_train_samples', type=int, default=None)
    p.add_argument('--max_valid_samples', type=int, default=10000)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed); torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n===== variant={args.variant}  device={device} =====")

    tok = GPT2Tokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    vocab_size = len(tok)

    train_ds = WikiTextDataset(tok, 'train', args.seq_length, args.stride, args.max_train_samples)
    valid_ds = WikiTextDataset(tok, 'validation', args.seq_length, args.stride, args.max_valid_samples)
    test_ds = WikiTextDataset(tok, 'test', args.seq_length, args.stride, args.max_valid_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    token_bias_init = None
    if args.variant == 'bias':
        print("  computing log-unigram bias init from training tokens...")
        token_bias_init = log_unigram_from_tokens(train_ds._tokens, vocab_size)
        print(f"  bias range: min={token_bias_init.min():.2f}  max={token_bias_init.max():.2f}  "
              f"mean={token_bias_init.mean():.2f}")

    mapping_file = os.path.join(args.output_dir, 'binary_mapping.pt')
    model = DecoupledBiasLM(
        vocab_size=vocab_size, embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_heads=args.num_heads, dropout=args.dropout,
        num_bits=args.num_bits, interdim=args.interdim,
        seq_length=args.seq_length, variant=args.variant,
        mapping_file=mapping_file, token_bias_init=token_bias_init,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    ffn_params = sum(p.numel() for p in model.ffn_in.parameters())
    if args.variant == 'decouple':
        ffn_params += sum(p.numel() for p in model.ffn_out.parameters())
    print(f"  total params: {n_params:,}")
    print(f"  ffn params (all): {ffn_params:,}")
    if model.token_bias is not None:
        print(f"  token_bias: {model.token_bias.numel():,}")

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
        print(f"  resuming from epoch {start_epoch+1}/{args.n_epochs}  (best {best_valid_ppl:.2f})")

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
              f"Train PPL {tr_ppl:.2f}  Valid PPL {va_ppl:.2f}  LR {scheduler.get_last_lr()[0]:.2e}")
        json.dump({
            'variant': args.variant,
            'train_losses': train_losses, 'train_ppls': train_ppls,
            'valid_losses': valid_losses, 'valid_ppls': valid_ppls,
            'epoch_times': epoch_times, 'best_valid_ppl': best_valid_ppl,
        }, open(partial_path, 'w'), indent=2)

    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    te_l, te_ppl = evaluate(test_loader, model, criterion, device)
    summary = {
        'variant': args.variant, 'vocab_size': vocab_size,
        'best_valid_ppl': best_valid_ppl, 'test_ppl': te_ppl,
        'n_params': n_params,
        'train_ppls': train_ppls, 'valid_ppls': valid_ppls,
        'epoch_times': epoch_times,
        'config': {k: getattr(args, k) for k in
                   ['n_epochs','batch_size','embedding_dim','hidden_dim',
                    'num_layers','num_heads','dropout','lr','num_bits',
                    'interdim','seq_length','stride','seed']},
    }
    json.dump(summary, open(os.path.join(args.output_dir, 'summary.json'), 'w'), indent=2)
    print(f"\n[{args.variant}] Best valid PPL {best_valid_ppl:.2f}   Test PPL {te_ppl:.2f}")


if __name__ == '__main__':
    main()
