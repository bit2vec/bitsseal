"""
Driver for 001 LM improvement experiments (ideas 3, 1, 2).

Reuses WikiTextDataset / train_epoch / evaluate / MemoryTracker from the
existing lm_wikitext_comparison module without modifying it. Only the
embedding module inside TransformerLM is swapped for a custom projector,
and (for idea 2) an auxiliary distillation loss is added.

Checkpoint + partial-results resume follow the same format as the baseline.
"""

import argparse
import json
import os
import sys
import time
import copy

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

from lm_wikitext_comparison import (  # noqa: E402
    MemoryTracker, WikiTextDataset, TransformerLM,
    train_epoch, evaluate,
)
from improved_projectors import BinaryResidualAE, BinarySemHash  # noqa: E402


def build_model(idea: str, vocab_size: int, embedding_dim: int, hidden_dim: int,
                num_layers: int, num_heads: int, dropout: float, num_bits: int,
                interdim: str, seq_length: int, output_dir: str,
                pretrained_emb_file: str) -> TransformerLM:
    # Build the baseline TransformerLM as binary_seb, then swap its embedding
    model = TransformerLM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        embedding_type='binary_seb',
        num_bits=num_bits,
        interdim=interdim,
        subword_file=os.path.join(output_dir, 'binary_seb_mapping.pt'),
        max_seq_len=seq_length + 10,
    )

    if idea == 'residual_ae':
        model.embedding = BinaryResidualAE(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            num_bits=num_bits,
            relu_dropout=dropout,
            subword_binary_file=os.path.join(output_dir, 'binary_seb_mapping.pt'),
        )
    elif idea == 'semhash':
        model.embedding = BinarySemHash(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            num_bits=num_bits,
            interdim=interdim,
            relu_dropout=dropout,
            pretrained_emb_file=pretrained_emb_file,
            subword_binary_file=os.path.join(output_dir, 'binary_seb_mapping.pt'),
        )
    elif idea == 'distill':
        # Idea 2 reuses the baseline binary_seb projector — only the training
        # loop differs (added MSE-to-subword-emb loss). No embedding swap.
        pass
    else:
        raise ValueError(f"unknown idea: {idea}")

    return model


def train_epoch_with_aux(loader, model, criterion, optimizer, device,
                         aux_fn=None, aux_weight: float = 0.0,
                         clip_grad: float = 1.0):
    """Copy of train_epoch (WikiTextDataset yields input_ids / target_ids)
    with an extra aux-loss term."""
    import tqdm, sys
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for batch in tqdm.tqdm(loader, desc='Training', file=sys.stdout):
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        task_loss = criterion(logits.view(-1, model.vocab_size), target_ids.view(-1))
        loss = task_loss
        if aux_fn is not None and aux_weight > 0:
            aux = aux_fn()
            if aux.device != loss.device:
                aux = aux.to(loss.device)
            loss = loss + aux_weight * aux
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        n_tok = target_ids.numel()
        total_loss += task_loss.item() * n_tok
        total_tokens += n_tok
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, float(np.exp(avg_loss))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--idea', choices=['residual_ae', 'semhash', 'distill'], required=True)
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
    p.add_argument('--pretrained_emb_file', type=str,
                   default='results/language_modeling/best_model_standard.pt')
    p.add_argument('--distill_weight', type=float, default=0.1)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed); torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}\n IDEA={args.idea}  device={device}\n{'='*70}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    train_ds = WikiTextDataset(tokenizer, split='train',
                               seq_length=args.seq_length, stride=args.stride,
                               max_samples=args.max_train_samples)
    valid_ds = WikiTextDataset(tokenizer, split='validation',
                               seq_length=args.seq_length, stride=args.stride,
                               max_samples=args.max_valid_samples)
    test_ds = WikiTextDataset(tokenizer, split='test',
                              seq_length=args.seq_length, stride=args.stride,
                              max_samples=args.max_valid_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(
        idea=args.idea, vocab_size=vocab_size,
        embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, num_heads=args.num_heads,
        dropout=args.dropout, num_bits=args.num_bits,
        interdim=args.interdim, seq_length=args.seq_length,
        output_dir=args.output_dir,
        pretrained_emb_file=args.pretrained_emb_file,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
    criterion = nn.CrossEntropyLoss()

    # Checkpoint / resume
    model_path = os.path.join(args.output_dir, 'best_model.pt')
    partial_path = os.path.join(args.output_dir, 'results.json')
    train_losses, train_ppls, valid_losses, valid_ppls, epoch_times = [], [], [], [], []
    best_valid_ppl = float('inf'); start_epoch = 0
    if os.path.isfile(model_path) and os.path.isfile(partial_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        with open(partial_path) as f:
            prev = json.load(f)
        train_losses = prev.get('train_losses', [])
        train_ppls = prev.get('train_ppls', [])
        valid_losses = prev.get('valid_losses', [])
        valid_ppls = prev.get('valid_ppls', [])
        epoch_times = prev.get('epoch_times', [0.0] * len(valid_ppls))
        best_valid_ppl = min(valid_ppls) if valid_ppls else float('inf')
        start_epoch = len(valid_ppls)
        for _ in range(start_epoch): scheduler.step()
        print(f"  Resuming from epoch {start_epoch+1}/{args.n_epochs} (best PPL {best_valid_ppl:.2f})")

    # Prepare distillation target for idea 2
    distill_target = None
    if args.idea == 'distill':
        ckpt = torch.load(args.pretrained_emb_file, map_location=device, weights_only=False)
        distill_target = ckpt['embedding.weight'][:vocab_size].detach().to(device)
        print(f"  distillation target: {tuple(distill_target.shape)}")

    def aux_fn():
        # Always returns a scalar tensor on the right device
        losses = []
        # Auto-encoder-style reconstruction loss (idea 3)
        if hasattr(model.embedding, 'aux_recon_loss'):
            losses.append(model.embedding.aux_recon_loss().to(device))
        # Distillation MSE (idea 2)
        if args.idea == 'distill' and distill_target is not None:
            proj = model.embedding(model.embedding.subword, dropout=0)
            losses.append(nn.functional.mse_loss(proj, distill_target))
        if not losses:
            return torch.tensor(0.0, device=device)
        return sum(losses)

    aux_weight = args.distill_weight if args.idea == 'distill' else 1.0

    start_wall = time.time()
    for epoch in range(start_epoch, args.n_epochs):
        t0 = time.time()
        tr_l, tr_ppl = train_epoch_with_aux(train_loader, model, criterion, optimizer, device,
                                            aux_fn=aux_fn, aux_weight=aux_weight)
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
        with open(partial_path, 'w') as f:
            json.dump({
                'idea': args.idea, 'train_losses': train_losses,
                'train_ppls': train_ppls, 'valid_losses': valid_losses,
                'valid_ppls': valid_ppls, 'epoch_times': epoch_times,
                'best_valid_ppl': best_valid_ppl,
            }, f, indent=2)

    # Final test eval
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    te_l, te_ppl = evaluate(test_loader, model, criterion, device)
    total_time = time.time() - start_wall
    summary = {
        'idea': args.idea, 'vocab_size': vocab_size,
        'best_valid_ppl': best_valid_ppl, 'test_ppl': te_ppl,
        'total_train_time': total_time,
        'avg_epoch_time': float(np.mean(epoch_times)) if epoch_times else 0.0,
        'n_params': n_params,
        'train_ppls': train_ppls, 'valid_ppls': valid_ppls,
        'epoch_times': epoch_times,
        'config': {k: getattr(args, k) for k in
                   ['n_epochs','batch_size','embedding_dim','hidden_dim',
                    'num_layers','num_heads','dropout','lr','num_bits',
                    'interdim','seq_length','stride','seed','distill_weight']},
    }
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{args.idea}] Best valid PPL: {best_valid_ppl:.2f}   Test PPL: {te_ppl:.2f}")


if __name__ == '__main__':
    main()
