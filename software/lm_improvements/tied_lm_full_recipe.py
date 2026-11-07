"""Paper-faithful tied 001 LM with full Fairseq-style training recipe.

Target: beat paper's reported PPL 29.3 on WikiText-103.

Recipe (combines fixes 1-6 we identified):
  1. Embedding scaled by sqrt(d_model)               [Vaswani 2017]
  2. Inverse-sqrt LR + 4000-step linear warmup       [Vaswani 2017 / Fairseq]
  3. AdamW betas=(0.9, 0.98), eps=1e-9               [Vaswani 2017]
  4. Pre-LN + Xavier init                            [already in code]
  5. (skip adaptive softmax — applies to word vocab) [our 50K BPE]
  6. Gradient accumulation 4x (~32K tokens/update)   [Fairseq standard]

Tying:
  ONE BinarySubwordConcateOnehot FFN serves both:
  - input path  : `embed = FFN(bits_input_token) * sqrt(d)`
  - output path : `W = FFN(all subword bits)` materialized per fwd pass
  - logits      : `h @ W^T` then standard cross-entropy softmax
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LM_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'lm'))
sys.path.insert(0, LM_DIR)

from lm_wikitext_comparison import WikiTextDataset, evaluate            # noqa: E402
from binary_subword_sentiment import BinarySubwordConcateOnehot         # noqa: E402


class PaperTiedLM(nn.Module):
    """Tied 001 LM with embedding-scale fix and FFN-materialized output."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int, num_heads: int, dropout: float,
                 num_bits: int = 64, interdim: str = '4096',
                 seq_length: int = 256, mapping_file: str = ''):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embed_scale = math.sqrt(embedding_dim)        # FIX #1

        # Single FFN used for both input and output
        self.embedding = BinarySubwordConcateOnehot(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            num_bits=num_bits,
            interdim=interdim,
            relu_dropout=dropout,
            subword_binary_file=mapping_file,
            use_layernorm=False,    # paper does not say; keep off for less constraint
            bias_init=0.0,
        )
        # Pre-allocate vocab indices for output projection materialization
        self.register_buffer('all_ids', torch.arange(vocab_size, dtype=torch.long))

        # Pre-LN transformer (FIX #4)
        self.pos_embedding = nn.Embedding(seq_length + 10, embedding_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=hidden_dim, dropout=dropout,
            batch_first=True, norm_first=True,             # pre-LN
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.final_ln = nn.LayerNorm(embedding_dim)

        # Xavier init for transformer matmul weights, normal for embeddings
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        # Input embedding (tied FFN), scaled
        token_embeds = self.embedding(input_ids) * self.embed_scale
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.drop(token_embeds + self.pos_embedding(positions))

        mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()
        h = self.transformer(x, mask=mask)
        h = self.final_ln(h)

        # Output projection materialized via the SAME FFN
        W = self.embedding(self.all_ids)            # (V, d) — paper's tied scheme
        logits = h @ W.t()                          # (B, T, V)
        return logits


def make_inv_sqrt_schedule(optimizer, warmup_steps: int, peak_lr: float):
    """Inverse-sqrt LR with linear warmup. Returns LambdaLR scheduler."""
    def lr_lambda(step):
        # peak at warmup_steps; lr_lambda is multiplicative on initial_lr
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return math.sqrt(warmup_steps / step)
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch_grad_accum(loader, model, criterion, optimizer, scheduler,
                           device, accum_steps=4, clip_grad=1.0):
    """Train one epoch with gradient accumulation for larger effective batch."""
    import tqdm, sys as _sys
    model.train()
    total_loss = 0.0
    total_tokens = 0
    optimizer.zero_grad()
    micro = 0
    for batch in tqdm.tqdm(loader, desc='Train', file=_sys.stdout):
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)

        logits = model(input_ids)
        loss = criterion(logits.view(-1, model.vocab_size), target_ids.view(-1))
        (loss / accum_steps).backward()
        micro += 1

        if micro == accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            micro = 0

        n_tok = target_ids.numel()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok

    # flush any leftover micro-batches
    if micro > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, float(np.exp(avg_loss))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', required=True)
    p.add_argument('--n_epochs', type=int, default=25)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--accum_steps', type=int, default=4)               # FIX #6
    p.add_argument('--embedding_dim', type=int, default=512)
    p.add_argument('--hidden_dim', type=int, default=2048)
    p.add_argument('--num_layers', type=int, default=6)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--peak_lr', type=float, default=7e-4)              # FIX #2
    p.add_argument('--warmup_steps', type=int, default=4000)           # FIX #2
    p.add_argument('--num_bits', type=int, default=64)
    p.add_argument('--interdim', type=str, default='4096')
    p.add_argument('--seq_length', type=int, default=256)
    p.add_argument('--stride', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n===== PAPER-FAITHFUL TIED 001 (full recipe) :: device={device} =====")

    tok = GPT2Tokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    vocab_size = len(tok)

    train_ds = WikiTextDataset(tok, 'train', args.seq_length, args.stride)
    valid_ds = WikiTextDataset(tok, 'validation', args.seq_length, args.stride, max_samples=10000)
    test_ds = WikiTextDataset(tok, 'test', args.seq_length, args.stride, max_samples=10000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    mapping_file = os.path.join(args.output_dir, 'binary_seb_mapping.pt')
    model = PaperTiedLM(
        vocab_size=vocab_size, embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_heads=args.num_heads, dropout=args.dropout,
        num_bits=args.num_bits, interdim=args.interdim,
        seq_length=args.seq_length, mapping_file=mapping_file,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_emb = sum(p.numel() for p in model.embedding.parameters())
    print(f"  total params: {n_params:,}")
    print(f"  embedding (FFN) params: {n_emb:,}")
    print(f"  effective batch size: {args.batch_size * args.accum_steps} sequences "
          f"≈ {args.batch_size * args.accum_steps * args.seq_length:,} tokens/update")

    # FIX #3: Adam betas=(0.9, 0.98), eps=1e-9
    optimizer = optim.AdamW(model.parameters(), lr=args.peak_lr,
                            betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    scheduler = make_inv_sqrt_schedule(optimizer, args.warmup_steps, args.peak_lr)
    criterion = nn.CrossEntropyLoss()

    model_path = os.path.join(args.output_dir, 'best_model.pt')
    partial_path = os.path.join(args.output_dir, 'results.json')
    tr_l, tr_p, va_l, va_p, ets = [], [], [], [], []
    best = float('inf'); start_epoch = 0
    if os.path.isfile(model_path) and os.path.isfile(partial_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        prev = json.load(open(partial_path))
        tr_l = prev.get('train_losses', []); tr_p = prev.get('train_ppls', [])
        va_l = prev.get('valid_losses', []); va_p = prev.get('valid_ppls', [])
        ets = prev.get('epoch_times', [0.0] * len(va_p))
        best = min(va_p) if va_p else float('inf')
        start_epoch = len(va_p)
        # advance scheduler — count effective updates
        steps_per_epoch = max(1, len(train_loader) // args.accum_steps)
        for _ in range(start_epoch * steps_per_epoch):
            scheduler.step()
        print(f"  resuming epoch {start_epoch+1}/{args.n_epochs}  best PPL {best:.2f}")

    start = time.time()
    for epoch in range(start_epoch, args.n_epochs):
        t0 = time.time()
        trl, trp = train_epoch_grad_accum(train_loader, model, criterion,
                                          optimizer, scheduler, device,
                                          accum_steps=args.accum_steps)
        vl, vp = evaluate(valid_loader, model, criterion, device)
        dt = time.time() - t0
        ets.append(dt)
        tr_l.append(trl); tr_p.append(trp); va_l.append(vl); va_p.append(vp)
        if vp < best:
            best = vp
            torch.save(model.state_dict(), model_path)
        print(f"Epoch {epoch+1}/{args.n_epochs} ({dt:.1f}s)  "
              f"Train PPL {trp:.2f}  Valid PPL {vp:.2f}  LR {scheduler.get_last_lr()[0]:.2e}")
        json.dump({
            'variant': 'paper_tied_full_recipe',
            'train_losses': tr_l, 'train_ppls': tr_p,
            'valid_losses': va_l, 'valid_ppls': va_p,
            'epoch_times': ets, 'best_valid_ppl': best,
        }, open(partial_path, 'w'), indent=2)

    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    tl, tp = evaluate(test_loader, model, criterion, device)
    summary = {
        'variant': 'paper_tied_full_recipe', 'vocab_size': vocab_size,
        'best_valid_ppl': best, 'test_ppl': tp,
        'n_params': n_params, 'n_embed_params': n_emb,
        'train_ppls': tr_p, 'valid_ppls': va_p, 'epoch_times': ets,
        'config': vars(args),
    }
    json.dump(summary, open(os.path.join(args.output_dir, 'summary.json'), 'w'), indent=2)
    print(f"\n[paper_tied_full_recipe] best valid PPL {best:.2f}   test PPL {tp:.2f}")


if __name__ == '__main__':
    main()
