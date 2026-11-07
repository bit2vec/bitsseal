"""Path A: UNTIED 001 LM with full Fairseq-style recipe + n=128 bits.

Goal: beat EmByte's 26.10 PPL on WikiText-103.

Differs from `tied_lm_full_recipe.py`:
- num_bits = 128 (richer per-token bit signal)
- output projection is an INDEPENDENT nn.Linear(d, V) — not materialised from FFN
- otherwise identical training recipe (six fixes)
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
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LM_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'lm'))
sys.path.insert(0, LM_DIR)
sys.path.insert(0, THIS_DIR)

from lm_wikitext_comparison import WikiTextDataset, evaluate            # noqa: E402
from binary_subword_sentiment import BinarySubwordConcateOnehot         # noqa: E402
from tied_lm_full_recipe import make_inv_sqrt_schedule, train_epoch_grad_accum  # noqa: E402


class UntiedFullRecipeLM(nn.Module):
    """Untied 001 LM with embedding-scale fix and independent output projection."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads,
                 dropout, num_bits=128, interdim='4096', seq_length=256, mapping_file=''):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embed_scale = math.sqrt(embedding_dim)        # FIX #1

        self.embedding = BinarySubwordConcateOnehot(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            num_bits=num_bits,
            interdim=interdim,
            relu_dropout=dropout,
            subword_binary_file=mapping_file,
            use_layernorm=False,
            bias_init=0.0,
        )

        self.pos_embedding = nn.Embedding(seq_length + 10, embedding_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=hidden_dim, dropout=dropout,
            batch_first=True, norm_first=True,             # FIX #4 pre-LN
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.final_ln = nn.LayerNorm(embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, vocab_size, bias=False)

        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.xavier_normal_(self.output_proj.weight)

    def forward(self, input_ids):
        B, T = input_ids.shape
        token_embeds = self.embedding(input_ids) * self.embed_scale
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.drop(token_embeds + self.pos_embedding(positions))
        mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()
        h = self.transformer(x, mask=mask)
        h = self.final_ln(h)
        logits = self.output_proj(h)
        return logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', required=True)
    p.add_argument('--n_epochs', type=int, default=25)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--accum_steps', type=int, default=4)
    p.add_argument('--embedding_dim', type=int, default=512)
    p.add_argument('--hidden_dim', type=int, default=2048)
    p.add_argument('--num_layers', type=int, default=6)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--peak_lr', type=float, default=7e-4)
    p.add_argument('--warmup_steps', type=int, default=4000)
    p.add_argument('--num_bits', type=int, default=128)               # path-A choice
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

    print(f"\n===== UNTIED 001 + FULL RECIPE + n={args.num_bits} bits :: device={device} =====")

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
    model = UntiedFullRecipeLM(
        vocab_size=vocab_size, embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_heads=args.num_heads, dropout=args.dropout,
        num_bits=args.num_bits, interdim=args.interdim,
        seq_length=args.seq_length, mapping_file=mapping_file,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_emb = sum(p.numel() for p in model.embedding.parameters())
    n_out = sum(p.numel() for p in model.output_proj.parameters())
    print(f"  total params: {n_params:,}")
    print(f"  embedding (FFN) params: {n_emb:,}")
    print(f"  output_proj params: {n_out:,}")
    print(f"  effective batch size: {args.batch_size * args.accum_steps} sequences "
          f"≈ {args.batch_size * args.accum_steps * args.seq_length:,} tokens/update")

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
            'variant': 'untied_full_recipe', 'num_bits': args.num_bits,
            'train_losses': tr_l, 'train_ppls': tr_p,
            'valid_losses': va_l, 'valid_ppls': va_p,
            'epoch_times': ets, 'best_valid_ppl': best,
        }, open(partial_path, 'w'), indent=2)

    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    tl, tp = evaluate(test_loader, model, criterion, device)
    summary = {
        'variant': 'untied_full_recipe', 'num_bits': args.num_bits,
        'vocab_size': vocab_size, 'best_valid_ppl': best, 'test_ppl': tp,
        'n_params': n_params, 'n_embed_params': n_emb, 'n_output_params': n_out,
        'train_ppls': tr_p, 'valid_ppls': va_p, 'epoch_times': ets,
        'config': vars(args),
    }
    json.dump(summary, open(os.path.join(args.output_dir, 'summary.json'), 'w'), indent=2)
    print(f"\n[untied_full_recipe n={args.num_bits}] best valid PPL {best:.2f}   test PPL {tp:.2f}")


if __name__ == '__main__':
    main()
