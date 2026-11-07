"""Symmetric-bit-I/O LM for 001.

Input side: same as baseline 001 (binary_seb) — random 64-bit mapping
of subwords, one-hot-expanded, FFN to embedding.

Output side (NEW): 64-dim Bernoulli bit-logit head. Per-subword log-prob
is sum of log P(bit_i = bits_w_i | context) over all 64 bits. Softmax
renormalizes over the 50,257 known subwords.

Output head size: d*64 + 64 ≈ 32K parameters, independent of vocab.
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

from lm_wikitext_comparison import WikiTextDataset, train_epoch, evaluate  # noqa: E402
from binary_subword_sentiment import BinarySubwordConcateOnehot            # noqa: E402


class BitOutputHead(nn.Module):
    """h (d-dim) -> 64 bit logits -> per-vocab log-prob via
    product over independent Bernoullis using the fixed bit mapping."""

    def __init__(self, d: int, num_bits: int, bits_map: torch.Tensor):
        super().__init__()
        # bits_map: (V, num_bits) int tensor in {0,1}
        self.d = d
        self.num_bits = num_bits
        self.register_buffer('B', bits_map.float())  # (V, num_bits)
        self.proj = nn.Linear(d, num_bits)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (..., d). Returns logits: (..., V)
        a = self.proj(h)                             # (..., num_bits)
        log_p1 = F.logsigmoid(a)                     # log σ(a)
        log_p0 = F.logsigmoid(-a)                    # log σ(-a)
        # per-word log prob:   B @ log_p1 + (1-B) @ log_p0
        # Using einsum for arbitrary leading dims:
        logits = torch.einsum('...i,vi->...v', log_p1, self.B) + \
                 torch.einsum('...i,vi->...v', log_p0, (1.0 - self.B))
        return logits


class BitOutputTransformerLM(nn.Module):
    """001 Transformer with bit input (FFN) and bit output (factorised)."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads,
                 dropout, num_bits=64, interdim='4096', seq_length=256,
                 mapping_file=''):
        super().__init__()
        self.vocab_size = vocab_size

        # Input embedding = existing BinarySubwordConcateOnehot
        self.embedding = BinarySubwordConcateOnehot(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            num_bits=num_bits,
            interdim=interdim,
            relu_dropout=dropout,
            subword_binary_file=mapping_file,
        )

        # Share the binary mapping with the output head
        bits_map = self.embedding.subword2binary  # (V, num_bits) long tensor
        self.output_head = BitOutputHead(embedding_dim, num_bits, bits_map.long())

        # Standard transformer backbone
        self.pos_embedding = nn.Embedding(seq_length + 10, embedding_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=hidden_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)

        # Xavier init
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.xavier_normal_(self.output_head.proj.weight)

    def forward(self, input_ids):
        B, T = input_ids.shape
        token_embeds = self.embedding(input_ids)
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.drop(token_embeds + self.pos_embedding(positions))

        mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()
        h = self.transformer(x, mask=mask)
        logits = self.output_head(h)  # (B, T, V)
        return logits


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
    print(f"\n===== BIT-OUTPUT LM :: device={device} =====")

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
    model = BitOutputTransformerLM(
        vocab_size=vocab_size, embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_heads=args.num_heads, dropout=args.dropout,
        num_bits=args.num_bits, interdim=args.interdim,
        seq_length=args.seq_length, mapping_file=mapping_file,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_output = sum(p.numel() for p in model.output_head.parameters())
    print(f"  total params: {n_params:,}")
    print(f"  output-head params: {n_output:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
    criterion = nn.CrossEntropyLoss()

    model_path = os.path.join(args.output_dir, 'best_model.pt')
    partial_path = os.path.join(args.output_dir, 'results.json')
    tr_l, tr_p, va_l, va_p, ets = [], [], [], [], []
    best = float('inf'); start_epoch = 0
    if os.path.isfile(model_path) and os.path.isfile(partial_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        prev = json.load(open(partial_path))
        tr_l = prev.get('train_losses', [])
        tr_p = prev.get('train_ppls', [])
        va_l = prev.get('valid_losses', [])
        va_p = prev.get('valid_ppls', [])
        ets = prev.get('epoch_times', [0.0] * len(va_p))
        best = min(va_p) if va_p else float('inf')
        start_epoch = len(va_p)
        for _ in range(start_epoch): scheduler.step()
        print(f"  resuming epoch {start_epoch+1}/{args.n_epochs} (best {best:.2f})")

    start_wall = time.time()
    for epoch in range(start_epoch, args.n_epochs):
        t0 = time.time()
        trl, trp = train_epoch(train_loader, model, criterion, optimizer, device)
        vl, vp = evaluate(valid_loader, model, criterion, device)
        scheduler.step()
        dt = time.time() - t0
        ets.append(dt)
        tr_l.append(trl); tr_p.append(trp); va_l.append(vl); va_p.append(vp)
        if vp < best:
            best = vp
            torch.save(model.state_dict(), model_path)
        print(f"Epoch {epoch+1}/{args.n_epochs} ({dt:.1f}s)  "
              f"Train PPL {trp:.2f}  Valid PPL {vp:.2f}")
        json.dump({
            'variant': 'bit_output', 'train_losses': tr_l, 'train_ppls': tr_p,
            'valid_losses': va_l, 'valid_ppls': va_p, 'epoch_times': ets,
            'best_valid_ppl': best,
        }, open(partial_path, 'w'), indent=2)

    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    tl, tp = evaluate(test_loader, model, criterion, device)
    summary = {
        'variant': 'bit_output', 'vocab_size': vocab_size,
        'best_valid_ppl': best, 'test_ppl': tp,
        'n_params_total': n_params, 'n_params_output_head': n_output,
        'train_ppls': tr_p, 'valid_ppls': va_p, 'epoch_times': ets,
    }
    json.dump(summary, open(os.path.join(args.output_dir, 'summary.json'), 'w'), indent=2)
    print(f"\n[bit_output] Best valid PPL {best:.2f}   Test PPL {tp:.2f}")


if __name__ == '__main__':
    main()
