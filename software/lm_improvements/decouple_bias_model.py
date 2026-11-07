"""
LM model variants that implement weight tying via FFN(bits) as in the
paper, then test two architectural changes:

- variant='tied'    : single FFN serves both input path and output rows
                      (paper-faithful reference; |V| x d output matrix is
                      materialized from FFN at every forward pass).
- variant='decouple': FFN_in != FFN_out. Input path uses ffn_in(bits),
                      output projection rows come from ffn_out(bits).
                      Removes the dual-role constraint at the cost of ~one
                      extra FFN worth of params (small).
- variant='bias'    : single tied FFN + learnable per-token logit bias b_i.
                      Captures the unigram prior that a shared FFN cannot
                      express cleanly.

All variants share the same binary mapping (random unique) and the same
Transformer backbone.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _one_hot_bits() -> nn.Embedding:
    oh = F.one_hot(torch.arange(2)).to(torch.float32)
    emb = nn.Embedding(2, 2, _weight=oh)
    emb.requires_grad_(False)
    return emb


def _build_unique_mapping(num_embeddings: int, num_bits: int) -> torch.Tensor:
    """Random unique binary mapping. Matches paper's setup."""
    mapping = torch.randint(0, 2, (num_embeddings, num_bits), dtype=torch.long)
    seen = set()
    for i in range(num_embeddings):
        seq = tuple(mapping[i].tolist())
        attempts = 0
        while seq in seen and attempts < 1000:
            mapping[i] = torch.randint(0, 2, (num_bits,), dtype=torch.long)
            seq = tuple(mapping[i].tolist())
            attempts += 1
        seen.add(seq)
    return mapping


class BitFFN(nn.Module):
    """Bits -> one-hot -> FFN -> embedding_dim."""

    def __init__(self, embedding_dim: int, num_bits: int = 64,
                 interdim: str = '4096', dropout: float = 0.1,
                 use_gelu: bool = False, use_layernorm: bool = True,
                 bias_init: float = 0.01):
        super().__init__()
        self.num_bits = num_bits
        self.dropout = dropout
        self.use_gelu = use_gelu

        self.binary_embedding = _one_hot_bits()
        in_dim = 2 * num_bits

        dims = [in_dim] + [int(x) for x in str(interdim).split('_')] + [embedding_dim]
        self.linears = nn.ModuleList([nn.Linear(dims[i-1], dims[i]) for i in range(1, len(dims))])
        self.layernum = len(self.linears)

        # Match paper's recommended init (Kaiming hidden, Xavier output, small +ve bias)
        for i, lin in enumerate(self.linears):
            if i + 1 != self.layernum:
                nn.init.kaiming_normal_(lin.weight, nonlinearity='relu', mode='fan_out')
                nn.init.constant_(lin.bias, bias_init)
            else:
                nn.init.xavier_normal_(lin.weight)
                nn.init.zeros_(lin.bias)

        self.out_norm = nn.LayerNorm(embedding_dim) if use_layernorm else None

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        """bits: (..., num_bits) int tensor; output: (..., embedding_dim)."""
        oh = self.binary_embedding(bits)
        x = oh.reshape(*bits.shape[:-1], -1)
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i + 1 != self.layernum:
                x = F.gelu(x) if self.use_gelu else F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.out_norm is not None:
            x = self.out_norm(x)
        return x


class DecoupledBiasLM(nn.Module):
    """Transformer LM with tied / decoupled / bias variants."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int, num_heads: int, dropout: float,
                 num_bits: int = 64, interdim: str = '4096',
                 seq_length: int = 256,
                 variant: str = 'tied',
                 mapping_file: str = '',
                 token_bias_init: Optional[torch.Tensor] = None):
        super().__init__()
        assert variant in ('tied', 'decouple', 'bias')
        self.variant = variant
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # --- Binary mapping (shared by all FFNs) ---
        if mapping_file and os.path.isfile(mapping_file):
            mapping = torch.load(mapping_file)
        else:
            mapping = _build_unique_mapping(vocab_size, num_bits)
            if mapping_file:
                os.makedirs(os.path.dirname(mapping_file) or '.', exist_ok=True)
                torch.save(mapping, mapping_file)
        self.register_buffer('subword2binary', mapping)
        self.register_buffer('all_ids', torch.arange(vocab_size))

        # --- FFNs ---
        self.ffn_in = BitFFN(embedding_dim, num_bits, interdim, dropout)
        if variant == 'decouple':
            self.ffn_out = BitFFN(embedding_dim, num_bits, interdim, dropout)
        else:
            # tied / bias: share the FFN (by reference)
            self.ffn_out = self.ffn_in

        # --- Transformer backbone ---
        self.pos_embedding = nn.Embedding(seq_length + 10, embedding_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=hidden_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)

        if variant == 'bias':
            bias = token_bias_init if token_bias_init is not None else torch.zeros(vocab_size)
            self.token_bias = nn.Parameter(bias.float())
        else:
            self.register_parameter('token_bias', None)

        # Xavier init for transformer + pos embedding
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def _bits_for(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.subword2binary[token_ids]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        token_embeds = self.ffn_in(self._bits_for(input_ids))
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        h = self.drop(token_embeds + self.pos_embedding(positions))

        mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()
        h = self.transformer(h, mask=mask)

        # Output projection: W = ffn_out(bits of all vocab ids)
        out_bits = self._bits_for(self.all_ids)   # (V, num_bits)
        W = self.ffn_out(out_bits)                # (V, d)
        logits = h @ W.t()                        # (B, T, V)
        if self.token_bias is not None:
            logits = logits + self.token_bias
        return logits
