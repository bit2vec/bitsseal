"""Custom EpiAgent body that mirrors flash-attn's BertEncoder structure
(pre-norm, fused Wqkv) so that pretrained weights from
`pretrained_EpiAgent.pth` load with no key mapping work.

The flash-attn BertEncoder layers are saved as:
  EpiAgent_transformer.layers.{i}.norm1.{w,b}            # pre-attn LN
  EpiAgent_transformer.layers.{i}.mixer.Wqkv.{w,b}       # fused Q/K/V
  EpiAgent_transformer.layers.{i}.mixer.out_proj.{w,b}
  EpiAgent_transformer.layers.{i}.norm2.{w,b}            # pre-ffn LN
  EpiAgent_transformer.layers.{i}.mlp.fc1.{w,b}
  EpiAgent_transformer.layers.{i}.mlp.fc2.{w,b}

This module's `EpiAgentLayer` keeps exactly the same param names so a
direct `load_state_dict(strict=False)` works.

Architecture: pre-norm transformer with GELU activation and fused QKV
projection. Equivalent math to flash-attn (just no flash kernel).
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Mixer(nn.Module):
    """Self-attention with fused Wqkv (matches flash-attn naming)."""

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.Wqkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        B, T, _ = x.shape
        qkv = self.Wqkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, h, T, d]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, h, T, T]
        if key_padding_mask is not None:
            # key_padding_mask: True at valid positions (mirrors flash usage in EpiAgent)
            mask = (~key_padding_mask).bool()  # invert to "to mask"
            attn = attn.masked_fill(mask[:, None, None, :], -1e4)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)  # [B, h, T, d]
        out = out.transpose(1, 2).reshape(B, T, self.hidden_size)
        out = self.out_proj(out)
        return out


class _MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class EpiAgentLayer(nn.Module):
    """Pre-norm transformer block with flash-attn-compatible param names."""

    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.mixer = _Mixer(hidden_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = _MLP(hidden_size, intermediate_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.dropout(self.mixer(self.norm1(x), key_padding_mask))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class EpiAgentBody(nn.Module):
    """Stack of EpiAgentLayer that mirrors `EpiAgent_transformer.layers.{i}`."""

    def __init__(self, hidden_size=512, num_layers=18, num_heads=8,
                 intermediate_size=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EpiAgentLayer(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


def load_pretrained_body(body: EpiAgentBody, ckpt_path: str,
                         prefix: str = 'EpiAgent_transformer.'):
    """Load the body weights from a pretrained_EpiAgent.pth checkpoint.

    Returns a dict with 'loaded' (count of matched keys) and 'unmatched'
    (list of unmatched checkpoint keys).
    """
    full_sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    target_sd = {k[len(prefix):]: v for k, v in full_sd.items()
                 if k.startswith(prefix)}
    missing, unexpected = body.load_state_dict(target_sd, strict=False)
    return {
        'loaded': len(target_sd) - len(unexpected),
        'missing_in_ckpt': list(missing),
        'unexpected_in_ckpt': list(unexpected),
        'pretrained_body_total': sum(v.numel() for v in target_sd.values()),
    }


def load_pretrained_embeddings(ckpt_path: str):
    """Pulls cCRE_embedding.weight and rank_embedding.weight as cpu tensors."""
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return {
        'cCRE_embedding': sd['cCRE_embedding.weight'].float().cpu(),
        'rank_embedding': sd['rank_embedding.weight'].float().cpu(),
    }
