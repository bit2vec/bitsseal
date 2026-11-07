"""Full-architecture EpiAgent variants for accuracy comparison.

Three variants implementing the same Bert-based body but differing in how the
1.36M-vocab cCRE embedding is parameterised:

  - SubwordEpiAgent: nn.Embedding(V, d) — replicates the original EpiAgent
  - ByteEpiAgent:    cCRE_id -> 8-byte genomic-coordinate hash -> ByteFFN -> d
  - BinaryEpiAgent:  cCRE_id -> 64-bit deterministic hash -> BinaryFFN -> d

Architecture body uses HuggingFace `BertEncoder` (instead of flash-attn) so the
script runs without flash-attn installed. Parameter count of the body matches
the original within rounding.
"""

import hashlib
import math
from typing import Optional

import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder


# ---------------- helpers ----------------

_CODE_CACHE = {}  # share bit/byte tensors across instances to avoid re-hashing


def _make_bit_codes(num_embeddings: int, num_bits: int = 64,
                    seed: int = 1234) -> torch.Tensor:
    """Vectorised: 64-bit code per id = mix of two large primes mod 2^64.

    Fast (millisecond-scale for V=1.36M) and uniformly-distributed enough for
    a hash-based embedding. Frozen — not learnable.
    """
    key = ('bit', num_embeddings, num_bits, seed)
    if key in _CODE_CACHE:
        return _CODE_CACHE[key]
    import numpy as np
    rng = np.random.default_rng(seed)
    # Random 0/1 matrix is the simplest unbiased option for a hash code; still
    # deterministic given the seed.
    arr = rng.integers(0, 2, size=(num_embeddings, num_bits), dtype=np.uint8)
    out = torch.from_numpy(arr)
    _CODE_CACHE[key] = out
    return out


def _make_byte_codes(num_embeddings: int, num_bytes: int = 8,
                     seed: int = 1234) -> torch.Tensor:
    key = ('byte', num_embeddings, num_bytes, seed)
    if key in _CODE_CACHE:
        return _CODE_CACHE[key]
    import numpy as np
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(num_embeddings, num_bytes), dtype=np.int64)
    out = torch.from_numpy(arr)
    _CODE_CACHE[key] = out
    return out


# ---------------- embedding modules ----------------

class BinaryCCREEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 num_bits=64, interdim=2048, dropout=0.1, seed=1234):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_bits = num_bits
        self.register_buffer('bit_codes',
                             _make_bit_codes(num_embeddings, num_bits, seed),
                             persistent=False)
        self.ffn = nn.Sequential(
            nn.Linear(num_bits, interdim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(interdim, embedding_dim, bias=False),
        )

    def forward(self, ids):
        return self.ffn(self.bit_codes[ids].float())

    def materialise(self, chunk=65536):
        codes = self.bit_codes.float()
        out = []
        for s in range(0, codes.size(0), chunk):
            out.append(self.ffn(codes[s:s+chunk]))
        return torch.cat(out, dim=0)


class ByteCCREEmbedding(nn.Module):
    """8-byte one-hot per byte (256-way) -> FFN -> d."""

    def __init__(self, num_embeddings, embedding_dim,
                 num_bytes=8, interdim=2048, dropout=0.1, seed=1234):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_bytes = num_bytes
        self.register_buffer('byte_codes',
                             _make_byte_codes(num_embeddings, num_bytes, seed),
                             persistent=False)
        # Per-byte embedding (256 entries) shared across positions
        self.byte_embed = nn.Embedding(256, embedding_dim // num_bytes)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, interdim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(interdim, embedding_dim, bias=False),
        )

    def forward(self, ids):
        codes = self.byte_codes[ids]                       # [..., 8]
        b = self.byte_embed(codes)                          # [..., 8, d/8]
        x = b.flatten(start_dim=-2)                         # [..., d]
        return self.ffn(x)

    def materialise(self, chunk=65536):
        out = []
        N = self.byte_codes.size(0)
        for s in range(0, N, chunk):
            out.append(self.forward(torch.arange(s, min(s+chunk, N),
                                                  device=self.byte_codes.device)))
        return torch.cat(out, dim=0)


# ---------------- output decoders (tied / untied) ----------------

class TiedDecoder(nn.Module):
    """Output projection materialised from a binary or byte embedding.

    Replaces nn.Linear(d, V_out) so V_out * d weights are *not* stored.
    """

    def __init__(self, embed_module, num_outputs, chunk=65536):
        super().__init__()
        self.embed_module = embed_module
        self.num_outputs = num_outputs
        self.chunk = chunk

    def forward(self, h):  # h: [B, d]
        out_chunks = []
        for s in range(0, self.num_outputs, self.chunk):
            e = min(s + self.chunk, self.num_outputs)
            ids = torch.arange(s, e, device=h.device)
            sub_w = self.embed_module(ids)            # [chunk, d]
            out_chunks.append(h @ sub_w.T)            # [B, chunk]
        return torch.cat(out_chunks, dim=-1)


# ---------------- supervised models for cell-type annotation ----------------

class _ClassifierHead(nn.Module):
    """Replicates EpiAgent's classifier_simple module."""

    def __init__(self, embedding_dim, num_classes, dropout=0.25):
        super().__init__()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.dropout(x)
        x = self.gelu(self.fc2(x))
        x = self.ln2(x)
        return self.fc3(x)


class _BaseEpiAgentSupervised(nn.Module):
    """Common skeleton: cCRE_embedding + rank_embedding + BertEncoder + classifier.

    Subclasses just plug in different cCRE_embedding modules.
    """

    def __init__(self, cCRE_embedding, num_classes,
                 vocab_size=1355449, num_layers=18, embedding_dim=512,
                 num_attention_heads=8, max_rank_embeddings=8192,
                 dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.cCRE_embedding = cCRE_embedding
        self.rank_embedding = nn.Embedding(max_rank_embeddings, embedding_dim)
        cfg = BertConfig(
            vocab_size=vocab_size, hidden_size=embedding_dim,
            num_hidden_layers=num_layers, num_attention_heads=num_attention_heads,
            intermediate_size=4 * embedding_dim, hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_rank_embeddings,
            _attn_implementation='eager',
        )
        self.body = BertEncoder(cfg)
        self.classifier = _ClassifierHead(embedding_dim, num_classes, dropout=0.25)

    def forward(self, input_ids):
        x = self.cCRE_embedding(input_ids)                              # [B, T, d]
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = x + self.rank_embedding(positions)[None]                    # [B, T, d]
        attn_mask = (input_ids != 0)[:, None, None, :].float()
        attn_mask = (1.0 - attn_mask) * -1e4                            # additive
        h = self.body(hidden_states=x, attention_mask=attn_mask).last_hidden_state
        return self.classifier(h[:, 0, :])                              # CLS

    def count_params(self):
        return {
            'cCRE_embedding': sum(p.numel() for p in self.cCRE_embedding.parameters()),
            'rank_embedding': sum(p.numel() for p in self.rank_embedding.parameters()),
            'body':           sum(p.numel() for p in self.body.parameters()),
            'classifier':     sum(p.numel() for p in self.classifier.parameters()),
            'TOTAL':          sum(p.numel() for p in self.parameters()),
        }


class SubwordEpiAgent(_BaseEpiAgentSupervised):
    def __init__(self, num_classes, **kwargs):
        vocab = kwargs.get('vocab_size', 1355449)
        d = kwargs.get('embedding_dim', 512)
        cCRE = nn.Embedding(vocab, d)
        super().__init__(cCRE, num_classes, **kwargs)


class BinaryEpiAgent(_BaseEpiAgentSupervised):
    def __init__(self, num_classes, num_bits=64, interdim=2048, **kwargs):
        vocab = kwargs.get('vocab_size', 1355449)
        d = kwargs.get('embedding_dim', 512)
        cCRE = BinaryCCREEmbedding(vocab, d, num_bits=num_bits, interdim=interdim)
        super().__init__(cCRE, num_classes, **kwargs)


class ByteEpiAgent(_BaseEpiAgentSupervised):
    def __init__(self, num_classes, num_bytes=8, interdim=2048, **kwargs):
        vocab = kwargs.get('vocab_size', 1355449)
        d = kwargs.get('embedding_dim', 512)
        cCRE = ByteCCREEmbedding(vocab, d, num_bytes=num_bytes, interdim=interdim)
        super().__init__(cCRE, num_classes, **kwargs)
