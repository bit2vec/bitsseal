"""Encoder variants for Bits ablations.

- StructuredBitsEncoder: Bits with binary-of-token-id codes (vs current random).
- HashEmbeddingEncoder: Svenstrup-style hash embeddings adapted to a Hyena-shaped
  (B, L) -> (B, L, d_model) encoder (vs Bits's V-independent FFN).
"""
import hashlib
import torch
import torch.nn as nn

from binary_subword_sentiment import BinarySubwordConcateOnehot


def _structured_bit_map(num_embeddings, num_bits):
    """Bit i = (token_id >> i) & 1. Unique by construction iff 2^num_bits >= V.

    For V > 2^num_bits we wrap via XOR with a fixed PRNG-seeded mask per overflow
    bucket (matches the structure-then-overflow pattern documented in the paper)."""
    assert num_bits >= 1
    capacity = 1 << num_bits
    ids = torch.arange(num_embeddings, dtype=torch.long)
    # Base structured code: binary representation of (ids % capacity).
    base = (ids.unsqueeze(1) >> torch.arange(num_bits, dtype=torch.long)) & 1
    if num_embeddings <= capacity:
        return base.to(torch.long)
    # Overflow: XOR with a per-bucket mask so each (id // capacity) shifts the pattern.
    bucket = (ids // capacity).unsqueeze(1)
    g = torch.Generator().manual_seed(0xB175C0DE)
    masks = torch.randint(0, 2, (max(int(bucket.max().item()) + 1, 1), num_bits),
                          dtype=torch.long, generator=g)
    return (base ^ masks[bucket.squeeze(1)]).to(torch.long)


class StructuredBinarySubwordConcateOnehot(BinarySubwordConcateOnehot):
    """Bits encoder where the bit map = binary decomposition of token ID
    (instead of random per-token bits). Everything else identical."""

    def _create_unique_mapping(self, num_embeddings, num_bits):
        return _structured_bit_map(num_embeddings, num_bits)


class HashEmbeddingEncoder(nn.Module):
    """Svenstrup hash embedding adapted to (B, L) -> (B, L, d_model).

    Forward:
      ids -> K hashed buckets per id -> sum of K learned vectors weighted by
      learned per-id K importance weights.

    Parameter count: V * K (weights) + B * d (pool). For V=50K, K=2, B=10K, d=128:
      100K + 1.28M params -- comparable to or larger than Bits's V-independent FFN.
    """
    def __init__(self, num_embeddings, embedding_dim, K=2, B_buckets=10000, **_):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.K = K
        self.B = B_buckets
        self.pool = nn.Embedding(self.B, embedding_dim, padding_idx=0)
        self.weights = nn.Embedding(num_embeddings, K, padding_idx=0)
        bucket_ids = torch.zeros(num_embeddings, K, dtype=torch.long)
        for v in range(num_embeddings):
            for k in range(K):
                h = hashlib.sha1(f"{v}_{k}".encode()).digest()
                bucket_ids[v, k] = int.from_bytes(h[:8], "big") % self.B
        self.register_buffer("bucket_ids", bucket_ids)

    def forward(self, ids):
        b_ids = self.bucket_ids[ids]                  # (B, L, K)
        bucket_vecs = self.pool(b_ids)                 # (B, L, K, D)
        wts = self.weights(ids).unsqueeze(-1)          # (B, L, K, 1)
        return (bucket_vecs * wts).sum(dim=2)          # (B, L, D)
