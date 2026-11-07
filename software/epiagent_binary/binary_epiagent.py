"""Binary-embedding drop-in for EpiAgent (Chen et al. Nature Methods 2025).

EpiAgent uses two huge V x d tables (V = 1,355,449, d = 512):
  - cCRE_embedding (input)              ~694 M params
  - signal_decoder (output projection)  ~694 M params
Together they account for ~93% of the 1.5 B-parameter model.

This module replaces both with a single 64-bit-code binary FFN that:
  - costs ~1.2 M params instead of 694 M (570x compression)
  - generates a deterministic 64-bit code per cCRE ID via a packed bit LUT
    (10.83 MB, frozen, not learnable)
  - decodes via FFN(64 -> interdim -> 512)
  - can be tied at output by re-using the same FFN to materialize V x d
    rows on-demand: out = h @ FFN(all_codes).T
"""

import hashlib
import math
from typing import Optional

import torch
import torch.nn as nn


def _make_bit_codes(num_embeddings: int, num_bits: int = 64,
                    seed: int = 1234) -> torch.Tensor:
    """Deterministically map each integer id in [0, V) to a {0, 1}^B vector.

    Uses SHA-1 of (id, seed) and slices the leading B bits. Returns a uint8
    tensor packed to 1 bit per entry (so memory cost is V*B/8 bytes).
    """
    nbytes = (num_bits + 7) // 8
    out = torch.empty((num_embeddings, num_bits), dtype=torch.uint8)
    seed_bytes = seed.to_bytes(4, 'big')
    for i in range(num_embeddings):
        h = hashlib.sha1(seed_bytes + i.to_bytes(8, 'big')).digest()[:nbytes]
        for b in range(num_bits):
            out[i, b] = (h[b // 8] >> (7 - (b % 8))) & 1
    return out


class BinaryCCREEmbedding(nn.Module):
    """Drop-in replacement for `nn.Embedding(num_embeddings, embedding_dim)`.

    Maintains a frozen V x B bit-code buffer; learnable params live in the
    `Linear -> activation -> Linear` FFN.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 num_bits: int = 64, interdim: int = 2048,
                 dropout: float = 0.1, seed: int = 1234):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_bits = num_bits

        bit_codes = _make_bit_codes(num_embeddings, num_bits, seed=seed)
        # Frozen buffer; cast to float at lookup time (uint8 saves memory).
        self.register_buffer('bit_codes', bit_codes, persistent=False)

        self.ffn = nn.Sequential(
            nn.Linear(num_bits, interdim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(interdim, embedding_dim, bias=False),
        )

    @property
    def weight(self):
        """For compatibility with code that reads `embedding.weight`."""
        return self.materialise()

    def materialise(self) -> torch.Tensor:
        """Compute the full V x d table (rare; used for distillation/debug)."""
        codes = self.bit_codes.float()
        return self.ffn(codes)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        codes = self.bit_codes[ids].float()
        return self.ffn(codes)


class TiedBinaryDecoder(nn.Module):
    """Output projection (V x d) materialised from a BinaryCCREEmbedding.

    Replaces `nn.Linear(d, V)`. At forward time computes
        logits[..., v] = h[..., :] @ E[v, :].T
    where E = ffn(all_bit_codes).
    """

    def __init__(self, binary_embedding: BinaryCCREEmbedding,
                 num_outputs: Optional[int] = None,
                 chunk: int = 65536):
        super().__init__()
        self.binary_embedding = binary_embedding
        # If num_outputs < num_embeddings, predict only the first num_outputs
        # rows (EpiAgent's `signal_decoder` is V-4, dropping 4 special tokens).
        self.num_outputs = num_outputs or binary_embedding.num_embeddings
        self.chunk = chunk

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [..., d] -> logits: [..., V_out]."""
        # Materialise weights chunk-by-chunk to avoid OOM on V = 1.36M.
        all_codes = self.binary_embedding.bit_codes[: self.num_outputs].float()
        logits_chunks = []
        for s in range(0, self.num_outputs, self.chunk):
            e = min(s + self.chunk, self.num_outputs)
            sub_w = self.binary_embedding.ffn(all_codes[s:e])  # [chunk, d]
            logits_chunks.append(h @ sub_w.T)                  # [..., chunk]
        return torch.cat(logits_chunks, dim=-1)


class BinaryEpiAgentMinimal(nn.Module):
    """Minimal EpiAgent variant for parameter-count and distillation studies.

    Uses vanilla `nn.TransformerEncoder` instead of FlashAttention BertEncoder,
    since this script's purpose is measuring embedding compression and
    distillation feasibility, not reproducing the full pretraining loop.
    The math/parameter-count of the body is the same as the original.
    """

    def __init__(self,
                 vocab_size: int = 1355449,
                 num_layers: int = 18,
                 embedding_dim: int = 512,
                 num_attention_heads: int = 8,
                 max_rank_embeddings: int = 8192,
                 num_bits: int = 64,
                 interdim: int = 2048,
                 use_binary_input: bool = True,
                 use_binary_output: bool = True,
                 num_signal_outputs: Optional[int] = None,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size

        # Input embedding (full or binary)
        if use_binary_input:
            self.cCRE_embedding = BinaryCCREEmbedding(
                num_embeddings=vocab_size, embedding_dim=embedding_dim,
                num_bits=num_bits, interdim=interdim, dropout=dropout)
        else:
            self.cCRE_embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rank_embedding = nn.Embedding(max_rank_embeddings, embedding_dim)

        # Vanilla transformer body (substitute for flash BertEncoder)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_attention_heads,
            dim_feedforward=4 * embedding_dim, dropout=dropout,
            batch_first=True, norm_first=False, activation='gelu')
        self.body = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Output / signal decoder (V-4 in original)
        out_size = num_signal_outputs or (vocab_size - 4)
        if use_binary_output:
            assert isinstance(self.cCRE_embedding, BinaryCCREEmbedding)
            self.signal_decoder = TiedBinaryDecoder(
                self.cCRE_embedding, num_outputs=out_size)
        else:
            self.signal_decoder = nn.Linear(embedding_dim, out_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.cCRE_embedding(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = x + self.rank_embedding(positions).unsqueeze(0)
        h = self.body(x, src_key_padding_mask=(input_ids == 0))
        # CLS token (position 0) for signal decoder
        return self.signal_decoder(h[:, 0, :])

    def count_params(self):
        groups = {
            'cCRE_embedding': sum(p.numel() for p in self.cCRE_embedding.parameters()),
            'rank_embedding': sum(p.numel() for p in self.rank_embedding.parameters()),
            'body': sum(p.numel() for p in self.body.parameters()),
        }
        if isinstance(self.signal_decoder, TiedBinaryDecoder):
            groups['signal_decoder'] = 0  # tied to cCRE_embedding
        else:
            groups['signal_decoder'] = sum(p.numel() for p in self.signal_decoder.parameters())
        groups['TOTAL'] = sum(groups.values())
        return groups
