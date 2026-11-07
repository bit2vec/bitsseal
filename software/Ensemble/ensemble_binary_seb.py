"""
Ensemble Binary SEB Implementations for Sentiment Analysis

This module implements two ensemble approaches:
1. Independent Ensemble: 10 completely independent Binary SEB models with different seeds
2. Multi-Head: Shared embedding with 10 different projection heads

"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class IndependentEnsembleBinarySEB(nn.Module):
    """
    Multi-Channel Binary SEB per Section 2.2 / Eq. 6 of the paper.

    Uses K independent stochastic mappings whose bit sequences are CONCATENATED
    into a single length-Kn sequence, then passed through ONE neural compressor:

        b_i = Concat(b_i^{(1)}, ..., b_i^{(K)}) in {0,1}^{Kn}
        e_i = FFN(one_hot(b_i))                 in R^{embedding_dim}

    This reduces variance introduced by a single stochastic mapping while
    keeping downstream input size (embedding_dim) unchanged.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_bits: int = 64,
        interdim: str = '128',
        relu_dropout: float = 0.1,
        num_models: int = 10,
        base_seed: int = 42,
        # Binary-SEB-only implementation knobs (defaults preserve old behaviour)
        use_gelu: bool = False,
        use_layernorm: bool = False,
        bias_init: float = 0.0,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_bits_per_channel = num_bits
        self.num_channels = num_models
        self.total_bits = num_bits * num_models  # K * n
        self.binary_vocab_size = 2
        self.use_gelu = use_gelu
        self.use_layernorm = use_layernorm

        print(f"\n=== Multi-Channel (Concat) Binary SEB — Paper Eq. 6 ===")
        print(f"Number of channels K: {num_models}")
        print(f"Bits per channel n: {num_bits}")
        print(f"Total bits Kn: {self.total_bits}")
        print(f"Embedding dim: {embedding_dim}")

        # Build K independent mappings using different seeds, then concat along bits.
        mapping_chunks = []
        for k in range(num_models):
            g = torch.Generator()
            g.manual_seed(base_seed + k)
            mapping_k = self._create_unique_mapping(num_embeddings, num_bits, g)
            mapping_chunks.append(mapping_k)

        # Final mapping shape: (num_embeddings, K*n)
        subword2binary = torch.cat(mapping_chunks, dim=1)
        self.register_buffer('subword2binary', subword2binary)

        # Shared one-hot binary embedding (frozen)
        onehot_binary = F.one_hot(torch.arange(0, self.binary_vocab_size)).to(torch.float32)
        self.binary_embedding = nn.Embedding(
            self.binary_vocab_size, self.binary_vocab_size, _weight=onehot_binary
        )
        self.binary_embedding.requires_grad_(False)

        # Single FFN compressor with input 2*K*n (one-hot over K*n bits)
        if interdim == '' or interdim == []:
            interdim_list = [embedding_dim]
        elif isinstance(interdim, int):
            interdim_list = [interdim]
        else:
            interdim_list = [int(i) for i in str(interdim).split('_')]

        input_dim = self.binary_vocab_size * self.total_bits  # 2 * K * n
        dims = [input_dim] + interdim_list + [embedding_dim]

        layers = []
        for j in range(1, len(dims)):
            linear = nn.Linear(dims[j-1], dims[j])
            if j < len(dims) - 1:
                nn.init.kaiming_normal_(linear.weight, nonlinearity='relu', mode='fan_out')
                nn.init.constant_(linear.bias, bias_init)
            else:
                nn.init.xavier_normal_(linear.weight)
                nn.init.zeros_(linear.bias)
            layers.append(linear)
            if j < len(dims) - 1:
                layers.append(nn.GELU() if use_gelu else nn.ReLU())
                if relu_dropout > 0:
                    layers.append(nn.Dropout(relu_dropout))
        self.ffn = nn.Sequential(*layers)
        self.output_norm = nn.LayerNorm(embedding_dim) if use_layernorm else None

        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def _create_unique_mapping(self, vocab_size: int, num_bits: int, generator: torch.Generator) -> torch.Tensor:
        mapping = torch.randint(0, 2, (vocab_size, num_bits), dtype=torch.long, generator=generator)
        seen = set()
        for i in range(vocab_size):
            seq = tuple(mapping[i].tolist())
            retry = 0
            while seq in seen and retry < 100:
                mapping[i] = torch.randint(0, 2, (num_bits,), dtype=torch.long, generator=generator)
                seq = tuple(mapping[i].tolist())
                retry += 1
            seen.add(seq)
        return mapping

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Eq. 6 forward: concat K channels of bits, then single FFN."""
        binary_seq = self.subword2binary[x]                  # (B, S, K*n)
        onehot = self.binary_embedding(binary_seq)           # (B, S, K*n, 2)
        flat = onehot.reshape(*x.shape, -1)                  # (B, S, 2*K*n)
        out = self.ffn(flat)                                 # (B, S, embedding_dim)
        if self.output_norm is not None:
            out = self.output_norm(out)
        return out


class MultiHeadBinarySEB(nn.Module):
    """
    Multi-Head Binary SEB with shared embedding and independent projection heads.

    Architecture:
    - Shared one-hot binary embedding (memory efficient)
    - 10 different random subword-to-binary mappings (diversity)
    - 10 independent projection heads (different features)
    - Concatenated output

    This is more parameter-efficient than independent ensemble while maintaining diversity.

    Args:
        num_embeddings: Vocabulary size
        embedding_dim: Total output dimension (split across heads)
        num_bits: Bits per subword
        interdim: FFN hidden dims per head
        relu_dropout: Dropout rate
        num_heads: Number of attention-style heads (default: 10)
        base_seed: Base seed for head diversity
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_bits: int = 64,
        interdim: str = '128',
        relu_dropout: float = 0.1,
        num_heads: int = 10,
        base_seed: int = 42,
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0, f"embedding_dim {embedding_dim} must be divisible by num_heads {num_heads}"

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_bits = num_bits
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.binary_vocab_size = 2

        print(f"\n=== Multi-Head Binary SEB ===")
        print(f"Number of heads: {num_heads}")
        print(f"Total embedding dim: {embedding_dim}")
        print(f"Dim per head: {self.head_dim}")
        print(f"Num bits: {num_bits}")

        # SHARED one-hot binary embedding (memory efficient!)
        onehot_binary = F.one_hot(torch.arange(0, self.binary_vocab_size)).to(torch.float32)
        self.binary_embedding = nn.Embedding(
            self.binary_vocab_size,
            self.binary_vocab_size,
            _weight=onehot_binary
        )
        self.binary_embedding.requires_grad_(False)

        # Different mappings for each head (diversity)
        self.mappings = nn.ModuleList()
        for i in range(num_heads):
            torch.manual_seed(base_seed + i)
            mapping = self._create_unique_mapping(num_embeddings, num_bits)
            # Wrap in a simple container to use ModuleList
            container = nn.Module()
            container.register_buffer('mapping', mapping)
            self.mappings.append(container)

        # Independent projection heads
        if interdim == '' or interdim == []:
            interdim_list = [self.head_dim]
        elif isinstance(interdim, int):
            interdim_list = [interdim]
        else:
            interdim_list = [int(i) for i in str(interdim).split('_')]

        input_dim = self.binary_vocab_size * num_bits
        dims = [input_dim] + interdim_list + [self.head_dim]

        self.projection_heads = nn.ModuleList()
        for i in range(num_heads):
            layers = []
            for j in range(1, len(dims)):
                layers.append(nn.Linear(dims[j-1], dims[j]))
                if j < len(dims) - 1:
                    layers.append(nn.ReLU())
                    if relu_dropout > 0:
                        layers.append(nn.Dropout(relu_dropout))

            self.projection_heads.append(nn.Sequential(*layers))

        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def _create_unique_mapping(self, vocab_size: int, num_bits: int) -> torch.Tensor:
        """Create unique binary mapping for each subword."""
        mapping = torch.randint(0, 2, (vocab_size, num_bits), dtype=torch.long)

        seen = set()
        for i in range(vocab_size):
            seq = tuple(mapping[i].tolist())
            retry_count = 0
            while seq in seen and retry_count < 100:
                mapping[i] = torch.randint(0, 2, (num_bits,), dtype=torch.long)
                seq = tuple(mapping[i].tolist())
                retry_count += 1
            seen.add(seq)

        return mapping

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all heads and concatenate.

        Args:
            x: Input token IDs [batch_size, seq_len]

        Returns:
            Concatenated head outputs [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len = x.shape
        head_outputs = []

        for mapping_container, projection_head in zip(self.mappings, self.projection_heads):
            # Different mapping for each head
            binary_seq = mapping_container.mapping[x]  # [batch, seq, num_bits]

            # SHARED binary embedding (efficient!)
            binary_embedded = self.binary_embedding(binary_seq)  # [batch, seq, num_bits, 2]
            binary_embedded = binary_embedded.view(batch_size, seq_len, -1)

            # Independent projection
            head_output = projection_head(binary_embedded)  # [batch, seq, head_dim]
            head_outputs.append(head_output)

        # Concatenate all heads
        return torch.cat(head_outputs, dim=-1)  # [batch, seq, embedding_dim]


# Wrapper for compatibility with existing sentiment code
class EnsembleBinaryEmbedding(nn.Module):
    """
    Wrapper to make ensemble embeddings compatible with LSTM sentiment analysis.

    Automatically handles the increased embedding dimension by adjusting LSTM input.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        ensemble_type: str = 'multihead',  # 'multihead' or 'independent'
        num_bits: int = 64,
        interdim: str = '128',
        relu_dropout: float = 0.1,
        num_models: int = 10,
        base_seed: int = 42,
        use_gelu: bool = False,
        use_layernorm: bool = False,
        bias_init: float = 0.0,
    ):
        super().__init__()

        self.ensemble_type = ensemble_type

        self.embedding_dim_per_model = embedding_dim
        self.num_models = num_models

        if ensemble_type == 'independent':
            # Eq. 6: K channels concat'd into length Kn, single FFN compresses to embedding_dim
            self.embedding = IndependentEnsembleBinarySEB(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                num_bits=num_bits,
                interdim=interdim,
                relu_dropout=relu_dropout,
                num_models=num_models,
                base_seed=base_seed,
                use_gelu=use_gelu,
                use_layernorm=use_layernorm,
                bias_init=bias_init,
            )
            self.output_dim = embedding_dim  # Mean-pooled output

        elif ensemble_type == 'multihead':
            self.embedding = MultiHeadBinarySEB(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim * num_models,
                num_bits=num_bits,
                interdim=interdim,
                relu_dropout=relu_dropout,
                num_heads=num_models,
                base_seed=base_seed,
            )
            self.output_dim = embedding_dim * num_models
        else:
            raise ValueError(f"Unknown ensemble_type: {ensemble_type}")

        print(f"\n{'='*60}")
        print(f"Ensemble Type: {ensemble_type.upper()}")
        print(f"Output dimension: {self.output_dim}")
        print(f"{'='*60}\n")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)
