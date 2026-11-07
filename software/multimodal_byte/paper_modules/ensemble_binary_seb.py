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
    Ensemble of 10 independent Binary SEB models, each with different random seed.

    Each model has:
    - Different random subword-to-binary mapping (diversity)
    - Independent binary embedding (one-hot, frozen)
    - Independent FFN projection

    Final output: Concatenation of all 10 model outputs

    Args:
        num_embeddings: Vocabulary size
        embedding_dim: Output dimension PER MODEL (total output = embedding_dim * num_models)
        num_bits: Bits per subword per model
        interdim: FFN hidden dims (e.g., '128' or '128_64')
        relu_dropout: Dropout rate
        num_models: Number of independent models (default: 10)
        base_seed: Base seed for reproducibility (each model gets base_seed + i)
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
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim_per_model = embedding_dim  # Each model outputs this dim
        self.total_embedding_dim = embedding_dim * num_models  # Concatenated output
        self.num_bits = num_bits
        self.num_models = num_models
        self.binary_vocab_size = 2

        print(f"\n=== Independent Ensemble Binary SEB ===")
        print(f"Number of models: {num_models}")
        print(f"Embedding dim per model: {embedding_dim}")
        print(f"Total output dim: {self.total_embedding_dim}")
        print(f"Num bits per model: {num_bits}")

        # Create independent models with different seeds
        self.models = nn.ModuleList()

        for i in range(num_models):
            model_seed = base_seed + i
            print(f"Creating model {i+1}/{num_models} with seed {model_seed}")

            model = self._create_single_model(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                num_bits=num_bits,
                interdim=interdim,
                relu_dropout=relu_dropout,
                seed=model_seed,
            )
            self.models.append(model)

        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def _create_single_model(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_bits: int,
        interdim: str,
        relu_dropout: float,
        seed: int,
    ) -> nn.Module:
        """Create a single Binary SEB model with specific seed."""

        # Set seed for this model's random mapping
        torch.manual_seed(seed)

        # Create unique binary mapping with this seed
        subword2binary = self._create_unique_mapping(num_embeddings, num_bits)

        # One-hot binary embedding (frozen)
        onehot_binary = F.one_hot(torch.arange(0, self.binary_vocab_size)).to(torch.float32)
        binary_embedding = nn.Embedding(
            self.binary_vocab_size,
            self.binary_vocab_size,
            _weight=onehot_binary
        )
        binary_embedding.requires_grad_(False)

        # Parse interdim for FFN
        if interdim == '' or interdim == []:
            interdim_list = [embedding_dim]
        elif isinstance(interdim, int):
            interdim_list = [interdim]
        else:
            interdim_list = [int(i) for i in str(interdim).split('_')]

        # Build FFN
        input_dim = self.binary_vocab_size * num_bits
        dims = [input_dim] + interdim_list + [embedding_dim]

        layers = []
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i-1], dims[i]))
            if i < len(dims) - 1:
                layers.append(nn.ReLU())
                if relu_dropout > 0:
                    layers.append(nn.Dropout(relu_dropout))

        ffn = nn.Sequential(*layers)

        # Create module container
        model = nn.Module()
        model.register_buffer('subword2binary', subword2binary)
        model.binary_embedding = binary_embedding
        model.ffn = ffn

        return model

    def _create_unique_mapping(self, vocab_size: int, num_bits: int) -> torch.Tensor:
        """Create unique binary mapping for each subword."""
        mapping = torch.randint(0, 2, (vocab_size, num_bits), dtype=torch.long)

        # Ensure uniqueness (attempt)
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
        Forward pass through all models and concatenate.

        Args:
            x: Input token IDs [batch_size, seq_len]

        Returns:
            Concatenated embeddings [batch_size, seq_len, total_embedding_dim]
        """
        outputs = []

        for model in self.models:
            # Map subwords to binary sequences
            binary_seq = model.subword2binary[x]  # [batch, seq, num_bits]

            # Get one-hot embeddings
            binary_embedded = model.binary_embedding(binary_seq)  # [batch, seq, num_bits, 2]

            # Flatten to [batch, seq, num_bits * 2]
            batch_size, seq_len = x.shape
            binary_embedded = binary_embedded.view(batch_size, seq_len, -1)

            # Project through FFN
            output = model.ffn(binary_embedded)  # [batch, seq, embedding_dim]
            outputs.append(output)

        # Concatenate all model outputs
        return torch.cat(outputs, dim=-1)  # [batch, seq, total_embedding_dim]


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
    ):
        super().__init__()

        self.ensemble_type = ensemble_type

        if ensemble_type == 'independent':
            # Each model outputs embedding_dim, total is embedding_dim * num_models
            self.embedding = IndependentEnsembleBinarySEB(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                num_bits=num_bits,
                interdim=interdim,
                relu_dropout=relu_dropout,
                num_models=num_models,
                base_seed=base_seed,
            )
            self.output_dim = embedding_dim * num_models

        elif ensemble_type == 'multihead':
            # Total output is embedding_dim (split across heads)
            self.embedding = MultiHeadBinarySEB(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim * num_models,  # Match independent total
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
