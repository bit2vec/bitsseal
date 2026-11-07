"""
Binary Subword Embedding (Binary SEB)

This module implements a binary version of the Subword Embedding from Bytes (SEB) method.
Instead of using 256 bytes to represent subwords, we use binary encoding (0 and 1).

Key differences from original SEB:
- Vocabulary size: 2 (binary) instead of 256 (bytes)
- Sequence length: Longer binary sequences needed (e.g., 64-128 bits vs 8 bytes)
- Privacy: Even higher privacy due to more ambiguous gradients
- One-hot dimension: 2 instead of 256

The binary approach provides:
1. Maximum ambiguity in gradients (only 2 possible values per position)
2. Even smaller embedding vocabulary (2 vs 256)
3. Potentially stronger privacy guarantees
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Optional, Tuple


class BinarySubwordEmbedding(nn.Module):
    """
    Binary Subword Embedding: Maps subwords to binary sequences and embeds them.
    
    Instead of using byte sequences (256 possible values), this uses binary sequences
    (only 0 and 1), providing maximum gradient ambiguity for privacy protection.
    
    Args:
        num_embeddings: Size of the subword vocabulary
        embedding_dim: Dimension of the output embedding
        num_bits: Number of binary bits per subword (default: 64)
        interdim: Hidden dimensions for FFN layers (string like "512" or "512_256")
        relu_dropout: Dropout rate for FFN layers
        subword_binary_file: Path to save/load the subword-to-binary mapping
        use_onehot: If True, use one-hot encoding for binary values
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_bits: int = 64,
        interdim: str = '',
        relu_dropout: float = 0.1,
        subword_binary_file: str = '',
        use_onehot: bool = True,
    ) -> None:
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_bits = num_bits
        self.relu_dropout = relu_dropout
        self.use_onehot = use_onehot
        
        # Binary vocabulary: only 0 and 1
        self.binary_vocab_size = 2
        
        self.subword_binary_file = subword_binary_file
        
        # Create or load the subword to binary sequence mapping
        if subword_binary_file and os.path.isfile(subword_binary_file):
            subword2binary = torch.load(subword_binary_file)
            # Verify dimensions match
            if subword2binary.shape[0] != num_embeddings or subword2binary.shape[1] != num_bits:
                print(f"Warning: Loaded mapping shape {subword2binary.shape} doesn't match expected ({num_embeddings}, {num_bits})")
                subword2binary = self._create_binary_mapping(num_embeddings, num_bits)
                if subword_binary_file:
                    self._save_mapping(subword2binary, subword_binary_file)
        else:
            subword2binary = self._create_binary_mapping(num_embeddings, num_bits)
            if subword_binary_file:
                self._save_mapping(subword2binary, subword_binary_file)
        
        # Register as buffer (not a parameter, doesn't need gradients)
        self.register_buffer('subword2binary', subword2binary)
        
        # Create binary embedding layer
        if use_onehot:
            # One-hot encoding: [0] -> [1, 0], [1] -> [0, 1]
            onehot_binary = F.one_hot(torch.arange(0, self.binary_vocab_size)).to(torch.float32)
            self.binary_embedding = nn.Embedding(
                self.binary_vocab_size, 
                self.binary_vocab_size, 
                _weight=onehot_binary
            )
            self.binary_embedding.requires_grad_(False)
            input_dim = self.binary_vocab_size * self.num_bits  # 2 * num_bits
        else:
            # Learnable embeddings for binary values
            self.binary_embedding = nn.Embedding(self.binary_vocab_size, embedding_dim)
            input_dim = embedding_dim * self.num_bits
        
        self.weight = self.binary_embedding.weight
        
        # Build FFN layers
        self.linears, self.layernum = self._build_ffn(input_dim, embedding_dim, interdim)
        
        # Register subword indices for construct_matrix_for_output_layer
        subword = torch.tensor([i for i in range(self.num_embeddings)])
        self.register_buffer('subword', subword)
    
    def _create_binary_mapping(self, num_embeddings: int, num_bits: int) -> torch.Tensor:
        """
        Create a mapping from subword indices to unique binary sequences.
        
        For privacy, we use random binary sequences rather than actual binary encoding
        of the subword index, similar to how SEB uses random byte sequences.
        """
        mapping = torch.randint(0, 2, (num_embeddings, num_bits), dtype=torch.long)
        
        # Ensure uniqueness (with high probability for large num_bits)
        # For num_bits >= 64, collision probability is negligible
        seen = set()
        for i in range(num_embeddings):
            seq_tuple = tuple(mapping[i].tolist())
            while seq_tuple in seen:
                mapping[i] = torch.randint(0, 2, (num_bits,), dtype=torch.long)
                seq_tuple = tuple(mapping[i].tolist())
            seen.add(seq_tuple)
        
        return mapping
    
    def _save_mapping(self, mapping: torch.Tensor, filepath: str) -> None:
        """Save the binary mapping to a file."""
        dirname = os.path.dirname(filepath)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)
        torch.save(mapping, filepath)
    
    def _build_ffn(self, input_dim: int, output_dim: int, interdim: str) -> Tuple[nn.ModuleList, int]:
        """Build the feed-forward network for aggregation."""
        if interdim == '' or interdim == []:
            interdim_list = [output_dim]
        elif isinstance(interdim, int):
            interdim_list = [interdim]
        else:
            # Parse string like "512" or "512_256"
            interdim_list = [int(i) for i in str(interdim).split('_')]
        
        dims = [input_dim] + interdim_list + [output_dim]
        
        layers = nn.ModuleList()
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i-1], dims[i]))
        
        return layers, len(dims) - 1
    
    def construct_matrix_for_output_layer(self) -> torch.Tensor:
        """
        Construct the embedding matrix for the decoder output layer.
        Used when sharing embeddings between encoder and decoder.
        """
        return self.forward(self.subword, dropout=0)
    
    def forward(self, input: torch.Tensor, dropout: Optional[float] = None) -> torch.Tensor:
        """
        Forward pass: Convert subword indices to embeddings via binary encoding.
        
        Args:
            input: Tensor of subword indices, shape (batch_size, seq_len) or (seq_len,)
            dropout: Optional dropout rate override
        
        Returns:
            Subword embeddings, shape (*input.shape, embedding_dim)
        """
        # Look up binary sequences for input subwords
        # input: (batch, seq) or (seq,)
        # binary_input: (batch, seq, num_bits) or (seq, num_bits)
        binary_input = self.subword2binary[input]
        
        # Get binary embeddings
        # embeddings: (batch, seq, num_bits, binary_embed_dim)
        embeddings = self.binary_embedding(binary_input)
        
        # Get dimensions for reshaping
        emb_shape = embeddings.size()
        
        # Concatenate binary embeddings
        if len(emb_shape) == 4:
            # Batch input: (batch, seq, num_bits, embed) -> (batch, seq, num_bits * embed)
            batch_size, seq_len = emb_shape[0], emb_shape[1]
            embed = embeddings.reshape(batch_size, seq_len, -1)
        elif len(emb_shape) == 3:
            # Non-batch input: (seq, num_bits, embed) -> (seq, num_bits * embed)
            seq_len = emb_shape[0]
            embed = embeddings.reshape(seq_len, -1)
        else:
            raise ValueError(f"Unexpected embedding shape: {emb_shape}")
        
        # Apply FFN layers
        relu_dropout = self.relu_dropout if dropout is None else dropout
        
        for i in range(self.layernum):
            embed = self.linears[i](embed)
            if i + 1 != self.layernum:
                embed = F.relu(embed)
                embed = F.dropout(embed, p=relu_dropout, training=self.training)
        
        return embed


class BinarySubwordEmbeddingSum(nn.Module):
    """
    Alternative Binary SEB using summation aggregation instead of concatenation.
    
    This variant sums the binary embeddings (with learned position embeddings)
    instead of concatenating them, resulting in smaller FFN input dimension.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_bits: int = 64,
        interdim: str = '',
        relu_dropout: float = 0.1,
        subword_binary_file: str = '',
        use_position_embedding: bool = True,
    ) -> None:
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_bits = num_bits
        self.relu_dropout = relu_dropout
        self.binary_vocab_size = 2
        
        self.subword_binary_file = subword_binary_file
        
        # Create or load mapping
        if subword_binary_file and os.path.isfile(subword_binary_file):
            subword2binary = torch.load(subword_binary_file)
        else:
            subword2binary = self._create_binary_mapping(num_embeddings, num_bits)
            if subword_binary_file:
                dirname = os.path.dirname(subword_binary_file)
                if dirname and not os.path.isdir(dirname):
                    os.makedirs(dirname)
                torch.save(subword2binary, subword_binary_file)
        
        self.register_buffer('subword2binary', subword2binary)
        
        # Binary embeddings (learnable)
        self.binary_embedding = nn.Embedding(self.binary_vocab_size, embedding_dim)
        self.weight = self.binary_embedding.weight
        
        # Optional position embeddings for bits within subword
        self.use_position_embedding = use_position_embedding
        if use_position_embedding:
            self.position_embedding = nn.Embedding(num_bits, embedding_dim)
        
        # Build FFN
        if interdim == '' or interdim == []:
            interdim_list = [embedding_dim]
        elif isinstance(interdim, int):
            interdim_list = [interdim]
        else:
            interdim_list = [int(i) for i in str(interdim).split('_')]
        
        dims = [embedding_dim] + interdim_list + [embedding_dim]
        
        self.linears = nn.ModuleList()
        for i in range(1, len(dims)):
            self.linears.append(nn.Linear(dims[i-1], dims[i]))
        self.layernum = len(dims) - 1
        
        subword = torch.tensor([i for i in range(self.num_embeddings)])
        self.register_buffer('subword', subword)
        
        # Position indices
        positions = torch.arange(num_bits)
        self.register_buffer('positions', positions)
    
    def _create_binary_mapping(self, num_embeddings: int, num_bits: int) -> torch.Tensor:
        mapping = torch.randint(0, 2, (num_embeddings, num_bits), dtype=torch.long)
        seen = set()
        for i in range(num_embeddings):
            seq_tuple = tuple(mapping[i].tolist())
            while seq_tuple in seen:
                mapping[i] = torch.randint(0, 2, (num_bits,), dtype=torch.long)
                seq_tuple = tuple(mapping[i].tolist())
            seen.add(seq_tuple)
        return mapping
    
    def construct_matrix_for_output_layer(self) -> torch.Tensor:
        return self.forward(self.subword, dropout=0)
    
    def forward(self, input: torch.Tensor, dropout: Optional[float] = None) -> torch.Tensor:
        binary_input = self.subword2binary[input]
        embeddings = self.binary_embedding(binary_input)
        
        # Add position embeddings
        if self.use_position_embedding:
            pos_emb = self.position_embedding(self.positions)
            embeddings = embeddings + pos_emb
        
        # Sum over bit dimension
        if len(embeddings.shape) == 4:
            embed = embeddings.sum(dim=2)
        else:
            embed = embeddings.sum(dim=1)
        
        # Apply FFN
        relu_dropout = self.relu_dropout if dropout is None else dropout
        
        for i in range(self.layernum):
            embed = self.linears[i](embed)
            if i + 1 != self.layernum:
                embed = F.relu(embed)
                embed = F.dropout(embed, p=relu_dropout, training=self.training)
        
        return embed


class BinarySubwordEmbeddingAttention(nn.Module):
    """
    Binary SEB with attention-based aggregation.
    
    Uses self-attention to aggregate binary embeddings, allowing the model
    to learn which bit positions are most important.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_bits: int = 64,
        interdim: str = '',
        relu_dropout: float = 0.1,
        subword_binary_file: str = '',
    ) -> None:
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_bits = num_bits
        self.relu_dropout = relu_dropout
        self.binary_vocab_size = 2
        
        # Create or load mapping
        if subword_binary_file and os.path.isfile(subword_binary_file):
            subword2binary = torch.load(subword_binary_file)
        else:
            subword2binary = torch.randint(0, 2, (num_embeddings, num_bits), dtype=torch.long)
            if subword_binary_file:
                dirname = os.path.dirname(subword_binary_file)
                if dirname and not os.path.isdir(dirname):
                    os.makedirs(dirname)
                torch.save(subword2binary, subword_binary_file)
        
        self.register_buffer('subword2binary', subword2binary)
        
        # Binary embeddings
        self.binary_embedding = nn.Embedding(self.binary_vocab_size, embedding_dim)
        self.weight = self.binary_embedding.weight
        
        # Attention weight
        self.W_a = Parameter(torch.Tensor(embedding_dim, 1))
        nn.init.normal_(self.W_a, mean=0, std=embedding_dim ** -0.5)
        
        # Build FFN
        if interdim == '' or interdim == []:
            interdim_list = [embedding_dim]
        elif isinstance(interdim, int):
            interdim_list = [interdim]
        else:
            interdim_list = [int(i) for i in str(interdim).split('_')]
        
        dims = [embedding_dim] + interdim_list + [embedding_dim]
        
        self.linears = nn.ModuleList()
        for i in range(1, len(dims)):
            self.linears.append(nn.Linear(dims[i-1], dims[i]))
        self.layernum = len(dims) - 1
        
        subword = torch.tensor([i for i in range(self.num_embeddings)])
        self.register_buffer('subword', subword)
    
    def construct_matrix_for_output_layer(self) -> torch.Tensor:
        return self.forward(self.subword, dropout=0)
    
    def forward(self, input: torch.Tensor, dropout: Optional[float] = None) -> torch.Tensor:
        binary_input = self.subword2binary[input]
        embeddings = self.binary_embedding(binary_input)
        
        # Compute attention weights
        # embeddings: (..., num_bits, embed_dim)
        # W_a: (embed_dim, 1)
        attn_scores = torch.matmul(embeddings, self.W_a)  # (..., num_bits, 1)
        attn_weights = F.softmax(attn_scores, dim=-2)  # (..., num_bits, 1)
        
        # Weighted sum
        embed = (attn_weights * embeddings).sum(dim=-2)  # (..., embed_dim)
        
        # Apply FFN
        relu_dropout = self.relu_dropout if dropout is None else dropout
        
        for i in range(self.layernum):
            embed = self.linears[i](embed)
            if i + 1 != self.layernum:
                embed = F.relu(embed)
                embed = F.dropout(embed, p=relu_dropout, training=self.training)
        
        return embed


# Utility functions
def calculate_collision_probability(num_bits: int) -> float:
    """
    Calculate the probability that two subwords map to the same binary sequence.
    
    With num_bits binary values, probability = 1 / 2^num_bits
    """
    return 1.0 / (2 ** num_bits)


def get_recommended_num_bits(vocab_size: int, target_collision_prob: float = 1e-15) -> int:
    """
    Get recommended number of bits based on vocabulary size and target collision probability.
    
    For a vocabulary of size V, we need enough bits to make collision probability acceptably low.
    With n bits, collision probability ≈ V^2 / (2 * 2^n) (birthday problem approximation)
    """
    import math
    # Solve for n: V^2 / (2 * 2^n) < target_collision_prob
    # 2^n > V^2 / (2 * target_collision_prob)
    # n > log2(V^2 / (2 * target_collision_prob))
    n = math.ceil(math.log2(vocab_size ** 2 / (2 * target_collision_prob)))
    return max(n, 32)  # Minimum 32 bits


if __name__ == "__main__":
    # Test the Binary SEB
    print("Testing Binary Subword Embedding...")
    
    vocab_size = 10000
    embed_dim = 512
    num_bits = 64
    
    # Test main class
    binary_seb = BinarySubwordEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=embed_dim,
        num_bits=num_bits,
        interdim='512',
        relu_dropout=0.1,
        use_onehot=True,
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    output = binary_seb(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, {embed_dim})")
    
    # Print collision probability
    print(f"\nCollision probability with {num_bits} bits: {calculate_collision_probability(num_bits):.2e}")
    print(f"Recommended bits for vocab size {vocab_size}: {get_recommended_num_bits(vocab_size)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in binary_seb.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Compare with byte-based SEB (256 vocabulary)
    byte_vocab_params = 256 * embed_dim
    binary_vocab_params = 2 * 2  # One-hot: 2x2
    print(f"Byte vocabulary params: {byte_vocab_params:,}")
    print(f"Binary vocabulary params: {binary_vocab_params:,}")
