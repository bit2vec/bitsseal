"""
Binary Subword Embedding for Sentiment Analysis

This module provides Binary SEB implementations compatible with the LSTM sentiment
analysis code structure used in the original SEB paper experiments.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class BinarySubwordConcateOnehot(nn.Module):
    """
    Binary SEB with one-hot encoding and concatenation aggregation.
    
    This is the binary equivalent of ByteSubwordConcateOnehot.
    Instead of 256 bytes, we use 2 binary values (0 and 1).
    
    Args:
        num_embeddings: Vocabulary size (number of subwords)
        embedding_dim: Output embedding dimension
        num_bits: Number of bits per subword (default: 64)
        interdim: Hidden dimensions for FFN (string like '128' or '128_64')
        relu_dropout: Dropout rate
        subword_binary_file: Path to save/load the subword-to-binary mapping
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_bits: int = 64,
        interdim: str = '',
        relu_dropout: float = 0.1,
        aggre: str = 'avg',  # kept for API compatibility
        subword_binary_file: str = '',
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_bits = num_bits
        self.relu_dropout = relu_dropout
        
        # Binary vocabulary: only 0 and 1
        self.binary_vocab_size = 2
        
        self.subword_binary_file = subword_binary_file
        
        # Create or load subword-to-binary mapping
        if subword_binary_file and os.path.isfile(subword_binary_file):
            subword2binary = torch.load(subword_binary_file)
            # Check if dimensions match
            if subword2binary.shape[0] < num_embeddings:
                # Extend mapping for larger vocabulary
                extra = torch.randint(0, 2, (num_embeddings - subword2binary.shape[0], num_bits), dtype=torch.long)
                subword2binary = torch.cat([subword2binary, extra], dim=0)
            elif subword2binary.shape[0] > num_embeddings:
                subword2binary = subword2binary[:num_embeddings]
        else:
            subword2binary = self._create_unique_mapping(num_embeddings, num_bits)
            if subword_binary_file:
                dirname = os.path.dirname(subword_binary_file)
                if dirname and not os.path.isdir(dirname):
                    os.makedirs(dirname)
                torch.save(subword2binary, subword_binary_file)
        
        self.register_buffer('subword2binary', subword2binary)
        
        # One-hot binary embedding (frozen): 0 -> [1,0], 1 -> [0,1]
        onehot_binary = F.one_hot(torch.arange(0, self.binary_vocab_size)).to(torch.float32)
        self.binary_embedding = nn.Embedding(
            self.binary_vocab_size,
            self.binary_vocab_size,
            _weight=onehot_binary
        )
        self.binary_embedding.requires_grad_(False)
        
        self.weight = self.binary_embedding.weight
        
        # Parse interdim
        if interdim == '' or interdim == []:
            interdim_list = [embedding_dim]
        elif isinstance(interdim, int):
            interdim_list = [interdim]
        else:
            interdim_list = [int(i) for i in str(interdim).split('_')]
        
        # Build FFN: input is (2 * num_bits) due to one-hot encoding
        input_dim = self.binary_vocab_size * self.num_bits  # 2 * num_bits
        dims = [input_dim] + interdim_list + [embedding_dim]
        
        self.layernum = len(dims) - 1
        
        layers = []
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i-1], dims[i]))
        
        self.linears = nn.ModuleList(layers)
        
        # Register subword indices for output layer construction
        subword = torch.tensor([i for i in range(self.num_embeddings)])
        self.register_buffer('subword', subword)
    
    def _create_unique_mapping(self, num_embeddings: int, num_bits: int) -> torch.Tensor:
        """Create unique random binary mapping for each subword."""
        mapping = torch.randint(0, 2, (num_embeddings, num_bits), dtype=torch.long)
        
        # Ensure uniqueness (with high probability for large num_bits)
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
    
    def construct_matrix_for_output_layer(self) -> torch.Tensor:
        """Construct embedding matrix for decoder output layer."""
        return self.forward(self.subword, dropout=0)
    
    def forward(self, input: torch.Tensor, dropout: float = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input: Token IDs, shape (batch, seq) or (seq,)
            dropout: Optional dropout override
        
        Returns:
            Embeddings, shape (*input.shape, embedding_dim)
        """
        # Look up binary sequences
        binary_input = self.subword2binary[input]
        
        # Get one-hot embeddings
        embeddings = self.binary_embedding(binary_input)
        
        relu_dropout = self.relu_dropout if dropout is None else dropout
        
        # Reshape: concatenate one-hot vectors
        emb_shape = embeddings.size()
        
        if len(emb_shape) == 4:
            # Batch input: (batch, seq, num_bits, 2) -> (batch, seq, num_bits * 2)
            batch_size, seq_len = emb_shape[0], emb_shape[1]
            embed = embeddings.reshape(batch_size, seq_len, -1)
        elif len(emb_shape) == 3:
            # Non-batch: (seq, num_bits, 2) -> (seq, num_bits * 2)
            seq_len = emb_shape[0]
            embed = embeddings.reshape(seq_len, -1)
        else:
            raise ValueError(f"Unexpected embedding shape: {emb_shape}")
        
        # Apply FFN
        for i in range(self.layernum):
            embed = self.linears[i](embed)
            if i + 1 != self.layernum:
                embed = F.relu(embed)
                embed = F.dropout(embed, p=relu_dropout, training=self.training)
        
        return embed


class BinarySubwordSum(nn.Module):
    """
    Binary SEB with summation aggregation (like ByteSubword).
    
    Instead of concatenating, this sums the binary embeddings.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_bits: int = 64,
        interdim: str = '',
        relu_dropout: float = 0.1,
        aggre: str = 'avg',
        subword_binary_file: str = '',
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_bits = num_bits
        self.relu_dropout = relu_dropout
        self.binary_vocab_size = 2
        
        # Create or load mapping
        if subword_binary_file and os.path.isfile(subword_binary_file):
            subword2binary = torch.load(subword_binary_file)
            if subword2binary.shape[0] < num_embeddings:
                extra = torch.randint(0, 2, (num_embeddings - subword2binary.shape[0], num_bits), dtype=torch.long)
                subword2binary = torch.cat([subword2binary, extra], dim=0)
            elif subword2binary.shape[0] > num_embeddings:
                subword2binary = subword2binary[:num_embeddings]
        else:
            subword2binary = torch.randint(0, 2, (num_embeddings, num_bits), dtype=torch.long)
            if subword_binary_file:
                dirname = os.path.dirname(subword_binary_file)
                if dirname and not os.path.isdir(dirname):
                    os.makedirs(dirname)
                torch.save(subword2binary, subword_binary_file)
        
        self.register_buffer('subword2binary', subword2binary)
        
        # Learnable binary embeddings
        self.binary_embedding = nn.Embedding(self.binary_vocab_size + 1, embedding_dim, padding_idx=self.binary_vocab_size)
        self.weight = self.binary_embedding.weight
        
        # Parse interdim
        if interdim == '' or interdim == []:
            interdim_list = [embedding_dim]
        elif isinstance(interdim, int):
            interdim_list = [interdim]
        else:
            interdim_list = [int(i) for i in str(interdim).split('_')]
        
        # FFN after summation
        dims = [embedding_dim] + interdim_list + [embedding_dim]
        self.layernum = len(dims) - 1
        
        layers = []
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i-1], dims[i]))
        
        self.linears = nn.ModuleList(layers)
        
        subword = torch.tensor([i for i in range(self.num_embeddings)])
        self.register_buffer('subword', subword)
    
    def construct_matrix_for_output_layer(self) -> torch.Tensor:
        return self.forward(self.subword, dropout=0)
    
    def forward(self, input: torch.Tensor, dropout: float = None) -> torch.Tensor:
        binary_input = self.subword2binary[input]
        embeddings = self.binary_embedding(binary_input)
        
        # Sum over bits dimension
        if len(embeddings.shape) == 4:
            embed = embeddings.sum(dim=2)
        else:
            embed = embeddings.sum(dim=1)
        
        relu_dropout = self.relu_dropout if dropout is None else dropout
        
        # Apply FFN
        for i in range(self.layernum):
            embed = self.linears[i](embed)
            if i + 1 != self.layernum:
                embed = F.gelu(embed)
                embed = F.dropout(embed, p=relu_dropout, training=self.training)
        
        return embed


class BinarySubwordAttention(nn.Module):
    """
    Binary SEB with attention-based aggregation.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_bits: int = 64,
        interdim: str = '',
        relu_dropout: float = 0.1,
        aggre: str = 'avg',
        subword_binary_file: str = '',
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_bits = num_bits
        self.relu_dropout = relu_dropout
        self.binary_vocab_size = 2
        
        # Create or load mapping
        if subword_binary_file and os.path.isfile(subword_binary_file):
            subword2binary = torch.load(subword_binary_file)
            if subword2binary.shape[0] < num_embeddings:
                extra = torch.randint(0, 2, (num_embeddings - subword2binary.shape[0], num_bits), dtype=torch.long)
                subword2binary = torch.cat([subword2binary, extra], dim=0)
        else:
            subword2binary = torch.randint(0, 2, (num_embeddings, num_bits), dtype=torch.long)
            if subword_binary_file:
                dirname = os.path.dirname(subword_binary_file)
                if dirname and not os.path.isdir(dirname):
                    os.makedirs(dirname)
                torch.save(subword2binary, subword_binary_file)
        
        self.register_buffer('subword2binary', subword2binary)
        
        # Learnable binary embeddings
        self.binary_embedding = nn.Embedding(self.binary_vocab_size + 1, embedding_dim, padding_idx=self.binary_vocab_size)
        self.weight = self.binary_embedding.weight
        
        # Attention weight
        self.W_a = Parameter(torch.Tensor(embedding_dim, 1))
        nn.init.normal_(self.W_a, mean=0, std=embedding_dim ** -0.5)
        
        # Parse interdim
        if interdim == '' or interdim == []:
            interdim_list = [embedding_dim]
        elif isinstance(interdim, int):
            interdim_list = [interdim]
        else:
            interdim_list = [int(i) for i in str(interdim).split('_')]
        
        dims = [embedding_dim] + interdim_list + [embedding_dim]
        self.layernum = len(dims) - 1
        
        layers = []
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i-1], dims[i]))
        
        self.linears = nn.ModuleList(layers)
        
        subword = torch.tensor([i for i in range(self.num_embeddings)])
        self.register_buffer('subword', subword)
    
    def construct_matrix_for_output_layer(self) -> torch.Tensor:
        return self.forward(self.subword, dropout=0)
    
    def forward(self, input: torch.Tensor, dropout: float = None) -> torch.Tensor:
        binary_input = self.subword2binary[input]
        embeddings = self.binary_embedding(binary_input)
        
        relu_dropout = self.relu_dropout if dropout is None else dropout
        
        # Attention
        attn_scores = torch.matmul(embeddings, self.W_a)
        attn_weights = F.softmax(attn_scores, dim=-2)
        embed = (attn_weights * embeddings).sum(dim=-2)
        
        # Apply FFN
        for i in range(self.layernum):
            embed = self.linears[i](embed)
            if i + 1 != self.layernum:
                embed = F.relu(embed)
                embed = F.dropout(embed, p=relu_dropout, training=self.training)
        
        return embed


# Convenience function matching the ByteWordEmbedding pattern
def BinaryWordEmbedding(
    num_embeddings: int,
    embedding_dim: int,
    num_bits: int = 64,
    interdim: str = '128',
    relu_dropout: float = 0.1,
    aggre: str = 'avg',
    subword_binary_file: str = '',
    variant: str = 'onehot',  # 'onehot', 'sum', 'attention'
):
    """
    Factory function to create Binary SEB embedding.
    
    Args:
        variant: Which aggregation method to use
            - 'onehot': Concatenation with one-hot encoding (recommended)
            - 'sum': Summation aggregation
            - 'attention': Attention-based aggregation
    """
    if variant == 'onehot':
        m = BinarySubwordConcateOnehot(
            num_embeddings, embedding_dim, num_bits,
            interdim, relu_dropout, aggre, subword_binary_file
        )
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        for layer in m.linears:
            nn.init.xavier_uniform_(layer.weight)
    
    elif variant == 'sum':
        m = BinarySubwordSum(
            num_embeddings, embedding_dim, num_bits,
            interdim, relu_dropout, aggre, subword_binary_file
        )
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        for layer in m.linears:
            nn.init.xavier_uniform_(layer.weight)
    
    elif variant == 'attention':
        m = BinarySubwordAttention(
            num_embeddings, embedding_dim, num_bits,
            interdim, relu_dropout, aggre, subword_binary_file
        )
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        for layer in m.linears:
            nn.init.xavier_uniform_(layer.weight)
    
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    return m


if __name__ == "__main__":
    # Test the module
    print("Testing BinarySubwordConcateOnehot...")
    
    vocab_size = 5000
    embed_dim = 256
    num_bits = 64
    
    model = BinarySubwordConcateOnehot(
        num_embeddings=vocab_size,
        embedding_dim=embed_dim,
        num_bits=num_bits,
        interdim='128',
        relu_dropout=0.1,
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    output = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {embed_dim})")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
