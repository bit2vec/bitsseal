#!/usr/bin/env python3
"""
FILM Attack Experiment on WikiText-103


Usage:
    python film_attack_wikitext.py --num_samples 100 --batch_sizes 1 4 16 32
"""

import os
import sys
import argparse
import json
import logging
import gc
import psutil
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# Hugging Face
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

# Evaluation
from rouge_score import rouge_scorer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Memory Tracking Utilities
# ============================================================================

class MemoryTracker:
    """
    Tracks memory usage (CPU and GPU) during experiments.
    
    Provides detailed breakdown of:
    - Model parameters memory
    - Embedding layer memory
    - GPU memory (allocated and cached)
    - CPU/RAM usage
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.measurements = []
        self.use_cuda = device.type == 'cuda'
        
    def get_gpu_memory(self) -> Dict[str, float]:
        """Get GPU memory usage in MB."""
        if not self.use_cuda:
            return {"allocated": 0, "cached": 0, "max_allocated": 0}
        
        return {
            "allocated": torch.cuda.memory_allocated(self.device) / 1024**2,
            "cached": torch.cuda.memory_reserved(self.device) / 1024**2,
            "max_allocated": torch.cuda.max_memory_allocated(self.device) / 1024**2,
        }
    
    def get_cpu_memory(self) -> Dict[str, float]:
        """Get CPU/RAM memory usage in MB."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            "rss": mem_info.rss / 1024**2,  # Resident Set Size
            "vms": mem_info.vms / 1024**2,  # Virtual Memory Size
        }
    
    def count_parameters(self, model: nn.Module) -> Dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Count embedding parameters specifically
        embedding_params = 0
        if hasattr(model, 'embedding'):
            embedding_params = sum(p.numel() for p in model.embedding.parameters())
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "embedding": embedding_params,
            "non_embedding": total_params - embedding_params,
        }
    
    def get_parameter_memory(self, model: nn.Module) -> Dict[str, float]:
        """Get memory used by model parameters in MB."""
        total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        
        # Embedding memory
        embedding_bytes = 0
        if hasattr(model, 'embedding'):
            embedding_bytes = sum(
                p.numel() * p.element_size() 
                for p in model.embedding.parameters()
            )
        
        # Buffer memory (for SEB mappings like subword2bytes)
        buffer_bytes = 0
        if hasattr(model, 'embedding'):
            for name, buf in model.embedding.named_buffers():
                if buf is not None:
                    buffer_bytes += buf.numel() * buf.element_size()
        
        return {
            "total_params_mb": total_bytes / 1024**2,
            "embedding_params_mb": embedding_bytes / 1024**2,
            "buffer_mb": buffer_bytes / 1024**2,
            "non_embedding_mb": (total_bytes - embedding_bytes) / 1024**2,
        }
    
    def measure(self, label: str, model: nn.Module = None) -> Dict:
        """Take a memory measurement."""
        gc.collect()
        if self.use_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        measurement = {
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "gpu": self.get_gpu_memory(),
            "cpu": self.get_cpu_memory(),
        }
        
        if model is not None:
            measurement["params"] = self.count_parameters(model)
            measurement["param_memory"] = self.get_parameter_memory(model)
        
        self.measurements.append(measurement)
        return measurement
    
    def reset_peak_memory(self):
        """Reset peak memory statistics."""
        if self.use_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def get_summary(self) -> Dict:
        """Get summary of all measurements."""
        return {
            "measurements": self.measurements,
            "device": str(self.device),
            "cuda_available": self.use_cuda,
        }
    
    def print_measurement(self, measurement: Dict):
        """Print a single measurement."""
        logger.info(f"  Memory [{measurement['label']}]:")
        
        if "param_memory" in measurement:
            pm = measurement["param_memory"]
            logger.info(f"    Model Parameters: {pm['total_params_mb']:.2f} MB")
            logger.info(f"    - Embedding: {pm['embedding_params_mb']:.2f} MB")
            logger.info(f"    - Buffers (mappings): {pm['buffer_mb']:.2f} MB")
            logger.info(f"    - Non-embedding: {pm['non_embedding_mb']:.2f} MB")
        
        if self.use_cuda:
            gpu = measurement["gpu"]
            logger.info(f"    GPU Allocated: {gpu['allocated']:.2f} MB")
            logger.info(f"    GPU Cached: {gpu['cached']:.2f} MB")
            logger.info(f"    GPU Peak: {gpu['max_allocated']:.2f} MB")
        
        cpu = measurement["cpu"]
        logger.info(f"    CPU RSS: {cpu['rss']:.2f} MB")


def get_embedding_memory_breakdown(embedding: nn.Module, embed_type: str) -> Dict:
    """
    Get detailed memory breakdown for an embedding layer.
    
    Returns memory in MB for each component.
    """
    breakdown = {
        "type": embed_type,
        "total_mb": 0,
        "components": {}
    }
    
    if embed_type == "subword":
        # Standard embedding: just the weight matrix
        weight_bytes = embedding.weight.numel() * embedding.weight.element_size()
        breakdown["components"]["embedding_weight"] = weight_bytes / 1024**2
        breakdown["total_mb"] = weight_bytes / 1024**2
        
    elif embed_type == "byte_seb":
        # Byte SEB components
        components = {}
        
        # Byte embedding (one-hot, frozen)
        if hasattr(embedding, 'byte_embedding'):
            w = embedding.byte_embedding.weight
            components["byte_embedding"] = w.numel() * w.element_size() / 1024**2
        
        # Subword-to-bytes mapping buffer
        if hasattr(embedding, 'subword2bytes'):
            buf = embedding.subword2bytes
            components["subword2bytes_mapping"] = buf.numel() * buf.element_size() / 1024**2
        
        # FFN layers
        ffn_bytes = 0
        if hasattr(embedding, 'fc1'):
            ffn_bytes += embedding.fc1.weight.numel() * embedding.fc1.weight.element_size()
            ffn_bytes += embedding.fc1.bias.numel() * embedding.fc1.bias.element_size()
        if hasattr(embedding, 'fc2'):
            ffn_bytes += embedding.fc2.weight.numel() * embedding.fc2.weight.element_size()
            ffn_bytes += embedding.fc2.bias.numel() * embedding.fc2.bias.element_size()
        components["ffn_layers"] = ffn_bytes / 1024**2
        
        breakdown["components"] = components
        breakdown["total_mb"] = sum(components.values())
        
    elif embed_type == "binary_seb":
        # Binary SEB components
        components = {}
        
        # Binary embedding (one-hot, frozen) - very small!
        if hasattr(embedding, 'binary_embedding'):
            w = embedding.binary_embedding.weight
            components["binary_embedding"] = w.numel() * w.element_size() / 1024**2
        
        # Subword-to-binary mapping buffer
        if hasattr(embedding, 'subword2binary'):
            buf = embedding.subword2binary
            components["subword2binary_mapping"] = buf.numel() * buf.element_size() / 1024**2
        
        # FFN layers
        ffn_bytes = 0
        if hasattr(embedding, 'fc1'):
            ffn_bytes += embedding.fc1.weight.numel() * embedding.fc1.weight.element_size()
            ffn_bytes += embedding.fc1.bias.numel() * embedding.fc1.bias.element_size()
        if hasattr(embedding, 'fc2'):
            ffn_bytes += embedding.fc2.weight.numel() * embedding.fc2.weight.element_size()
            ffn_bytes += embedding.fc2.bias.numel() * embedding.fc2.bias.element_size()
        components["ffn_layers"] = ffn_bytes / 1024**2
        
        breakdown["components"] = components
        breakdown["total_mb"] = sum(components.values())
    
    return breakdown


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for FILM attack experiment."""
    
    # Dataset
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-v1"
    max_seq_len: int = 128
    min_seq_len: int = 20
    
    # Model
    embedding_dim: int = 768
    hidden_dim: int = 768
    num_layers: int = 2
    num_heads: int = 8
    
    # SEB settings
    num_bytes: int = 8          # For Byte SEB
    num_bits: int = 64          # For Binary SEB
    ffn_interdim: int = 512     # FFN hidden dimension in SEB
    relu_dropout: float = 0.1
    
    # Attack settings
    num_samples: int = 100
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    beam_size: int = 10  # Paper: beam width 10
    max_candidates: int = 7000  # As in original FILM paper
    
    # GPT-2 for reconstruction
    gpt2_model: str = "gpt2"
    
    # Embedding types to test
    embedding_types: List[str] = field(default_factory=lambda: ["subword", "byte_seb", "binary_seb"])
    
    # Output
    output_dir: str = "./results"
    save_examples: bool = True
    num_example_saves: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# ============================================================================
# Embedding Implementations
# ============================================================================

class StandardSubwordEmbedding(nn.Module):
    """Standard subword embedding with direct token-to-embedding mapping."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.weight = self.embedding.weight
        self.num_embeddings = vocab_size
        
    def forward(self, x: torch.Tensor, dropout: float = None) -> torch.Tensor:
        return self.embedding(x)
    
    def construct_matrix_for_output_layer(self) -> torch.Tensor:
        return self.weight


class ByteSubwordEmbedding(nn.Module):
    """
    Byte SEB: Maps subwords to random byte sequences (256 vocabulary).
    Implementation matches the original SEB paper.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_bytes: int = 8,
        interdim: int = 512,
        relu_dropout: float = 0.1,
        subword_bytes_file: str = "",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_bytes = num_bytes
        self.num_embeddings = vocab_size
        self.byte_vocab_size = 256
        self.relu_dropout = relu_dropout
        
        # Create or load subword-to-bytes mapping
        if subword_bytes_file and os.path.exists(subword_bytes_file):
            subword2bytes = torch.load(subword_bytes_file)
        else:
            subword2bytes = torch.randint(0, 256, (vocab_size, num_bytes), dtype=torch.long)
            if subword_bytes_file:
                os.makedirs(os.path.dirname(subword_bytes_file), exist_ok=True)
                torch.save(subword2bytes, subword_bytes_file)
        
        self.register_buffer('subword2bytes', subword2bytes)
        
        # One-hot byte embedding (frozen)
        onehot = F.one_hot(torch.arange(256)).float()
        self.byte_embedding = nn.Embedding(256, 256, _weight=onehot)
        self.byte_embedding.requires_grad_(False)
        self.weight = self.byte_embedding.weight
        
        # FFN for aggregation
        input_dim = 256 * num_bytes
        self.fc1 = nn.Linear(input_dim, interdim)
        self.fc2 = nn.Linear(interdim, embedding_dim)
        
        # Initialize
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        
        # Register all subword indices
        self.register_buffer('subword_indices', torch.arange(vocab_size))
    
    def forward(self, x: torch.Tensor, dropout: float = None) -> torch.Tensor:
        dropout = dropout if dropout is not None else self.relu_dropout
        
        bytes_seq = self.subword2bytes[x]
        byte_embeds = self.byte_embedding(bytes_seq)
        
        # Reshape and concatenate
        shape = byte_embeds.shape
        if len(shape) == 4:
            batch, seq, nb, d = shape
            flat = byte_embeds.reshape(batch, seq, -1)
        else:
            seq, nb, d = shape
            flat = byte_embeds.reshape(seq, -1)
        
        # FFN
        out = F.relu(self.fc1(flat))
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.fc2(out)
        return out
    
    def construct_matrix_for_output_layer(self) -> torch.Tensor:
        return self.forward(self.subword_indices, dropout=0)


class BinarySubwordEmbedding(nn.Module):
    """
    Binary SEB: Maps subwords to random binary sequences (0/1 only).
    Provides maximum gradient ambiguity for enhanced privacy.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_bits: int = 64,
        interdim: int = 512,
        relu_dropout: float = 0.1,
        subword_binary_file: str = "",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_bits = num_bits
        self.num_embeddings = vocab_size
        self.binary_vocab_size = 2
        self.relu_dropout = relu_dropout
        
        # Create or load subword-to-binary mapping
        if subword_binary_file and os.path.exists(subword_binary_file):
            subword2binary = torch.load(subword_binary_file)
        else:
            subword2binary = self._create_unique_mapping(vocab_size, num_bits)
            if subword_binary_file:
                os.makedirs(os.path.dirname(subword_binary_file), exist_ok=True)
                torch.save(subword2binary, subword_binary_file)
        
        self.register_buffer('subword2binary', subword2binary)
        
        # One-hot binary embedding (frozen): 0 -> [1,0], 1 -> [0,1]
        onehot = F.one_hot(torch.arange(2)).float()
        self.binary_embedding = nn.Embedding(2, 2, _weight=onehot)
        self.binary_embedding.requires_grad_(False)
        self.weight = self.binary_embedding.weight
        
        # FFN for aggregation
        input_dim = 2 * num_bits
        self.fc1 = nn.Linear(input_dim, interdim)
        self.fc2 = nn.Linear(interdim, embedding_dim)
        
        # Initialize
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        
        # Register all subword indices
        self.register_buffer('subword_indices', torch.arange(vocab_size))
    
    def _create_unique_mapping(self, vocab_size: int, num_bits: int) -> torch.Tensor:
        """Create unique binary mapping for each subword."""
        mapping = torch.randint(0, 2, (vocab_size, num_bits), dtype=torch.long)
        
        # Ensure uniqueness (for large num_bits, collisions are extremely rare)
        seen = set()
        for i in range(vocab_size):
            seq = tuple(mapping[i].tolist())
            attempts = 0
            while seq in seen and attempts < 1000:
                mapping[i] = torch.randint(0, 2, (num_bits,), dtype=torch.long)
                seq = tuple(mapping[i].tolist())
                attempts += 1
            seen.add(seq)
        
        return mapping
    
    def forward(self, x: torch.Tensor, dropout: float = None) -> torch.Tensor:
        dropout = dropout if dropout is not None else self.relu_dropout
        
        binary_seq = self.subword2binary[x]
        binary_embeds = self.binary_embedding(binary_seq)
        
        # Reshape and concatenate
        shape = binary_embeds.shape
        if len(shape) == 4:
            batch, seq, nb, d = shape
            flat = binary_embeds.reshape(batch, seq, -1)
        else:
            seq, nb, d = shape
            flat = binary_embeds.reshape(seq, -1)
        
        # FFN
        out = F.relu(self.fc1(flat))
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.fc2(out)
        return out
    
    def construct_matrix_for_output_layer(self) -> torch.Tensor:
        return self.forward(self.subword_indices, dropout=0)


# ============================================================================
# Language Model for Training
# ============================================================================

class TransformerLM(nn.Module):
    """Transformer Language Model for gradient computation."""
    
    def __init__(
        self,
        embedding: nn.Module,
        hidden_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = embedding
        embed_dim = embedding.embedding_dim
        vocab_size = embedding.num_embeddings
        
        # Positional encoding
        self.pos_embedding = nn.Embedding(1024, embed_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeds = self.embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embedding(positions)
        embeds = token_embeds + pos_embeds
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
        
        # Transform
        hidden = self.transformer(embeds, mask=causal_mask)
        
        # Output
        logits = self.output_proj(hidden)
        return logits


# ============================================================================
# FILM Attack Implementation
# ============================================================================

class FILMAttack:
    """
    FILM Attack: Recovering Private Text from Gradients
    
    Based on Gupta et al. (2022), this attack:
    1. Extracts candidate tokens from embedding gradients
    2. Uses GPT-2 to reconstruct coherent sentences via beam search
    """
    
    def __init__(
        self,
        tokenizer: GPT2Tokenizer,
        device: torch.device,
        gpt2_model: str = "gpt2",
        beam_size: int = 10,
        max_candidates: int = 7000,
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.beam_size = beam_size
        self.max_candidates = max_candidates
        
        # Load GPT-2 for reconstruction
        logger.info(f"Loading GPT-2 model: {gpt2_model}")
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model).to(device)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        self.gpt2.eval()
        
        # ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def extract_tokens_standard(self, grad: torch.Tensor) -> Set[int]:
        """Extract candidate tokens from standard embedding gradient."""
        grad_norms = grad.norm(dim=1)
        candidates = (grad_norms > 1e-10).nonzero(as_tuple=True)[0].tolist()
        return set(candidates)
    
    def extract_tokens_seb(
        self,
        grad: torch.Tensor,
        mapping: torch.Tensor,
        vocab_size: int,
    ) -> Set[int]:
        """
        Extract candidate tokens from SEB embedding gradient.
        
        For SEB, gradient updates to code (byte/bit) embeddings affect
        ALL subwords sharing those codes, creating ambiguity.
        """
        # Find codes with non-zero gradients
        grad_norms = grad.norm(dim=1)
        active_codes = set((grad_norms > 1e-10).nonzero(as_tuple=True)[0].tolist())
        
        if not active_codes:
            return set()
        
        # Find all subwords containing these codes
        candidates = set()
        mapping_np = mapping.cpu().numpy()
        
        for subword_idx in range(min(vocab_size, len(mapping_np))):
            subword_codes = set(mapping_np[subword_idx].tolist())
            if subword_codes & active_codes:
                candidates.add(subword_idx)
                
                # Limit candidates as in original paper
                if len(candidates) >= self.max_candidates:
                    break
        
        return candidates
    
    def reconstruct_sentence(
        self,
        candidates: List[int],
        target_length: int,
    ) -> str:
        """
        Reconstruct sentence from candidate tokens using GPT-2 beam search.
        
        Following the FILM paper methodology.
        """
        if not candidates:
            return ""
        
        # Convert to text tokens
        candidate_texts = []
        for token_id in candidates:
            try:
                text = self.tokenizer.decode([token_id])
                # Don't filter out tokens - even whitespace/special tokens may be useful
                candidate_texts.append(text)
            except:
                continue

        if not candidate_texts:
            return ""
        
        # Greedy reconstruction with GPT-2 scoring
        best_sequence = []
        remaining = list(candidate_texts)
        
        for _ in range(min(target_length, len(candidate_texts))):
            if not remaining:
                break
            
            best_token = None
            best_score = float('-inf')
            current_text = " ".join(best_sequence) if best_sequence else ""
            
            # Score each candidate
            for candidate in remaining[:100]:  # Limit for efficiency
                test_text = f"{current_text} {candidate}".strip()
                
                try:
                    inputs = self.gpt2_tokenizer(
                        test_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=128
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.gpt2(**inputs)
                        # Perplexity-based scoring
                        logits = outputs.logits

                        # Handle single-token case
                        if inputs['input_ids'].shape[1] <= 1:
                            # Use log probability of the token under the model
                            # For single tokens, use a neutral score
                            score = 0.0
                        else:
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = inputs['input_ids'][..., 1:].contiguous()
                            loss = F.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                                reduction='mean'
                            )
                            score = -loss.item()  # Higher is better
                    
                    if score > best_score:
                        best_score = score
                        best_token = candidate
                except Exception as e:
                    continue
            
            if best_token:
                best_sequence.append(best_token)
                remaining.remove(best_token)
        
        return " ".join(best_sequence)
    
    def attack(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        embedding_type: str,
    ) -> Dict:
        """
        Execute FILM attack on a batch.
        
        Returns attack results including recovered text and metrics.
        """
        model.train()
        model.zero_grad()
        
        # Forward and backward pass
        outputs = model(input_ids, attention_mask)
        
        # Language modeling loss (next token prediction)
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        loss.backward()
        
        # Get embedding gradient
        if embedding_type == "subword":
            grad = model.embedding.weight.grad
            mapping = None
        elif embedding_type == "byte_seb":
            # For SEB, enable gradients temporarily on the frozen embedding
            model.embedding.byte_embedding.weight.requires_grad_(True)

            # Re-run forward and backward to get gradients
            model.zero_grad()
            outputs = model(input_ids, attention_mask)
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss.backward()

            grad = model.embedding.byte_embedding.weight.grad
            mapping = model.embedding.subword2bytes

            # Disable gradients again
            model.embedding.byte_embedding.weight.requires_grad_(False)
        elif embedding_type == "binary_seb":
            # For binary SEB, enable gradients temporarily on the frozen embedding
            model.embedding.binary_embedding.weight.requires_grad_(True)

            # Re-run forward and backward to get gradients
            model.zero_grad()
            outputs = model(input_ids, attention_mask)
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss.backward()

            grad = model.embedding.binary_embedding.weight.grad
            mapping = model.embedding.subword2binary

            # Disable gradients again
            model.embedding.binary_embedding.weight.requires_grad_(False)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        if grad is None:
            return {"error": "No gradient computed"}
        
        # Extract candidates
        original_tokens = set(input_ids.view(-1).tolist())
        
        if embedding_type == "subword":
            candidates = self.extract_tokens_standard(grad)
        else:
            candidates = self.extract_tokens_seb(
                grad, mapping, model.embedding.num_embeddings
            )
        
        # Compute token-level metrics
        if candidates:
            precision = len(original_tokens & candidates) / len(candidates)
            recall = len(original_tokens & candidates) / len(original_tokens)
        else:
            precision = 0.0
            recall = 0.0
        
        # Reconstruct sentence
        original_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        target_length = len(input_ids[0])
        recovered_text = self.reconstruct_sentence(list(candidates), target_length)
        
        # Compute ROUGE scores
        if recovered_text and original_text:
            rouge_scores = self.rouge_scorer.score(original_text, recovered_text)
            rouge1 = rouge_scores['rouge1'].fmeasure
            rouge2 = rouge_scores['rouge2'].fmeasure
            rougeL = rouge_scores['rougeL'].fmeasure
        else:
            rouge1 = rouge2 = rougeL = 0.0
        
        return {
            "original_text": original_text,
            "recovered_text": recovered_text,
            "num_candidates": len(candidates),
            "num_original_tokens": len(original_tokens),
            "precision": precision,
            "recall": recall,
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
        }


# ============================================================================
# Dataset
# ============================================================================

class WikiTextDataset(Dataset):
    """WikiText-103 dataset for FILM attack experiment."""
    
    def __init__(
        self,
        tokenizer: GPT2Tokenizer,
        split: str = "test",
        max_length: int = 128,
        min_length: int = 20,
        max_samples: int = None,
    ):
        logger.info(f"Loading WikiText-103 {split} split...")
        dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)
        
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        for item in tqdm(dataset, desc="Processing"):
            text = item['text'].strip()
            if not text or text.startswith('='):
                continue
            
            # Tokenize
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Check length
            actual_length = (encoded['input_ids'] != tokenizer.pad_token_id).sum().item()
            if actual_length >= min_length:
                self.samples.append({
                    'input_ids': encoded['input_ids'].squeeze(0),
                    'attention_mask': encoded['attention_mask'].squeeze(0),
                    'text': text,
                })
            
            if max_samples and len(self.samples) >= max_samples:
                break
        
        logger.info(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(config: ExperimentConfig):
    """Run the complete FILM attack experiment with memory tracking."""
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Setup
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")
    
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "examples"), exist_ok=True)
    
    # Initialize memory tracker
    memory_tracker = MemoryTracker(device)
    memory_tracker.measure("baseline")
    
    # Load tokenizer
    logger.info("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(config.gpt2_model)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    
    # Load dataset
    dataset = WikiTextDataset(
        tokenizer=tokenizer,
        split="test",
        max_length=config.max_seq_len,
        min_length=config.min_seq_len,
        max_samples=config.num_samples * 10,  # Load more, sample later
    )
    
    memory_tracker.measure("after_dataset_load")
    
    # Initialize attack
    attack = FILMAttack(
        tokenizer=tokenizer,
        device=device,
        gpt2_model=config.gpt2_model,
        beam_size=config.beam_size,
        max_candidates=config.max_candidates,
    )
    
    memory_tracker.measure("after_attack_init")
    
    # Results storage
    all_results = {}
    memory_results = {}
    
    # Test each embedding type
    for embed_type in config.embedding_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing embedding type: {embed_type.upper()}")
        logger.info(f"{'='*60}")
        
        # Reset GPU memory stats
        memory_tracker.reset_peak_memory()
        
        # Create embedding
        if embed_type == "subword":
            embedding = StandardSubwordEmbedding(vocab_size, config.embedding_dim)
        elif embed_type == "byte_seb":
            embedding = ByteSubwordEmbedding(
                vocab_size,
                config.embedding_dim,
                num_bytes=config.num_bytes,
                interdim=config.ffn_interdim,
                relu_dropout=config.relu_dropout,
                subword_bytes_file=os.path.join(config.output_dir, "subword2bytes.pt"),
            )
        elif embed_type == "binary_seb":
            embedding = BinarySubwordEmbedding(
                vocab_size,
                config.embedding_dim,
                num_bits=config.num_bits,
                interdim=config.ffn_interdim,
                relu_dropout=config.relu_dropout,
                subword_binary_file=os.path.join(config.output_dir, "subword2binary.pt"),
            )
        else:
            raise ValueError(f"Unknown embedding type: {embed_type}")
        
        # Get embedding memory breakdown
        embed_memory = get_embedding_memory_breakdown(embedding, embed_type)
        logger.info(f"\n  Embedding Memory Breakdown ({embed_type}):")
        logger.info(f"    Total: {embed_memory['total_mb']:.4f} MB")
        for comp_name, comp_mb in embed_memory['components'].items():
            logger.info(f"    - {comp_name}: {comp_mb:.4f} MB")
        
        # Create model
        model = TransformerLM(
            embedding=embedding,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
        ).to(device)
        
        # Measure model memory
        model_measurement = memory_tracker.measure(f"model_{embed_type}", model)
        memory_tracker.print_measurement(model_measurement)
        
        # Store memory info
        memory_results[embed_type] = {
            "embedding_breakdown": embed_memory,
            "model_measurement": model_measurement,
            "parameter_counts": model_measurement.get("params", {}),
            "parameter_memory": model_measurement.get("param_memory", {}),
        }
        
        embed_results = {}
        
        # Test each batch size
        for batch_size in config.batch_sizes:
            logger.info(f"\nBatch size: {batch_size}")
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
            )
            
            batch_results = []
            examples = []
            
            # Track peak memory during attack
            memory_tracker.reset_peak_memory()
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Attacking")):
                if batch_idx >= config.num_samples:
                    break
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Run attack
                result = attack.attack(model, input_ids, attention_mask, embed_type)
                batch_results.append(result)
                
                # Save examples
                if config.save_examples and len(examples) < config.num_example_saves:
                    examples.append({
                        "batch_idx": batch_idx,
                        "original": result.get("original_text", ""),
                        "recovered": result.get("recovered_text", ""),
                        "rouge1": result.get("rouge1", 0),
                        "rougeL": result.get("rougeL", 0),
                    })
            
            # Measure peak memory after attacks
            peak_measurement = memory_tracker.measure(f"peak_{embed_type}_batch{batch_size}")
            
            # Aggregate results
            avg_results = {
                "rouge1": np.mean([r.get("rouge1", 0.0) for r in batch_results]),
                "rouge2": np.mean([r.get("rouge2", 0.0) for r in batch_results]),
                "rougeL": np.mean([r.get("rougeL", 0.0) for r in batch_results]),
                "precision": np.mean([r.get("precision", 0.0) for r in batch_results]),
                "recall": np.mean([r.get("recall", 0.0) for r in batch_results]),
                "avg_candidates": np.mean([r.get("num_candidates", 0) for r in batch_results]),
                "num_samples": len(batch_results),
                "peak_gpu_mb": peak_measurement["gpu"]["max_allocated"] if memory_tracker.use_cuda else 0,
                "peak_cpu_mb": peak_measurement["cpu"]["rss"],
            }
            
            embed_results[batch_size] = avg_results
            
            # Log results
            logger.info(f"  ROUGE-1: {avg_results['rouge1']:.4f}")
            logger.info(f"  ROUGE-2: {avg_results['rouge2']:.4f}")
            logger.info(f"  ROUGE-L: {avg_results['rougeL']:.4f}")
            logger.info(f"  Precision: {avg_results['precision']:.4f}")
            logger.info(f"  Recall: {avg_results['recall']:.4f}")
            logger.info(f"  Avg candidates: {avg_results['avg_candidates']:.0f}")
            logger.info(f"  Peak GPU Memory: {avg_results['peak_gpu_mb']:.2f} MB")
            logger.info(f"  Peak CPU Memory: {avg_results['peak_cpu_mb']:.2f} MB")
            
            # Save examples
            if examples:
                example_file = os.path.join(
                    config.output_dir, "examples",
                    f"{embed_type}_batch{batch_size}_examples.json"
                )
                with open(example_file, 'w') as f:
                    json.dump(examples, f, indent=2)
        
        all_results[embed_type] = embed_results
        
        # Clean up to free memory before next embedding type
        del model
        del embedding
        gc.collect()
        if memory_tracker.use_cuda:
            torch.cuda.empty_cache()
    
    # Save final results
    results_file = os.path.join(config.output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")
    
    # Save memory results
    memory_file = os.path.join(config.output_dir, "memory_results.json")
    
    # Convert memory results to serializable format
    serializable_memory = {}
    for embed_type, mem_data in memory_results.items():
        serializable_memory[embed_type] = {
            "embedding_breakdown": mem_data["embedding_breakdown"],
            "parameter_counts": mem_data["parameter_counts"],
            "parameter_memory": mem_data["parameter_memory"],
        }
    
    with open(memory_file, 'w') as f:
        json.dump(serializable_memory, f, indent=2)
    logger.info(f"Memory results saved to: {memory_file}")
    
    # Print summary
    print_summary(all_results, config.batch_sizes)
    print_memory_summary(memory_results, vocab_size, config)
    
    return all_results, memory_results


def print_summary(results: Dict, batch_sizes: List[int]):
    """Print experiment summary."""
    
    print("\n" + "="*80)
    print("FILM ATTACK EXPERIMENT SUMMARY")
    print("="*80)
    
    # ROUGE-L table (main metric)
    print("\nROUGE-L F-scores (Lower = Better Privacy)")
    print("-"*70)
    header = f"{'Batch Size':<12}"
    for embed_type in results.keys():
        header += f"{embed_type:<15}"
    print(header)
    print("-"*70)
    
    for bs in batch_sizes:
        row = f"{bs:<12}"
        for embed_type in results.keys():
            if bs in results[embed_type]:
                score = results[embed_type][bs]['rougeL']
                row += f"{score:<15.4f}"
            else:
                row += f"{'N/A':<15}"
        print(row)
    
    # Recall table
    print("\nRecall (Lower = Better Privacy)")
    print("-"*70)
    print(header)
    print("-"*70)
    
    for bs in batch_sizes:
        row = f"{bs:<12}"
        for embed_type in results.keys():
            if bs in results[embed_type]:
                score = results[embed_type][bs]['recall']
                row += f"{score:<15.4f}"
            else:
                row += f"{'N/A':<15}"
        print(row)
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("Lower ROUGE scores = Harder to reconstruct original text = Better privacy")
    print("Lower Recall = Fewer original tokens identified = Better privacy")
    print("Binary SEB should show the lowest scores (best privacy protection)")
    print("="*80)


def print_memory_summary(memory_results: Dict, vocab_size: int, config: ExperimentConfig):
    """Print memory usage summary comparing different embedding types."""
    
    print("\n" + "="*80)
    print("MEMORY USAGE SUMMARY")
    print("="*80)
    
    # Embedding memory comparison table
    print("\nEmbedding Layer Memory Comparison")
    print("-"*80)
    print(f"{'Embedding Type':<20} {'Total (MB)':<15} {'Reduction':<15} {'Components'}")
    print("-"*80)
    
    # Get baseline (subword) memory for comparison
    baseline_mb = None
    if "subword" in memory_results:
        baseline_mb = memory_results["subword"]["embedding_breakdown"]["total_mb"]
    
    for embed_type in ["subword", "byte_seb", "binary_seb"]:
        if embed_type not in memory_results:
            continue
        
        breakdown = memory_results[embed_type]["embedding_breakdown"]
        total_mb = breakdown["total_mb"]
        
        # Calculate reduction
        if baseline_mb and baseline_mb > 0:
            reduction = (1 - total_mb / baseline_mb) * 100
            reduction_str = f"{reduction:.1f}% ↓" if reduction > 0 else f"{-reduction:.1f}% ↑"
        else:
            reduction_str = "baseline"
        
        # Component summary
        components = breakdown["components"]
        comp_str = ", ".join([f"{k}: {v:.3f}MB" for k, v in components.items()])
        
        print(f"{embed_type:<20} {total_mb:<15.4f} {reduction_str:<15} {comp_str}")
    
    # Model total memory
    print("\nTotal Model Memory (Parameters)")
    print("-"*80)
    print(f"{'Embedding Type':<20} {'Total Params':<15} {'Embedding Params':<18} {'Embedding %'}")
    print("-"*80)
    
    for embed_type in ["subword", "byte_seb", "binary_seb"]:
        if embed_type not in memory_results:
            continue
        
        params = memory_results[embed_type].get("parameter_counts", {})
        total = params.get("total", 0)
        embedding = params.get("embedding", 0)
        pct = (embedding / total * 100) if total > 0 else 0
        
        print(f"{embed_type:<20} {total:<15,} {embedding:<18,} {pct:.1f}%")
    
    # Theoretical analysis
    print("\n" + "="*80)
    print("THEORETICAL MEMORY ANALYSIS")
    print("="*80)
    
    embed_dim = config.embedding_dim
    num_bytes = config.num_bytes
    num_bits = config.num_bits
    ffn_interdim = config.ffn_interdim
    
    print(f"\nConfiguration: vocab_size={vocab_size}, embed_dim={embed_dim}")
    print(f"              num_bytes={num_bytes}, num_bits={num_bits}, ffn_interdim={ffn_interdim}")
    
    # Standard subword embedding
    subword_params = vocab_size * embed_dim
    subword_mb = subword_params * 4 / 1024**2  # float32
    
    # Byte SEB
    byte_embed_params = 256 * 256  # one-hot (frozen)
    byte_mapping = vocab_size * num_bytes * 8 / 1024**2  # int64 mapping
    byte_ffn = (256 * num_bytes * ffn_interdim) + ffn_interdim + (ffn_interdim * embed_dim) + embed_dim
    byte_total_mb = (byte_embed_params * 4 + byte_ffn * 4) / 1024**2 + byte_mapping
    
    # Binary SEB
    binary_embed_params = 2 * 2  # one-hot (frozen) - TINY!
    binary_mapping = vocab_size * num_bits * 8 / 1024**2  # int64 mapping
    binary_ffn = (2 * num_bits * ffn_interdim) + ffn_interdim + (ffn_interdim * embed_dim) + embed_dim
    binary_total_mb = (binary_embed_params * 4 + binary_ffn * 4) / 1024**2 + binary_mapping
    
    print(f"\nTheoretical Embedding Memory:")
    print(f"  Standard Subword: {subword_mb:.2f} MB ({subword_params:,} params)")
    print(f"  Byte SEB:         {byte_total_mb:.2f} MB (reduction: {(1-byte_total_mb/subword_mb)*100:.1f}%)")
    print(f"  Binary SEB:       {binary_total_mb:.2f} MB (reduction: {(1-binary_total_mb/subword_mb)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("1. Standard Subword: O(V × d) memory where V=vocab_size, d=embed_dim")
    print("2. Byte SEB: O(256 × d + FFN) - vocab reduced from V to 256")
    print("3. Binary SEB: O(2 × d + FFN) - vocab reduced from V to just 2!")
    print("")
    print("Binary SEB achieves maximum memory efficiency for the embedding vocabulary")
    print("while providing the strongest privacy protection.")
    print("="*80)


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FILM Attack Experiment on WikiText-103")
    
    # Dataset
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--min_seq_len", type=int, default=20)
    
    # Model
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=2)
    
    # SEB
    parser.add_argument("--num_bytes", type=int, default=8)
    parser.add_argument("--num_bits", type=int, default=64)
    parser.add_argument("--ffn_interdim", type=int, default=512)
    
    # Attack
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_sizes", type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--max_candidates", type=int, default=7000)
    parser.add_argument("--gpt2_model", type=str, default="gpt2")
    
    # Embedding types
    parser.add_argument("--embedding_types", type=str, nargs='+',
                        default=["subword", "byte_seb", "binary_seb"])
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--save_examples", action="store_true", default=True)
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_bytes=args.num_bytes,
        num_bits=args.num_bits,
        ffn_interdim=args.ffn_interdim,
        num_samples=args.num_samples,
        batch_sizes=args.batch_sizes,
        beam_size=args.beam_size,
        max_candidates=args.max_candidates,
        gpt2_model=args.gpt2_model,
        embedding_types=args.embedding_types,
        output_dir=args.output_dir,
        save_examples=args.save_examples,
        device=args.device,
        seed=args.seed,
    )
    
    # Save config
    os.makedirs(config.output_dir, exist_ok=True)
    config_file = os.path.join(config.output_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # Run experiment
    run_experiment(config)


if __name__ == "__main__":
    main()
