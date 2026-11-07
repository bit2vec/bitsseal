#!/usr/bin/env python3
"""

Metrics:
- Perplexity (PPL) - lower is better
- Memory usage
- Training time

Usage:
    python lm_wikitext_comparison.py --run_all
    python lm_wikitext_comparison.py --embedding_type binary_seb --num_bits 64
"""

import functools
import sys
import os
import gc
import time
import json
import math
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import tqdm
import psutil

# Hugging Face
from datasets import load_dataset
from transformers import GPT2Tokenizer

# Import Binary SEB
from binary_subword_sentiment import (
    BinarySubwordConcateOnehot,
    BinaryWordEmbedding,
)

# Try to import Byte SEB
try:
    from byte_subword_concate_onehot import ByteSubwordConcateOnehot
    BYTE_SEB_AVAILABLE = True
except ImportError:
    BYTE_SEB_AVAILABLE = False
    print("Warning: ByteSubwordConcateOnehot not found. Byte SEB comparison will be skipped.")


# ============================================================================
# Memory Tracking
# ============================================================================

class MemoryTracker:
    """Track CPU and GPU memory usage."""

    def __init__(self, device):
        self.device = device
        self.use_cuda = device.type == 'cuda'
        self.measurements = []

    def get_memory(self):
        """Get current memory usage."""
        cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2

        if self.use_cuda:
            torch.cuda.synchronize()
            gpu_allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            gpu_cached = torch.cuda.memory_reserved(self.device) / 1024**2
            gpu_peak = torch.cuda.max_memory_allocated(self.device) / 1024**2
        else:
            gpu_allocated = gpu_cached = gpu_peak = 0

        return {
            'cpu_mb': cpu_mem,
            'gpu_allocated_mb': gpu_allocated,
            'gpu_cached_mb': gpu_cached,
            'gpu_peak_mb': gpu_peak,
        }

    def measure(self, label):
        """Take a memory measurement."""
        gc.collect()
        if self.use_cuda:
            torch.cuda.empty_cache()

        mem = self.get_memory()
        mem['label'] = label
        mem['timestamp'] = datetime.now().isoformat()
        self.measurements.append(mem)
        return mem

    def reset_peak(self):
        """Reset peak memory stats."""
        if self.use_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_model_memory(self, model):
        """Get memory breakdown for a model."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        param_memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        buffer_memory_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**2

        embed_params = 0
        embed_memory = 0
        if hasattr(model, 'embedding'):
            embed_params = sum(p.numel() for p in model.embedding.parameters())
            embed_memory = sum(p.numel() * p.element_size() for p in model.embedding.parameters()) / 1024**2

            if hasattr(model.embedding, 'subword2binary'):
                buf = model.embedding.subword2binary
                buffer_memory_mb += buf.numel() * buf.element_size() / 1024**2
            if hasattr(model.embedding, 'subword2bytes'):
                buf = model.embedding.subword2bytes
                buffer_memory_mb += buf.numel() * buf.element_size() / 1024**2

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'embedding_params': embed_params,
            'param_memory_mb': param_memory_mb,
            'buffer_memory_mb': buffer_memory_mb,
            'embedding_memory_mb': embed_memory,
            'total_memory_mb': param_memory_mb + buffer_memory_mb,
        }


# ============================================================================
# Dataset
# ============================================================================

class WikiTextDataset(Dataset):
    """WikiText-103 dataset for language modeling."""

    def __init__(
        self,
        tokenizer,
        split: str = 'train',
        seq_length: int = 256,
        stride: int = 128,
        max_samples: int = None,
        chunk_size: int = 1024 * 1024,  # 1 MB of text per tokenization call
    ):
        print(f"Loading WikiText-103 {split} split...")
        dataset = load_dataset('wikitext', 'wikitext-103-v1', split=split)

        # Tokenize in chunks to avoid OOM from one giant tokenizer.encode() call.
        # We accumulate filtered text into a buffer and flush to the tokenizer
        # every ~chunk_size characters.
        print("Tokenizing in chunks...")
        token_chunks: List[List[int]] = []
        total_chars = 0
        buf: List[str] = []
        buf_chars = 0

        def _flush():
            nonlocal buf, buf_chars
            if buf:
                chunk_text = ' '.join(buf)
                token_chunks.append(tokenizer.encode(chunk_text))
                buf = []
                buf_chars = 0

        for item in dataset:
            text = item['text'].strip()
            if text and not text.startswith('='):
                total_chars += len(text)
                buf.append(text)
                buf_chars += len(text)
                if buf_chars >= chunk_size:
                    _flush()

        _flush()  # remaining text

        print(f"Total characters: {total_chars:,}")

        # Merge all token chunks into a single numpy array (int64) to avoid
        # materialising thousands of individual torch tensors.
        total_tokens = sum(len(c) for c in token_chunks)
        print(f"Total tokens: {total_tokens:,}")

        self._tokens = np.empty(total_tokens, dtype=np.int64)
        offset = 0
        for c in token_chunks:
            self._tokens[offset:offset + len(c)] = c
            offset += len(c)
        del token_chunks  # free intermediate lists

        # Pre-compute valid sample count (sliding window) instead of
        # materialising every sample.
        self.seq_length = seq_length
        self.stride = stride

        num_possible = (total_tokens - seq_length) // stride
        if max_samples is not None:
            self._num_samples = min(num_possible, max_samples)
        else:
            self._num_samples = max(num_possible, 0)

        print(f"Created {self._num_samples:,} samples")

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        input_ids = torch.as_tensor(
            self._tokens[start:start + self.seq_length].copy(), dtype=torch.long
        )
        target_ids = torch.as_tensor(
            self._tokens[start + 1:start + self.seq_length + 1].copy(), dtype=torch.long
        )
        return {'input_ids': input_ids, 'target_ids': target_ids}


# ============================================================================
# Model Definition
# ============================================================================

class TransformerLM(nn.Module):
    """Transformer Language Model for perplexity evaluation."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        embedding_type: str = 'standard',
        num_bits: int = 64,
        num_bytes: int = 8,
        interdim: str = '512',
        subword_file: str = '',
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.embedding_type = embedding_type
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Create embedding
        if embedding_type == 'standard':
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        elif embedding_type == 'binary_seb':
            self.embedding = BinarySubwordConcateOnehot(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                num_bits=num_bits,
                interdim=interdim,
                relu_dropout=dropout,
                subword_binary_file=subword_file,
            )

        elif embedding_type == 'byte_seb':
            if not BYTE_SEB_AVAILABLE:
                raise ImportError("ByteSubwordConcateOnehot not available")
            self.embedding = ByteSubwordConcateOnehot(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                interdim=interdim,
                relu_dropout=dropout,
                subword_bytes_file=subword_file,
            )

        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        # Positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection - shared with embedding (paper B.4)
        self.output_proj = nn.Linear(embedding_dim, vocab_size, bias=False)

        # Tie weights for standard embedding
        if embedding_type == 'standard':
            self.output_proj.weight = self.embedding.weight
        # For SEB types: output projection uses construct_matrix_for_output_layer
        # which is computed dynamically via the FFN, so weights aren't literally tied
        # but the same embedding function is used

        self.dropout = nn.Dropout(dropout)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        token_embeds = self.embedding(input_ids)

        # Add positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embedding(positions)

        embeds = self.dropout(token_embeds + pos_embeds)

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device),
            diagonal=1
        ).bool()

        # Transform
        hidden = self.transformer(embeds, mask=causal_mask)

        # Project to vocabulary
        logits = self.output_proj(hidden)

        return logits


class LSTMLM(nn.Module):
    """LSTM Language Model (simpler alternative)."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        embedding_type: str = 'standard',
        num_bits: int = 64,
        num_bytes: int = 8,
        interdim: str = '512',
        subword_file: str = '',
    ):
        super().__init__()

        self.embedding_type = embedding_type
        self.vocab_size = vocab_size

        # Create embedding
        if embedding_type == 'standard':
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        elif embedding_type == 'binary_seb':
            self.embedding = BinarySubwordConcateOnehot(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                num_bits=num_bits,
                interdim=interdim,
                relu_dropout=dropout,
                subword_binary_file=subword_file,
            )

        elif embedding_type == 'byte_seb':
            if not BYTE_SEB_AVAILABLE:
                raise ImportError("ByteSubwordConcateOnehot not available")
            self.embedding = ByteSubwordConcateOnehot(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                interdim=interdim,
                relu_dropout=dropout,
                subword_bytes_file=subword_file,
            )

        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Tie weights if standard
        if embedding_type == 'standard':
            self.output_proj.weight = self.embedding.weight

    def forward(self, input_ids, hidden=None):
        embeds = self.dropout(self.embedding(input_ids))
        output, hidden = self.lstm(embeds, hidden)
        logits = self.output_proj(self.dropout(output))
        return logits, hidden


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(dataloader, model, criterion, optimizer, device, clip_grad=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch in tqdm.tqdm(dataloader, desc='Training', file=sys.stdout):
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)

        optimizer.zero_grad()

        if isinstance(model, LSTMLM):
            logits, _ = model(input_ids)
        else:
            logits = model(input_ids)

        # Compute loss
        loss = criterion(logits.view(-1, model.vocab_size), target_ids.view(-1))

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        total_loss += loss.item() * target_ids.numel()
        total_tokens += target_ids.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity


def evaluate(dataloader, model, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc='Evaluating', file=sys.stdout):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            if isinstance(model, LSTMLM):
                logits, _ = model(input_ids)
            else:
                logits = model(input_ids)

            loss = criterion(logits.view(-1, model.vocab_size), target_ids.view(-1))

            total_loss += loss.item() * target_ids.numel()
            total_tokens += target_ids.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(
    embedding_type: str,
    model_type: str = 'transformer',  # 'transformer' or 'lstm'
    num_bits: int = 64,
    num_bytes: int = 8,
    seed: int = 42,
    n_epochs: int = 20,
    batch_size: int = 32,
    embedding_dim: int = 512,
    hidden_dim: int = 2048,
    num_layers: int = 6,
    num_heads: int = 8,
    dropout: float = 0.1,
    lr: float = 1e-4,
    seq_length: int = 256,
    stride: int = 128,
    interdim: str = '512',
    output_dir: str = './results_lm',
    max_train_samples: int = None,
    max_valid_samples: int = 10000,
):
    """Run a single language modeling experiment."""

    print(f"\n{'='*70}")
    print(f"Language Modeling Experiment: {embedding_type.upper()} ({model_type.upper()})")
    print(f"{'='*70}")

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Memory tracker
    memory_tracker = MemoryTracker(device)
    memory_tracker.measure('start')

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    # Load datasets
    train_dataset = WikiTextDataset(
        tokenizer, split='train',
        seq_length=seq_length, stride=stride,
        max_samples=max_train_samples
    )
    valid_dataset = WikiTextDataset(
        tokenizer, split='validation',
        seq_length=seq_length, stride=stride,
        max_samples=max_valid_samples
    )
    test_dataset = WikiTextDataset(
        tokenizer, split='test',
        seq_length=seq_length, stride=stride,
        max_samples=max_valid_samples
    )

    memory_tracker.measure('after_data_load')

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    os.makedirs(output_dir, exist_ok=True)
    subword_file = os.path.join(output_dir, f'{embedding_type}_mapping.pt')

    if model_type == 'transformer':
        model = TransformerLM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            embedding_type=embedding_type,
            num_bits=num_bits,
            num_bytes=num_bytes,
            interdim=interdim,
            subword_file=subword_file,
            max_seq_len=seq_length + 10,
        ).to(device)
    else:
        model = LSTMLM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            embedding_type=embedding_type,
            num_bits=num_bits,
            num_bytes=num_bytes,
            interdim=interdim,
            subword_file=subword_file,
        ).to(device)

    # Model memory info
    model_memory = memory_tracker.get_model_memory(model)
    memory_tracker.measure('after_model_init')

    print(f"\nModel Parameters:")
    print(f"  Total: {model_memory['total_params']:,}")
    print(f"  Trainable: {model_memory['trainable_params']:,}")
    print(f"  Embedding: {model_memory['embedding_params']:,}")
    print(f"  Parameter Memory: {model_memory['param_memory_mb']:.2f} MB")
    print(f"  Buffer Memory: {model_memory['buffer_memory_mb']:.2f} MB")
    print(f"  Total Memory: {model_memory['total_memory_mb']:.2f} MB")

    # Optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    # Training
    best_valid_ppl = float('inf')
    train_losses, train_ppls = [], []
    valid_losses, valid_ppls = [], []
    epoch_times = []

    model_path = os.path.join(output_dir, f'best_model_{embedding_type}.pt')

    # Resume from checkpoint if available
    start_epoch = 0
    if os.path.isfile(model_path):
        print(f"\nFound existing checkpoint: {model_path}")
        checkpoint_state = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint_state)
        # Check for saved partial results to determine start epoch
        partial_results_file = os.path.join(output_dir, f'results_{embedding_type}.json')
        if os.path.isfile(partial_results_file):
            with open(partial_results_file, 'r') as f:
                partial = json.load(f)
            if 'valid_ppls' in partial and len(partial['valid_ppls']) > 0:
                start_epoch = len(partial['valid_ppls'])
                train_losses = partial.get('train_losses', [])
                train_ppls = partial.get('train_ppls', [])
                valid_losses = partial.get('valid_losses', [])
                valid_ppls = partial.get('valid_ppls', [])
                epoch_times = partial.get('epoch_times', [0.0] * start_epoch)
                best_valid_ppl = min(valid_ppls) if valid_ppls else float('inf')
                print(f"  Resuming from epoch {start_epoch + 1}/{n_epochs} (best valid PPL: {best_valid_ppl:.2f})")
        if start_epoch == 0:
            # Checkpoint exists but no results JSON — evaluate to get baseline PPL
            _, resume_ppl = evaluate(valid_loader, model, criterion, device)
            best_valid_ppl = resume_ppl
            print(f"  Checkpoint loaded (valid PPL: {resume_ppl:.2f}), training from epoch 1")
        # Advance scheduler to the correct position
        for _ in range(start_epoch):
            scheduler.step()

    memory_tracker.reset_peak()
    total_train_start = time.time()

    if start_epoch >= n_epochs:
        print(f"\nAlready completed {n_epochs} epochs, skipping training.")
    else:
        print(f"\nTraining for epochs {start_epoch + 1} to {n_epochs}...")

    for epoch in range(start_epoch, n_epochs):
        epoch_start = time.time()

        train_loss, train_ppl = train_epoch(train_loader, model, criterion, optimizer, device)
        valid_loss, valid_ppl = evaluate(valid_loader, model, criterion, device)

        scheduler.step()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        train_losses.append(train_loss)
        train_ppls.append(train_ppl)
        valid_losses.append(valid_loss)
        valid_ppls.append(valid_ppl)

        if valid_ppl < best_valid_ppl:
            best_valid_ppl = valid_ppl
            torch.save(model.state_dict(), model_path)

        print(f"Epoch {epoch+1}/{n_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f}")
        print(f"  Valid Loss: {valid_loss:.4f}, Valid PPL: {valid_ppl:.2f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save partial results after each epoch for resume support
        partial_results = {
            'embedding_type': embedding_type,
            'train_losses': train_losses,
            'train_ppls': train_ppls,
            'valid_losses': valid_losses,
            'valid_ppls': valid_ppls,
            'epoch_times': epoch_times,
            'best_valid_ppl': best_valid_ppl,
        }
        partial_file = os.path.join(output_dir, f'results_{embedding_type}.json')
        with open(partial_file, 'w') as f:
            json.dump(partial_results, f, indent=2)

    total_train_time = time.time() - total_train_start
    final_memory = memory_tracker.measure('after_training')

    # Test evaluation
    model.load_state_dict(torch.load(model_path))
    test_loss, test_ppl = evaluate(test_loader, model, criterion, device)

    print(f"\n{'='*70}")
    print(f"RESULTS: {embedding_type.upper()}")
    print(f"{'='*70}")
    print(f"Best Valid PPL: {best_valid_ppl:.2f}")
    print(f"Test PPL: {test_ppl:.2f}")
    print(f"Total Training Time: {total_train_time:.1f}s")
    print(f"Avg Epoch Time: {np.mean(epoch_times):.1f}s")
    print(f"Peak GPU Memory: {final_memory['gpu_peak_mb']:.2f} MB")

    # Save results
    results = {
        'embedding_type': embedding_type,
        'model_type': model_type,
        'num_bits': num_bits if embedding_type == 'binary_seb' else None,
        'num_bytes': num_bytes if embedding_type == 'byte_seb' else None,
        'vocab_size': vocab_size,
        'best_valid_ppl': best_valid_ppl,
        'test_ppl': test_ppl,
        'total_train_time': total_train_time,
        'avg_epoch_time': np.mean(epoch_times),
        'model_memory': model_memory,
        'peak_gpu_mb': final_memory['gpu_peak_mb'],
        'peak_cpu_mb': final_memory['cpu_mb'],
        'train_losses': train_losses,
        'train_ppls': train_ppls,
        'valid_losses': valid_losses,
        'valid_ppls': valid_ppls,
        'config': {
            'seed': seed,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'dropout': dropout,
            'lr': lr,
            'seq_length': seq_length,
            'interdim': interdim,
        }
    }

    results_file = os.path.join(output_dir, f'results_{embedding_type}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return results


def run_all_comparisons(output_dir: str = './results_lm', **kwargs):
    """Run all three embedding types for comparison."""

    all_results = {}

    # Standard embedding
    all_results['standard'] = run_experiment(
        embedding_type='standard',
        output_dir=output_dir,
        **kwargs
    )

    # Byte SEB
    if BYTE_SEB_AVAILABLE:
        all_results['byte_seb'] = run_experiment(
            embedding_type='byte_seb',
            output_dir=output_dir,
            **kwargs
        )

    # Binary SEB
    all_results['binary_seb'] = run_experiment(
        embedding_type='binary_seb',
        output_dir=output_dir,
        **kwargs
    )

    # Print Table 5 style comparison
    print_table5(all_results)

    # Save comparison
    comparison_file = os.path.join(output_dir, 'comparison_table5.json')
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nComparison saved to: {comparison_file}")

    return all_results


def print_table5(results: Dict):
    """Print results in Table 5 format from the paper."""

    print("\n")
    print("=" * 90)
    print("TABLE 5: Language Modeling Results on WikiText-103")
    print("=" * 90)
    print()

    # Header
    print(f"{'Method':<20} {'Valid PPL':<12} {'Test PPL':<12} {'Params':<15} {'Memory (MB)':<15} {'Time (s)':<12}")
    print("-" * 90)

    # Baseline memory for reduction calculation
    baseline_mem = results.get('standard', {}).get('model_memory', {}).get('total_memory_mb', 1)
    baseline_params = results.get('standard', {}).get('model_memory', {}).get('total_params', 1)

    for name in ['standard', 'byte_seb', 'binary_seb']:
        if name not in results:
            continue

        r = results[name]
        valid_ppl = r['best_valid_ppl']
        test_ppl = r['test_ppl']
        params = r['model_memory']['total_params']
        memory = r['model_memory']['total_memory_mb']
        train_time = r['total_train_time']

        print(f"{name:<20} {valid_ppl:<12.2f} {test_ppl:<12.2f} {params:<15,} {memory:<15.2f} {train_time:<12.1f}")

    print("-" * 90)

    # Memory reduction summary
    print("\nMemory & Parameter Reduction vs Standard:")
    for name in ['byte_seb', 'binary_seb']:
        if name not in results:
            continue

        r = results[name]
        mem = r['model_memory']['total_memory_mb']
        params = r['model_memory']['total_params']
        embed_params = r['model_memory']['embedding_params']

        mem_reduction = (1 - mem / baseline_mem) * 100
        param_reduction = (1 - params / baseline_params) * 100

        print(f"  {name}:")
        print(f"    Memory reduction: {mem_reduction:.1f}%")
        print(f"    Parameter reduction: {param_reduction:.1f}%")
        print(f"    Embedding params: {embed_params:,}")

    print()
    print("=" * 90)
    print("Note: PPL = Perplexity (lower is better)")
    print("=" * 90)


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='WikiText-103 Language Modeling with Binary SEB')

    parser.add_argument('--embedding_type', type=str, default='binary_seb',
                        choices=['standard', 'byte_seb', 'binary_seb'])
    parser.add_argument('--model_type', type=str, default='transformer',
                        choices=['transformer', 'lstm'])
    parser.add_argument('--run_all', action='store_true',
                        help='Run all embedding types for comparison')
    parser.add_argument('--num_bits', type=int, default=64)
    parser.add_argument('--num_bytes', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=2048)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seq_length', type=int, default=256)
    parser.add_argument('--stride', type=int, default=128)
    parser.add_argument('--interdim', type=str, default='4096')
    parser.add_argument('--output_dir', type=str, default='./results_lm')
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_valid_samples', type=int, default=10000)

    args = parser.parse_args()

    kwargs = {
        'model_type': args.model_type,
        'num_bits': args.num_bits,
        'num_bytes': args.num_bytes,
        'seed': args.seed,
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'dropout': args.dropout,
        'lr': args.lr,
        'seq_length': args.seq_length,
        'stride': args.stride,
        'interdim': args.interdim,
        'max_train_samples': args.max_train_samples,
        'max_valid_samples': args.max_valid_samples,
    }

    if args.run_all:
        run_all_comparisons(output_dir=args.output_dir, **kwargs)
    else:
        results = run_experiment(
            embedding_type=args.embedding_type,
            output_dir=args.output_dir,
            **kwargs
        )
        print_table5({args.embedding_type: results})


if __name__ == '__main__':
    main()
