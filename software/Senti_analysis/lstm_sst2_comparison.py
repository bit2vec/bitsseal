#!/usr/bin/env python3
"""
SST2 Sentiment Analysis with Binary SEB


Tracks accuracy, training time, and memory usage.

Usage:
    python lstm_sst2_comparison.py --embedding_type binary_seb --num_bits 64
    python lstm_sst2_comparison.py --run_all  # Run all three for comparison
"""

import functools
import sys
import os
import gc
import time
import json
import argparse
from datetime import datetime

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import psutil

# Use compatibility layer for torchtext (handles missing/changed APIs)
import sys as _sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
_sys.path.insert(0, _script_dir)
_sys.path.insert(0, os.path.join(_script_dir, '..'))
try:
    from vocab_compat import build_vocab_from_iterator, get_tokenizer
except ImportError:
    import re
    from collections import Counter as _Counter

    class _SimpleVocab:
        def __init__(self, stoi, itos, default_index=0):
            self._stoi, self._itos, self._default_index = stoi, itos, default_index
        def __getitem__(self, token): return self._stoi.get(token, self._default_index)
        def __len__(self): return len(self._itos)
        def __contains__(self, token): return token in self._stoi
        def set_default_index(self, index): self._default_index = index

    def build_vocab_from_iterator(iterator, min_freq=1, specials=None):
        if specials is None: specials = []
        counter = _Counter()
        for tokens in iterator: counter.update(tokens)
        itos = list(specials)
        stoi = {s: i for i, s in enumerate(specials)}
        for token, freq in counter.most_common():
            if freq >= min_freq and token not in stoi:
                stoi[token] = len(itos); itos.append(token)
        return _SimpleVocab(stoi, itos, default_index=0)

    def get_tokenizer(name='basic_english'):
        def _tok(text): return re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower()).split()
        return _tok

# Import embedding modules
from binary_subword_sentiment import (
    BinarySubwordConcateOnehot,
    BinarySubwordSum,
    BinaryWordEmbedding,
)

# Try to import byte SEB (if available)
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
        
        # Parameter memory (float32 = 4 bytes)
        param_memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        
        # Buffer memory
        buffer_memory_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**2
        
        # Embedding-specific
        embed_params = 0
        embed_memory = 0
        if hasattr(model, 'embedding'):
            embed_params = sum(p.numel() for p in model.embedding.parameters())
            embed_memory = sum(p.numel() * p.element_size() for p in model.embedding.parameters()) / 1024**2
            
            # Add buffer memory for SEB mappings
            if hasattr(model.embedding, 'subword2binary'):
                buffer_memory_mb += model.embedding.subword2binary.numel() * model.embedding.subword2binary.element_size() / 1024**2
            if hasattr(model.embedding, 'subword2bytes'):
                buffer_memory_mb += model.embedding.subword2bytes.numel() * model.embedding.subword2bytes.element_size() / 1024**2
        
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
# Model Definition
# ============================================================================

class LSTM(nn.Module):
    """BiLSTM classifier for sentiment analysis."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        bidirectional: bool,
        dropout_rate: float,
        pad_index: int,
        embedding_type: str = 'standard',
        num_bits: int = 64,
        num_bytes: int = 8,
        interdim: str = '128',
        subword_file: str = '',
    ):
        super().__init__()
        
        self.embedding_type = embedding_type
        
        # Create embedding layer based on type
        if embedding_type == 'standard':
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        
        elif embedding_type == 'binary_seb':
            self.embedding = BinarySubwordConcateOnehot(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                num_bits=num_bits,
                interdim=interdim,
                relu_dropout=dropout_rate,
                subword_binary_file=subword_file if subword_file else f'binary_sst2_{num_bits}bits.pt',
            )
        
        elif embedding_type == 'byte_seb':
            if not BYTE_SEB_AVAILABLE:
                raise ImportError("ByteSubwordConcateOnehot not available")
            self.embedding = ByteSubwordConcateOnehot(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                interdim=interdim,
                relu_dropout=dropout_rate,
                subword_bytes_file=subword_file if subword_file else 'bytes_sst2.pt',
            )
        
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
        
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, n_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate if n_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, ids, length):
        # Get embeddings
        embedded = self.dropout(self.embedding(ids))
        
        # Pack for LSTM
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, length.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Get final hidden state
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        else:
            hidden = self.dropout(hidden[-1])
        
        prediction = self.fc(hidden)
        return prediction


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(dataloader, model, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    epoch_losses = []
    epoch_accs = []
    
    for batch in tqdm.tqdm(dataloader, desc='Training', file=sys.stdout):
        ids = batch['ids'].to(device)
        length = batch['length']
        label = batch['label'].to(device)
        
        optimizer.zero_grad()
        prediction = model(ids, length)
        loss = criterion(prediction, label)
        
        accuracy = get_accuracy(prediction, label)
        
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    
    return np.mean(epoch_losses), np.mean(epoch_accs)


def evaluate(dataloader, model, criterion, device):
    """Evaluate the model using sample-weighted accuracy (not per-batch mean).

    The previous implementation averaged per-batch accuracies, which biases
    the reported score whenever the last batch is smaller than the others.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc='Evaluating', file=sys.stdout):
            ids = batch['ids'].to(device)
            length = batch['length']
            label = batch['label'].to(device)

            prediction = model(ids, length)
            loss = criterion(prediction, label)
            preds = prediction.argmax(dim=-1)
            bs = label.numel()
            total_loss += loss.item() * bs
            total_correct += preds.eq(label).sum().item()
            total_samples += bs

    total_samples = max(total_samples, 1)
    return total_loss / total_samples, total_correct / total_samples


def get_accuracy(prediction, label):
    """Calculate accuracy."""
    batch_size = prediction.shape[0]
    predicted_classes = prediction.argmax(dim=-1)
    correct = predicted_classes.eq(label).sum()
    return correct / batch_size


def collate_fn(batch, pad_index):
    """Collate function for DataLoader."""
    # Filter out zero-length sequences to avoid pack_padded_sequence errors
    batch = [item for item in batch if item['length'] > 0]
    if len(batch) == 0:
        # Return a dummy batch with a single padding token if all items were empty
        return {
            'ids': torch.tensor([[pad_index]]),
            'length': torch.tensor([1]),
            'label': torch.tensor([0]).long(),
        }
    batch_ids = [item['ids'] for item in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_length = torch.stack([item['length'] for item in batch])
    batch_label = torch.stack([item['label'] for item in batch]).long()

    return {
        'ids': batch_ids,
        'length': batch_length,
        'label': batch_label,
    }


def initialize_weights(m):
    """Initialize model weights."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.orthogonal_(param)


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(
    embedding_type: str,
    num_bits: int = 64,
    num_bytes: int = 8,
    seed: int = 42,
    n_epochs: int = 50,
    batch_size: int = 512,
    embedding_dim: int = 256,
    hidden_dim: int = 300,
    n_layers: int = 2,
    dropout_rate: float = 0.4,
    lr: float = 5e-4,
    interdim: str = '128',
    output_dir: str = './results_sst2',
):
    """Run a single experiment."""
    
    print(f"\n{'='*60}")
    print(f"Running SST2 Experiment: {embedding_type.upper()}")
    print(f"{'='*60}")
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize memory tracker
    memory_tracker = MemoryTracker(device)
    memory_tracker.measure('start')
    
    # Load dataset
    print("\nLoading SST2 dataset...")
    train_data, test_data = datasets.load_dataset('sst2', split=['train', 'validation'])
    
    # Tokenize
    tokenizer = get_tokenizer('basic_english')
    max_length = 256
    
    def tokenize_data(example):
        tokens = tokenizer(example['sentence'])[:max_length]
        if len(tokens) == 0:
            tokens = ['<unk>']
        return {'tokens': tokens, 'length': len(tokens)}
    
    # Split train/valid
    train_valid = train_data.train_test_split(test_size=0.25, seed=seed)
    train_data = train_valid['train']
    valid_data = train_valid['test']
    
    train_data = train_data.map(tokenize_data)
    valid_data = valid_data.map(tokenize_data)
    test_data = test_data.map(tokenize_data)
    
    # Build vocabulary
    min_freq = 5
    vocab = build_vocab_from_iterator(
        train_data['tokens'],
        min_freq=min_freq,
        specials=['<unk>', '<pad>']
    )
    vocab.set_default_index(vocab['<unk>'])
    pad_index = vocab['<pad>']
    vocab_size = len(vocab)
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Numericalize
    def numericalize(example):
        return {'ids': [vocab[token] for token in example['tokens']]}
    
    train_data = train_data.map(numericalize)
    valid_data = valid_data.map(numericalize)
    test_data = test_data.map(numericalize)
    
    # Convert to torch format
    train_data = train_data.with_format(type='torch', columns=['ids', 'label', 'length'])
    valid_data = valid_data.with_format(type='torch', columns=['ids', 'label', 'length'])
    test_data = test_data.with_format(type='torch', columns=['ids', 'label', 'length'])
    
    memory_tracker.measure('after_data_load')
    
    # Create dataloaders
    collate = functools.partial(collate_fn, pad_index=pad_index)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, collate_fn=collate, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, collate_fn=collate)
    
    # Create model
    output_dim = len(train_data.unique('label'))
    
    os.makedirs(output_dir, exist_ok=True)
    subword_file = os.path.join(output_dir, f'{embedding_type}_mapping.pt')
    
    model = LSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        bidirectional=True,
        dropout_rate=dropout_rate,
        pad_index=pad_index,
        embedding_type=embedding_type,
        num_bits=num_bits,
        num_bytes=num_bytes,
        interdim=interdim,
        subword_file=subword_file,
    ).to(device)
    
    # Initialize weights (but not embedding for SEB)
    model.lstm.apply(initialize_weights)
    model.fc.apply(initialize_weights)
    
    if embedding_type == 'standard':
        nn.init.normal_(model.embedding.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(model.embedding.weight[pad_index], 0)
    
    # Get model memory info
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
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_valid_loss = float('inf')
    best_valid_acc = 0
    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []
    epoch_times = []
    
    model_path = os.path.join(output_dir, f'best_model_{embedding_type}.pt')
    
    memory_tracker.reset_peak()
    total_train_start = time.time()
    
    print(f"\nTraining for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(train_loader, model, criterion, optimizer, device)
        valid_loss, valid_acc = evaluate(valid_loader, model, criterion, device)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), model_path)
        
        print(f"Epoch {epoch+1}/{n_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")
    
    total_train_time = time.time() - total_train_start
    
    # Final memory measurement
    final_memory = memory_tracker.measure('after_training')
    
    # Load best model and test
    model.load_state_dict(torch.load(model_path))
    test_loss, test_acc = evaluate(test_loader, model, criterion, device)
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {embedding_type.upper()}")
    print(f"{'='*60}")
    print(f"Best Valid Acc: {best_valid_acc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Total Training Time: {total_train_time:.1f}s")
    print(f"Avg Epoch Time: {np.mean(epoch_times):.1f}s")
    print(f"Peak GPU Memory: {final_memory['gpu_peak_mb']:.2f} MB")
    print(f"Peak CPU Memory: {final_memory['cpu_mb']:.2f} MB")
    
    # Save results
    results = {
        'embedding_type': embedding_type,
        'num_bits': num_bits if embedding_type == 'binary_seb' else None,
        'num_bytes': num_bytes if embedding_type == 'byte_seb' else None,
        'vocab_size': vocab_size,
        'best_valid_acc': best_valid_acc,
        'test_acc': test_acc,
        'total_train_time': total_train_time,
        'avg_epoch_time': np.mean(epoch_times),
        'model_memory': model_memory,
        'peak_gpu_mb': final_memory['gpu_peak_mb'],
        'peak_cpu_mb': final_memory['cpu_mb'],
        'train_losses': train_losses,
        'train_accs': train_accs,
        'valid_losses': valid_losses,
        'valid_accs': valid_accs,
        'config': {
            'seed': seed,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'dropout_rate': dropout_rate,
            'lr': lr,
            'interdim': interdim,
        }
    }
    
    results_file = os.path.join(output_dir, f'results_{embedding_type}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


def run_all_comparisons(output_dir: str = './results_sst2', **kwargs):
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
    
    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY - SST2")
    print("="*80)
    
    print(f"\n{'Method':<20} {'Test Acc':<12} {'Params':<15} {'Memory (MB)':<15} {'Time (s)':<12}")
    print("-"*80)
    
    for name, result in all_results.items():
        test_acc = result['test_acc']
        params = result['model_memory']['total_params']
        memory = result['model_memory']['total_memory_mb']
        train_time = result['total_train_time']
        print(f"{name:<20} {test_acc:<12.4f} {params:<15,} {memory:<15.2f} {train_time:<12.1f}")
    
    # Save comparison
    comparison_file = os.path.join(output_dir, 'comparison_summary.json')
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nComparison saved to: {comparison_file}")
    
    return all_results


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SST2 Sentiment Analysis with Binary SEB')
    
    parser.add_argument('--embedding_type', type=str, default='binary_seb',
                        choices=['standard', 'byte_seb', 'binary_seb'],
                        help='Type of embedding to use')
    parser.add_argument('--run_all', action='store_true',
                        help='Run all embedding types for comparison')
    parser.add_argument('--num_bits', type=int, default=64,
                        help='Number of bits for Binary SEB')
    parser.add_argument('--num_bytes', type=int, default=8,
                        help='Number of bytes for Byte SEB')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--dropout_rate', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--interdim', type=str, default='128')
    parser.add_argument('--output_dir', type=str, default='./results_sst2')
    
    args = parser.parse_args()
    
    if args.run_all:
        run_all_comparisons(
            output_dir=args.output_dir,
            num_bits=args.num_bits,
            num_bytes=args.num_bytes,
            seed=args.seed,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            lr=args.lr,
            interdim=args.interdim,
        )
    else:
        run_experiment(
            embedding_type=args.embedding_type,
            num_bits=args.num_bits,
            num_bytes=args.num_bytes,
            seed=args.seed,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            lr=args.lr,
            interdim=args.interdim,
            output_dir=args.output_dir,
        )


if __name__ == '__main__':
    main()
