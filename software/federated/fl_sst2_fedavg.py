#!/usr/bin/env python3
"""
Federated Learning on SST2 with FedAvg Framework


Usage:
    python fl_sst2_fedavg.py --run_all
    python fl_sst2_fedavg.py --embedding_type binary_seb --participation_ratios 0.2 0.4 0.6 0.8 1.0
    python fl_sst2_fedavg.py --quick_test  # Quick test with fewer rounds
"""

import os
import sys
import gc
import json
import copy
import random
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import tqdm
import psutil

# Hugging Face
import datasets

# Use compatibility layer for torchtext (handles missing/changed APIs)
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
sys.path.insert(0, os.path.join(_script_dir, '..'))
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

# Import Binary SEB
from binary_subword_sentiment import BinarySubwordConcateOnehot

# Try to import Byte SEB
try:
    from byte_subword_concate_onehot import ByteSubwordConcateOnehot
    BYTE_SEB_AVAILABLE = True
except ImportError:
    BYTE_SEB_AVAILABLE = False
    print("Warning: ByteSubwordConcateOnehot not found. Byte SEB comparison will be skipped.")


# ============================================================================
# Configuration
# ============================================================================

@dataclass 
class FLConfig:
    """Configuration for federated learning experiment."""
    
    # Dataset
    max_seq_len: int = 256
    min_freq: int = 5
    
    # Model architecture (matching paper's BiLSTM)
    embedding_dim: int = 256
    hidden_dim: int = 300
    num_layers: int = 2
    dropout: float = 0.4
    
    # SEB settings
    num_bytes: int = 8
    num_bits: int = 64
    interdim: str = '128'
    
    # Federated Learning settings
    num_clients: int = 20
    participation_ratios: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0])
    num_rounds: int = 50          # Communication rounds
    local_epochs: int = 1         # Local training epochs per round
    local_batch_size: int = 32
    
    # Training
    lr: float = 5e-4
    
    # Embedding types to test
    embedding_types: List[str] = field(default_factory=lambda: ['standard', 'byte_seb', 'binary_seb'])
    
    # Output
    output_dir: str = './results_fl_sst2'
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42


# ============================================================================
# Memory Tracking
# ============================================================================

class MemoryTracker:
    """Track memory usage."""
    
    def __init__(self, device):
        self.device = device
        self.use_cuda = device.type == 'cuda'
    
    def get_memory(self):
        cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        if self.use_cuda:
            torch.cuda.synchronize()
            gpu_mem = torch.cuda.max_memory_allocated(self.device) / 1024**2
        else:
            gpu_mem = 0
        return {'cpu_mb': cpu_mem, 'gpu_mb': gpu_mem}
    
    def reset_peak(self):
        if self.use_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def get_model_memory(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        buffer_memory_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**2
        
        embed_params = 0
        if hasattr(model, 'embedding'):
            embed_params = sum(p.numel() for p in model.embedding.parameters())
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'embedding_params': embed_params,
            'param_memory_mb': param_memory_mb,
            'buffer_memory_mb': buffer_memory_mb,
            'total_memory_mb': param_memory_mb + buffer_memory_mb,
        }


# ============================================================================
# Dataset for Federated Learning
# ============================================================================

class SST2ClientDataset(Dataset):
    """SST2 dataset wrapper for a single client."""
    
    def __init__(self, data_indices, full_dataset):
        self.indices = data_indices
        self.full_dataset = full_dataset
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.full_dataset[self.indices[idx]]


def prepare_sst2_data(config: FLConfig, seed: int = 42):
    """
    Prepare SST2 dataset and split among clients.
    
    Returns:
        client_datasets: List of datasets, one per client
        test_dataset: Test dataset
        vocab: Vocabulary
        pad_index: Padding index
    """
    print("Loading SST2 dataset...")
    train_data, test_data = datasets.load_dataset('sst2', split=['train', 'validation'])

    # Tokenize
    tokenizer = get_tokenizer('basic_english')
    
    def tokenize_data(example):
        tokens = tokenizer(example['sentence'])[:config.max_seq_len]
        return {'tokens': tokens, 'length': len(tokens)}
    
    train_data = train_data.map(tokenize_data)
    test_data = test_data.map(tokenize_data)

    # Filter out samples with length 0 (pack_padded_sequence requires length > 0)
    train_data = train_data.filter(lambda x: x['length'] > 0)
    test_data = test_data.filter(lambda x: x['length'] > 0)

    # Build vocabulary from all training data
    vocab = build_vocab_from_iterator(
        train_data['tokens'],
        min_freq=config.min_freq,
        specials=['<unk>', '<pad>']
    )
    vocab.set_default_index(vocab['<unk>'])
    pad_index = vocab['<pad>']
    
    # Numericalize
    def numericalize(example):
        return {'ids': [vocab[token] for token in example['tokens']]}
    
    train_data = train_data.map(numericalize)
    test_data = test_data.map(numericalize)
    
    # Convert to torch format
    train_data = train_data.with_format(type='torch', columns=['ids', 'label', 'length'])
    test_data = test_data.with_format(type='torch', columns=['ids', 'label', 'length'])
    
    # Split training data among clients (uniform distribution as in paper)
    num_samples = len(train_data)
    indices = list(range(num_samples))
    random.seed(seed)
    random.shuffle(indices)
    
    # Distribute samples uniformly to clients
    samples_per_client = num_samples // config.num_clients
    client_indices = []
    for i in range(config.num_clients):
        start_idx = i * samples_per_client
        if i == config.num_clients - 1:
            # Last client gets remaining samples
            end_idx = num_samples
        else:
            end_idx = start_idx + samples_per_client
        client_indices.append(indices[start_idx:end_idx])
    
    # Create client datasets
    client_datasets = [
        SST2ClientDataset(client_indices[i], train_data)
        for i in range(config.num_clients)
    ]
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Total training samples: {num_samples}")
    print(f"Number of clients: {config.num_clients}")
    print(f"Samples per client: ~{samples_per_client}")
    print(f"Test samples: {len(test_data)}")
    
    return client_datasets, test_data, vocab, pad_index


# ============================================================================
# Model Definition
# ============================================================================

class BiLSTMClassifier(nn.Module):
    """
    BiLSTM Classifier for sentiment analysis.
    Same architecture as in the paper for SST2.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        pad_index: int,
        embedding_type: str = 'standard',
        num_bits: int = 64,
        num_bytes: int = 8,
        interdim: str = '128',
        subword_file: str = '',
    ):
        super().__init__()
        
        self.embedding_type = embedding_type
        self.vocab_size = vocab_size
        
        # Create embedding layer
        if embedding_type == 'standard':
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        
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
        
        # BiLSTM
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, ids, length):
        # Embedding
        embedded = self.dropout(self.embedding(ids))
        
        # Pack sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, length.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Concatenate final hidden states from both directions
        hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        
        # Output
        output = self.fc(hidden)
        return output


# ============================================================================
# Federated Learning with FedAvg
# ============================================================================

class FedAvgServer:
    """
    FedAvg Server for coordinating federated learning.
    
    Implements the FedAvg algorithm from McMahan et al. (2017):
    1. Server sends global model to selected clients
    2. Clients train locally for E epochs
    3. Server aggregates client updates
    """
    
    def __init__(
        self,
        global_model: nn.Module,
        client_datasets: List[Dataset],
        test_dataset: Dataset,
        config: FLConfig,
        pad_index: int,
    ):
        self.global_model = global_model
        self.client_datasets = client_datasets
        self.test_dataset = test_dataset
        self.config = config
        self.pad_index = pad_index
        self.device = torch.device(config.device)
        
        # Create test dataloader
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=lambda b: self.collate_fn(b),
        )
        
        self.num_clients = len(client_datasets)
        self.history = {'round': [], 'train_loss': [], 'test_acc': []}
    
    def collate_fn(self, batch):
        """Collate function for DataLoader."""
        batch_ids = [item['ids'] for item in batch]
        batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=self.pad_index, batch_first=True)
        batch_length = torch.stack([item['length'] for item in batch])
        batch_label = torch.stack([item['label'] for item in batch]).long()
        return {'ids': batch_ids, 'length': batch_length, 'label': batch_label}
    
    def select_clients(self, participation_ratio: float) -> List[int]:
        """Randomly select clients based on participation ratio."""
        num_selected = max(1, int(self.num_clients * participation_ratio))
        return random.sample(range(self.num_clients), num_selected)
    
    def client_update(self, client_id: int) -> Tuple[Dict, int, float]:
        """
        Perform local training on a client.
        
        Returns:
            state_dict: Updated model parameters
            num_samples: Number of training samples
            avg_loss: Average training loss
        """
        # Create local model as copy of global model
        local_model = copy.deepcopy(self.global_model)
        local_model.to(self.device)
        local_model.train()
        
        # Create client dataloader
        client_loader = DataLoader(
            self.client_datasets[client_id],
            batch_size=self.config.local_batch_size,
            shuffle=True,
            collate_fn=lambda b: self.collate_fn(b),
        )
        
        # Local optimizer: AdamW with weight decay for better generalization
        optimizer = optim.AdamW(local_model.parameters(), lr=self.config.lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0
        num_batches = 0

        # Local training
        for epoch in range(self.config.local_epochs):
            for batch in client_loader:
                ids = batch['ids'].to(self.device)
                length = batch['length']
                label = batch['label'].to(self.device)

                optimizer.zero_grad()
                output = local_model(ids, length)
                loss = criterion(output, label)
                loss.backward()
                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        num_samples = len(self.client_datasets[client_id])
        
        return local_model.state_dict(), num_samples, avg_loss
    
    def aggregate(self, client_updates: List[Tuple[Dict, int]]):
        """
        Aggregate client updates using FedAvg (weighted by number of samples).
        
        Args:
            client_updates: List of (state_dict, num_samples) tuples
        """
        total_samples = sum(num_samples for _, num_samples in client_updates)

        # Only average LEARNABLE parameters. Buffers (e.g. subword2binary,
        # subword2bytes) are identical across clients (never updated by
        # optimizer), but averaging them via .float()*weight + load_state_dict
        # silently corrupts them: with unequal client sizes, the weighted sum
        # becomes 0.9999... which truncates to 0 when cast back to long.
        # Empirically this flips ~50% of bits after 50 rounds, destroying
        # binary_seb training at any c < 1.0.
        global_state = self.global_model.state_dict()
        trainable_keys = {name for name, p in self.global_model.named_parameters()
                          if p.requires_grad}

        aggregated_state = {}
        for key, global_val in global_state.items():
            if key in trainable_keys:
                aggregated_state[key] = sum(
                    state_dict[key].to(global_val.dtype) * (num_samples / total_samples)
                    for state_dict, num_samples in client_updates
                )
            else:
                aggregated_state[key] = global_val

        self.global_model.load_state_dict(aggregated_state)
    
    def evaluate(self) -> float:
        """Evaluate global model on test set."""
        self.global_model.to(self.device)
        self.global_model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                ids = batch['ids'].to(self.device)
                length = batch['length']
                label = batch['label'].to(self.device)
                
                output = self.global_model(ids, length)
                pred = output.argmax(dim=-1)
                correct += (pred == label).sum().item()
                total += label.size(0)
        
        return correct / total
    
    def train(self, participation_ratio: float, num_rounds: int, verbose: bool = True) -> Dict:
        """
        Run federated learning training.
        
        Args:
            participation_ratio: Fraction of clients participating each round
            num_rounds: Number of communication rounds
            verbose: Whether to print progress
        
        Returns:
            Dictionary with training history and final metrics
        """
        self.history = {'round': [], 'train_loss': [], 'test_acc': []}
        best_acc = 0
        
        pbar = tqdm.tqdm(range(num_rounds), desc=f'FL (c={participation_ratio})', file=sys.stdout)
        
        for round_idx in pbar:
            # Select clients
            selected_clients = self.select_clients(participation_ratio)
            
            # Collect client updates
            client_updates = []
            round_loss = 0
            
            for client_id in selected_clients:
                state_dict, num_samples, avg_loss = self.client_update(client_id)
                client_updates.append((state_dict, num_samples))
                round_loss += avg_loss
            
            # Aggregate updates
            self.aggregate(client_updates)
            
            # Evaluate
            test_acc = self.evaluate()
            avg_round_loss = round_loss / len(selected_clients)
            
            # Record history
            self.history['round'].append(round_idx + 1)
            self.history['train_loss'].append(avg_round_loss)
            self.history['test_acc'].append(test_acc)
            
            if test_acc > best_acc:
                best_acc = test_acc
            
            if verbose:
                pbar.set_postfix({
                    'loss': f'{avg_round_loss:.4f}',
                    'acc': f'{test_acc:.4f}',
                    'best': f'{best_acc:.4f}'
                })
        
        return {
            'final_acc': test_acc,
            'best_acc': best_acc,
            'history': self.history,
        }


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(
    embedding_type: str,
    participation_ratios: List[float],
    config: FLConfig,
    client_datasets: List[Dataset],
    test_dataset: Dataset,
    vocab_size: int,
    pad_index: int,
    output_dir: str,
) -> Dict:
    """Run federated learning experiment for one embedding type."""
    
    print(f"\n{'='*70}")
    print(f"Federated Learning Experiment: {embedding_type.upper()}")
    print(f"{'='*70}")
    
    device = torch.device(config.device)
    memory_tracker = MemoryTracker(device)
    
    results = {
        'embedding_type': embedding_type,
        'participation_results': {},
    }
    
    for c in participation_ratios:
        print(f"\n--- Participation ratio c = {c} ---")

        # Reset seeds so each participation ratio starts from the same model
        # initialisation. Without this, binary_seb's FFN sometimes lands on a
        # dead-ReLU initialisation and training collapses to ~51% (random).
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)

        # Create fresh model for each participation ratio
        os.makedirs(output_dir, exist_ok=True)
        subword_file = os.path.join(output_dir, f'{embedding_type}_mapping.pt')

        model = BiLSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            output_dim=2,  # Binary classification
            num_layers=config.num_layers,
            dropout=config.dropout,
            pad_index=pad_index,
            embedding_type=embedding_type,
            num_bits=config.num_bits,
            num_bytes=config.num_bytes,
            interdim=config.interdim,
            subword_file=subword_file,
        )
        
        # Initialize weights
        if embedding_type == 'standard':
            nn.init.normal_(model.embedding.weight, mean=0, std=config.embedding_dim ** -0.5)
            nn.init.constant_(model.embedding.weight[pad_index], 0)
        
        # Get model memory info (only once per embedding type)
        if c == participation_ratios[0]:
            model_memory = memory_tracker.get_model_memory(model)
            results['model_memory'] = model_memory
            print(f"\nModel Parameters:")
            print(f"  Total: {model_memory['total_params']:,}")
            print(f"  Embedding: {model_memory['embedding_params']:,}")
            print(f"  Memory: {model_memory['total_memory_mb']:.2f} MB")
        
        # Create FedAvg server
        server = FedAvgServer(
            global_model=model,
            client_datasets=client_datasets,
            test_dataset=test_dataset,
            config=config,
            pad_index=pad_index,
        )
        
        # Train
        train_results = server.train(
            participation_ratio=c,
            num_rounds=config.num_rounds,
            verbose=True,
        )
        
        results['participation_results'][c] = {
            'final_acc': train_results['final_acc'],
            'best_acc': train_results['best_acc'],
            'history': train_results['history'],
        }
        
        print(f"  Final Accuracy: {train_results['final_acc']:.4f}")
        print(f"  Best Accuracy: {train_results['best_acc']:.4f}")
        
        # Clean up
        del model, server
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return results


def run_all_experiments(config: FLConfig) -> Dict:
    """Run experiments for all embedding types."""
    
    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Prepare data
    client_datasets, test_dataset, vocab, pad_index = prepare_sst2_data(config)
    vocab_size = len(vocab)
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    all_results = {}
    
    for embedding_type in config.embedding_types:
        if embedding_type == 'byte_seb' and not BYTE_SEB_AVAILABLE:
            print(f"\nSkipping {embedding_type} (module not available)")
            continue
        
        results = run_experiment(
            embedding_type=embedding_type,
            participation_ratios=config.participation_ratios,
            config=config,
            client_datasets=client_datasets,
            test_dataset=test_dataset,
            vocab_size=vocab_size,
            pad_index=pad_index,
            output_dir=config.output_dir,
        )
        
        all_results[embedding_type] = results
        
        # Save individual results
        results_file = os.path.join(config.output_dir, f'results_{embedding_type}.json')
        with open(results_file, 'w') as f:
            # Convert history lists to be JSON serializable
            save_results = copy.deepcopy(results)
            json.dump(save_results, f, indent=2, default=str)
    
    # Print Table 7
    print_table7(all_results, config.participation_ratios)
    
    # Save all results
    all_results_file = os.path.join(config.output_dir, 'all_results.json')
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    return all_results


def print_table7(results: Dict, participation_ratios: List[float]):
    """Print results in Table 7 format from the paper."""
    
    print("\n")
    print("=" * 90)
    print("TABLE 7: Accuracies for Federated Learning on SST2")
    print("c is the participation ratio in each communication round")
    print("=" * 90)
    print()
    
    # Header
    header = f"{'Embeddings':<15}"
    for c in participation_ratios:
        header += f"{'c='+str(c):<12}"
    header += f"{'Avg':<12}"
    print(header)
    print("-" * 90)
    
    # Results for each embedding type
    for embed_type in ['standard', 'byte_seb', 'binary_seb']:
        if embed_type not in results:
            continue
        
        row = f"{embed_type:<15}"
        accs = []
        
        for c in participation_ratios:
            if c in results[embed_type]['participation_results']:
                acc = results[embed_type]['participation_results'][c]['best_acc']
                accs.append(acc)
                row += f"{acc*100:.1f}%{'':<7}"
            else:
                row += f"{'N/A':<12}"
        
        # Average
        if accs:
            avg_acc = sum(accs) / len(accs)
            row += f"{avg_acc*100:.1f}%"
        
        print(row)
    
    print("-" * 90)
    
    # Memory comparison
    print("\nModel Memory Comparison:")
    print("-" * 60)
    print(f"{'Embeddings':<15} {'Total Params':<15} {'Embed Params':<15} {'Memory (MB)':<12}")
    print("-" * 60)
    
    baseline_embed = None
    for embed_type in ['standard', 'byte_seb', 'binary_seb']:
        if embed_type not in results:
            continue
        
        mem = results[embed_type].get('model_memory', {})
        total = mem.get('total_params', 0)
        embed = mem.get('embedding_params', 0)
        memory = mem.get('total_memory_mb', 0)
        
        if embed_type == 'standard':
            baseline_embed = embed
        
        reduction = ""
        if baseline_embed and embed_type != 'standard':
            red = (1 - embed / baseline_embed) * 100
            reduction = f" (↓{red:.0f}%)"
        
        print(f"{embed_type:<15} {total:<15,} {embed:<15,}{reduction}")
    
    print()
    print("=" * 90)
    print("Note: Higher accuracy is better. SEB methods should achieve comparable")
    print("or better accuracy while providing privacy protection.")
    print("=" * 90)


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Federated Learning on SST2 with FedAvg')
    
    # Experiment settings
    parser.add_argument('--embedding_type', type=str, default=None,
                        choices=['standard', 'byte_seb', 'binary_seb'],
                        help='Run single embedding type (default: run all)')
    parser.add_argument('--run_all', action='store_true',
                        help='Run all embedding types')
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test with fewer rounds')
    
    # FL settings
    parser.add_argument('--num_clients', type=int, default=20,
                        help='Number of clients (default: 20 as in paper)')
    parser.add_argument('--participation_ratios', type=float, nargs='+',
                        default=[0.2, 0.4, 0.6, 0.8, 1.0],
                        help='Participation ratios to test')
    parser.add_argument('--num_rounds', type=int, default=50,
                        help='Number of communication rounds')
    parser.add_argument('--local_epochs', type=int, default=1,
                        help='Local training epochs per round')
    parser.add_argument('--local_batch_size', type=int, default=32,
                        help='Local batch size')
    
    # Model settings
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=5e-4)
    
    # SEB settings
    parser.add_argument('--num_bits', type=int, default=64)
    parser.add_argument('--num_bytes', type=int, default=8)
    parser.add_argument('--interdim', type=str, default='128')
    
    # Other
    parser.add_argument('--output_dir', type=str, default='./results_fl_sst2')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick_test:
        args.num_rounds = 10
        args.participation_ratios = [0.4, 1.0]
    
    # Create config
    config = FLConfig(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_bytes=args.num_bytes,
        num_bits=args.num_bits,
        interdim=args.interdim,
        num_clients=args.num_clients,
        participation_ratios=args.participation_ratios,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        local_batch_size=args.local_batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    # Determine which embedding types to run
    if args.embedding_type:
        config.embedding_types = [args.embedding_type]
    elif not args.run_all and not args.quick_test:
        # Default: run all
        config.embedding_types = ['standard', 'byte_seb', 'binary_seb']
    
    # Save config
    os.makedirs(config.output_dir, exist_ok=True)
    config_file = os.path.join(config.output_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # Run experiments
    run_all_experiments(config)


if __name__ == '__main__':
    main()
