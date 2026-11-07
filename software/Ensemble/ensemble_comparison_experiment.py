#!/usr/bin/env python3
"""
Ensemble Binary SEB Comparison Experiment

Compares three approaches on SST-2 sentiment analysis:
1. Baseline: Single Binary SEB model
2. Independent Ensemble: 10 independent Binary SEB models (different seeds)
3. Multi-Head: Shared embedding with 10 projection heads

"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ensemble_binary_seb import EnsembleBinaryEmbedding
from binary_subword_sentiment import BinarySubwordConcateOnehot


class LSTMSentimentClassifier(nn.Module):
    """LSTM-based sentiment classifier."""

    def __init__(
        self,
        embedding: nn.Module,
        embedding_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.embedding = embedding
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x, lengths=None):
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embedding_dim]

        # Pack if lengths provided
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)

        # Use last hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        hidden = self.dropout(hidden)
        logits = self.fc(hidden)

        return logits


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        lengths = batch.get('length', None)

        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            lengths = batch.get('length', None)

            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def run_experiment(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    vocab_size: int,
    device: torch.device,
    config: dict,
) -> dict:
    """
    Run training experiment for one model type.

    Args:
        model_type: 'baseline', 'independent', or 'multihead'
        train_loader: Training data
        val_loader: Validation data
        test_loader: Test data
        vocab_size: Vocabulary size
        device: torch device
        config: Experiment configuration

    Returns:
        Dictionary with results
    """

    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {model_type.upper()}")
    print(f"{'='*80}\n")

    # Create embedding
    base_embed_dim = config['embedding_dim']
    num_models = config['num_models']
    seb_kwargs = dict(
        use_gelu=config.get('use_gelu', False),
        use_layernorm=config.get('use_layernorm', False),
        bias_init=config.get('bias_init', 0.0),
    )

    if model_type == 'baseline':
        embedding = BinarySubwordConcateOnehot(
            num_embeddings=vocab_size,
            embedding_dim=base_embed_dim,
            num_bits=config['num_bits'],
            interdim=config['interdim'],
            relu_dropout=config['relu_dropout'],
            **seb_kwargs,
        ).to(device)
        total_embed_dim = base_embed_dim

    elif model_type == 'independent':
        embedding = EnsembleBinaryEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=base_embed_dim,
            ensemble_type='independent',
            num_bits=config['num_bits'],
            interdim=config['interdim'],
            relu_dropout=config['relu_dropout'],
            num_models=num_models,
            base_seed=config['seed'],
            **seb_kwargs,
        ).to(device)
        # Algorithm 3: mean pool → same dim as single model
        total_embed_dim = base_embed_dim

    elif model_type == 'multihead':
        embedding = EnsembleBinaryEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=base_embed_dim,
            ensemble_type='multihead',
            num_bits=config['num_bits'],
            interdim=config['interdim'],
            relu_dropout=config['relu_dropout'],
            num_models=num_models,
            base_seed=config['seed'],
        ).to(device)
        total_embed_dim = base_embed_dim * num_models

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Create classifier
    model = LSTMSentimentClassifier(
        embedding=embedding,
        embedding_dim=total_embed_dim,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional'],
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Optional cosine LR with linear warmup (binary-SEB-only knob)
    scheduler = None
    if config.get('use_cosine_lr', False):
        import math as _math
        total_steps = config['epochs'] * len(train_loader)
        warmup_steps = min(500, max(1, total_steps // 20))
        def _lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + _math.cos(_math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # Training loop
    best_val_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    print(f"\nTraining for {config['epochs']} epochs...")
    start_time = time.time()

    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scheduler=scheduler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    training_time = time.time() - start_time

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\nTraining completed in {training_time:.2f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    return {
        'model_type': model_type,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'embedding_dim': total_embed_dim,
        'best_val_acc': best_val_acc,
        'final_train_acc': history['train_acc'][-1],
        'test_acc': test_acc,
        'test_loss': test_loss,
        'training_time': training_time,
        'history': history,
    }


def tokenize_data_map(example, tokenizer, max_length):
    """Tokenize a single example (SST-2 format with 'sentence')."""
    text = example.get('sentence') or example.get('text', '')
    tokens = tokenizer(text)[:max_length]
    length = len(tokens)
    return {'tokens': tokens, 'length': length}


def numericalize_data_map(example, vocab_obj):
    """Convert tokens to IDs."""
    ids = [vocab_obj[token] for token in example['tokens']]
    return {'ids': ids}


def collate_batch(batch, pad_index):
    """Collate batch for DataLoader."""
    batch_ids = [torch.tensor(i['ids'], dtype=torch.long) for i in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_length = torch.tensor([i['length'] for i in batch], dtype=torch.long)
    batch_label = torch.tensor([i['label'] for i in batch], dtype=torch.long)
    return {'input_ids': batch_ids, 'length': batch_length, 'label': batch_label}


def load_sst2_data(max_length, min_freq=5, test_size=0.25, seed=42):
    """
    Load SST-2 dataset with proper preprocessing.

    Returns:
        train_data, valid_data, test_data, vocab_obj, pad_index
    """
    print("Loading SST-2 dataset...")

    # Load dataset
    try:
        raw_datasets = datasets.load_dataset('sst2')
        train_data_raw, test_data_raw = raw_datasets['train'], raw_datasets['validation']
    except Exception as e:
        print(f"Trying 'glue', 'sst2'...")
        raw_datasets = datasets.load_dataset('glue', 'sst2')
        train_data_raw, test_data_raw = raw_datasets['train'], raw_datasets['validation']

    print(f"  Train samples: {len(train_data_raw)}")
    print(f"  Test samples: {len(test_data_raw)}")

    # Tokenizer
    tokenizer_func = get_tokenizer('basic_english')

    # Split train into train/valid
    train_valid_data = train_data_raw.train_test_split(test_size=test_size, seed=seed)
    train_data, valid_data = train_valid_data['train'], train_valid_data['test']

    print(f"  Train split: {len(train_data)}")
    print(f"  Valid split: {len(valid_data)}")

    # Tokenize
    fn_kwargs_tokenize = {'tokenizer': tokenizer_func, 'max_length': max_length}
    train_data = train_data.map(tokenize_data_map, fn_kwargs=fn_kwargs_tokenize)
    valid_data = valid_data.map(tokenize_data_map, fn_kwargs=fn_kwargs_tokenize)
    test_data = test_data_raw.map(tokenize_data_map, fn_kwargs=fn_kwargs_tokenize)

    # Build vocabulary
    print("Building vocabulary...")
    special_tokens = ['<unk>', '<pad>']
    vocab_obj = build_vocab_from_iterator(
        train_data['tokens'],
        min_freq=min_freq,
        specials=special_tokens
    )
    pad_index = vocab_obj['<pad>']
    vocab_obj.set_default_index(vocab_obj['<unk>'])
    vocab_size = len(vocab_obj)

    print(f"  Vocabulary size: {vocab_size}")
    print(f"  PAD index: {pad_index}")

    # Numericalize
    fn_kwargs_numericalize = {'vocab_obj': vocab_obj}
    train_data = train_data.map(numericalize_data_map, fn_kwargs=fn_kwargs_numericalize)
    valid_data = valid_data.map(numericalize_data_map, fn_kwargs=fn_kwargs_numericalize)
    test_data = test_data.map(numericalize_data_map, fn_kwargs=fn_kwargs_numericalize)

    # Set format for PyTorch
    train_data.set_format(type='torch', columns=['ids', 'length', 'label'])
    valid_data.set_format(type='torch', columns=['ids', 'length', 'label'])
    test_data.set_format(type='torch', columns=['ids', 'length', 'label'])

    return train_data, valid_data, test_data, vocab_obj, pad_index, vocab_size


def load_imdb_data(max_length, min_freq=5, test_size=0.1, seed=42):
    """
    Load IMDB dataset with proper preprocessing.

    Returns:
        train_data, valid_data, test_data, vocab_obj, pad_index, vocab_size
    """
    print("Loading IMDB dataset...")

    # Load dataset
    raw_datasets = datasets.load_dataset('imdb')
    train_data_raw = raw_datasets['train']
    test_data_raw = raw_datasets['test']

    print(f"  Train samples: {len(train_data_raw)}")
    print(f"  Test samples: {len(test_data_raw)}")

    # Tokenizer
    tokenizer_func = get_tokenizer('basic_english')

    # Split train into train/valid
    train_valid_data = train_data_raw.train_test_split(test_size=test_size, seed=seed)
    train_data, valid_data = train_valid_data['train'], train_valid_data['test']

    print(f"  Train split: {len(train_data)}")
    print(f"  Valid split: {len(valid_data)}")

    # Tokenize
    fn_kwargs_tokenize = {'tokenizer': tokenizer_func, 'max_length': max_length}
    train_data = train_data.map(tokenize_data_map, fn_kwargs=fn_kwargs_tokenize)
    valid_data = valid_data.map(tokenize_data_map, fn_kwargs=fn_kwargs_tokenize)
    test_data = test_data_raw.map(tokenize_data_map, fn_kwargs=fn_kwargs_tokenize)

    # Build vocabulary
    print("Building vocabulary...")
    special_tokens = ['<unk>', '<pad>']
    vocab_obj = build_vocab_from_iterator(
        train_data['tokens'],
        min_freq=min_freq,
        specials=special_tokens
    )
    pad_index = vocab_obj['<pad>']
    vocab_obj.set_default_index(vocab_obj['<unk>'])
    vocab_size = len(vocab_obj)

    print(f"  Vocabulary size: {vocab_size}")
    print(f"  PAD index: {pad_index}")

    # Numericalize
    fn_kwargs_numericalize = {'vocab_obj': vocab_obj}
    train_data = train_data.map(numericalize_data_map, fn_kwargs=fn_kwargs_numericalize)
    valid_data = valid_data.map(numericalize_data_map, fn_kwargs=fn_kwargs_numericalize)
    test_data = test_data.map(numericalize_data_map, fn_kwargs=fn_kwargs_numericalize)

    # Set format for PyTorch
    train_data.set_format(type='torch', columns=['ids', 'length', 'label'])
    valid_data.set_format(type='torch', columns=['ids', 'length', 'label'])
    test_data.set_format(type='torch', columns=['ids', 'length', 'label'])

    return train_data, valid_data, test_data, vocab_obj, pad_index, vocab_size


def main():
    parser = argparse.ArgumentParser(description='Ensemble Binary SEB Comparison')

    # Data
    parser.add_argument('--dataset', type=str, default='sst2', choices=['sst2', 'imdb'])
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--min_freq', type=int, default=5)
    parser.add_argument('--test_size', type=float, default=0.25)

    # Model
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dim PER MODEL (total will be this * num_models)')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_bits', type=int, default=64)
    parser.add_argument('--interdim', type=str, default='128')
    parser.add_argument('--num_models', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--relu_dropout', type=float, default=0.1)
    parser.add_argument('--bidirectional', action='store_true', default=True)

    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)

    # Experiment
    parser.add_argument('--models', nargs='+', default=['baseline', 'independent', 'multihead'],
                        help='Which models to train')
    parser.add_argument('--output_dir', type=str, default='./ensemble_results')

    # Binary-SEB-only implementation tweaks (off by default; algorithm unchanged)
    parser.add_argument('--use_gelu', action='store_true', help='GELU instead of ReLU')
    parser.add_argument('--use_layernorm', action='store_true', help='LayerNorm after FFN output')
    parser.add_argument('--bias_init', type=float, default=0.0, help='Hidden-layer bias init')
    parser.add_argument('--use_cosine_lr', action='store_true', help='Cosine LR + linear warmup')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare config
    config = {
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_classes': 2,
        'num_bits': args.num_bits,
        'interdim': args.interdim,
        'num_models': args.num_models,
        'dropout': args.dropout,
        'relu_dropout': args.relu_dropout,
        'bidirectional': args.bidirectional,
        'epochs': args.epochs,
        'lr': args.lr,
        'seed': args.seed,
        'use_gelu': args.use_gelu,
        'use_layernorm': args.use_layernorm,
        'bias_init': args.bias_init,
        'use_cosine_lr': args.use_cosine_lr,
    }

    print("\n" + "="*80)
    print("ENSEMBLE BINARY SEB COMPARISON EXPERIMENT")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Models to compare: {args.models}")
    print(f"Embedding dim per model: {args.embedding_dim}")
    print(f"Number of models in ensemble: {args.num_models}")
    print(f"Total embedding dim (ensemble): {args.embedding_dim * args.num_models}")
    print(f"Epochs: {args.epochs}")
    print("="*80 + "\n")

    # Load data
    if args.dataset == 'sst2':
        train_data, valid_data, test_data, vocab_obj, pad_index, vocab_size = load_sst2_data(
            max_length=args.max_len,
            min_freq=args.min_freq,
            test_size=args.test_size,
            seed=args.seed,
        )
    elif args.dataset == 'imdb':
        train_data, valid_data, test_data, vocab_obj, pad_index, vocab_size = load_imdb_data(
            max_length=args.max_len,
            min_freq=args.min_freq,
            test_size=args.test_size,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"\nDataset loaded:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Valid: {len(valid_data)} samples")
    print(f"  Test: {len(test_data)} samples")

    # Create collate function with pad_index
    import functools
    collate_fn = functools.partial(collate_batch, pad_index=pad_index)

    # Create DataLoaders
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    # Run experiments
    results = {}

    for model_type in args.models:
        result = run_experiment(
            model_type=model_type,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            vocab_size=vocab_size,
            device=device,
            config=config,
        )
        results[model_type] = result

    # Save results
    output_file = os.path.join(args.output_dir, 'comparison_results.json')

    # Convert history to lists for JSON serialization
    for model_type in results:
        results[model_type]['history'] = {
            k: [float(v) for v in vals]
            for k, vals in results[model_type]['history'].items()
        }

    with open(output_file, 'w') as f:
        json.dump({
            'config': config,
            'args': vars(args),
            'results': results,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Params':<15} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Time (s)':<10}")
    print("-"*80)

    for model_type in args.models:
        if model_type in results:
            r = results[model_type]
            print(f"{model_type:<20} {r['total_params']:<15,} "
                  f"{r['final_train_acc']:<12.4f} {r['best_val_acc']:<12.4f} "
                  f"{r['test_acc']:<12.4f} {r['training_time']:<10.2f}")

    print("="*80)

    # Print insights
    print("\nKEY INSIGHTS:")
    if 'baseline' in results and 'independent' in results:
        baseline_acc = results['baseline']['test_acc']
        independent_acc = results['independent']['test_acc']
        diff = (independent_acc - baseline_acc) * 100
        print(f"  Independent vs Baseline: {diff:+.2f}% test accuracy")

    if 'baseline' in results and 'multihead' in results:
        baseline_acc = results['baseline']['test_acc']
        multihead_acc = results['multihead']['test_acc']
        diff = (multihead_acc - baseline_acc) * 100
        print(f"  Multi-Head vs Baseline: {diff:+.2f}% test accuracy")

    if 'independent' in results and 'multihead' in results:
        ind_params = results['independent']['total_params']
        mh_params = results['multihead']['total_params']
        param_ratio = mh_params / ind_params
        print(f"  Multi-Head uses {param_ratio:.2f}x parameters of Independent")


if __name__ == '__main__':
    main()
