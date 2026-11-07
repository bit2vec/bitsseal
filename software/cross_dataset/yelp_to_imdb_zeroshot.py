"""
Zero-Shot Cross-Dataset Test: Yelp → IMDB
Pretrain on Yelp reviews , test directly on IMDB without fine-tuning
"""

import math
import torch
import torch.nn as nn
import datasets
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from tqdm import tqdm
import json
import os
import random
import numpy as np
from dataclasses import dataclass
import argparse

# Import Binary SEB modules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary_subword_sentiment import BinarySubwordConcateOnehot
from byte_subword_concate_onehot import ByteSubwordConcateOnehot


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Config:
    # Dataset settings
    yelp_train_samples: int = 100000  # Use only 100K samples
    max_length: int = 256
    min_freq: int = 5

    # Model architecture
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3

    # Binary SEB settings
    num_bits: int = 64
    interdim: int = 512

    # Pretraining on Yelp
    pretrain_epochs: int = 30  # Increased from 5: binary SEB FFN needs ~30 epochs
                                # to fully converge (matches IMDB sentiment benchmark).
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5  # AdamW weight decay for better generalization
    grad_clip_norm: float = 1.0  # Gradient clipping for stability
    warmup_steps: int = 500  # Linear LR warmup steps

    # Other settings
    embedding_types: list = None
    output_dir: str = './results_yelp_to_imdb'
    checkpoint_dir: str = './checkpoints_yelp_to_imdb'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42

    def __post_init__(self):
        if self.embedding_types is None:
            self.embedding_types = ['standard', 'byte_seb', 'binary_seb']


class ReviewDataset(Dataset):
    """Dataset for reviews with variable length."""

    def __init__(self, texts, labels, max_length, pad_index):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.pad_index = pad_index

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx][:self.max_length]
        label = self.labels[idx]
        length = len(text)

        # Pad to max_length
        padded_text = text + [self.pad_index] * (self.max_length - length)

        return {
            'text': torch.LongTensor(padded_text),
            'label': torch.LongTensor([label]),
            'length': torch.LongTensor([length])
        }


def build_vocab(texts, min_freq=5):
    """Build vocabulary from texts."""
    counter = Counter()
    for text in texts:
        counter.update(text)

    # Add special tokens
    vocab = {'<pad>': 0, '<unk>': 1}

    # Add words that meet min_freq threshold
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def tokenize_text(text, vocab_for_tokenization=None):
    """Tokenize text using simple whitespace splitting."""
    tokens = text.lower().split()
    if vocab_for_tokenization is None:
        return tokens
    return [vocab_for_tokenization.get(token, vocab_for_tokenization.get('<unk>', 1)) for token in tokens]


def load_yelp_data(config):
    """Load and preprocess Yelp dataset (100K samples)."""
    print("\n" + "=" * 60)
    print("LOADING YELP DATASET (Pretraining - 100K samples)")
    print("=" * 60)

    # Load only 100K training samples and 10K test samples
    train_data = datasets.load_dataset('yelp_polarity', split=f'train[:{config.yelp_train_samples}]')
    test_data = datasets.load_dataset('yelp_polarity', split='test[:10000]')

    print(f"Loaded {len(train_data)} training samples")
    print(f"Loaded {len(test_data)} test samples")

    # Tokenize
    print("Tokenizing Yelp data...")
    train_texts = [tokenize_text(example['text']) for example in tqdm(train_data, desc="Tokenizing train")]
    test_texts = [tokenize_text(example['text']) for example in tqdm(test_data, desc="Tokenizing test")]

    # Build vocabulary
    print("Building vocabulary from Yelp...")
    vocab = build_vocab(train_texts, min_freq=config.min_freq)
    print(f"Vocabulary size: {len(vocab)}")

    # Convert to indices
    train_texts = [[vocab.get(token, vocab['<unk>']) for token in text] for text in train_texts]
    test_texts = [[vocab.get(token, vocab['<unk>']) for token in text] for text in test_texts]

    train_labels = [example['label'] for example in train_data]
    test_labels = [example['label'] for example in test_data]

    print(f"Yelp Train: {len(train_texts)} samples")
    print(f"Yelp Test:  {len(test_texts)} samples")

    return train_texts, train_labels, test_texts, test_labels, vocab


def load_imdb_data(config, vocab):
    """Load and preprocess IMDB dataset using Yelp vocabulary."""
    print("\n" + "=" * 60)
    print("LOADING IMDB DATASET (Zero-Shot Test)")
    print("=" * 60)

    raw_datasets = datasets.load_dataset('imdb')

    train_data = raw_datasets['train']
    test_data = raw_datasets['test']

    # Tokenize
    print("Tokenizing IMDB data with Yelp vocabulary...")
    train_texts = [tokenize_text(example['text'], vocab) for example in tqdm(train_data, desc="Tokenizing train")]
    test_texts = [tokenize_text(example['text'], vocab) for example in tqdm(test_data, desc="Tokenizing test")]

    train_labels = [example['label'] for example in train_data]
    test_labels = [example['label'] for example in test_data]

    print(f"IMDB Train: {len(train_texts)} samples")
    print(f"IMDB Test: {len(test_texts)} samples")

    # Calculate OOV rate
    total_tokens = sum(len(text) for text in train_texts + test_texts)
    oov_tokens = sum(text.count(vocab['<unk>']) for text in train_texts + test_texts)
    oov_rate = (oov_tokens / total_tokens) * 100
    print(f"OOV Rate: {oov_rate:.2f}%")

    return train_texts, train_labels, test_texts, test_labels


class SentimentClassifier(nn.Module):
    """Sentiment classifier with different embedding types."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 dropout, embedding_type='standard', num_bits=64, interdim=128, pad_idx=0, checkpoint_dir='./checkpoints_yelp_to_imdb'):
        super().__init__()

        self.embedding_type = embedding_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Create embedding layer based on type
        if embedding_type == 'standard':
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        elif embedding_type == 'byte_seb':
            self.embedding = ByteSubwordConcateOnehot(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                interdim=interdim,
                relu_dropout=dropout,
                subword_bytes_file=os.path.join(checkpoint_dir, 'byte_seb_subword_bytes.pt'),
                embeddings_file=os.path.join(checkpoint_dir, 'byte_seb_embeddings.pt')
            )
        elif embedding_type == 'binary_seb':
            self.embedding = BinarySubwordConcateOnehot(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                num_bits=num_bits,
                interdim=interdim,
                relu_dropout=dropout
            )
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        # LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x, lengths):
        # Embed
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]

        # Pack sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed)

        # Use final hidden state
        hidden = hidden[-1]  # [batch, hidden_dim]

        # Classify
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)  # [batch, 2]

        return logits

    def count_parameters(self):
        """Count trainable parameters."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embedding = sum(p.numel() for p in self.embedding.parameters() if p.requires_grad)
        return total, embedding


def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        text = batch['text'].to(device)
        labels = batch['label'].squeeze(1).to(device)
        lengths = batch['length'].squeeze(1)

        # Forward
        optimizer.zero_grad()
        logits = model(text, lengths)
        loss = criterion(logits, labels)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
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
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            text = batch['text'].to(device)
            labels = batch['label'].squeeze(1).to(device)
            lengths = batch['length'].squeeze(1)

            logits = model(text, lengths)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def run_zero_shot_experiment(config):
    """Run the zero-shot cross-dataset test."""

    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Load Yelp data
    yelp_train_texts, yelp_train_labels, yelp_test_texts, yelp_test_labels, vocab = load_yelp_data(config)

    # Load IMDB data with Yelp vocabulary
    imdb_train_texts, imdb_train_labels, imdb_test_texts, imdb_test_labels = load_imdb_data(config, vocab)

    vocab_size = len(vocab)
    pad_idx = vocab['<pad>']

    results = {
        'config': {
            'yelp_train_samples': config.yelp_train_samples,
            'max_length': config.max_length,
            'vocab_size': vocab_size,
            'embedding_dim': config.embedding_dim,
            'hidden_dim': config.hidden_dim,
            'pretrain_epochs': config.pretrain_epochs,
        },
        'pretrain_results': {},
        'zero_shot_results': {}
    }

    criterion = nn.CrossEntropyLoss()

    # Run for each embedding type
    for embedding_type in config.embedding_types:
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {embedding_type.upper()}")
        print("=" * 80)

        # Create model
        model = SentimentClassifier(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            embedding_type=embedding_type,
            num_bits=config.num_bits,
            interdim=config.interdim,
            pad_idx=pad_idx,
            checkpoint_dir=config.checkpoint_dir
        ).to(config.device)

        total_params, embed_params = model.count_parameters()
        print(f"Total parameters: {total_params:,}")
        print(f"Embedding parameters: {embed_params:,}")

        # ===== PHASE 1: PRETRAIN ON YELP =====
        print("\n" + "-" * 60)
        print("PHASE 1: PRETRAINING ON YELP (100K samples)")
        print("-" * 60)

        # Create Yelp datasets
        yelp_train_dataset = ReviewDataset(yelp_train_texts, yelp_train_labels, config.max_length, pad_idx)
        yelp_test_dataset = ReviewDataset(yelp_test_texts, yelp_test_labels, config.max_length, pad_idx)

        yelp_train_loader = DataLoader(yelp_train_dataset, batch_size=config.batch_size, shuffle=True)
        yelp_test_loader = DataLoader(yelp_test_dataset, batch_size=config.batch_size, shuffle=False)

        # AdamW with weight decay for better generalization
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=getattr(config, 'weight_decay', 1e-5),
        )

        # Linear warmup -> cosine decay scheduler. Warmup helps binary SEB
        # avoid early instability from the random FFN init; cosine decay
        # gives smoother late-training convergence.
        total_steps = config.pretrain_epochs * len(yelp_train_loader)
        warmup_steps = min(getattr(config, 'warmup_steps', 500), total_steps // 10)

        def _lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

        pretrain_history = []
        best_pretrain_acc = 0

        for epoch in range(config.pretrain_epochs):
            train_loss, train_acc = train_epoch(model, yelp_train_loader, optimizer, criterion, config.device, scheduler)
            test_loss, test_acc = evaluate(model, yelp_test_loader, criterion, config.device)

            pretrain_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc
            })

            if test_acc > best_pretrain_acc:
                best_pretrain_acc = test_acc
                # Save checkpoint
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'test_acc': test_acc
                }, os.path.join(config.checkpoint_dir, f'{embedding_type}_pretrained.pt'))

            print(f"Epoch {epoch+1}/{config.pretrain_epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

        results['pretrain_results'][embedding_type] = {
            'total_params': total_params,
            'embed_params': embed_params,
            'best_test_acc': best_pretrain_acc,
            'history': pretrain_history
        }

        # ===== PHASE 2: ZERO-SHOT TEST ON IMDB =====
        print("\n" + "-" * 60)
        print("PHASE 2: ZERO-SHOT TEST ON IMDB")
        print("-" * 60)

        # Load best pretrained checkpoint
        checkpoint = torch.load(os.path.join(config.checkpoint_dir, f'{embedding_type}_pretrained.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])

        # Create IMDB datasets
        imdb_train_dataset = ReviewDataset(imdb_train_texts, imdb_train_labels, config.max_length, pad_idx)
        imdb_test_dataset = ReviewDataset(imdb_test_texts, imdb_test_labels, config.max_length, pad_idx)

        imdb_train_loader = DataLoader(imdb_train_dataset, batch_size=config.batch_size, shuffle=False)
        imdb_test_loader = DataLoader(imdb_test_dataset, batch_size=config.batch_size, shuffle=False)

        # Evaluate on IMDB (zero-shot)
        print("Evaluating on IMDB train set (zero-shot)...")
        train_loss, train_acc = evaluate(model, imdb_train_loader, criterion, config.device)
        print(f"IMDB Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} ({train_acc*100:.2f}%)")

        print("\nEvaluating on IMDB test set (zero-shot)...")
        test_loss, test_acc = evaluate(model, imdb_test_loader, criterion, config.device)
        print(f"IMDB Test: Loss={test_loss:.4f}, Acc={test_acc:.4f} ({test_acc*100:.2f}%)")

        results['zero_shot_results'][embedding_type] = {
            'imdb_train_acc': train_acc,
            'imdb_train_loss': train_loss,
            'imdb_test_acc': test_acc,
            'imdb_test_loss': test_loss
        }

        print(f"\n{embedding_type.upper()} completed:")
        print(f"  Yelp Test Acc (pretrained):  {best_pretrain_acc:.4f}")
        print(f"  IMDB Train Acc (zero-shot):  {train_acc:.4f}")
        print(f"  IMDB Test Acc (zero-shot):   {test_acc:.4f}")

    # Save results
    results_path = os.path.join(config.output_dir, 'yelp_to_imdb_zeroshot_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("ZERO-SHOT EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"Results saved to: {results_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: ZERO-SHOT TRANSFER (YELP → IMDB)")
    print("=" * 80)
    print(f"{'Embedding':<15} {'Yelp Test':<15} {'IMDB Train':<15} {'IMDB Test':<15} {'Params':<15}")
    print("-" * 80)
    for emb_type in config.embedding_types:
        yelp_acc = results['pretrain_results'][emb_type]['best_test_acc']
        imdb_train_acc = results['zero_shot_results'][emb_type]['imdb_train_acc']
        imdb_test_acc = results['zero_shot_results'][emb_type]['imdb_test_acc']
        params = results['pretrain_results'][emb_type]['total_params']
        print(f"{emb_type:<15} {yelp_acc:>6.2%}         {imdb_train_acc:>6.2%}         {imdb_test_acc:>6.2%}         {params:>12,}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-Shot Cross-Dataset Test: Yelp → IMDB')
    parser.add_argument('--embedding_types', nargs='+', default=['standard', 'byte_seb', 'binary_seb'])
    parser.add_argument('--yelp_train_samples', type=int, default=100000)
    parser.add_argument('--pretrain_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='./results_yelp_to_imdb')

    args = parser.parse_args()

    config = Config(
        embedding_types=args.embedding_types,
        yelp_train_samples=args.yelp_train_samples,
        pretrain_epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )

    run_zero_shot_experiment(config)
