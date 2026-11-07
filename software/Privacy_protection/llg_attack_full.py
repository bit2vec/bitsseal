#!/usr/bin/env python3
"""


Key differences from quick version:
- 10K samples (10x more data)
- 10 epochs (more training for model convergence)
- 128 sequence length (longer context)
- Attack at different training stages (untrained, early, mid, late)
- Proper parameter tuning
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
from collections import Counter
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binary_subword_embedding import BinarySubwordEmbedding


class TextClassifier(nn.Module):
    """Text classifier for LLG attack - proper architecture for full experiment"""
    def __init__(self, vocab_size, embedding_dim, num_classes, embedding_type='standard', num_bits=64):
        super().__init__()
        self.embedding_type = embedding_type

        if embedding_type == 'binary_seb':
            self.embedding = BinarySubwordEmbedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                num_bits=num_bits,
                interdim='128',
                relu_dropout=0.1,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Proper architecture with dropout
        self.lstm = nn.LSTM(embedding_dim, 256, num_layers=2, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        lstm_out, (hidden, _) = self.lstm(embedded)  # hidden: [2, batch, 256]
        hidden = hidden[-1]  # Take last layer: [batch, 256]
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)  # [batch, num_classes]
        return logits


def extract_labels_llg(gradients, batch_size, num_classes):
    """
    LLG Attack Algorithm (from paper Section 5.2)
    Same as quick version - proven to work
    """
    # Sum gradients across hidden dimension to get g_i for each class
    g = gradients.sum(dim=1).cpu().numpy()  # [num_classes]

    extracted = []

    # Step 1: Extract labels based on negative gradients (Property 1)
    impact = 0.0
    for i in range(num_classes):
        if g[i] < 0:
            extracted.append(i)
            impact += g[i]

    if len(extracted) > 0:
        impact = impact / batch_size * (1 + 1/num_classes)
    else:
        impact = g[g < 0].mean() / batch_size if (g < 0).any() else -0.01

    # Step 2: Calibrate gradients
    g_calibrated = g.copy()

    # Subtract impact for already extracted labels
    for label in extracted:
        g_calibrated[label] -= impact

    # Step 3: Extract remaining labels until we reach batch_size
    while len(extracted) < batch_size:
        min_idx = np.argmin(g_calibrated)
        extracted.append(min_idx)
        g_calibrated[min_idx] -= impact

    return extracted


def evaluate_attack_accuracy(predicted_labels, true_labels):
    """Calculate Attack Success Rate (ASR) metric"""
    pred_counter = Counter(predicted_labels)
    true_counter = Counter(true_labels)

    correct = 0
    total = len(predicted_labels)

    for label in pred_counter:
        correct += min(pred_counter[label], true_counter[label])

    asr = (correct / total * 100) if total > 0 else 0.0
    return asr


def run_full_llg_attack(
    embedding_type='binary_seb',
    num_bits=64,
    batch_size=16,  # Larger batch for full experiment
    num_samples=10000,  # 10K samples
    max_length=128,
    learning_rate=5e-5,  # Lower LR for stability
    epochs=10,  # Full training
    num_attack_batches=200,  # Attack more batches
    attack_frequency=2,  # Attack every N epochs
    device=None
):
    """
    Run FULL LLG attack experiment

    Attacks at different training stages to see how model convergence affects privacy:
    - Epoch 0 (untrained): Most vulnerable per paper
    - Epoch 2, 4, 6, 8 (during training): Decreasing vulnerability
    - Epoch 10 (trained): Least vulnerable but still leaks
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*80}")
    print(f"FULL LLG ATTACK ON {embedding_type.upper()}")
    print(f"{'='*80}")
    print(f"Embedding type: {embedding_type}")
    print(f"Training samples: {num_samples:,}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Attack batches per stage: {num_attack_batches}")
    print(f"Attack frequency: Every {attack_frequency} epochs")
    print(f"{'='*80}\n")

    # Load IMDB dataset
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Loading IMDB dataset...")
    dataset = load_dataset('imdb', split='train')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Use 10K samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset = Subset(dataset, indices)

    # Prepare data
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch])

        encoded = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        return encoded['input_ids'], labels

    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Create model
    vocab_size = tokenizer.vocab_size
    embedding_dim = 256  # Larger for full experiment
    num_classes = 2

    print(f"Creating model...")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Embedding dim: {embedding_dim}")
    print(f"   Num classes: {num_classes}")

    model = TextClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        embedding_type=embedding_type,
        num_bits=num_bits
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training phase with periodic attacks
    print(f"\n{'='*80}")
    print(f"Training model with periodic LLG attacks")
    print(f"{'='*80}\n")

    all_attack_results = []
    epoch_summaries = []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        batches_processed = 0

        # Determine if we should attack this epoch
        should_attack = (epoch % attack_frequency == 0) or (epoch == epochs - 1)
        attacks_this_epoch = 0

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()

            # ATTACK: Extract gradients of last layer
            if should_attack and batch_idx < num_attack_batches:
                last_layer_gradients = model.fc.weight.grad.clone()  # [num_classes, hidden_dim]

                with torch.no_grad():
                    predicted_labels = extract_labels_llg(
                        last_layer_gradients,
                        batch_size=len(labels),
                        num_classes=num_classes
                    )

                    true_labels = labels.cpu().numpy().tolist()
                    asr = evaluate_attack_accuracy(predicted_labels, true_labels)

                    all_attack_results.append({
                        'epoch': int(epoch),
                        'batch': int(batch_idx),
                        'asr': float(asr),
                        'predicted': [int(x) for x in predicted_labels],
                        'true': [int(x) for x in true_labels]
                    })
                    attacks_this_epoch += 1

            # Update model
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            epoch_acc += (preds == labels).float().mean().item()
            batches_processed += 1

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Attacks: {attacks_this_epoch}")

        avg_loss = epoch_loss / batches_processed
        avg_acc = epoch_acc / batches_processed * 100

        # Calculate ASR for this epoch if we attacked
        epoch_asrs = [r['asr'] for r in all_attack_results if r['epoch'] == epoch]
        epoch_avg_asr = np.mean(epoch_asrs) if epoch_asrs else 0.0

        epoch_summary = {
            'epoch': int(epoch),
            'loss': float(avg_loss),
            'accuracy': float(avg_acc),
            'num_attacks': attacks_this_epoch,
            'avg_asr': float(epoch_avg_asr) if attacks_this_epoch > 0 else None
        }
        epoch_summaries.append(epoch_summary)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   Accuracy: {avg_acc:.2f}%")
        if attacks_this_epoch > 0:
            print(f"   Attacks performed: {attacks_this_epoch}")
            print(f"   Average ASR: {epoch_avg_asr:.2f}%")
        print()

    training_time = time.time() - start_time

    # Analyze attack results by training stage
    print(f"\n{'='*80}")
    print(f"LLG ATTACK RESULTS BY TRAINING STAGE")
    print(f"{'='*80}\n")

    # Group attacks by epoch
    for epoch_num in sorted(set(r['epoch'] for r in all_attack_results)):
        epoch_attacks = [r for r in all_attack_results if r['epoch'] == epoch_num]
        asrs = [r['asr'] for r in epoch_attacks]

        print(f"Epoch {epoch_num} ({'Untrained' if epoch_num == 0 else 'Training' if epoch_num < epochs-1 else 'Trained'}):")
        print(f"   Average ASR: {np.mean(asrs):.2f}%")
        print(f"   Std Dev: {np.std(asrs):.2f}%")
        print(f"   Min ASR: {np.min(asrs):.2f}%")
        print(f"   Max ASR: {np.max(asrs):.2f}%")
        print(f"   Num attacks: {len(epoch_attacks)}")
        print()

    # Overall statistics
    all_asrs = [r['asr'] for r in all_attack_results]
    avg_asr = np.mean(all_asrs)
    std_asr = np.std(all_asrs)

    print(f"Overall Statistics:")
    print(f"   Average ASR (all stages): {avg_asr:.2f}%")
    print(f"   Std Dev: {std_asr:.2f}%")
    print(f"   Total attacks: {len(all_attack_results)}")
    print(f"   Training time: {training_time/3600:.2f} hours")

    # Save results
    results = {
        'config': {
            'embedding_type': embedding_type,
            'num_bits': num_bits,
            'batch_size': batch_size,
            'num_samples': num_samples,
            'max_length': max_length,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'num_attack_batches': num_attack_batches,
            'attack_frequency': attack_frequency,
        },
        'summary': {
            'avg_asr_overall': float(avg_asr),
            'std_asr_overall': float(std_asr),
            'min_asr': float(np.min(all_asrs)),
            'max_asr': float(np.max(all_asrs)),
            'total_attacks': len(all_attack_results),
            'training_time_hours': training_time / 3600
        },
        'epoch_summaries': epoch_summaries,
        'attack_results': all_attack_results  # Detailed per-batch results
    }

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Full LLG attack on Binary SEB')
    parser.add_argument('--samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--attack_batch_sizes', type=int, nargs='+',
                        default=[1, 2, 4, 8, 12, 16, 32],
                        help='Batch sizes to test for LLG attack (Figure 4)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--attack_batches', type=int, default=200, help='Batches to attack per stage')
    parser.add_argument('--output_dir', type=str, default='./results_llg_full')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # Resolve device with safe CUDA fallback
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        torch.zeros(1, device=args.device)
    except (AssertionError, RuntimeError):
        print(f"Warning: device '{args.device}' unavailable, falling back to CPU")
        args.device = 'cpu'

    print("\n" + "="*80)
    print("FULL LLG ATTACK EXPERIMENT")
    print(f"Training on {args.samples:,} samples")
    print(f"Attacking at multiple training stages over {args.epochs} epochs")
    print(f"Estimated time: ~3-4 hours per embedding type, ~6-8 hours total")
    print("="*80 + "\n")

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    # Run for each embedding type at each batch size (matching paper Figure 4)
    for emb_type in ['standard', 'binary_seb']:
        print("\n" + "="*80)
        print(f"EXPERIMENT: {emb_type.upper()}")
        print("="*80)

        emb_results = {}
        for bs in args.attack_batch_sizes:
            print(f"\n--- Batch size: {bs} ---")
            start_time = time.time()

            result = run_full_llg_attack(
                embedding_type=emb_type,
                num_bits=64,
                batch_size=bs,
                num_samples=args.samples,
                epochs=args.epochs,
                num_attack_batches=args.attack_batches,
                device=args.device
            )

            result['training_time_seconds'] = time.time() - start_time
            emb_results[str(bs)] = result

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results[emb_type] = emb_results

        # Save per-type results
        output_file = os.path.join(args.output_dir, f'results_{emb_type}_full.json')
        with open(output_file, 'w') as f:
            json.dump(emb_results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    # Save combined results
    output_file = os.path.join(args.output_dir, 'all_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print Figure 4 style summary: ASR vs batch size
    print("\n" + "="*80)
    print("FIGURE 4: LLG Attack Success Rate (ASR) vs Batch Size")
    print("="*80)
    print(f"\n{'Batch Size':<12} {'Standard ASR':<18} {'Binary SEB ASR':<18}")
    print("-"*50)
    for bs in args.attack_batch_sizes:
        bs_key = str(bs)
        std_asr = all_results.get('standard', {}).get(bs_key, {}).get('summary', {}).get('avg_asr_overall', 0)
        bin_asr = all_results.get('binary_seb', {}).get(bs_key, {}).get('summary', {}).get('avg_asr_overall', 0)
        print(f"{bs:<12} {std_asr:<18.2f} {bin_asr:<18.2f}")
    print("="*80)
