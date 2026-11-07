#!/usr/bin/env python3
"""
Table 11: Neural Compression Projector Variants on SST-2

Compares three projector architectures for Binary SEB (001):
1. Original FFN (2-layer, 128 hidden)
2. MLP-512 (2-layer, 512 hidden)
3. Autoencoder (512->256 bottleneck with reconstruction loss)

Usage:
    python projector_comparison_sst2.py --output_dir ./results_projectors
"""

import functools
import sys
import os
import time
import json
import argparse

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

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

from binary_subword_sentiment import BinarySubwordConcateOnehot


class BinarySubwordMLP512(nn.Module):
    """Binary SEB with MLP-512 projector (Table 11 variant 2)."""

    def __init__(self, num_embeddings, embedding_dim, num_bits=64, relu_dropout=0.1,
                 subword_binary_file='',
                 use_gelu=False, use_layernorm=False, bias_init=0.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_bits = num_bits
        self.binary_vocab_size = 2
        self.use_gelu = use_gelu
        self.use_layernorm = use_layernorm

        if subword_binary_file and os.path.isfile(subword_binary_file):
            subword2binary = torch.load(subword_binary_file)
            if subword2binary.shape[0] < num_embeddings:
                extra = torch.randint(0, 2, (num_embeddings - subword2binary.shape[0], num_bits), dtype=torch.long)
                subword2binary = torch.cat([subword2binary, extra], dim=0)
        else:
            subword2binary = torch.randint(0, 2, (num_embeddings, num_bits), dtype=torch.long)
            if subword_binary_file:
                os.makedirs(os.path.dirname(subword_binary_file) or '.', exist_ok=True)
                torch.save(subword2binary, subword_binary_file)

        self.register_buffer('subword2binary', subword2binary)

        import torch.nn.functional as F
        onehot_binary = F.one_hot(torch.arange(0, 2)).to(torch.float32)
        self.binary_embedding = nn.Embedding(2, 2, _weight=onehot_binary)
        self.binary_embedding.requires_grad_(False)

        input_dim = 2 * num_bits  # 128
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, embedding_dim)
        self.dropout = nn.Dropout(relu_dropout)
        # Hidden layer init: Kaiming normal + small positive bias
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu', mode='fan_out')
        nn.init.constant_(self.fc1.bias, bias_init)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        # Optional output LayerNorm
        self.output_norm = nn.LayerNorm(embedding_dim) if use_layernorm else None

    def forward(self, input, dropout=None):
        import torch.nn.functional as F
        binary_input = self.subword2binary[input]
        embeddings = self.binary_embedding(binary_input)
        shape = embeddings.size()
        if len(shape) == 4:
            embed = embeddings.reshape(shape[0], shape[1], -1)
        else:
            embed = embeddings.reshape(shape[0], -1)
        activation = F.gelu if self.use_gelu else torch.relu
        embed = activation(self.fc1(embed))
        embed = self.dropout(embed)
        embed = self.fc2(embed)
        if self.output_norm is not None:
            embed = self.output_norm(embed)
        return embed


class BinarySubwordAutoencoder(nn.Module):
    """Binary SEB with Autoencoder projector (Table 11 variant 3).
    Encoder: 128 -> 512 -> 256. Decoder reconstructs 128-dim input."""

    def __init__(self, num_embeddings, embedding_dim, num_bits=64, relu_dropout=0.1,
                 subword_binary_file='',
                 use_gelu=False, use_layernorm=False, bias_init=0.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_bits = num_bits
        self.use_gelu = use_gelu
        self.use_layernorm = use_layernorm

        if subword_binary_file and os.path.isfile(subword_binary_file):
            subword2binary = torch.load(subword_binary_file)
            if subword2binary.shape[0] < num_embeddings:
                extra = torch.randint(0, 2, (num_embeddings - subword2binary.shape[0], num_bits), dtype=torch.long)
                subword2binary = torch.cat([subword2binary, extra], dim=0)
        else:
            subword2binary = torch.randint(0, 2, (num_embeddings, num_bits), dtype=torch.long)
            if subword_binary_file:
                os.makedirs(os.path.dirname(subword_binary_file) or '.', exist_ok=True)
                torch.save(subword2binary, subword_binary_file)

        self.register_buffer('subword2binary', subword2binary)

        import torch.nn.functional as F
        onehot_binary = F.one_hot(torch.arange(0, 2)).to(torch.float32)
        self.binary_embedding = nn.Embedding(2, 2, _weight=onehot_binary)
        self.binary_embedding.requires_grad_(False)

        act_layer = nn.GELU() if use_gelu else nn.ReLU()
        input_dim = 2 * num_bits
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            act_layer,
            nn.Dropout(relu_dropout),
            nn.Linear(512, embedding_dim),
        )
        # Decoder (for reconstruction loss during training)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.GELU() if use_gelu else nn.ReLU(),
            nn.Linear(512, input_dim),
        )
        self.reconstruction_loss_weight = 0.1
        # Init: hidden Kaiming + bias_init, output Xavier
        for layer in [self.encoder[0], self.decoder[0]]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu', mode='fan_out')
            nn.init.constant_(layer.bias, bias_init)
        for layer in [self.encoder[3], self.decoder[2]]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.output_norm = nn.LayerNorm(embedding_dim) if use_layernorm else None

    def forward(self, input, dropout=None):
        binary_input = self.subword2binary[input]
        embeddings = self.binary_embedding(binary_input)
        shape = embeddings.size()
        if len(shape) == 4:
            flat = embeddings.reshape(shape[0], shape[1], -1)
        else:
            flat = embeddings.reshape(shape[0], -1)

        encoded = self.encoder(flat)

        if self.training:
            decoded = self.decoder(encoded)
            self._reconstruction_target = flat.detach()
            self._reconstruction_output = decoded

        if self.output_norm is not None:
            encoded = self.output_norm(encoded)
        return encoded

    def get_reconstruction_loss(self):
        if hasattr(self, '_reconstruction_output') and hasattr(self, '_reconstruction_target'):
            return nn.functional.mse_loss(self._reconstruction_output, self._reconstruction_target)
        return torch.tensor(0.0)


class LSTM(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, output_dim, n_layers,
                 dropout_rate, pad_index):
        super().__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=True,
                            dropout=dropout_rate if n_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        embedded = self.dropout(self.embedding(ids))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, length.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        return self.fc(hidden)


def collate_fn(batch, pad_index):
    # Drop zero-length samples (they can't go through pack_padded_sequence and
    # their presence in a batch used to trigger a dummy all-zero sample that
    # polluted the reported accuracy).
    batch = [item for item in batch if item['length'] > 0]
    if len(batch) == 0:
        # Return None; caller skips None batches when evaluating / training.
        return None
    batch_ids = [item['ids'] for item in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_length = torch.stack([item['length'] for item in batch])
    batch_label = torch.stack([item['label'] for item in batch]).long()
    return {'ids': batch_ids, 'length': batch_length, 'label': batch_label}


def count_correct(prediction, label):
    """Return (num_correct, num_total) for this batch so the caller can
    aggregate with correct per-sample weighting instead of per-batch mean."""
    preds = prediction.argmax(dim=-1)
    return preds.eq(label).sum().item(), label.numel()


def run_projector_experiment(projector_name, embedding_module, train_loader, valid_loader, test_loader,
                             embedding_dim, hidden_dim, output_dim, n_layers, dropout_rate,
                             pad_index, n_epochs, lr, device, use_cosine_lr=False):
    import math
    model = LSTM(embedding_module, embedding_dim, hidden_dim, output_dim, n_layers,
                 dropout_rate, pad_index).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Optional cosine LR schedule with brief linear warmup. Same compute,
    # often better convergence in late training.
    scheduler = None
    if use_cosine_lr:
        total_steps = n_epochs * len(train_loader)
        warmup_steps = min(500, max(1, total_steps // 20))
        def _lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    best_valid_acc = 0
    best_model_state = None
    epoch_times = []

    for epoch in range(n_epochs):
        model.train()
        start = time.time()
        for batch in train_loader:
            if batch is None:
                continue
            ids = batch['ids'].to(device)
            length = batch['length']
            label = batch['label'].to(device)
            optimizer.zero_grad()
            pred = model(ids, length)
            loss = criterion(pred, label)
            if hasattr(embedding_module, 'get_reconstruction_loss'):
                loss = loss + embedding_module.reconstruction_loss_weight * embedding_module.get_reconstruction_loss().to(device)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        epoch_times.append(time.time() - start)

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for batch in valid_loader:
                if batch is None:
                    continue
                ids = batch['ids'].to(device)
                label = batch['label'].to(device)
                length = batch['length']
                pred = model(ids, length)
                c, t = count_correct(pred, label)
                v_correct += c
                v_total += t
        valid_acc = v_correct / max(v_total, 1)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            import copy
            best_model_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            print(f"  [{projector_name}] Epoch {epoch+1}: Valid Acc = {valid_acc:.4f}")

    # Reload best model for test evaluation (paper B.4: "best model parameters
    # evaluated on validation data are applied for testing")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    t_correct, t_total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            ids = batch['ids'].to(device)
            label = batch['label'].to(device)
            length = batch['length']
            pred = model(ids, length)
            c, t = count_correct(pred, label)
            t_correct += c
            t_total += t
    test_acc = t_correct / max(t_total, 1)

    total_time = sum(epoch_times)
    params = sum(p.numel() for p in model.parameters())

    return {
        'projector': projector_name,
        'test_acc': test_acc,
        'best_valid_acc': best_valid_acc,
        'total_time': total_time,
        'avg_epoch_time': np.mean(epoch_times),
        'total_params': params,
    }


def main():
    parser = argparse.ArgumentParser(description='Table 11: Projector Comparison on SST-2')
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--num_bits', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--dropout_rate', type=float, default=0.4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default='./results_projectors')
    # Binary-SEB-only implementation tweaks (off by default)
    parser.add_argument('--use_gelu', action='store_true', help='GELU instead of ReLU in FFN')
    parser.add_argument('--use_layernorm', action='store_true', help='LayerNorm after FFN output')
    parser.add_argument('--bias_init', type=float, default=0.0, help='Bias init for hidden layers (try 0.01)')
    parser.add_argument('--use_cosine_lr', action='store_true', help='Cosine LR schedule with warmup')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load SST-2
    print("Loading SST-2...")
    train_data, test_data = datasets.load_dataset('sst2', split=['train', 'validation'])
    tokenizer = get_tokenizer('basic_english')

    def tokenize_data(example):
        tokens = tokenizer(example['sentence'])[:256]
        if len(tokens) == 0:
            tokens = ['<unk>']
        return {'tokens': tokens, 'length': len(tokens)}

    train_valid = train_data.train_test_split(test_size=0.25, seed=seed)
    train_data = train_valid['train'].map(tokenize_data)
    valid_data = train_valid['test'].map(tokenize_data)
    test_data = test_data.map(tokenize_data)

    vocab = build_vocab_from_iterator(train_data['tokens'], min_freq=5, specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    pad_index = vocab['<pad>']
    vocab_size = len(vocab)

    def numericalize(example):
        return {'ids': [vocab[t] for t in example['tokens']]}

    train_data = train_data.map(numericalize).with_format('torch', columns=['ids', 'label', 'length'])
    valid_data = valid_data.map(numericalize).with_format('torch', columns=['ids', 'label', 'length'])
    test_data = test_data.map(numericalize).with_format('torch', columns=['ids', 'label', 'length'])

    collate = functools.partial(collate_fn, pad_index=pad_index)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, collate_fn=collate, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate)

    os.makedirs(args.output_dir, exist_ok=True)

    # Pull binary-SEB-only knobs from CLI (default off so behaviour is unchanged
    # unless explicitly enabled). All four are pure implementation tweaks; no
    # change to the algorithm or architecture sizes.
    seb_kwargs = dict(
        use_gelu=getattr(args, 'use_gelu', False),
        use_layernorm=getattr(args, 'use_layernorm', False),
        bias_init=getattr(args, 'bias_init', 0.0),
    )

    projectors = {
        'Original FFN': BinarySubwordConcateOnehot(
            num_embeddings=vocab_size, embedding_dim=args.embedding_dim,
            num_bits=args.num_bits, interdim='128', relu_dropout=args.dropout_rate,
            subword_binary_file=os.path.join(args.output_dir, 'binary_mapping_ffn.pt'),
            **seb_kwargs,
        ),
        'MLP-512': BinarySubwordMLP512(
            num_embeddings=vocab_size, embedding_dim=args.embedding_dim,
            num_bits=args.num_bits, relu_dropout=args.dropout_rate,
            subword_binary_file=os.path.join(args.output_dir, 'binary_mapping_mlp512.pt'),
            **seb_kwargs,
        ),
        'Autoencoder': BinarySubwordAutoencoder(
            num_embeddings=vocab_size, embedding_dim=args.embedding_dim,
            num_bits=args.num_bits, relu_dropout=args.dropout_rate,
            subword_binary_file=os.path.join(args.output_dir, 'binary_mapping_ae.pt'),
            **seb_kwargs,
        ),
    }

    all_results = {}
    for name, embedding in projectors.items():
        print(f"\n{'='*60}")
        print(f"Projector: {name}")
        print(f"{'='*60}")
        result = run_projector_experiment(
            name, embedding, train_loader, valid_loader, test_loader,
            args.embedding_dim, args.hidden_dim, 2, 2, args.dropout_rate,
            pad_index, args.n_epochs, args.lr, device,
            use_cosine_lr=getattr(args, 'use_cosine_lr', False),
        )
        all_results[name] = result
        print(f"  Test Acc: {result['test_acc']:.4f}, Valid Acc: {result['best_valid_acc']:.4f}, Time: {result['total_time']:.0f}s")

    # Print Table 11
    print("\n" + "="*70)
    print("TABLE 11: 001 with different neural compression projectors on SST-2")
    print("="*70)
    print(f"{'Projector':<20} {'Test Acc':<12} {'Valid Acc':<12} {'Time':<10}")
    print("-"*55)
    for name, r in all_results.items():
        print(f"{name:<20} {r['test_acc']*100:<12.2f}% {r['best_valid_acc']*100:<12.2f}% {r['total_time']:<10.0f}s")

    with open(os.path.join(args.output_dir, 'table11_projector_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    main()
