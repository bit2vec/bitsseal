#!/usr/bin/env python3
"""Public-healthcare text classification comparing Subword / EmByte / 001.

Datasets (all public on HuggingFace):
- ade         : ade_corpus_v2 / Ade_corpus_v2_classification (binary ADE sentence classification)
- pubmedqa   : pubmed_qa / pqa_labeled (3-way yes/no/maybe from question + context)
- medmcqa    : openlifescienceai/medmcqa (4-way MCQ; predict correct option index)

Uses the same BiLSTM architecture as the IMDB/SST-2 sentiment experiments and
the same three embedding modules: standard nn.Embedding, ByteSubwordConcateOnehot,
BinarySubwordConcateOnehot.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SENTI_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'Senti_analysis'))
sys.path.insert(0, SENTI_DIR)

from binary_subword_sentiment import BinarySubwordConcateOnehot  # noqa: E402
from byte_subword_concate_onehot import ByteSubwordConcateOnehot  # noqa: E402

PAD = '<pad>'
UNK = '<unk>'


# -------------- dataset loaders --------------

def tokenize(text: str):
    """Simple lowercase + alnum split, matching the IMDB/SST-2 pipeline."""
    import re
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower()).split()


def load_ade(val_fraction=0.1, seed=42):
    ds = datasets.load_dataset('ade_corpus_v2', 'Ade_corpus_v2_classification', split='train')
    ds = ds.shuffle(seed=seed)
    n = len(ds)
    n_val = max(int(n * val_fraction), 1)
    n_test = max(int(n * val_fraction), 1)
    train = ds.select(range(0, n - n_val - n_test))
    val = ds.select(range(n - n_val - n_test, n - n_test))
    test = ds.select(range(n - n_test, n))

    def to_examples(split):
        return [(e['text'], int(e['label'])) for e in split]
    return to_examples(train), to_examples(val), to_examples(test), 2


def load_pubmedqa(val_fraction=0.15, seed=42):
    ds = datasets.load_dataset('pubmed_qa', 'pqa_labeled', split='train').shuffle(seed=seed)
    decision_map = {'yes': 0, 'no': 1, 'maybe': 2}
    n = len(ds)
    n_val = max(int(n * val_fraction), 1)
    n_test = max(int(n * val_fraction), 1)
    train = ds.select(range(0, n - n_val - n_test))
    val = ds.select(range(n - n_val - n_test, n - n_test))
    test = ds.select(range(n - n_test, n))

    def to_examples(split):
        out = []
        for e in split:
            q = e['question'] or ''
            ctx = ' '.join(e['context']['contexts']) if isinstance(e.get('context'), dict) else ''
            text = q + ' ' + ctx
            out.append((text, decision_map.get(e['final_decision'], 0)))
        return out
    return to_examples(train), to_examples(val), to_examples(test), 3


def load_medmcqa(seed=42, subsample=None):
    ds = datasets.load_dataset('openlifescienceai/medmcqa')
    def to_examples(split, name):
        rows = ds[name]
        if subsample and name == 'train':
            rows = rows.shuffle(seed=seed).select(range(min(subsample, len(rows))))
        out = []
        for e in rows:
            text = (e['question'] or '') + ' [SEP] ' + \
                   (e.get('opa') or '') + ' [SEP] ' + (e.get('opb') or '') + ' [SEP] ' + \
                   (e.get('opc') or '') + ' [SEP] ' + (e.get('opd') or '')
            out.append((text, int(e['cop'])))
        return out
    train = to_examples(ds, 'train')
    val = to_examples(ds, 'validation')
    test = to_examples(ds, 'test')
    return train, val, test, 4


DATASET_LOADERS = {
    'ade': load_ade,
    'pubmedqa': load_pubmedqa,
    'medmcqa': load_medmcqa,
}


# -------------- vocab + encoding --------------

def build_vocab(texts, min_freq=2, max_tokens=None):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    itos = [PAD, UNK]
    stoi = {PAD: 0, UNK: 1}
    for tok, freq in counter.most_common():
        if freq < min_freq:
            break
        if max_tokens and len(itos) >= max_tokens:
            break
        stoi[tok] = len(itos)
        itos.append(tok)
    return stoi, itos


class TextClassificationDataset(Dataset):
    def __init__(self, examples, stoi, max_length=256):
        self.examples = examples
        self.stoi = stoi
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, label = self.examples[idx]
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in tokenize(text)[:self.max_length]]
        if not ids:
            ids = [self.stoi[UNK]]
        return {'ids': torch.tensor(ids, dtype=torch.long),
                'length': torch.tensor(len(ids), dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long)}


def collate(batch, pad_index=0):
    batch = [b for b in batch if b['length'].item() > 0]
    if not batch:
        return None
    ids = nn.utils.rnn.pad_sequence([b['ids'] for b in batch], batch_first=True, padding_value=pad_index)
    length = torch.stack([b['length'] for b in batch])
    label = torch.stack([b['label'] for b in batch]).long()
    return {'ids': ids, 'length': length, 'label': label}


# -------------- model --------------

class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, output_dim, n_layers, dropout, pad_index):
        super().__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids, length):
        embedded = self.dropout(self.embedding(ids))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, length.cpu(),
                                                    batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        return self.fc(hidden)


def build_embedding(kind, vocab_size, embedding_dim, num_bits=64, num_bytes=8,
                    interdim='128', dropout=0.4, mapping_file=''):
    if kind == 'standard':
        return nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    if kind == 'byte_seb':
        # ByteSubwordConcateOnehot's signature does not take num_bytes (fixed at 8 internally)
        return ByteSubwordConcateOnehot(
            num_embeddings=vocab_size, embedding_dim=embedding_dim,
            interdim=interdim, relu_dropout=dropout,
            subword_bytes_file=mapping_file,
        )
    if kind == 'binary_seb':
        return BinarySubwordConcateOnehot(
            num_embeddings=vocab_size, embedding_dim=embedding_dim,
            num_bits=num_bits, interdim=interdim, relu_dropout=dropout,
            subword_binary_file=mapping_file,
        )
    raise ValueError(kind)


# -------------- training --------------

def run_one(method, dataset_name, out_dir, seed=42,
            embedding_dim=64, hidden_dim=64, dropout=0.4, lr=5e-4, batch_size=64,
            num_bits=64, num_bytes=8, interdim='128', n_epochs=20, max_length=256):
    print(f"\n====== {dataset_name.upper()} :: {method} ======")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader = DATASET_LOADERS[dataset_name]
    kwargs = {'seed': seed}
    if dataset_name == 'medmcqa':
        kwargs['subsample'] = 100000
    train_ex, val_ex, test_ex, num_classes = loader(**kwargs)
    print(f"  train={len(train_ex)} val={len(val_ex)} test={len(test_ex)} classes={num_classes}")

    stoi, itos = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(itos)
    print(f"  vocab_size={vocab_size}")

    os.makedirs(out_dir, exist_ok=True)
    mapping_file = os.path.join(out_dir, f'{method}_mapping.pt')

    train_ds = TextClassificationDataset(train_ex, stoi, max_length)
    val_ds = TextClassificationDataset(val_ex, stoi, max_length)
    test_ds = TextClassificationDataset(test_ex, stoi, max_length)
    import functools
    coll = functools.partial(collate, pad_index=stoi[PAD])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)

    embedding = build_embedding(method, vocab_size, embedding_dim,
                                num_bits=num_bits, num_bytes=num_bytes,
                                interdim=interdim, dropout=dropout, mapping_file=mapping_file)
    model = BiLSTMClassifier(embedding, embedding_dim, hidden_dim, num_classes, 2, dropout, stoi[PAD]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    embed_params = sum(p.numel() for p in embedding.parameters())
    print(f"  total_params={total_params:,}  embed_params={embed_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    start = time.time()

    def eval_loader(loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in loader:
                if batch is None: continue
                ids = batch['ids'].to(device); length = batch['length']
                label = batch['label'].to(device)
                pred = model(ids, length)
                correct += pred.argmax(-1).eq(label).sum().item()
                total += label.numel()
        return correct / max(total, 1)

    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_start = time.time()
        for batch in train_loader:
            if batch is None: continue
            ids = batch['ids'].to(device); length = batch['length']
            label = batch['label'].to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, length), label)
            loss.backward()
            optimizer.step()
        val_acc = eval_loader(val_loader)
        ep_time = time.time() - ep_start
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            import copy
            best_state = copy.deepcopy(model.state_dict())
        if epoch % max(1, n_epochs // 5) == 0 or epoch == 1:
            print(f"  epoch {epoch}/{n_epochs}  val_acc={val_acc:.4f}  ({ep_time:.1f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc = eval_loader(test_loader)
    total_time = time.time() - start
    print(f"  FINAL  test_acc={test_acc:.4f}  best_val={best_val_acc:.4f}  total_time={total_time:.1f}s")

    return {
        'dataset': dataset_name, 'method': method,
        'num_classes': num_classes, 'vocab_size': vocab_size,
        'total_params': total_params, 'embed_params': embed_params,
        'best_val_acc': best_val_acc, 'test_acc': test_acc,
        'train_size': len(train_ex), 'val_size': len(val_ex), 'test_size': len(test_ex),
        'total_time_s': total_time, 'n_epochs': n_epochs,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['ade', 'pubmedqa', 'medmcqa'])
    p.add_argument('--method', required=True, choices=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--output_dir', required=True)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--embedding_dim', type=int, default=64)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.4)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_bits', type=int, default=64)
    p.add_argument('--num_bytes', type=int, default=8)
    p.add_argument('--interdim', type=str, default='128')
    p.add_argument('--max_length', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    result = run_one(args.method, args.dataset, args.output_dir,
                     seed=args.seed, embedding_dim=args.embedding_dim,
                     hidden_dim=args.hidden_dim, dropout=args.dropout, lr=args.lr,
                     batch_size=args.batch_size, num_bits=args.num_bits,
                     num_bytes=args.num_bytes, interdim=args.interdim,
                     n_epochs=args.n_epochs, max_length=args.max_length)
    out_file = os.path.join(args.output_dir, f'{args.dataset}_{args.method}.json')
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_file}")


if __name__ == '__main__':
    main()
