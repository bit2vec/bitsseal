#!/usr/bin/env python3
"""
Table 12: Cosine similarity of subword embeddings based on 001

Computes cosine similarity between word embeddings for sentiment-related
words (good, great, funny, bad, worse, boring) using a trained Binary SEB model.

Usage:
    python cosine_similarity_analysis.py --model_path ./results_imdb/best_model_binary_seb.pt \
        --mapping_path ./results_imdb/binary_seb_mapping.pt --output_dir ./results_cosine
"""

import functools
import sys
import os
import json
import argparse

import datasets
import numpy as np
import torch
import torch.nn as nn

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


def main():
    parser = argparse.ArgumentParser(description='Table 12: Cosine Similarity Analysis')
    parser.add_argument('--model_path', type=str, default='',
                        help='Path to trained binary_seb model checkpoint')
    parser.add_argument('--mapping_path', type=str, default='',
                        help='Path to binary mapping .pt file')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--num_bits', type=int, default=64)
    parser.add_argument('--interdim', type=str, default='128')
    parser.add_argument('--output_dir', type=str, default='./results_cosine')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load IMDB and build vocabulary (same as training)
    print("Loading IMDB dataset...")
    train_data = datasets.load_dataset('imdb', split='train')
    tokenizer = get_tokenizer('basic_english')

    def tokenize_data(example):
        tokens = tokenizer(example['text'])[:256]
        return {'tokens': tokens}

    train_data = train_data.map(tokenize_data)
    train_valid = train_data.train_test_split(test_size=0.25, seed=seed)
    train_data = train_valid['train']

    vocab = build_vocab_from_iterator(train_data['tokens'], min_freq=5, specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Create Binary SEB embedding
    mapping_path = args.mapping_path if args.mapping_path else os.path.join(args.output_dir, 'binary_mapping.pt')
    embedding = BinarySubwordConcateOnehot(
        num_embeddings=vocab_size,
        embedding_dim=args.embedding_dim,
        num_bits=args.num_bits,
        interdim=args.interdim,
        relu_dropout=0.0,
        subword_binary_file=mapping_path,
    ).to(device)

    # If model checkpoint provided, load trained weights
    if args.model_path and os.path.isfile(args.model_path):
        print(f"Loading model from {args.model_path}")
        state_dict = torch.load(args.model_path, map_location=device)
        # Extract embedding weights only
        embed_keys = {k: v for k, v in state_dict.items() if k.startswith('embedding.')}
        embed_keys = {k.replace('embedding.', ''): v for k, v in embed_keys.items()}
        embedding.load_state_dict(embed_keys, strict=False)
        print("Loaded trained embedding weights")
    else:
        print("No trained model provided. Using randomly initialized embedding.")
        print("For meaningful results, train IMDB first and provide --model_path")

    # Target words from Table 12
    target_words = ['good', 'great', 'funny', 'bad', 'worse', 'boring']

    # Get word indices
    word_ids = []
    for word in target_words:
        idx = vocab[word]
        if idx == vocab['<unk>']:
            print(f"Warning: '{word}' not in vocabulary")
        word_ids.append(idx)

    # Compute embeddings
    embedding.eval()
    with torch.no_grad():
        word_tensor = torch.tensor(word_ids, dtype=torch.long).to(device)
        word_embeddings = embedding(word_tensor)  # (num_words, embedding_dim)

    # Compute cosine similarity matrix
    norms = word_embeddings.norm(dim=1, keepdim=True)
    normalized = word_embeddings / (norms + 1e-8)
    cos_sim = torch.mm(normalized, normalized.t()).cpu().numpy()

    # Print Table 12
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("TABLE 12: Cosine similarity of subword embeddings based on 001")
    print("="*70)
    print()

    # Header
    header = f"{'':>10}"
    for w in target_words:
        header += f"{w:>10}"
    print(header)

    for i, w1 in enumerate(target_words):
        row = f"{w1:>10}"
        for j, w2 in enumerate(target_words):
            row += f"{cos_sim[i, j]:>10.2f}"
        print(row)

    # Save
    results = {
        'words': target_words,
        'cosine_similarity': cos_sim.tolist(),
    }
    with open(os.path.join(args.output_dir, 'table12_cosine_similarity.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/table12_cosine_similarity.json")


if __name__ == '__main__':
    main()
