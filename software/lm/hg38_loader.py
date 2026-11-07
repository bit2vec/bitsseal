#!/usr/bin/env python3
"""hg38 loader for k-mer pretraining.

Downloads chromosomes from UCSC, parses FASTA, k-merizes (k=6 default),
builds a numpy array of k-mer token IDs.

Cache layout:
  software/lm/hg38_cache/chr<N>.fa.gz       (downloaded)
  software/lm/hg38_cache/chr<N>_k<k>.npy    (tokenized, cached)
  software/lm/hg38_cache/vocab_k<k>.json    (k-mer to ID map)
"""
import argparse
import gzip
import json
import os
import urllib.request
from collections import Counter

import numpy as np

CACHE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/hg38_cache'
UCSC_URL = 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/{chrom}.fa.gz'

# Standard human chromosomes
ALL_CHROMS = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
LARGE_CHROMS = [f'chr{i}' for i in range(1, 11)]   # chr1-chr10, ~50% of genome
SMALL_CHROMS = ['chr21', 'chr22']                   # quick test


def download_chrom(chrom):
    """Download a single chromosome FASTA if not already cached."""
    fn = os.path.join(CACHE_DIR, f'{chrom}.fa.gz')
    if os.path.exists(fn):
        return fn
    os.makedirs(CACHE_DIR, exist_ok=True)
    url = UCSC_URL.format(chrom=chrom)
    print(f'  downloading {url}', flush=True)
    urllib.request.urlretrieve(url, fn)
    return fn


def load_chrom_seq(chrom):
    """Returns the chromosome sequence as a single uppercase string with Ns."""
    fn = download_chrom(chrom)
    seqs = []
    with gzip.open(fn, 'rt') as f:
        for line in f:
            if line.startswith('>'):
                continue
            seqs.append(line.strip().upper())
    return ''.join(seqs)


def kmerize_to_tokens(seq, vocab_stoi, k=6, stride=1, skip_n=True):
    """Convert sequence to a numpy array of k-mer token IDs.
    Skips windows containing N (unknown) by default."""
    L = len(seq)
    if L < k: return np.array([], dtype=np.int32)
    n_windows = (L - k) // stride + 1
    tokens = np.zeros(n_windows, dtype=np.int32)
    j = 0
    pad_id = vocab_stoi.get('<pad>', 0)
    unk_id = vocab_stoi.get('<unk>', 1)
    for i in range(0, L - k + 1, stride):
        w = seq[i:i+k]
        if skip_n and 'N' in w:
            continue
        tokens[j] = vocab_stoi.get(w, unk_id)
        j += 1
    return tokens[:j]


def build_canonical_kmer_vocab(k=6):
    """Returns a deterministic vocab over all 4^k k-mers (no N).
    Stable across runs and corpora."""
    bases = 'ACGT'
    stoi = {'<pad>': 0, '<unk>': 1}
    if k == 1:
        for b in 'ACGTN':
            stoi[b] = len(stoi)
    else:
        # All possible k-mers in lexicographic order (deterministic)
        def enumerate_kmers(prefix, depth):
            if depth == 0:
                stoi[prefix] = len(stoi)
                return
            for b in bases:
                enumerate_kmers(prefix + b, depth - 1)
        enumerate_kmers('', k)
    return stoi


def build_corpus(chroms, k=6, stride=1):
    """Returns a single 1-D numpy array of k-mer token IDs across all chroms."""
    stoi = build_canonical_kmer_vocab(k=k)
    print(f'  vocab_size={len(stoi)} (k={k})', flush=True)
    all_tokens = []
    for chrom in chroms:
        cache_fn = os.path.join(CACHE_DIR, f'{chrom}_k{k}.npy')
        if os.path.exists(cache_fn):
            print(f'  cached {chrom}_k{k}', flush=True)
            tokens = np.load(cache_fn)
        else:
            print(f'  parsing {chrom}...', flush=True)
            seq = load_chrom_seq(chrom)
            print(f'    {chrom}: {len(seq):,} bp', flush=True)
            tokens = kmerize_to_tokens(seq, stoi, k=k, stride=stride)
            print(f'    {len(tokens):,} k-mers tokenized', flush=True)
            np.save(cache_fn, tokens)
        all_tokens.append(tokens)
    full = np.concatenate(all_tokens) if all_tokens else np.array([], dtype=np.int32)
    print(f'  total k-mer tokens across {len(chroms)} chromosomes: {len(full):,}',
          flush=True)
    return full, stoi


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--chroms', nargs='+', default=['chr22'])
    p.add_argument('--k', type=int, default=6)
    p.add_argument('--stride', type=int, default=1)
    args = p.parse_args()
    tokens, stoi = build_corpus(args.chroms, k=args.k, stride=args.stride)
    print(f'Done. Vocab size {len(stoi)}, total tokens {len(tokens):,}')


if __name__ == '__main__':
    main()
