#!/usr/bin/env python3
"""Text classification on 4 additional privacy-sensitive domains comparing
Subword / EmByte / 001. Reuses the healthcare-pipeline infrastructure.

Datasets (all public, no credentials):
- finance   : zeroshot/twitter-financial-news-sentiment (3-way bullish/bearish/neutral)
- education : allenai/openbookqa/main (4-way elementary-science MCQ)
- legal     : lex_glue/ledgar (100-way legal clause topic classification)
- mental    : go_emotions/simplified (take primary label; 28-way emotion)
"""

import argparse
import json
import os
import sys

import datasets
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from health_classification import run_one, DATASET_LOADERS  # noqa: E402


# ---------------- loaders ----------------

def load_finance(seed=42):
    ds = datasets.load_dataset('zeroshot/twitter-financial-news-sentiment')
    train = [(e['text'], int(e['label'])) for e in ds['train']]
    val = [(e['text'], int(e['label'])) for e in ds['validation']]
    # No test split; split val in half
    half = len(val) // 2
    test = val[half:]
    val = val[:half]
    return train, val, test, 3


def load_education(seed=42):
    ds = datasets.load_dataset('allenai/openbookqa', 'main')

    def to_examples(split):
        out = []
        for e in split:
            q = e.get('question_stem', '') or ''
            choices = e.get('choices', {})
            texts = choices.get('text', [])
            labels = choices.get('label', [])
            answer_key = e.get('answerKey', '').strip()
            label_to_idx = {lab: i for i, lab in enumerate(labels)}
            try:
                idx = label_to_idx[answer_key]
            except KeyError:
                continue
            text = q + ' [SEP] ' + ' [SEP] '.join(texts)
            out.append((text, idx))
        return out

    train = to_examples(ds['train'])
    val = to_examples(ds['validation'])
    test = to_examples(ds['test'])
    return train, val, test, 4


def load_legal(seed=42, subsample=None):
    ds = datasets.load_dataset('lex_glue', 'ledgar')

    def to_examples(split, subsample_here=None):
        out = []
        if subsample_here and len(split) > subsample_here:
            split = split.shuffle(seed=seed).select(range(subsample_here))
        for e in split:
            text = e.get('text', '')
            label = e.get('label')
            if isinstance(label, int):
                out.append((text, label))
        return out

    train = to_examples(ds['train'], subsample_here=subsample)
    val = to_examples(ds['validation'])
    test = to_examples(ds['test'])
    return train, val, test, 100  # LEDGAR has 100 classes


def load_mental(seed=42, subsample=None):
    ds = datasets.load_dataset('go_emotions', 'simplified')

    def to_examples(split, subsample_here=None):
        out = []
        if subsample_here and len(split) > subsample_here:
            split = split.shuffle(seed=seed).select(range(subsample_here))
        for e in split:
            labels = e.get('labels') or []
            if not labels:
                continue
            # multi-label dataset; take the first (primary) label
            out.append((e['text'], int(labels[0])))
        return out

    train = to_examples(ds['train'], subsample_here=subsample)
    val = to_examples(ds['validation'])
    test = to_examples(ds['test'])
    return train, val, test, 28  # 27 emotions + neutral


def load_contract(seed=42, subsample=None):
    """lex_glue/unfair_tos: binary unfair-clause detection (consumer protection)."""
    import datasets
    ds = datasets.load_dataset('lex_glue', 'unfair_tos')
    def to_examples(split):
        out = []
        for e in split:
            label = 1 if (e.get('labels') and len(e['labels']) > 0) else 0
            out.append((e['text'], label))
        return out
    return to_examples(ds['train']), to_examples(ds['validation']), to_examples(ds['test']), 2


def load_ecthr(seed=42, subsample=None):
    """lex_glue/ecthr_a: 10-way ECtHR human-rights violation classification.
    Multi-label in source; we take the first article (or 0 if none) for single-label."""
    import datasets
    ds = datasets.load_dataset('lex_glue', 'ecthr_a')
    def to_examples(split, sample=None):
        if sample and len(split) > sample:
            split = split.shuffle(seed=seed).select(range(sample))
        out = []
        for e in split:
            txt = e['text']
            if isinstance(txt, list):
                txt = ' '.join(txt[:5])  # truncate to first 5 paragraphs
            labels = e.get('labels') or []
            label = labels[0] if labels else 0
            out.append((txt, int(label)))
        return out
    train = to_examples(ds['train'], sample=subsample)
    return train, to_examples(ds['validation']), to_examples(ds['test']), 10


NEW_LOADERS = {
    'finance': load_finance,
    'education': load_education,
    'legal': load_legal,
    'mental': load_mental,
    'contract': load_contract,
    'ecthr': load_ecthr,
}

# Register with the healthcare-pipeline loader dict so run_one() picks them up
DATASET_LOADERS.update(NEW_LOADERS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True,
                   choices=['finance','education','legal','mental','contract','ecthr'])
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
    print(f"Wrote {out_file}")


if __name__ == '__main__':
    main()
