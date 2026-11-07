#!/usr/bin/env python3
"""On-device intent-classification benchmarks comparing Subword / EmByte / 001.

Targets the use case of LLMs running locally on mobile devices (Apple Foundation
Model, Gemini Nano, Llama 3.2 1B, etc.). Datasets are the standard intent
classification benchmarks for on-device assistants:

- clinc150  : clinc_oos / plus, 150-class intent + out-of-scope
- banking77 : 77-class banking intent classification
- massive   : 5-language (en-US, es-ES, fr-FR, de-DE, zh-CN) concatenated into
              one multilingual model; 60-class intent.
"""

import argparse
import json
import os
import sys

import datasets

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import health_classification as _hc  # noqa: E402
from health_classification import run_one, DATASET_LOADERS  # noqa: E402


def _unicode_tokenize(text: str):
    """Unicode-aware tokenizer: keep word chars from any script, split on whitespace.

    For Chinese and other scriptio-continua languages we additionally split on
    every CJK character so they become individual 'tokens'. This keeps a single
    pipeline working across Latin / accented / Chinese inputs.
    """
    import re
    if not text:
        return []
    # Insert space around CJK characters so they tokenize one-per-token
    text = re.sub(r'([぀-ヿ㐀-䶿一-鿿豈-﫿])',
                  r' \1 ', text)
    # Strip punctuation but keep all unicode word characters
    text = re.sub(r'[^\w\s]', ' ', text.lower(), flags=re.UNICODE)
    return text.split()


# Monkey-patch the tokenizer for on-device runs (multilingual data needs unicode)
_hc.tokenize = _unicode_tokenize


# ---------------- loaders ----------------

def load_clinc150(seed=42):
    """clinc_oos plus config: 150 in-scope intents + out-of-scope class."""
    ds = datasets.load_dataset('clinc_oos', 'plus')

    def to_examples(split):
        out = []
        for e in split:
            text = e.get('text', '')
            label = e.get('intent')
            if label is None:
                continue
            out.append((text, int(label)))
        return out

    train = to_examples(ds['train'])
    val = to_examples(ds['validation'])
    test = to_examples(ds['test'])
    # 150 in-scope + 1 oos class
    return train, val, test, 151


def load_banking77(seed=42, val_fraction=0.1):
    """Banking77: 77-class banking intent. mteb parquet mirror (train/test only)."""
    ds = datasets.load_dataset('mteb/banking77')

    def to_examples(split):
        out = []
        for e in split:
            text = e.get('text', '')
            label = e.get('label')
            if label is None:
                continue
            out.append((text, int(label)))
        return out

    full_train = to_examples(ds['train'])
    test = to_examples(ds['test'])

    import random as _r
    rng = _r.Random(seed)
    rng.shuffle(full_train)
    n_val = max(1, int(len(full_train) * val_fraction))
    val = full_train[:n_val]
    train = full_train[n_val:]
    return train, val, test, 77


def load_massive(seed=42, val_fraction=0.1):
    """MASSIVE multilingual intent classification (mteb parquet mirror).

    Concatenates 5 locales (en, es, fr, de, zh-CN) into one multilingual corpus.
    The mteb mirror stores the intent as a string label, so we build a unified
    label vocab across the union of locales (60 intents in MASSIVE v1.1).
    """
    locales = ['en', 'es', 'fr', 'de', 'zh-CN']
    raw = {'train': [], 'validation': [], 'test': []}

    for loc in locales:
        ds = datasets.load_dataset('mteb/amazon_massive_intent', loc)
        for split_name in raw:
            split = ds.get(split_name) or ds.get('dev') or ds['test']
            for e in split:
                text = e.get('text') or e.get('utt') or ''
                label = e.get('label')  # string in mteb mirror
                if label is None or text == '':
                    continue
                raw[split_name].append((text, str(label)))

    # Build unified label vocab from training labels (covers all in practice)
    label_vocab = sorted({lab for _, lab in raw['train']})
    label_to_id = {lab: i for i, lab in enumerate(label_vocab)}
    num_classes = len(label_vocab)

    def to_int(rows):
        out = []
        for txt, lab in rows:
            if lab in label_to_id:
                out.append((txt, label_to_id[lab]))
        return out

    import random as _r
    rng = _r.Random(seed)
    train_all = to_int(raw['train']); rng.shuffle(train_all)
    val_all = to_int(raw['validation']); rng.shuffle(val_all)
    test_all = to_int(raw['test']); rng.shuffle(test_all)

    return train_all, val_all, test_all, num_classes


NEW_LOADERS = {
    'clinc150': load_clinc150,
    'banking77': load_banking77,
    'massive': load_massive,
}

DATASET_LOADERS.update(NEW_LOADERS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True,
                   choices=['clinc150', 'banking77', 'massive'])
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
    p.add_argument('--max_length', type=int, default=128)
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
