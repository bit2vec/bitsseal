#!/usr/bin/env python3
"""Genome / biomed text classification benchmarks comparing Subword / EmByte / 001.

DNA datasets are k-merized (k=6, stride=1) so that vocab=4096 — the regime where
001's bit-level compression structurally beats standard's V x d table. Without
k-merization, DNA's 4-letter alphabet would make standard far cheaper than 001.

Datasets:
- gb_human_promoters     : Genomic Benchmarks human_nontata_promoters (binary)
- gb_human_enhancers     : Genomic Benchmarks human_enhancers_cohn (binary)
- gb_coding_vs_intergen  : Genomic Benchmarks demo_coding_vs_intergenomic_seqs (binary)
- gb_human_ocr           : Genomic Benchmarks human_ocr_ensembl (binary)
- gb_drosophila_enh      : Genomic Benchmarks drosophila_enhancers_stark (binary)
- emotion                : dair-ai/emotion 6-class (psychological text)
- med_abstracts          : TimSchopf/medical_abstracts (5-class condition labels)
- pubmed_mesh            : owaiskha9654/PubMed_MultiLabel_MeSH (50k, multi-label;
                            we use the first MeSH root as a single-label proxy)
"""

import argparse
import json
import os
import random
import sys

import datasets

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import health_classification as _hc  # noqa: E402
from health_classification import run_one, DATASET_LOADERS  # noqa: E402


# --------------- DNA k-merization helpers ---------------

def kmerize(seq: str, k: int = 6, stride: int = 1) -> str:
    """Convert raw DNA/protein sequence into space-separated k-mers."""
    if not seq:
        return ''
    seq = seq.upper()
    if len(seq) < k:
        return seq
    return ' '.join(seq[i:i+k] for i in range(0, len(seq) - k + 1, stride))


def _split_train_for_val(rows, seed=42, val_fraction=0.1):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_fraction))
    return rows[n_val:], rows[:n_val]


# --------------- DNA loaders (Genomic Benchmarks) ---------------

_GB_DATASETS = {
    'gb_human_promoters':    'katarinagresova/Genomic_Benchmarks_human_nontata_promoters',
    'gb_human_enhancers':    'katarinagresova/Genomic_Benchmarks_human_enhancers_cohn',
    'gb_coding_vs_intergen': 'katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs',
    'gb_human_ocr':          'katarinagresova/Genomic_Benchmarks_human_ocr_ensembl',
    'gb_drosophila_enh':     'katarinagresova/Genomic_Benchmarks_drosophila_enhancers_stark',
}


def _make_gb_loader(hf_id):
    def loader(seed=42, val_fraction=0.1):
        ds = datasets.load_dataset(hf_id)
        train_test = ds['train']
        test = ds['test']

        def to_examples(split):
            return [(kmerize(e['seq'], k=6, stride=1), int(e['label']))
                    for e in split]

        full_train = to_examples(train_test)
        full_test = to_examples(test)
        train, val = _split_train_for_val(full_train, seed=seed, val_fraction=val_fraction)
        return train, val, full_test, 2  # all GB datasets here are binary
    return loader


# --------------- biomed / psych loaders ---------------

def load_emotion(seed=42):
    """dair-ai/emotion: 6-class English emotion (Twitter)."""
    ds = datasets.load_dataset('dair-ai/emotion', 'split')

    def to_examples(split):
        return [(e['text'], int(e['label'])) for e in split]
    return (to_examples(ds['train']), to_examples(ds['validation']),
            to_examples(ds['test']), 6)


def load_med_abstracts(seed=42, val_fraction=0.1):
    """TimSchopf/medical_abstracts: condition_label (5 classes), medical_abstract."""
    ds = datasets.load_dataset('TimSchopf/medical_abstracts')

    def to_examples(split):
        out = []
        for e in split:
            text = e.get('medical_abstract', '') or ''
            label = e.get('condition_label')
            if label is None:
                continue
            out.append((text, int(label) - 1))  # labels are 1..5; shift to 0..4
        return out

    full_train = to_examples(ds['train'])
    test = to_examples(ds['test'])
    train, val = _split_train_for_val(full_train, seed=seed, val_fraction=val_fraction)
    # Confirm class count from data
    n_classes = max(lab for _, lab in (full_train + test)) + 1
    return train, val, test, int(n_classes)


def load_pubmed_mesh(seed=42, val_fraction=0.05, test_fraction=0.05, subsample=20000):
    """owaiskha9654/PubMed_MultiLabel_MeSH: 50k abstracts, MeSH multi-label.

    For single-label classification we take meshroot[0] as the primary topic.
    Sub-sampled to 20k for tractable runtime.
    """
    ds = datasets.load_dataset(
        'owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH',
        split='train')

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    if subsample and subsample < len(indices):
        indices = indices[:subsample]

    rows = []
    label_set = set()
    for i in indices:
        e = ds[i]
        title = e.get('Title', '') or ''
        abstract = e.get('abstractText', '') or ''
        text = (title + ' ' + abstract).strip()
        roots = e.get('meshroot') or []
        if isinstance(roots, str):
            # field may be encoded as a string list
            try:
                import ast
                parsed = ast.literal_eval(roots)
                if isinstance(parsed, list):
                    roots = parsed
                else:
                    roots = [str(parsed)]
            except Exception:
                roots = [roots]
        if not roots:
            continue
        primary = roots[0]
        if not isinstance(primary, str):
            primary = str(primary)
        rows.append((text, primary))
        label_set.add(primary)

    label_to_id = {lab: i for i, lab in enumerate(sorted(label_set))}
    rows = [(t, label_to_id[lab]) for t, lab in rows if lab in label_to_id]

    n = len(rows)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))
    test = rows[:n_test]
    val = rows[n_test:n_test + n_val]
    train = rows[n_test + n_val:]
    return train, val, test, len(label_to_id)


# --------------- register + main ---------------

NEW_LOADERS = {name: _make_gb_loader(hf_id)
               for name, hf_id in _GB_DATASETS.items()}
NEW_LOADERS.update({
    'emotion': load_emotion,
    'med_abstracts': load_med_abstracts,
    'pubmed_mesh': load_pubmed_mesh,
})

DATASET_LOADERS.update(NEW_LOADERS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=list(NEW_LOADERS.keys()))
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
