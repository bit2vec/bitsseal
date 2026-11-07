#!/usr/bin/env python3
"""Loader for Rfam-derived RNA + text + family-type dataset.

Parses Rfam.seed (Stockholm format) for ~4227 RNA families. Each family has
- rfam_acc / rfam_id
- type (e.g., "Gene; rRNA;", "Gene; sRNA;", "Gene; tRNA;")
- comment (long biological description text — what we want as the text feature)
- seed sequences (typically 5-50 per family, ~50-500 nt each)

Each sample is one (sequence, comment, family-type) triple.

Cache layout:
    software/multimodal_byte/rfam_cache/Rfam.seed       (downloaded by loader)
    software/multimodal_byte/rfam_cache/rfam_samples.json  (parsed cache)
"""
import argparse
import collections
import json
import os
import random
import re
import urllib.request

CACHE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/rfam_cache'
RFAM_SEED_URL = 'https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz'


def _download_seed(force=False):
    seed = os.path.join(CACHE_DIR, 'Rfam.seed')
    if os.path.exists(seed) and not force:
        return seed
    seed_gz = os.path.join(CACHE_DIR, 'Rfam.seed.gz')
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(seed_gz):
        print(f'[rfam_loader] downloading {RFAM_SEED_URL} ...', flush=True)
        urllib.request.urlretrieve(RFAM_SEED_URL, seed_gz)
    import gzip
    with gzip.open(seed_gz, 'rt') as f, open(seed, 'w') as g:
        g.write(f.read())
    return seed


def _parse_stockholm(path):
    """Yield {acc, id, type, comment, sequences=[(name, seq), ...]} per family."""
    cur = {'acc': None, 'id': None, 'type': '', 'comment': '',
           'description': '', 'sequences': []}
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if line == '# STOCKHOLM 1.0':
                if cur['acc']:
                    yield cur
                cur = {'acc': None, 'id': None, 'type': '',
                       'comment': '', 'description': '', 'sequences': []}
                continue
            if line.startswith('#=GF '):
                tag = line[5:7]
                rest = line[8:].strip() if len(line) >= 8 else ''
                if tag == 'AC': cur['acc'] = rest
                elif tag == 'ID': cur['id'] = rest
                elif tag == 'TP': cur['type'] = rest
                elif tag == 'DE': cur['description'] = rest
                elif tag == 'CC':
                    # multiple CC lines concat with space
                    cur['comment'] = (cur['comment'] + ' ' + rest).strip()
                continue
            if line.startswith('#') or line.startswith('//') or not line.strip():
                continue
            # Sequence line: "name SEQUENCE"
            parts = line.split(None, 1)
            if len(parts) == 2:
                name, seq = parts
                # Stockholm uses '.' and '-' for gaps; remove them for raw seq
                clean = seq.replace('.', '').replace('-', '').upper()
                if clean:
                    cur['sequences'].append((name, clean))
    if cur['acc']:
        yield cur


def parse_label(tp_field):
    """`Gene; rRNA;` -> `rRNA`. `Cis-reg; riboswitch;` -> `riboswitch`."""
    if not tp_field:
        return 'unknown'
    parts = [p.strip() for p in tp_field.split(';') if p.strip()]
    return parts[-1] if parts else 'unknown'


def build_dataset(max_seqs_per_family=5, min_seqs_per_family=1,
                  min_seq_len=20, max_seq_len=512,
                  exclude_labels=None, seed=42, force=False):
    """Returns list of dicts {sequence, comment, description, family, type}."""
    cache = os.path.join(CACHE_DIR, 'rfam_samples.json')
    if os.path.exists(cache) and not force:
        with open(cache) as f:
            return json.load(f)
    seed_file = _download_seed(force=force)
    rng = random.Random(seed)
    samples = []
    for fam in _parse_stockholm(seed_file):
        label = parse_label(fam['type'])
        if exclude_labels and label in exclude_labels:
            continue
        seqs = [(n, s) for n, s in fam['sequences']
                if min_seq_len <= len(s) <= max_seq_len]
        if len(seqs) < min_seqs_per_family:
            continue
        if len(seqs) > max_seqs_per_family:
            seqs = rng.sample(seqs, max_seqs_per_family)
        # Use the long comment as text; fall back to short description
        text = fam['comment'] if fam['comment'].strip() else fam['description']
        for name, seq in seqs:
            samples.append({
                'name': name,
                'sequence': seq,
                'comment': text,
                'description_short': fam['description'],
                'family_acc': fam['acc'],
                'family_id': fam['id'],
                'family_type': label,
            })
    rng.shuffle(samples)
    with open(cache, 'w') as f:
        json.dump(samples, f)
    return samples


def label_distribution(samples):
    return collections.Counter(s['family_type'] for s in samples)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max_seqs_per_family', type=int, default=5)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    samples = build_dataset(max_seqs_per_family=args.max_seqs_per_family,
                             force=args.force)
    print(f'#samples = {len(samples)}')
    print('label distribution:')
    for label, n in label_distribution(samples).most_common(25):
        print(f'  {label:25s}: {n}')


if __name__ == '__main__':
    main()
