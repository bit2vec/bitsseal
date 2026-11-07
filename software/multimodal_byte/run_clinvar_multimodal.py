#!/usr/bin/env python3
"""ClinVar multi-modal classification with byte/binary tokenization.

Modalities per variant: condition (free text) + gene + variant + review_status
+ variant_type. Label: clinical significance (top-N classes).
"""
import argparse, csv, functools, json, os, random, re, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))
sys.path.insert(0, THIS_DIR)

from clinvar_loader import load_clinvar, LABEL_SET
from run_civic_multimodal import PAD, UNK, SEP, build_vocab, collate
from run_civic_extended import (
    MultiModalClassifier, film_attack, tag_attack, llg_attack, lamp_attack,
)


def tokenize(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', (text or '').lower()).split()


def extract_gene_from_variant(variant_str):
    m = re.search(r'\(([A-Z][A-Z0-9]+)\)', variant_str or '')
    return m.group(1) if m else ''


# Modalities for ClinVar
MODALITIES = [
    ('condition',     32),
    ('gene',           4),
    ('variant',       16),
    ('variant_type',   8),
    ('review_status', 16),
]


class CVMultiModalDataset(Dataset):
    def __init__(self, examples, stoi, vocab_meta, mods, ablation):
        self.exs = examples; self.stoi = stoi
        self.vocab_meta = vocab_meta; self.mods = mods
        self.ablation = ablation

    def __len__(self): return len(self.exs)

    def __getitem__(self, i):
        ex = self.exs[i]
        ids = []
        for mod, budget in self.mods:
            if self.ablation == 'gene-only' and mod != 'gene':
                tok_ids = [self.stoi[PAD]] * budget
            elif self.ablation == 'gene+variant' and mod not in ('gene', 'variant'):
                tok_ids = [self.stoi[PAD]] * budget
            else:
                toks = tokenize(ex[mod])
                if self.vocab_meta['mode'] == 'separator':
                    tok_ids = [self.stoi.get(t, self.stoi[UNK]) for t in toks]
                else:
                    tok_ids = [self.stoi.get(f'{mod}::{t}', self.stoi[UNK]) for t in toks]
                if len(tok_ids) > budget: tok_ids = tok_ids[:budget]
                else: tok_ids = tok_ids + [self.stoi[PAD]] * (budget - len(tok_ids))
            if self.vocab_meta['mode'] == 'separator':
                ids.append(self.stoi[f'{SEP}_{mod}'])
            ids.extend(tok_ids)
        return torch.tensor(ids, dtype=torch.long), ex['label_idx']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/multimodal_byte/clinvar')
    p.add_argument('--n_epochs', type=int, default=4)
    p.add_argument('--bodies', nargs='+', default=['A', 'C', 'D', 'E'])
    p.add_argument('--methods', nargs='+', default=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ablations', nargs='+', default=['gene-only', 'gene+variant', 'all'])
    p.add_argument('--n_target', type=int, default=3000)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading ClinVar...')
    examples, _ = load_clinvar(n_target=args.n_target)

    # Backfill gene from variant string if empty
    for ex in examples:
        if not ex.get('gene'):
            ex['gene'] = extract_gene_from_variant(ex.get('variant', ''))

    # Filter to top-3 most-common classes (Benign too rare to learn)
    top3 = ['Pathogenic', 'Likely benign', 'Uncertain significance']
    examples = [e for e in examples if e['label'] in top3]
    label_to_idx = {l: i for i, l in enumerate(top3)}
    for ex in examples: ex['label_idx'] = label_to_idx[ex['label']]
    rng = random.Random(42); rng.shuffle(examples)

    print(f'Filtered to top-3 classes: {len(examples)} examples')
    print('Class dist:', Counter(e['label'] for e in examples))

    # Empirical lengths
    print('\nEmpirical lengths:')
    for mod, _ in MODALITIES:
        lens = [len(tokenize(ex[mod])) for ex in examples]
        print(f'  {mod}: max={max(lens)} p95={np.percentile(lens, 95):.0f} mean={np.mean(lens):.1f}')
    final_modalities = []
    for mod, _ in MODALITIES:
        lens = [len(tokenize(ex[mod])) for ex in examples]
        m = max(lens); p95 = max(int(np.percentile(lens, 95)), 1)
        if mod in ('condition',): budget = min(512, m)
        else: budget = m if m <= 1.5 * p95 else p95
        final_modalities.append((mod, max(budget, 1)))
        print(f'  budget[{mod}] = {final_modalities[-1][1]}')

    n = len(examples)
    n_tr = int(0.7 * n); n_va = int(0.1 * n)
    train_ex = examples[:n_tr]; val_ex = examples[n_tr:n_tr+n_va]
    test_ex = examples[n_tr+n_va:]

    # Build vocab (separator mode for simplicity)
    counter = Counter()
    for ex in train_ex:
        for mod, _ in final_modalities:
            counter.update(tokenize(ex[mod]))
    stoi = {PAD: 0, UNK: 1}
    for mod, _ in final_modalities: stoi[f'{SEP}_{mod}'] = len(stoi)
    for tok, c in counter.most_common():
        if c >= 2: stoi[tok] = len(stoi)
    vocab_meta = {'mode': 'separator'}
    vocab_size = len(stoi)
    _spin = [m for m in args.methods if m in ('simhash_seb', 'repulse_seb')]
    if _spin:
        import tempfile, run_civic_extended as _rce
        from run_table7_repro import simhash_table, repulse_table
        _itos = [None] * vocab_size
        for _t, _i in stoi.items(): _itos[_i] = _t
        _td = tempfile.mkdtemp()
        if 'simhash_seb' in _spin:
            _p = os.path.join(_td, 'sh.pt'); torch.save(simhash_table(_itos, num_bits=64), _p); _rce.SPINHASH_CODES['simhash_seb'] = _p
        if 'repulse_seb' in _spin:
            _tt = [sum((tokenize(ex.get(m, '')) for m, _ in final_modalities), []) for ex in train_ex]
            _p = os.path.join(_td, 'rp.pt'); torch.save(repulse_table(_itos, _tt, num_bits=64), _p); _rce.SPINHASH_CODES['repulse_seb'] = _p
    print(f'\nvocab={vocab_size}  train={len(train_ex)}/val={len(val_ex)}/test={len(test_ex)}')

    summary = {}
    for method in args.methods:
        for ablation in args.ablations:
            for body in args.bodies:
                key = f'{method}_{ablation}_{body}'
                print(f'\n>>> {key}')
                try:
                    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
                    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
                    coll = functools.partial(collate, pad_index=stoi[PAD])
                    bs = 16
                    train_ds = CVMultiModalDataset(train_ex, stoi, vocab_meta, final_modalities, ablation)
                    val_ds = CVMultiModalDataset(val_ex, stoi, vocab_meta, final_modalities, ablation)
                    test_ds = CVMultiModalDataset(test_ex, stoi, vocab_meta, final_modalities, ablation)
                    tr_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=coll)
                    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=coll)
                    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=coll)

                    model = MultiModalClassifier(vocab_size, 64, len(top3),
                                                 method=method, body_kind=body,
                                                 pad_idx=stoi[PAD]).to(device)
                    opt = optim.Adam(model.parameters(), lr=5e-4)
                    crit = nn.CrossEntropyLoss()

                    t0 = time.time()
                    best_val, best_state = 0.0, None
                    for ep in range(args.n_epochs):
                        model.train()
                        for batch in tr_loader:
                            opt.zero_grad()
                            crit(model(batch['ids'].to(device)),
                                 batch['label'].to(device)).backward()
                            opt.step()
                        model.eval()
                        c = t = 0
                        with torch.no_grad():
                            for batch in val_loader:
                                pred = model(batch['ids'].to(device)).argmax(-1).cpu()
                                c += (pred == batch['label']).sum().item()
                                t += batch['label'].size(0)
                        v = c / max(t, 1)
                        if v > best_val:
                            best_val = v
                            best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
                    if best_state is not None: model.load_state_dict(best_state)
                    train_time = time.time() - t0

                    model.eval()
                    c = t = 0
                    t0 = time.time()
                    with torch.no_grad():
                        for batch in test_loader:
                            pred = model(batch['ids'].to(device)).argmax(-1).cpu()
                            c += (pred == batch['label']).sum().item()
                            t += batch['label'].size(0)
                    inf_time = time.time() - t0
                    test_acc = c / max(t, 1)

                    # FILM + TAG attacks
                    model.train()
                    film_f1s, tag_f1s = [], []
                    n_atk = 0
                    for batch in tr_loader:
                        if n_atk >= 12: break
                        try: film_f1s.append(film_attack(model, batch['ids'], batch['label'], crit, device))
                        except: pass
                        try: tag_f1s.append(tag_attack(model, batch['ids'], batch['label'], crit, device, vocab_size))
                        except: pass
                        n_atk += 1

                    r = {
                        'method': method, 'ablation': ablation, 'body': body,
                        'val_acc': float(best_val), 'test_acc': float(test_acc),
                        'n_params': sum(p.numel() for p in model.parameters()),
                        'train_time_s': train_time, 'inference_time_s': inf_time,
                        'film_f1':  float(np.mean(film_f1s)) if film_f1s else 0.0,
                        'tag_f1':   float(np.mean(tag_f1s))  if tag_f1s  else 0.0,
                    }
                    summary[key] = r
                    print(f'  acc={r["test_acc"]:.4f}  FILM={r["film_f1"]:.3f} '
                          f'TAG={r["tag_f1"]:.3f}  params={r["n_params"]}')
                except Exception as e:
                    import traceback; traceback.print_exc()
                    summary[key] = {'error': str(e)}
                with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                    json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
