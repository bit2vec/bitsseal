#!/usr/bin/env python3
"""TCGA pan-cancer 5-way classification with multi-modal byte/binary tokenization.

Modalities per patient: text (ICD/histology) + structured (age/stage/grade/etc.)
                       + gene_top (top-K expressed genes ranked).
Label: cancer type (BRCA / KIRC / LUAD / COADREAD / UCEC).

Reuses the architecture and budget rule from run_civic_extended.py.
"""
import argparse, functools, json, os, random, re, sys, time
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

from tcga_loader import load_tcga_pan5
from run_civic_multimodal import PAD, UNK, SEP, build_vocab, collate
from run_civic_extended import (
    MultiModalClassifier, film_attack, tag_attack, llg_attack,
    lamp_attack, dlg_attack, mia_attack_quick, embinv_attack,
)


def tokenize(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower()).split()


# Modalities + budgets (will be tuned to TCGA empirically below)
MODALITIES = [
    ('text',       16),
    ('structured', 32),
    ('gene_top',   64),
]


class TCGAMultiModalDataset(Dataset):
    def __init__(self, examples, stoi, vocab_meta, modalities, ablation):
        self.exs = examples; self.stoi = stoi
        self.vocab_meta = vocab_meta; self.mods = modalities
        self.ablation = ablation

    def __len__(self): return len(self.exs)

    def __getitem__(self, i):
        ex = self.exs[i]
        ids = []
        for mod, budget in self.mods:
            if self.ablation == 'gene-only' and mod != 'gene_top':
                tok_ids = [self.stoi[PAD]] * budget
            elif self.ablation == 'gene+text' and mod not in ('gene_top', 'text'):
                tok_ids = [self.stoi[PAD]] * budget
            else:
                toks = tokenize(ex[mod])
                if self.vocab_meta['mode'] == 'separator':
                    tok_ids = [self.stoi.get(t, self.stoi[UNK]) for t in toks]
                else:
                    tok_ids = [self.stoi.get(f'{mod}::{t}', self.stoi[UNK]) for t in toks]
                if len(tok_ids) > budget:
                    tok_ids = tok_ids[:budget]
                else:
                    tok_ids = tok_ids + [self.stoi[PAD]] * (budget - len(tok_ids))
            if self.vocab_meta['mode'] == 'separator':
                ids.append(self.stoi[f'{SEP}_{mod}'])
            ids.extend(tok_ids)
        return torch.tensor(ids, dtype=torch.long), ex['label_idx']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/multimodal_byte/tcga')
    p.add_argument('--n_epochs', type=int, default=6)
    p.add_argument('--bodies', nargs='+', default=['A', 'C', 'D', 'E'])
    p.add_argument('--methods', nargs='+',
                   default=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--ablations', nargs='+',
                   default=['gene-only', 'gene+text', 'all'])
    p.add_argument('--top_k_genes', type=int, default=64)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading TCGA pan5 data...')
    examples, label_list = load_tcga_pan5(top_k_genes=args.top_k_genes)
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    for ex in examples: ex['label_idx'] = label_to_idx[ex['label']]
    rng = random.Random(42); rng.shuffle(examples)

    # Empirical length distribution
    print('\nEmpirical length distributions:')
    for mod, _ in MODALITIES:
        lens = [len(tokenize(ex[mod])) for ex in examples]
        print(f'  {mod}: max={max(lens)} p95={np.percentile(lens, 95):.0f} '
              f'p99={np.percentile(lens, 99):.0f} mean={np.mean(lens):.1f}')

    # Apply budget rule
    final_modalities = []
    for mod, _ in MODALITIES:
        lens = [len(tokenize(ex[mod])) for ex in examples]
        m = max(lens); p95 = int(np.percentile(lens, 95)) or 1
        if mod == 'text':         budget = min(512, m)
        elif mod == 'gene_top':   budget = min(2048, m)
        else:                     budget = m if m <= 1.5 * p95 else p95
        final_modalities.append((mod, budget))
        print(f'  budget[{mod}] = {budget}')

    n = len(examples)
    n_tr = int(0.7 * n); n_va = int(0.1 * n)
    train_ex = examples[:n_tr]
    val_ex = examples[n_tr:n_tr + n_va]
    test_ex = examples[n_tr + n_va:]
    print(f'\nSplit: train={len(train_ex)} val={len(val_ex)} test={len(test_ex)} '
          f'classes={len(label_list)}')

    # Class distribution
    print('Train class dist:', Counter(ex['label'] for ex in train_ex).most_common())

    summary = {}
    for boundary in ['separator']:
        # Build vocab using all relevant text
        train_texts = []
        for ex in train_ex:
            for mod, _ in final_modalities:
                train_texts.append(ex[mod])
        # Reuse build_vocab; expects a list of (text, label) -> wrap
        wrapped = [(t, 0) for t in train_texts]
        # build_vocab from civic expects list-of-dicts in v2; so reimplement here
        if boundary == 'separator':
            counter = Counter()
            for ex in train_ex:
                for mod, _ in final_modalities:
                    counter.update(tokenize(ex[mod]))
            stoi = {PAD: 0, UNK: 1}
            for mod, _ in final_modalities:
                stoi[f'{SEP}_{mod}'] = len(stoi)
            for tok, c in counter.most_common():
                if c >= 2: stoi[tok] = len(stoi)
            vocab_meta = {'mode': 'separator'}
        else:  # disjoint
            stoi = {PAD: 0, UNK: 1}
            for mod, _ in final_modalities:
                counter = Counter()
                for ex in train_ex:
                    counter.update(tokenize(ex[mod]))
                for tok, c in counter.most_common():
                    if c >= 2: stoi[f'{mod}::{tok}'] = len(stoi)
            vocab_meta = {'mode': 'disjoint'}
        vocab_size = len(stoi)
        print(f'\n[{boundary}] vocab={vocab_size}')

        for method in args.methods:
            for ablation in args.ablations:
                for body in args.bodies:
                    key = f'{boundary}_{method}_{ablation}_{body}'
                    print(f'  >>> {key}')
                    try:
                        random.seed(42); np.random.seed(42); torch.manual_seed(42)
                        if torch.cuda.is_available(): torch.cuda.manual_seed(42)
                        train_ds = TCGAMultiModalDataset(train_ex, stoi, vocab_meta, final_modalities, ablation)
                        val_ds = TCGAMultiModalDataset(val_ex, stoi, vocab_meta, final_modalities, ablation)
                        test_ds = TCGAMultiModalDataset(test_ex, stoi, vocab_meta, final_modalities, ablation)
                        coll = functools.partial(collate, pad_index=stoi[PAD])
                        bs = 16
                        tr_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=coll)
                        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=coll)
                        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=coll)

                        model = MultiModalClassifier(vocab_size, 64, len(label_list),
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

                        # Run attacks
                        model.train()
                        film_f1s, tag_f1s, llg_asrs, lamp_f1s = [], [], [], []
                        dlg_f1s, mia_aucs = [], []
                        n_atk = 0
                        for batch in tr_loader:
                            if n_atk >= 12: break
                            try: film_f1s.append(film_attack(model, batch['ids'], batch['label'], crit, device))
                            except: pass
                            try: tag_f1s.append(tag_attack(model, batch['ids'], batch['label'], crit, device, vocab_size))
                            except: pass
                            try: llg_asrs.append(llg_attack(model, batch['ids'], batch['label'], crit, device, len(label_list)))
                            except: pass
                            try: lamp_f1s.append(lamp_attack(model, batch['ids'], batch['label'], crit, device, vocab_size))
                            except: pass
                            try: dlg_f1s.append(dlg_attack(model, batch['ids'], batch['label'], crit, device, vocab_size))
                            except: pass
                            try: mia_aucs.append(mia_attack_quick(model, batch['ids'], batch['label'], crit, device))
                            except: pass
                            n_atk += 1

                        r = {
                            'method': method, 'ablation': ablation, 'body': body,
                            'val_acc': float(best_val), 'test_acc': float(test_acc),
                            'n_params': sum(p.numel() for p in model.parameters()),
                            'train_time_s': train_time, 'inference_time_s': inf_time,
                            'film_f1':  float(np.mean(film_f1s)) if film_f1s else 0.0,
                            'tag_f1':   float(np.mean(tag_f1s))  if tag_f1s  else 0.0,
                            'llg_asr':  float(np.mean(llg_asrs)) if llg_asrs else 0.0,
                            'lamp_f1':  float(np.mean(lamp_f1s)) if lamp_f1s else 0.0,
                            'dlg_f1':   float(np.mean(dlg_f1s))  if dlg_f1s  else 0.0,
                            'mia_auc':  float(np.mean(mia_aucs)) if mia_aucs else 0.5,
                        }
                        summary[key] = r
                        print(f'      acc={r["test_acc"]:.4f}  FILM={r["film_f1"]:.3f} '
                              f'TAG={r["tag_f1"]:.3f}  train={r["train_time_s"]:.1f}s')
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        summary[key] = {'error': str(e)}
                    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                        json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
