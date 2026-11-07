#!/usr/bin/env python3
"""Compare sequence-only vs sequence+text classification.

For each dataset (RNA family classification from Rfam, optionally DNA from
NCBI), we train two models with the *same* architecture and budget and
compare:

  A. seq_only   : only the sequence modality is non-PAD
  B. seq+text   : both sequence and text modalities are non-PAD

Both trained from scratch with random init. Same lr, same epochs. The only
difference is the available input. If text helps, A < B in accuracy.

Output: results/multimodality/seq_vs_seqtext/{dataset}/summary.json
"""
import argparse, copy, functools, json, os, random, re, sys, time
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

from run_civic_multimodal import PAD, UNK, SEP, collate
from run_civic_extended import MultiModalClassifier
import rfam_loader


def tokenize(text):
    return re.sub(r'[^A-Za-z0-9\s]', ' ', text.lower()).split()


def build_separator_vocab(train_examples, modalities, min_freq=2):
    counter = Counter()
    for ex in train_examples:
        for mod, _ in modalities:
            counter.update(tokenize(ex[mod]))
    stoi = {PAD: 0, UNK: 1}
    for mod, _ in modalities:
        stoi[f'{SEP}_{mod}'] = len(stoi)
    for tok, c in counter.most_common():
        if c >= min_freq:
            stoi[tok] = len(stoi)
    return stoi


def encode_with_mask(ex, stoi, modalities, mask_all_but=None):
    out = []
    for mod, budget in modalities:
        if mask_all_but is not None and mod not in mask_all_but:
            tok_ids = [stoi[PAD]] * budget
        else:
            toks = tokenize(ex[mod])
            tok_ids = [stoi.get(t, stoi[UNK]) for t in toks]
            if len(tok_ids) > budget:
                tok_ids = tok_ids[:budget]
            else:
                tok_ids = tok_ids + [stoi[PAD]] * (budget - len(tok_ids))
        out.append(stoi[f'{SEP}_{mod}'])
        out.extend(tok_ids)
    return out


class FlexibleDataset(Dataset):
    def __init__(self, examples, stoi, modalities, label_field, mask_all_but=None):
        self.exs = examples
        self.stoi = stoi
        self.mods = modalities
        self.lf = label_field
        self.mask = mask_all_but

    def __len__(self):
        return len(self.exs)

    def __getitem__(self, i):
        ex = self.exs[i]
        ids = encode_with_mask(ex, self.stoi, self.mods, mask_all_but=self.mask)
        return torch.tensor(ids, dtype=torch.long), ex[self.lf]


def train_eval(model, train_ds, val_ds, test_ds, stoi, device,
               n_epochs=6, batch_size=32, lr=5e-4):
    coll = functools.partial(collate, pad_index=stoi[PAD])
    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    va = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    te = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    t0 = time.time()
    best_val, best_state = 0.0, None
    for ep in range(n_epochs):
        model.train()
        for batch in tr:
            opt.zero_grad()
            crit(model(batch['ids'].to(device)),
                 batch['label'].to(device)).backward()
            opt.step()
        model.eval()
        c = t = 0
        with torch.no_grad():
            for batch in va:
                pred = model(batch['ids'].to(device)).argmax(-1).cpu()
                c += (pred == batch['label']).sum().item()
                t += batch['label'].size(0)
        v = c / max(t, 1)
        if v > best_val:
            best_val = v
            best_state = {k: vv.detach().cpu().clone()
                          for k, vv in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.time() - t0
    model.eval()
    c = t = 0
    with torch.no_grad():
        for batch in te:
            pred = model(batch['ids'].to(device)).argmax(-1).cpu()
            c += (pred == batch['label']).sum().item()
            t += batch['label'].size(0)
    return {
        'val_acc': float(best_val),
        'test_acc': float(c / max(t, 1)),
        'train_time_s': train_time,
        'n_params': sum(p.numel() for p in model.parameters()),
    }


def load_rfam(top_n_classes=10, max_samples=None, seed=42):
    samples = rfam_loader.build_dataset(max_seqs_per_family=5)
    counts = Counter(s['family_type'] for s in samples)
    keep = set(t for t, _ in counts.most_common(top_n_classes))
    keep.discard('Gene')  # too generic / catch-all
    samples = [s for s in samples if s['family_type'] in keep]
    label_list = sorted({s['family_type'] for s in samples})
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    for s in samples:
        s['label'] = label_to_idx[s['family_type']]
    rng = random.Random(seed)
    rng.shuffle(samples)
    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]
    n = len(samples)
    n_tr = int(0.7 * n); n_va = int(0.1 * n)
    return samples[:n_tr], samples[n_tr:n_tr + n_va], samples[n_tr + n_va:], label_list


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['rfam'], default='rfam')
    p.add_argument('--out_dir', default='results/multimodality/seq_vs_seqtext/rfam')
    p.add_argument('--n_epochs', type=int, default=6)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--method', default='binary_seb',
                   choices=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--body', default='D',
                   choices=['A', 'B', 'C', 'D', 'E'])
    p.add_argument('--seq_budget', type=int, default=256)
    p.add_argument('--text_budget', type=int, default=256)
    p.add_argument('--top_n_classes', type=int, default=10)
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  dataset={args.dataset}  method={args.method}  body={args.body}',
          flush=True)

    if args.dataset == 'rfam':
        train, val, test, labels = load_rfam(top_n_classes=args.top_n_classes,
                                              max_samples=args.max_samples,
                                              seed=args.seed)
        # Map field names so encode_with_mask sees uniform "seq" / "text"
        for split in (train, val, test):
            for s in split:
                s['seq'] = s['sequence']
                s['text'] = s['comment']
    else:
        raise ValueError(args.dataset)

    print(f'  train={len(train)}  val={len(val)}  test={len(test)}  '
          f'labels={labels}', flush=True)
    train_dist = Counter(s['family_type'] for s in train)
    print(f'  train label dist: {dict(train_dist)}', flush=True)

    modalities = [('seq', args.seq_budget), ('text', args.text_budget)]
    stoi = build_separator_vocab(train, modalities)
    vocab_size = len(stoi)
    print(f'  vocab={vocab_size}', flush=True)

    summary = {'config': vars(args), 'modalities': [m[0] for m in modalities],
               'vocab_size': vocab_size, 'n_classes': len(labels),
               'labels': labels, 'splits': {'train': len(train),
                                              'val': len(val), 'test': len(test)},
               'conditions': {}}

    def run_condition(name, mask_all_but):
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        tr_ds = FlexibleDataset(train, stoi, modalities, 'label', mask_all_but=mask_all_but)
        va_ds = FlexibleDataset(val,   stoi, modalities, 'label', mask_all_but=mask_all_but)
        te_ds = FlexibleDataset(test,  stoi, modalities, 'label', mask_all_but=mask_all_but)
        m = MultiModalClassifier(vocab_size=vocab_size, embedding_dim=64,
                                  num_classes=len(labels), method=args.method,
                                  body_kind=args.body, num_bits=64,
                                  interdim='128', pad_idx=stoi[PAD]).to(device)
        r = train_eval(m, tr_ds, va_ds, te_ds, stoi, device,
                        n_epochs=args.n_epochs, lr=args.lr)
        summary['conditions'][name] = r
        print(f'  [{name:15s}] test_acc={r["test_acc"]:.4f}  '
              f'val={r["val_acc"]:.4f}  time={r["train_time_s"]:.1f}s', flush=True)
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    print('\n[A] seq_only', flush=True)
    run_condition('seq_only', mask_all_but={'seq'})

    print('\n[B] seq+text', flush=True)
    run_condition('seq+text', mask_all_but=None)

    # Optional bonus: text-only (does text alone solve the task?)
    print('\n[C] text_only', flush=True)
    run_condition('text_only', mask_all_but={'text'})

    a = summary['conditions']['seq_only']['test_acc']
    b = summary['conditions']['seq+text']['test_acc']
    c = summary['conditions']['text_only']['test_acc']
    summary['delta_seqtext_minus_seqonly'] = b - a
    summary['delta_textonly_minus_seqonly'] = c - a
    print(f'\n=== {args.dataset} comparison ===', flush=True)
    print(f'  seq only       : {a:.4f}', flush=True)
    print(f'  seq + text     : {b:.4f}  (delta vs seq_only: {b-a:+.4f})', flush=True)
    print(f'  text only      : {c:.4f}  (delta vs seq_only: {c-a:+.4f})', flush=True)
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {args.out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
