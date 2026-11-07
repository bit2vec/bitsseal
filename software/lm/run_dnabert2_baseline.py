#!/usr/bin/env python3
"""DNABERT-2 baseline on Genomic Benchmarks Tier-1.

Loads zhihan1996/DNABERT-2-117M and fine-tunes per-task. Uses their published
BPE-on-DNA tokenizer (V~4096 learned merges) and BERT-style classifier head.

Standard baseline, reported in DNABERT-2's own paper Table 2.
"""
import argparse, functools, json, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

GB_HF_IDS = {
    'gb_human_promoters':       'katarinagresova/Genomic_Benchmarks_human_nontata_promoters',
    'gb_human_enhancers':       'katarinagresova/Genomic_Benchmarks_human_enhancers_cohn',
    'gb_human_enhancers_ensembl': 'katarinagresova/Genomic_Benchmarks_human_enhancers_ensembl',
    'gb_human_ocr_ensembl':     'katarinagresova/Genomic_Benchmarks_human_ocr_ensembl',
    'gb_human_regulatory':      'katarinagresova/Genomic_Benchmarks_human_ensembl_regulatory',
    'gb_demo_human_or_worm':    'katarinagresova/Genomic_Benchmarks_demo_human_or_worm',
}


def load_gb(name, seed=42):
    ds = datasets.load_dataset(GB_HF_IDS[name])
    full_train = list(ds['train']); test = list(ds['test'])
    rng = random.Random(seed); rng.shuffle(full_train)
    nv = max(1, int(len(full_train) * 0.1))
    val = full_train[:nv]; train = full_train[nv:]
    seq_key = 'seq' if 'seq' in full_train[0] else 'sequence'
    n_classes = max(int(r['label']) for r in full_train) + 1
    return ([(r[seq_key], int(r['label'])) for r in train],
            [(r[seq_key], int(r['label'])) for r in val],
            [(r[seq_key], int(r['label'])) for r in test],
            n_classes)


class DNADataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len):
        self.pairs = pairs; self.tok = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        seq, lab = self.pairs[i]
        enc = self.tok(seq, padding='max_length', truncation=True,
                        max_length=self.max_len, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'label': torch.tensor(lab, dtype=torch.long)}


def evaluate(model, loader, device):
    model.eval(); c = t = 0
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for batch in loader:
            ids = batch['input_ids'].to(device); mask = batch['attention_mask'].to(device)
            lab = batch['label'].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            c += (out.logits.argmax(-1) == lab).sum().item(); t += lab.size(0)
    return c / max(t, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(GB_HF_IDS.keys()), required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--max_len', type=int, default=128)  # DNABERT-2 paper uses ~128
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=3e-5)  # DNABERT-2 paper lr
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'DNABERT-2 baseline on {args.dataset}  device={device}', flush=True)

    train, val, test, n_classes = load_gb(args.dataset, seed=args.seed)
    print(f'  train={len(train)}  val={len(val)}  test={len(test)}  n_classes={n_classes}',
          flush=True)

    tokenizer = AutoTokenizer.from_pretrained('zhihan1996/DNABERT-2-117M', trust_remote_code=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 3
    config = AutoConfig.from_pretrained('zhihan1996/DNABERT-2-117M', trust_remote_code=True,
                                          num_labels=n_classes,
                                          problem_type='single_label_classification')
    config.pad_token_id = pad_id  # DNABERT-2's custom BertConfig doesn't wire this through kwargs
    model = AutoModelForSequenceClassification.from_pretrained(
        'zhihan1996/DNABERT-2-117M', config=config, trust_remote_code=True,
        ignore_mismatched_sizes=True).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  DNABERT-2 params: {n_params:,}', flush=True)

    tr = DataLoader(DNADataset(train, tokenizer, args.max_len), batch_size=args.batch_size, shuffle=True)
    va = DataLoader(DNADataset(val, tokenizer, args.max_len), batch_size=args.batch_size, shuffle=False)
    te = DataLoader(DNADataset(test, tokenizer, args.max_len), batch_size=args.batch_size, shuffle=False)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    t0 = time.time(); best_val = -1; best_state = None
    for ep in range(args.n_epochs):
        model.train()
        for batch in tr:
            ids = batch['input_ids'].to(device); mask = batch['attention_mask'].to(device)
            lab = batch['label'].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=mask, labels=lab)
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        v = evaluate(model, va, device)
        print(f'  epoch {ep+1}/{args.n_epochs}  val_acc={v:.4f}', flush=True)
        if v > best_val:
            best_val = v
            best_state = {k: vv.detach().cpu().clone() for k, vv in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    test_acc = evaluate(model, te, device)
    print(f'  test_acc={test_acc:.4f}  time={time.time()-t0:.1f}s', flush=True)

    summary = {'config': vars(args), 'n_params': n_params, 'best_val': float(best_val),
               'test_acc': float(test_acc), 'time_s': time.time() - t0,
               'splits': {'train': len(train), 'val': len(val), 'test': len(test)},
               'n_classes': n_classes}
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved {args.out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
