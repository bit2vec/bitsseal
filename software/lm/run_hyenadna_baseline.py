#!/usr/bin/env python3
"""HyenaDNA-tiny baseline on Genomic Benchmarks.

Loads LongSafari/hyenadna-tiny-1k-seqlen-hf and fine-tunes for sequence
classification. Single-nucleotide tokenization (vocab=5: A C G T N), Hyena
operator (not transformer attention), max context 1024 nt — fits the ~250 bp
GB human_promoters task without modification.

Reports test accuracy + parameter count for direct comparison with our
WikiText→DNA fine-tune experiment.
"""
import argparse, functools, json, os, random, re, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import datasets
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification


GB_HF_IDS = {
    'gb_human_promoters': 'katarinagresova/Genomic_Benchmarks_human_nontata_promoters',
    'gb_human_enhancers': 'katarinagresova/Genomic_Benchmarks_human_enhancers_cohn',
    'gb_human_enhancers_ensembl': 'katarinagresova/Genomic_Benchmarks_human_enhancers_ensembl',
    'gb_human_ocr_ensembl': 'katarinagresova/Genomic_Benchmarks_human_ocr_ensembl',
    'gb_human_regulatory': 'katarinagresova/Genomic_Benchmarks_human_ensembl_regulatory',
    'gb_demo_human_or_worm': 'katarinagresova/Genomic_Benchmarks_demo_human_or_worm',
    'gb_demo_coding_vs_intergenomic': 'katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs',
    'gb_dummy_mouse_enhancers_ensembl': 'katarinagresova/Genomic_Benchmarks_dummy_mouse_enhancers_ensembl',
}

# Nucleotide Transformer downstream tasks (Dalla-Torre et al., NeurIPS 2023).
_NT_REPO = 'InstaDeepAI/nucleotide_transformer_downstream_tasks'
for _c in ['H3', 'H3K14ac', 'H3K36me3', 'H3K4me1', 'H3K4me2', 'H3K4me3',
           'H3K79me3', 'H3K9ac', 'H4', 'H4ac',
           'enhancers', 'enhancers_types',
           'promoter_all', 'promoter_no_tata', 'promoter_tata',
           'splice_sites_donors', 'splice_sites_acceptors', 'splice_sites_all']:
    GB_HF_IDS[f'nt_{_c}'] = (_NT_REPO, _c)


def load_gb(name='gb_human_promoters', max_len=512, seed=42):
    entry = GB_HF_IDS[name]
    if isinstance(entry, tuple):
        repo, task_filter = entry
        ds = datasets.load_dataset(repo)
        ds = {sp: ds[sp].filter(lambda r: r['task'] == task_filter) for sp in ds}
    else:
        ds = datasets.load_dataset(entry)
    full_train = list(ds['train'])
    test = list(ds['test'])
    rng = random.Random(seed); rng.shuffle(full_train)
    n_val = max(1, int(len(full_train) * 0.1))
    val = full_train[:n_val]; train = full_train[n_val:]
    n_classes = max(int(r['label']) for r in full_train) + 1
    seq_key = 'seq' if 'seq' in full_train[0] else 'sequence'

    def proc(rows):
        out = []
        for r in rows:
            seq = r[seq_key].upper()
            if len(seq) > max_len:
                seq = seq[:max_len]
            out.append({'seq': seq, 'label': int(r['label'])})
        return out
    return proc(train), proc(val), proc(test), n_classes


_RC_TBL = str.maketrans('ACGTN', 'TGCAN')

def revcomp(s): return s.translate(_RC_TBL)[::-1]


class HyenaDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=1024, rc_aug=False, training=False):
        self.exs = examples
        self.tok = tokenizer
        self.max_len = max_len
        self.rc_aug = rc_aug
        self.training = training

    def __len__(self): return len(self.exs)

    def __getitem__(self, i):
        ex = self.exs[i]
        seq = ex['seq']
        if self.rc_aug and self.training and random.random() < 0.5:
            seq = revcomp(seq)
        enc = self.tok(seq, padding='max_length',
                        truncation=True, max_length=self.max_len,
                        return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0)
                              if 'attention_mask' in enc else None,
            'label': torch.tensor(ex['label'], dtype=torch.long),
        }


def collate(batch):
    out = {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
    }
    if batch[0]['attention_mask'] is not None:
        out['attention_mask'] = torch.stack([b['attention_mask'] for b in batch])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(GB_HF_IDS.keys()),
                   default='gb_human_promoters')
    p.add_argument('--out_dir', default=None)
    p.add_argument('--model_id', default='LongSafari/hyenadna-tiny-1k-seqlen-hf')
    p.add_argument('--max_len', type=int, default=512)
    p.add_argument('--n_epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=6e-4)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--from_pretrained', action='store_true', default=True,
                   help='Load HyenaDNA pretrained weights (default)')
    p.add_argument('--from_scratch', action='store_true',
                   help='Override: random init at same architecture')
    p.add_argument('--rc_aug', action='store_true',
                   help='Reverse-complement augmentation (paper protocol)')
    args = p.parse_args()

    if args.out_dir is None:
        suffix = 'pretrained' if not args.from_scratch else 'random'
        args.out_dir = f'results/multimodality/hyenadna_baseline/{args.dataset}_{suffix}'
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  dataset={args.dataset}  '
          f'init={"pretrained" if not args.from_scratch else "random"}',
          flush=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

    train, val, test, n_classes = load_gb(args.dataset, max_len=args.max_len,
                                            seed=args.seed)
    print(f'  train={len(train)}  val={len(val)}  test={len(test)}  '
          f'n_classes={n_classes}', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True,
                                         num_labels=n_classes,
                                         problem_type='single_label_classification')

    if args.from_scratch:
        model = AutoModelForSequenceClassification.from_config(
            config, trust_remote_code=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_id, config=config, trust_remote_code=True,
            ignore_mismatched_sizes=True)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  HyenaDNA params: total={n_params:,}  trainable={n_trainable:,}',
          flush=True)
    model = model.to(device)

    coll = functools.partial(collate)
    tr_loader = DataLoader(HyenaDataset(train, tokenizer, args.max_len,
                                          rc_aug=args.rc_aug, training=True),
                            batch_size=args.batch_size, shuffle=True,
                            collate_fn=coll)
    va_loader = DataLoader(HyenaDataset(val, tokenizer, args.max_len),
                            batch_size=args.batch_size, shuffle=False,
                            collate_fn=coll)
    te_loader = DataLoader(HyenaDataset(test, tokenizer, args.max_len),
                            batch_size=args.batch_size, shuffle=False,
                            collate_fn=coll)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def evaluate(loader):
        model.eval()
        c = t = 0
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for batch in loader:
                ids = batch['input_ids'].to(device)
                lab = batch['label'].to(device)
                out = model(input_ids=ids)
                pred = out.logits.argmax(-1)
                c += (pred == lab).sum().item()
                t += lab.size(0)
        return c / max(t, 1)

    t0 = time.time()
    best_val, best_state = 0.0, None
    for ep in range(args.n_epochs):
        model.train()
        for batch in tr_loader:
            ids = batch['input_ids'].to(device)
            lab = batch['label'].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(input_ids=ids, labels=lab)
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = evaluate(va_loader)
        print(f'    epoch {ep+1}/{args.n_epochs}  val_acc={v:.4f}', flush=True)
        if v > best_val:
            best_val = v
            best_state = {k: vv.detach().cpu().clone()
                          for k, vv in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.time() - t0
    test_acc = evaluate(te_loader)

    ckpt_path = os.path.join(args.out_dir, 'model.pt')
    torch.save({
        'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'optimizer_state_dict': opt.state_dict(),
        'epochs_completed': args.n_epochs,
        'best_val_acc': float(best_val),
        'test_acc': float(test_acc),
        'config': vars(args),
    }, ckpt_path)
    print(f'Saved checkpoint to {ckpt_path}', flush=True)

    summary = {
        'config': vars(args), 'model_id': args.model_id,
        'init': 'random' if args.from_scratch else 'pretrained',
        'n_params': n_params, 'n_trainable': n_trainable,
        'val_acc': float(best_val), 'test_acc': float(test_acc),
        'train_time_s': train_time,
        'epochs_completed': args.n_epochs,
        'splits': {'train': len(train), 'val': len(val), 'test': len(test)},
        'n_classes': n_classes,
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n=== HyenaDNA-tiny on {args.dataset} ===', flush=True)
    print(f'  init: {summary["init"]}  params: {n_params:,}', flush=True)
    print(f'  test_acc: {test_acc:.4f}  val_acc: {best_val:.4f}  '
          f'time: {train_time:.1f}s', flush=True)
    print(f'Wrote {args.out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
