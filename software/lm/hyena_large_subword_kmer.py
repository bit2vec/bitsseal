#!/usr/bin/env python3
"""HyenaDNA-large + custom k-mer subword embedding TABLE (Setup A for Paper B).

The "fair subword-at-large-V baseline" against HyenaDNA-large+Bits at k=22:
  - Same body: HyenaDNA-large pretrained from HF (1m-seqlen)
  - Same k: k=22 (large V regime, ~50-500K unique 22-mers in training set)
  - Different encoder: standard nn.Embedding(V, d_model) instead of Bits FFN

Saves model.pt for extension.
"""
import argparse, functools, json, os, random, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hyena_wikitext_to_dna import (
    GB_HF_IDS, load_gb, build_kmer_vocab, TokenDataset, kmerize, revcomp
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(GB_HF_IDS.keys()), required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--model_id', default='LongSafari/hyenadna-large-1m-seqlen-hf')
    p.add_argument('--k', type=int, default=22)
    p.add_argument('--max_len', type=int, default=512)
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=6e-4)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rc_aug', action='store_true')
    p.add_argument('--min_freq', type=int, default=2)
    p.add_argument('--random_body', action='store_true',
                   help='Skip from_pretrained; instantiate body from config (random init).')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  dataset={args.dataset}  model={args.model_id}  k={args.k}', flush=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

    train, val, test, n_classes = load_gb(
        args.dataset, k=args.k, max_len=args.max_len, seed=args.seed,
        keep_raw=args.rc_aug)
    stoi = build_kmer_vocab([s['tokens'] for s in train], min_freq=args.min_freq)
    vocab_size = len(stoi)
    print(f'  train={len(train)}  val={len(val)}  test={len(test)}  n_classes={n_classes}  V={vocab_size}', flush=True)

    tr_ds = TokenDataset(train, stoi, max_len=args.max_len,
                         rc_aug=args.rc_aug, training=True)
    va_ds = TokenDataset(val, stoi, max_len=args.max_len)
    te_ds = TokenDataset(test, stoi, max_len=args.max_len)

    def collate(batch):
        ids = torch.stack([b[0] for b in batch])
        lab = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return {'input_ids': ids, 'label': lab}

    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate, num_workers=4, pin_memory=True,
                    persistent_workers=True)
    va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate, num_workers=2, pin_memory=True,
                    persistent_workers=True)
    te = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate, num_workers=2, pin_memory=True,
                    persistent_workers=True)

    # Build model: load HyenaDNA-large pretrained body, swap embedding for custom k-mer table
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True,
                                        num_labels=n_classes,
                                        problem_type='single_label_classification')
    if args.random_body:
        # Random body init — for the "no pretrain anywhere" V-dep control
        model = AutoModelForSequenceClassification.from_config(
            config, trust_remote_code=True)
        print('  [random_body] body initialised from config (no HF pretrained weights)', flush=True)
    else:
        # Use default HF vocab size for from_pretrained, then swap the embedding
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_id, config=config, trust_remote_code=True,
            ignore_mismatched_sizes=True)
    d_model = getattr(config, 'd_model', getattr(config, 'hidden_size', 256))
    # Replace with new k-mer-sized embedding table (random init)
    new_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
    nn.init.normal_(new_emb.weight, mean=0, std=d_model ** -0.5)
    model.set_input_embeddings(new_emb)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb_params = vocab_size * d_model
    print(f'  HyenaDNA-large + k-mer table: total={n_params:,}  '
          f'trainable={n_train:,}  d={d_model}  emb_table={emb_params:,} '
          f'({100*emb_params/n_params:.1f}% of params)', flush=True)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def evaluate(loader):
        model.eval(); c, t = 0, 0
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for b in loader:
                ids = b['input_ids'].to(device); lab = b['label'].to(device)
                pred = model(input_ids=ids).logits.argmax(-1)
                c += (pred == lab).sum().item(); t += lab.size(0)
        return c / max(t, 1)

    t0 = time.time(); best_val, best_state = 0.0, None
    for ep in range(args.n_epochs):
        model.train()
        for b in tr:
            ids = b['input_ids'].to(device); lab = b['label'].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(input_ids=ids, labels=lab)
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = evaluate(va)
        print(f'    epoch {ep+1}/{args.n_epochs}  val={v:.4f}', flush=True)
        if v > best_val:
            best_val = v
            best_state = {k: vv.detach().cpu().clone() for k, vv in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    train_time = time.time() - t0
    test_acc = evaluate(te)

    ckpt_path = os.path.join(args.out_dir, 'model.pt')
    torch.save({
        'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'optimizer_state_dict': opt.state_dict(),
        'epochs_completed': args.n_epochs,
        'best_val_acc': float(best_val),
        'test_acc': float(test_acc),
        'config': vars(args),
        'vocab_size': vocab_size,
    }, ckpt_path)
    print(f'Saved checkpoint to {ckpt_path}', flush=True)

    summary = {
        'config': vars(args), 'model_id': args.model_id,
        'n_params': n_params, 'n_trainable': n_train,
        'emb_table_params': emb_params, 'vocab_size': vocab_size,
        'val_acc': float(best_val), 'test_acc': float(test_acc),
        'train_time_s': train_time, 'epochs_completed': args.n_epochs,
        'splits': {'train': len(train), 'val': len(val), 'test': len(test)},
        'n_classes': n_classes,
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n=== HyenaDNA-large + k={args.k} subword table on {args.dataset} ===', flush=True)
    print(f'  V={vocab_size}  emb={emb_params:,}  test={test_acc:.4f}  val={best_val:.4f}  time={train_time:.1f}s', flush=True)


if __name__ == '__main__':
    main()
