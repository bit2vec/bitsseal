#!/usr/bin/env python3
"""HyenaDNA-large + Bits encoder fine-tune at k=22 on Genomic Benchmarks.

Identical to hyena_large_bits_finetune.py except the tokenization is k=22
k-mers (not k=1 single nucleotides). This isolates fix-1a from the recipe-
mismatch analysis: HF hg38-pretrained body + flat lr 6e-4 + 10 epochs, but
with k=22 tokenization swapping out the 5-token nucleotide vocab for a
~500K-observed-vocab 22-mer space encoded via the V-indep Bits FFN.

Pairs with:
  - hyena_large_bits_finetune.py        (k=1 "broken" version)
  - hyena_large_subword_kmer.py         (k=22 V-dep Subword TABLE)
  - hyena_wikitext_to_dna.py            (full proposed recipe: k=22 + WT pretrain + lr 1e-4)
"""
import argparse, functools, json, os, random, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hyena_wikitext_to_dna import (
    GB_HF_IDS, load_gb, build_kmer_vocab, TokenDataset, kmerize, revcomp
)
from binary_subword_sentiment import BinarySubwordConcateOnehot


def _replace_input_embeddings(model, new_emb):
    if hasattr(model, 'set_input_embeddings'):
        model.set_input_embeddings(new_emb); return
    raise RuntimeError('cannot find input embeddings on HyenaDNA model')


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
    p.add_argument('--num_bits', type=int, default=64)
    p.add_argument('--interdim', type=str, default='256')
    p.add_argument('--rc_aug', action='store_true')
    p.add_argument('--min_freq', type=int, default=2)
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

    # Build model: HF hg38-pretrained Hyena-large body + Bits encoder at k=22
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True,
                                        num_labels=n_classes,
                                        problem_type='single_label_classification')
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, config=config, trust_remote_code=True,
        ignore_mismatched_sizes=True)
    d_model = getattr(config, 'd_model', getattr(config, 'hidden_size', 256))
    bits_emb = BinarySubwordConcateOnehot(
        num_embeddings=vocab_size, embedding_dim=d_model,
        num_bits=args.num_bits, interdim=args.interdim, relu_dropout=0.1,
        subword_binary_file='')
    _replace_input_embeddings(model, bits_emb)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  HyenaDNA-large + Bits k={args.k}: total={n_params:,}  trainable={n_train:,}  d={d_model}  V={vocab_size}', flush=True)

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
            best_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    test_acc = evaluate(te)
    elapsed = time.time() - t0
    print(f'\n=== HyenaDNA-large + Bits k={args.k} on {args.dataset} ===', flush=True)
    print(f'  V={vocab_size}  test={test_acc:.4f}  val={best_val:.4f}  time={elapsed:.1f}s', flush=True)

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump({
            'config': vars(args),
            'n_classes': n_classes, 'vocab_size': vocab_size,
            'n_params': n_params, 'n_trainable': n_train,
            'test_acc': test_acc, 'val_acc': best_val,
            'train_time_s': elapsed,
            'splits': {'train': len(train), 'val': len(val), 'test': len(test)},
        }, f, indent=2)


if __name__ == '__main__':
    main()
