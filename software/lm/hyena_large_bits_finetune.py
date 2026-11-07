#!/usr/bin/env python3
"""HyenaDNA-large + Bits encoder fine-tune on Genomic Benchmarks.

Loads HyenaDNA-large pretrained body from HF (no WikiText pretrain, no tiny
checkpoint), replaces input_embeddings with a fresh Bits encoder, fine-tunes
end-to-end on a GB classification task. Saves model.pt for later extension.

Pairs with run_hyenadna_baseline.py (baseline) and caduceus_with_bits.py
(Caduceus side of the body-agnostic Bits demonstration).
"""
import argparse, functools, json, os, random, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_hyenadna_baseline import GB_HF_IDS, load_gb, HyenaDataset, collate
from binary_subword_sentiment import BinarySubwordConcateOnehot


def _replace_input_embeddings(model, new_emb):
    if hasattr(model, 'set_input_embeddings'):
        model.set_input_embeddings(new_emb); return
    raise RuntimeError('cannot find input embeddings on HyenaDNA model')


def build_hyena_large_bits(vocab_size, n_classes, model_id, num_bits=64, interdim='256'):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True,
                                        num_labels=n_classes,
                                        problem_type='single_label_classification')
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, config=config, trust_remote_code=True,
        ignore_mismatched_sizes=True)
    d_model = getattr(config, 'd_model', getattr(config, 'hidden_size', 256))
    bits_emb = BinarySubwordConcateOnehot(
        num_embeddings=vocab_size, embedding_dim=d_model,
        num_bits=num_bits, interdim=interdim, relu_dropout=0.1,
        subword_binary_file='')
    _replace_input_embeddings(model, bits_emb)
    return model, d_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(GB_HF_IDS.keys()), required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--model_id', default='LongSafari/hyenadna-large-1m-seqlen-hf')
    p.add_argument('--max_len', type=int, default=1024)
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=6e-4)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_bits', type=int, default=64)
    p.add_argument('--interdim', type=str, default='256')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  dataset={args.dataset}  model={args.model_id}', flush=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

    train, val, test, n_classes = load_gb(args.dataset, max_len=args.max_len, seed=args.seed)
    print(f'  train={len(train)} val={len(val)} test={len(test)} n_classes={n_classes}', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size or len(tokenizer)
    model, d_model = build_hyena_large_bits(
        vocab_size, n_classes, args.model_id,
        num_bits=args.num_bits, interdim=args.interdim)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  HyenaDNA-large+Bits: total={n_params:,}  trainable={n_train:,}  d={d_model}', flush=True)

    coll = functools.partial(collate)
    tr = DataLoader(HyenaDataset(train, tokenizer, args.max_len, training=True),
                    batch_size=args.batch_size, shuffle=True, collate_fn=coll)
    va = DataLoader(HyenaDataset(val, tokenizer, args.max_len),
                    batch_size=args.batch_size, shuffle=False, collate_fn=coll)
    te = DataLoader(HyenaDataset(test, tokenizer, args.max_len),
                    batch_size=args.batch_size, shuffle=False, collate_fn=coll)

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
    }, ckpt_path)
    print(f'Saved checkpoint to {ckpt_path}', flush=True)

    summary = {
        'config': vars(args), 'model_id': args.model_id,
        'n_params': n_params, 'n_trainable': n_train,
        'val_acc': float(best_val), 'test_acc': float(test_acc),
        'train_time_s': train_time, 'epochs_completed': args.n_epochs,
        'splits': {'train': len(train), 'val': len(val), 'test': len(test)},
        'n_classes': n_classes,
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n=== HyenaDNA-large + Bits on {args.dataset} ===', flush=True)
    print(f'  test={test_acc:.4f}  val={best_val:.4f}  time={train_time:.1f}s', flush=True)
    print(f'Wrote {args.out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
