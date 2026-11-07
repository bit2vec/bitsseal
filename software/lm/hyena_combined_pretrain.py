#!/usr/bin/env python3
"""Simultaneous A: pretrain Hyena+Bits from random init on a UNIFIED corpus
combining WikiText-103 (BPE-50K) and hg38 (k=6 mer, V=4K), with a single
Bits encoder over V=54,353 (50,257 BPE + 4,096 k-mer + specials).

Mixing strategy: PyTorch ConcatDataset over the two pre-tokenized streams.
Samples are drawn proportionally to corpus size — DNA dominates because hg38
is far larger than WikiText.

Then fine-tune on GB human_promoters at DNA-only k=6 (V=4,098) — the
classifier sees a fresh sub-vocab (just the k-mer portion) and the Bits
encoder regenerates the bit-mapping for it. Body weights transfer.
"""
import argparse, copy, functools, json, os, random, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset

import datasets
from transformers import GPT2Tokenizer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, THIS_DIR)

from hyena_wikitext_to_dna import (
    HYENA_MODEL_ID, GB_HF_IDS, build_hyena_causal_lm, build_hyena_classifier,
    transfer_pretrain_to_classifier,
    kmerize, build_kmer_vocab, load_gb,
    TokenDataset, collate as gb_collate,
)
import hg38_loader


# Vocab layout:
#   BPE 0..50256                  (50,257 entries from GPT-2 tokenizer)
#   k=6 mer offset 50257..54354   (4,098 entries from build_canonical_kmer_vocab)
#                                 (= 4,096 unique k-mers + <pad> + <unk>)
BPE_VOCAB_SIZE = 50_257
KMER_VOCAB_SIZE = 4_098                                  # k=6: <pad>+<unk>+4^6
KMER_OFFSET = BPE_VOCAB_SIZE                             # 50,257
UNIFIED_VOCAB_SIZE = BPE_VOCAB_SIZE + KMER_VOCAB_SIZE    # 54,355


def load_wikitext_tokens(seq_length=256, stride=128, max_chars=None):
    """Return numpy int32 array of GPT-2 BPE token IDs for WikiText-103 train."""
    print('Loading WikiText-103...', flush=True)
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    ds = datasets.load_dataset('wikitext', 'wikitext-103-v1', split='train')
    chunks = []
    buf = []; buf_chars = 0; total_chars = 0
    chunk_size = 1024 * 1024
    def _flush():
        nonlocal buf, buf_chars
        if buf:
            chunks.append(tok.encode(' '.join(buf)))
            buf = []; buf_chars = 0
    for item in ds:
        text = item['text'].strip()
        if text and not text.startswith('='):
            buf.append(text); buf_chars += len(text); total_chars += len(text)
            if buf_chars >= chunk_size:
                _flush()
            if max_chars and total_chars >= max_chars:
                break
    _flush()
    total = sum(len(c) for c in chunks)
    arr = np.empty(total, dtype=np.int32)
    o = 0
    for c in chunks:
        arr[o:o+len(c)] = c; o += len(c)
    print(f'  WikiText tokens: {len(arr):,}', flush=True)
    return arr


class StreamDataset(Dataset):
    def __init__(self, tokens, seq_length=256, stride=128, max_samples=None):
        self.tokens = tokens
        self.seq_length = seq_length; self.stride = stride
        n = (len(tokens) - seq_length) // stride
        self._n = min(n, max_samples) if max_samples else max(n, 0)
    def __len__(self): return self._n
    def __getitem__(self, i):
        s = i * self.stride
        x = torch.as_tensor(self.tokens[s:s+self.seq_length].copy(), dtype=torch.long)
        y = torch.as_tensor(self.tokens[s+1:s+self.seq_length+1].copy(), dtype=torch.long)
        return {'input_ids': x, 'labels': y}


def transfer_combined_pretrain_to_classifier(model, pretrain_state, verbose=True):
    """Like transfer_pretrain_to_classifier but accepts unified-vocab pretrain
    state. Same: skip vocab-dependent buffer + LM head."""
    md = model.state_dict()
    transferred, skipped = [], []
    for k, v in pretrain_state.items():
        if 'subword2binary' in k or 'subword2bytes' in k or 'subword' in k:
            skipped.append((k, 'vocab-dependent buffer'))
            continue
        if k.startswith('lm_head.'):
            skipped.append((k, 'LM head replaced by classifier'))
            continue
        if k in md and md[k].shape == v.shape:
            md[k] = v.clone()
            transferred.append((k, tuple(v.shape)))
        else:
            skipped.append((k, 'shape mismatch / absent'))
    model.load_state_dict(md, strict=False)
    if verbose:
        n_t = sum(int(np.prod(t[1])) for t in transferred)
        print(f'  transferred {len(transferred)} tensors ({n_t:,} params)',
              flush=True)


def pretrain(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

    # Build hg38 k=6 corpus
    print('\n[hg38 corpus]', flush=True)
    dna_tokens, dna_stoi = hg38_loader.build_corpus(args.chroms, k=args.k,
                                                     stride=1)
    # Offset DNA token IDs into the unified vocab
    dna_tokens_offset = dna_tokens.astype(np.int64) + KMER_OFFSET

    # Build WikiText BPE corpus
    print('\n[WikiText corpus]', flush=True)
    text_tokens = load_wikitext_tokens(seq_length=args.seq_length,
                                        stride=args.stride)

    # Build datasets and combine via ConcatDataset
    print(f'\n[combined dataset]', flush=True)
    text_ds = StreamDataset(text_tokens, seq_length=args.seq_length,
                             stride=args.stride, max_samples=args.max_text_samples)
    dna_ds = StreamDataset(dna_tokens_offset, seq_length=args.seq_length,
                            stride=args.stride, max_samples=args.max_dna_samples)
    print(f'  text samples: {len(text_ds):,}', flush=True)
    print(f'  DNA samples : {len(dna_ds):,}', flush=True)
    combined = ConcatDataset([text_ds, dna_ds])
    print(f'  combined    : {len(combined):,}', flush=True)
    print(f'  text fraction: {len(text_ds) / max(len(combined), 1):.1%}',
          flush=True)

    # Build model — random init, vocab=54,353
    print(f'\n[model] random init, V={UNIFIED_VOCAB_SIZE}', flush=True)
    model, config = build_hyena_causal_lm(UNIFIED_VOCAB_SIZE, args.encoder,
                                            num_bits=args.num_bits,
                                            interdim=args.interdim,
                                            dropout=args.dropout,
                                            max_seq_len=args.seq_length)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  trainable params: {n_params:,}', flush=True)
    model = model.to(device)

    # Train
    tr = DataLoader(combined, batch_size=args.batch_size, shuffle=True,
                     num_workers=2, pin_memory=True)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                      betas=(0.9, 0.98))
    crit = nn.CrossEntropyLoss()
    n_steps_per_epoch = len(tr)
    total_steps = args.n_epochs * n_steps_per_epoch
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                            total_steps=max(total_steps, 1),
                                            pct_start=0.05)
    print(f'\n[train] {n_steps_per_epoch} steps/epoch, '
          f'{total_steps} total', flush=True)

    t0 = time.time()
    for ep in range(args.n_epochs):
        model.train()
        running = 0.0; n_batch = 0
        ep_t0 = time.time()
        for step, batch in enumerate(tr):
            ids = batch['input_ids'].to(device, non_blocking=True)
            lab = batch['labels'].to(device, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(input_ids=ids).logits
                loss = crit(logits.reshape(-1, logits.size(-1)),
                            lab.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            running += float(loss.item()); n_batch += 1
            if step % max(1, n_steps_per_epoch // 8) == 0:
                print(f'  ep {ep+1} step {step}/{n_steps_per_epoch} '
                      f'loss={loss.item():.3f}', flush=True)
            ckpt_every = getattr(args, 'ckpt_every', 0)
            if ckpt_every > 0 and step > 0 and step % ckpt_every == 0:
                torch.save({'model_state_dict':
                            {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()},
                            'config': vars(args), 'vocab_size': UNIFIED_VOCAB_SIZE,
                            'epoch': ep + 1, 'step': step, 'partial': True},
                           os.path.join(args.out_dir, 'combined_pretrain_partial.pt'))
                print(f'  [ckpt] saved partial at ep {ep+1} step {step}',
                      flush=True)
        ep_time = time.time() - ep_t0
        avg_loss = running / max(n_batch, 1)
        print(f'  ep {ep+1}/{args.n_epochs}  avg_loss={avg_loss:.3f}  ({ep_time:.1f}s)',
              flush=True)
        torch.save({'model_state_dict': {k: v.detach().cpu().clone()
                                          for k, v in model.state_dict().items()},
                    'config': vars(args), 'vocab_size': UNIFIED_VOCAB_SIZE,
                    'avg_loss': avg_loss, 'epoch': ep + 1},
                   os.path.join(args.out_dir, 'combined_pretrain.pt'))
    print(f'\npretrain done in {time.time() - t0:.1f}s', flush=True)
    return os.path.join(args.out_dir, 'combined_pretrain.pt')


def finetune(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
    train, val, test, n_classes = load_gb('gb_human_promoters', k=args.k,
                                            max_len=args.max_len, seed=args.seed)
    print(f'  train={len(train)} val={len(val)} test={len(test)}', flush=True)
    stoi = build_kmer_vocab([s['tokens'] for s in train])
    print(f'  vocab={len(stoi)}', flush=True)
    tr_ds = TokenDataset(train, stoi, max_len=args.max_len)
    va_ds = TokenDataset(val, stoi, max_len=args.max_len)
    te_ds = TokenDataset(test, stoi, max_len=args.max_len)

    summary = {'config': vars(args), 'vocab_size': len(stoi),
               'n_classes': n_classes, 'splits': {'train': len(train),
                                                   'val': len(val),
                                                   'test': len(test)},
               'conditions': {}}

    def make_classifier():
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
        m, _ = build_hyena_classifier(
            vocab_size=len(stoi), encoder_type=args.encoder,
            num_classes=n_classes, num_bits=args.num_bits,
            interdim=args.interdim, dropout=args.dropout, max_seq_len=args.max_len)
        return m

    def eval_loader(model, loader):
        model.eval()
        c = t = 0
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for batch in loader:
                ids = batch['input_ids'].to(device); lab = batch['label'].to(device)
                pred = model(input_ids=ids).logits.argmax(-1)
                c += (pred == lab).sum().item(); t += lab.size(0)
        return c / max(t, 1)

    def train_one(model, name):
        coll = functools.partial(gb_collate)
        tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=coll)
        va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)
        te = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)
        opt = optim.AdamW(model.parameters(), lr=args.lr_finetune, weight_decay=0.01)
        t0 = time.time()
        best_val, best_state = 0.0, None
        for ep in range(args.n_epochs_finetune):
            model.train()
            for batch in tr:
                ids = batch['input_ids'].to(device); lab = batch['label'].to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    out = model(input_ids=ids, labels=lab)
                opt.zero_grad(); out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            v = eval_loader(model, va)
            print(f'    [{name}] epoch {ep+1}/{args.n_epochs_finetune} val={v:.4f}',
                  flush=True)
            if v > best_val:
                best_val = v
                best_state = {k: vv.detach().cpu().clone()
                              for k, vv in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        train_time = time.time() - t0
        return {'val_acc': float(best_val),
                'test_acc': float(eval_loader(model, te)),
                'train_time_s': train_time,
                'n_params': sum(p.numel() for p in model.parameters())}

    if args.pretrain_init and args.pretrain_init.lower() != 'none':
        print(f'\n[W] load combined pretrain {args.pretrain_init}', flush=True)
        ck = torch.load(args.pretrain_init, map_location='cpu', weights_only=False)
        pretrain_state = ck.get('model_state_dict', ck)
        m = make_classifier()
        transfer_combined_pretrain_to_classifier(m, pretrain_state)
        m = m.to(device)
        rW = train_one(m, 'W')
        summary['conditions']['W_combined_pretrain'] = rW
        print(f'  W: test={rW["test_acc"]:.4f}  val={rW["val_acc"]:.4f}', flush=True)
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    print(f'\n=== Combined pretrain on gb_human_promoters (k={args.k}) ===',
          flush=True)
    if 'W_combined_pretrain' in summary['conditions']:
        print(f'  W (combined): {summary["conditions"]["W_combined_pretrain"]["test_acc"]:.4f}',
              flush=True)
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['pretrain', 'finetune', 'both'],
                   default='both')
    p.add_argument('--encoder', default='binary_seb',
                   choices=['standard', 'binary_seb', 'byte_seb'])
    p.add_argument('--out_dir', required=True)
    p.add_argument('--pretrain_init', default=None,
                   help='For finetune phase: combined-pretrain ckpt')
    p.add_argument('--chroms', nargs='+',
                   default=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10',
                            'chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20',
                            'chr21','chr22','chrX','chrY'])
    p.add_argument('--k', type=int, default=6)

    p.add_argument('--n_epochs', type=int, default=1)
    p.add_argument('--ckpt_every', type=int, default=50000,
                   help='Save partial checkpoint every N steps (0=disable)')
    p.add_argument('--n_epochs_finetune', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=6e-4)
    p.add_argument('--lr_finetune', type=float, default=1e-4)
    p.add_argument('--seq_length', type=int, default=256)
    p.add_argument('--stride', type=int, default=128)
    p.add_argument('--max_text_samples', type=int, default=None)
    p.add_argument('--max_dna_samples', type=int, default=None)
    p.add_argument('--max_len', type=int, default=512)
    p.add_argument('--num_bits', type=int, default=64)
    p.add_argument('--interdim', type=str, default='256')
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.phase in ('pretrain', 'both'):
        ckpt = pretrain(args)
        if args.phase == 'both':
            args.pretrain_init = ckpt
    if args.phase in ('finetune', 'both'):
        if args.phase == 'finetune' and not args.pretrain_init:
            args.pretrain_init = os.path.join(args.out_dir, 'combined_pretrain.pt')
        finetune(args)


if __name__ == '__main__':
    main()
