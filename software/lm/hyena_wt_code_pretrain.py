#!/usr/bin/env python3
"""Mixed Wiki + Code causal LM pretrain for Hyena+Bits.

WikiText-103 (default 80%) + Python code (default 20%) interleaved at sample
time. Same model architecture and BPE-50,257 vocab as hyena_wikitext_to_dna.py
pretrain. Saves pretrain.pt that can be loaded by --pretrain_ckpt in the
fine-tune phase.
"""
import argparse, json, os, random, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import datasets
from transformers import GPT2Tokenizer

# Re-use the model build + transfer helpers from the main script.
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hyena_wikitext_to_dna import build_hyena_causal_lm, make_encoder


def _tokenize_text_lines(tokenizer, lines, chunk_size=1024 * 1024):
    token_chunks, buf, buf_chars = [], [], 0
    def _flush():
        nonlocal buf, buf_chars
        if buf:
            token_chunks.append(tokenizer.encode(' '.join(buf)))
            buf = []; buf_chars = 0
    for text in lines:
        text = text.strip()
        if not text: continue
        buf.append(text); buf_chars += len(text)
        if buf_chars >= chunk_size:
            _flush()
    _flush()
    total = sum(len(c) for c in token_chunks)
    arr = np.empty(total, dtype=np.int64)
    off = 0
    for c in token_chunks:
        arr[off:off + len(c)] = c; off += len(c)
    return arr


def load_wikitext_tokens(tokenizer):
    print('Loading WikiText-103 train...', flush=True)
    ds = datasets.load_dataset('wikitext', 'wikitext-103-v1', split='train')
    lines = (item['text'] for item in ds
             if item['text'].strip() and not item['text'].strip().startswith('='))
    arr = _tokenize_text_lines(tokenizer, lines)
    print(f'  WikiText tokens: {len(arr):,}', flush=True)
    return arr


def load_code_tokens(tokenizer, max_lines=200_000):
    """Load Python code from CodeParrot-clean. Cap at max_lines to keep
    download and tokenization tractable."""
    print(f'Loading CodeParrot Python code (cap {max_lines} files)...', flush=True)
    ds = datasets.load_dataset('codeparrot/codeparrot-clean', split='train',
                                streaming=True)
    lines = []
    for i, item in enumerate(ds):
        if i >= max_lines: break
        lines.append(item['content'])
    arr = _tokenize_text_lines(tokenizer, lines)
    print(f'  Code tokens: {len(arr):,}', flush=True)
    return arr


class MixedDataset(Dataset):
    """Interleaves two token streams probabilistically at sample time."""
    def __init__(self, wiki_tokens, code_tokens, seq_length=256, stride=128,
                 code_ratio=0.2):
        self.wiki = wiki_tokens
        self.code = code_tokens
        self.seq_length = seq_length
        self.stride = stride
        self.code_ratio = code_ratio
        nw = (len(wiki_tokens) - seq_length) // stride
        nc = (len(code_tokens) - seq_length) // stride
        # Effective length: wiki dominates, code samples are virtual replicas
        self._n_wiki = max(nw, 0)
        self._n_code = max(nc, 0)
        # Per-epoch length = wiki + code samples, but we sample probabilistically
        self._n = self._n_wiki + int(self._n_wiki * code_ratio / (1 - code_ratio + 1e-9))
        print(f'  wiki samples={self._n_wiki:,}  code samples={self._n_code:,}',
              flush=True)
        print(f'  total samples / epoch={self._n:,}  code ratio={code_ratio:.2f}',
              flush=True)

    def __len__(self): return self._n

    def __getitem__(self, i):
        if random.random() < self.code_ratio and self._n_code > 0:
            j = random.randint(0, self._n_code - 1)
            arr = self.code; s = j * self.stride
        else:
            j = random.randint(0, self._n_wiki - 1)
            arr = self.wiki; s = j * self.stride
        x = torch.as_tensor(arr[s:s + self.seq_length].copy(), dtype=torch.long)
        y = torch.as_tensor(arr[s + 1:s + self.seq_length + 1].copy(), dtype=torch.long)
        return {'input_ids': x, 'labels': y}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--encoder', choices=['standard', 'binary_seb', 'byte_seb'],
                   default='binary_seb')
    p.add_argument('--code_ratio', type=float, default=0.2)
    p.add_argument('--code_max_files', type=int, default=100_000)
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=6e-4)
    p.add_argument('--seq_length', type=int, default=256)
    p.add_argument('--stride', type=int, default=128)
    p.add_argument('--num_bits', type=int, default=64)
    p.add_argument('--interdim', type=str, default='256')
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    vocab_size = tokenizer.vocab_size
    wiki = load_wikitext_tokens(tokenizer)
    code = load_code_tokens(tokenizer, max_lines=args.code_max_files)
    train_ds = MixedDataset(wiki, code, seq_length=args.seq_length,
                              stride=args.stride, code_ratio=args.code_ratio)

    model, config = build_hyena_causal_lm(vocab_size, args.encoder,
                                            num_bits=args.num_bits,
                                            interdim=args.interdim,
                                            dropout=args.dropout,
                                            max_seq_len=args.seq_length)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  trainable params: {n_params:,}', flush=True)
    print(f'  d_model={config.d_model}  n_layer={config.n_layer}  '
          f'vocab={config.vocab_size}', flush=True)
    model = model.to(device)

    tr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                     num_workers=2, pin_memory=True)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                      betas=(0.9, 0.98))
    crit = nn.CrossEntropyLoss()
    n_steps_per_epoch = len(tr)
    total_steps = args.n_epochs * n_steps_per_epoch
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                            total_steps=max(total_steps, 1),
                                            pct_start=0.05)
    print(f'  steps per epoch={n_steps_per_epoch}  total={total_steps}',
          flush=True)

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
            if step % max(1, n_steps_per_epoch // 5) == 0:
                print(f'  ep {ep+1} step {step}/{n_steps_per_epoch} '
                      f'loss={float(loss.item()):.3f}', flush=True)
        avg = running / max(n_batch, 1)
        print(f'  ep {ep+1}/{args.n_epochs}  train_loss={avg:.3f}  '
              f'({time.time()-ep_t0:.1f}s)', flush=True)

    train_time = time.time() - t0
    out_pt = os.path.join(args.out_dir, 'pretrain.pt')
    torch.save({'model_state_dict': model.state_dict(),
                'config': {'d_model': config.d_model, 'n_layer': config.n_layer,
                            'vocab_size': vocab_size, 'encoder': args.encoder,
                            'num_bits': args.num_bits, 'interdim': args.interdim,
                            'code_ratio': args.code_ratio}}, out_pt)
    summary = {'config': vars(args), 'n_params': n_params,
                'train_time_s': train_time,
                'wiki_tokens': int(len(wiki)), 'code_tokens': int(len(code))}
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nWrote {out_pt}', flush=True)
    print(f'  pretrain time: {train_time:.1f}s', flush=True)


if __name__ == '__main__':
    main()
