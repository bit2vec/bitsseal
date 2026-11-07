#!/usr/bin/env python3
"""Feasibility experiment: the necessity argument for vocabulary independence.

Fuse ICD-10-CM + ICD-10-PCS + SNOMED CT + RxNorm + drug names + clinical text.
The union vocabulary is ~892K symbols (public terminology sizes). A standard
subword model must allocate a |V| x d embedding table PLUS its Adam optimizer
states (2x); at realistic embedding widths this exceeds a single GPU and OOMs
*before any useful compute*. The Bits encoder is a constant ~33K-param module
regardless of |V| or the number of fused modalities, so it fits and trains.

For a sweep of embedding widths d, we instantiate both models, run one
train step on a small batch, and record peak GPU memory + fit/OOM status.
This is data-agnostic: peak memory is governed by |V|*d, a public fact.
"""
import argparse, gc, json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# public terminology sizes (rows that a subword model must embed)
TERMS = {'ICD-10-CM': 74000, 'ICD-10-PCS': 78000, 'SNOMED CT': 350000,
         'RxNorm': 280000, 'NDC/drug names': 60000, 'clinical text (subword)': 50000}
UNION_VOCAB = sum(TERMS.values())   # ~892000


class SubwordModel(nn.Module):
    def __init__(self, vocab, d, n_classes=10):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)          # the |V| x d table
        self.fc1 = nn.Linear(d, d); self.fc2 = nn.Linear(d, n_classes)
    def forward(self, ids):
        e = self.emb(ids).mean(1)
        return self.fc2(F.relu(self.fc1(e)))


class BitsModel(nn.Module):
    """byte->bits->one-hot(2)->FFN; constant params, independent of |V|."""
    def __init__(self, d, num_bits=64, n_classes=10):
        super().__init__()
        onehot = F.one_hot(torch.arange(2)).float()
        self.bit = nn.Embedding(2, 2, _weight=onehot); self.bit.requires_grad_(False)
        self.proj = nn.Linear(2 * num_bits, d)
        self.fc1 = nn.Linear(d, d); self.fc2 = nn.Linear(d, n_classes)
        self.num_bits = num_bits
    def byte_to_bits(self, b):
        B, T = b.shape
        nb = (self.num_bits + 7) // 8
        padded = F.pad(b, (0, nb - 1), value=0)
        w = padded.unfold(1, nb, 1)
        bits = torch.stack([(w >> (7 - i)) & 1 for i in range(8)], dim=-1).view(B, T, -1)
        return bits[:, :, :self.num_bits]
    def forward(self, byte_ids):
        bits = self.byte_to_bits(byte_ids)
        e = self.proj(self.bit(bits).view(*byte_ids.shape, -1)).mean(1)
        return self.fc2(F.relu(self.fc1(e)))


def trial(method, d, vocab, device, seq_len=64, bs=16, n_classes=10):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        if method == 'subword':
            model = SubwordModel(vocab, d, n_classes).to(device)
            x = torch.randint(0, vocab, (bs, seq_len), device=device)
        else:
            model = BitsModel(d, n_classes=n_classes).to(device)
            x = torch.randint(0, 256, (bs, seq_len * 8), device=device)
        y = torch.randint(0, n_classes, (bs,), device=device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)   # Adam => 2x states
        crit = nn.CrossEntropyLoss()
        for _ in range(3):                                    # a few real steps
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        params = sum(p.numel() for p in model.parameters())
        emb_params = sum(p.numel() for p in (model.emb.parameters() if method == 'subword'
                                             else model.proj.parameters()))
        del model, opt, x, y; gc.collect(); torch.cuda.empty_cache()
        return {'status': 'fit', 'peak_gb': round(peak, 3),
                'params': params, 'emb_params': emb_params}
    except torch.cuda.OutOfMemoryError as e:
        gc.collect(); torch.cuda.empty_cache()
        return {'status': 'OOM', 'peak_gb': None, 'error': str(e)[:80]}
    except RuntimeError as e:
        gc.collect(); torch.cuda.empty_cache()
        s = 'OOM' if 'out of memory' in str(e).lower() else 'ERROR'
        return {'status': s, 'error': str(e)[:80]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results/medcode_feasibility/summary.json')
    p.add_argument('--vocab', type=int, default=UNION_VOCAB)
    p.add_argument('--dims', nargs='+', type=int, default=[256, 512, 768, 1024, 2048, 4096])
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = 'cuda'
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {name}  total={total:.1f} GB')
    print(f'union vocab = {args.vocab:,}  (terminologies: {TERMS})')
    summary = {'gpu': name, 'gpu_total_gb': round(total, 1), 'vocab': args.vocab,
               'terminologies': TERMS, 'results': {}}
    print(f'\n{"d":>6} {"subword":>26} {"bits":>26}')
    for d in args.dims:
        sw = trial('subword', d, args.vocab, device)
        bi = trial('bits', d, args.vocab, device)
        summary['results'][d] = {'subword': sw, 'bits': bi}
        def fmt(r):
            if r['status'] == 'fit':
                return f"fit {r['peak_gb']:.2f}GB ({r['emb_params']/1e6:.0f}M emb)"
            return r['status']
        print(f'{d:>6} {fmt(sw):>26} {fmt(bi):>26}')
        json.dump(summary, open(args.out, 'w'), indent=2)
    print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()
