"""Llama-3.2 1B perplexity comparison: Subword vs EmByte vs 001.

Pipeline:
  1. Load Llama-3.2 1B (LlamaForCausalLM).
  2. Distill embed_tokens.weight (V=128256, d=2048) into:
     - 001: 64-bit hash + FFN (interdim=4096) -> ~520K params (vs 263M -> 504x)
     - EmByte: 8-byte one-hot + FFN (interdim=4096) -> ~17M params (vs 263M -> 15x)
  3. Replace embed_tokens with distilled module.
  4. Evaluate WikiText-2 perplexity for original / EmByte / 001.

For LLMs, the lm_head is also tied to embed_tokens by default. To preserve the
output projection we materialise it from the binary FFN at forward time.

Memory: 1B params * 2 (BF16) ~= 2.5 GB plus distill buffers.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from binary_geneformer import (
    BinaryGeneTokenEmbedding, ByteGeneTokenEmbedding,
)

from transformers import AutoModelForCausalLM, AutoTokenizer


def fmt_n(n):
    if n > 1e9: return f'{n/1e9:.2f}B'
    if n > 1e6: return f'{n/1e6:.2f}M'
    if n > 1e3: return f'{n/1e3:.2f}K'
    return str(n)


def distill(target, embed_module, n_steps=2000, batch=4096, lr=3e-4, device='cuda'):
    V = target.size(0); target = target.to(device)
    embed_module = embed_module.to(device)
    learnable = [p for p in embed_module.parameters() if p.requires_grad]
    @torch.no_grad()
    def full_mse():
        chunk = 32768; sse = 0; n = 0
        for s in range(0, V, chunk):
            e = min(s + chunk, V)
            pred = embed_module(torch.arange(s, e, device=device))
            sse += ((pred - target[s:e]) ** 2).sum().item()
            n   += pred.numel()
        return sse / n

    init = full_mse()
    print(f"  init MSE: {init:.6f}")
    opt = torch.optim.AdamW(learnable, lr=lr)
    t0 = time.time()
    for step in range(1, n_steps+1):
        ids = torch.randint(0, V, (batch,), device=device)
        loss = ((embed_module(ids) - target[ids])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, n_steps//10) == 0 or step == 1:
            print(f"    step {step:5d}/{n_steps}  batch_mse={loss.item():.6f}  ({time.time()-t0:.1f}s)")
    final = full_mse()
    print(f"  final MSE: {final:.6f}  ({init/max(final,1e-12):.1f}x)")
    return {'init_mse': init, 'final_mse': final}


def evaluate_ppl(model, tokenizer, text, device, max_length=2048, stride=1024):
    """Standard sliding-window PPL eval."""
    encodings = tokenizer(text, return_tensors='pt')
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end = 0
    n_eval = 0
    model.eval()
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        input_ids = encodings.input_ids[:, begin:end].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            out = model(input_ids, labels=target_ids)
            neg_log_likelihood = out.loss * trg_len
        nlls.append(neg_log_likelihood.item())
        n_eval += trg_len
        prev_end = end
        if end == seq_len: break
    avg_nll = sum(nlls) / max(n_eval, 1)
    return float(np.exp(avg_nll))


class TiedDecoderEmbed(nn.Module):
    """Wrap a BinaryGeneTokenEmbedding so it materialises its full V x d table
    on demand for use as Llama's lm_head (which expects nn.Linear(d, V) where
    weight is the embedding transposed). We expose `weight` property for tying.
    """
    def __init__(self, embed_module: nn.Module, padding_idx=None):
        super().__init__()
        self.embed_module = embed_module
        self.padding_idx = padding_idx

    def forward(self, input_ids):
        return self.embed_module(input_ids)

    @property
    def weight(self):
        # Materialise full V x d -- expensive but rare (only called by lm_head tying path)
        device = next(self.embed_module.parameters()).device
        V = self.embed_module.num_embeddings
        chunk = 16384
        rows = []
        for s in range(0, V, chunk):
            e = min(s + chunk, V)
            rows.append(self.embed_module(torch.arange(s, e, device=device)))
        return torch.cat(rows, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_dir', default='./third_party/llama32_1b')
    p.add_argument('--out_dir', default='./results/llama_binary')
    p.add_argument('--distill_steps', type=int, default=2000)
    p.add_argument('--num_bits', type=int, default=64)
    p.add_argument('--interdim', type=int, default=4096)
    p.add_argument('--max_length', type=int, default=2048)
    p.add_argument('--stride', type=int, default=1024)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print(f"Loading Llama-3.2 1B from {args.ckpt_dir}")
    print("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)
    model_orig = AutoModelForCausalLM.from_pretrained(
        args.ckpt_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    target_embed = model_orig.model.embed_tokens.weight.detach().clone()
    V, D = target_embed.shape
    total = sum(p.numel() for p in model_orig.parameters())
    print(f"  V={V}, d={D}, total params={fmt_n(total)}")
    print(f"  embed share = {V*D / total * 100:.2f}%")

    # Load eval text (WikiText-2 test as a sample)
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        text = '\n\n'.join([row['text'] for row in ds])
        text = text[:200000]  # keep PPL eval tractable: ~50k tokens
        print(f"  eval text: {len(text)} chars")
    except Exception as e:
        print(f'  WARN cannot load wikitext: {e}; using small sample')
        text = "The quick brown fox jumps over the lazy dog. " * 5000

    # ---- Subword baseline ----
    print(f"\n=== Subword (original Llama-3.2 1B) ===")
    model_orig = model_orig.to(device)
    t0 = time.time()
    ppl_sub = evaluate_ppl(model_orig, tokenizer, text, device,
                            max_length=args.max_length, stride=args.stride)
    print(f"  PPL = {ppl_sub:.2f}  ({time.time()-t0:.1f}s)")
    sub_params = sum(p.numel() for p in model_orig.parameters())
    del model_orig; torch.cuda.empty_cache()

    # ---- Distill 001 ----
    print(f"\n=== Distill 001 (64-bit, interdim={args.interdim}) ===")
    bin_embed = BinaryGeneTokenEmbedding(V, D, num_bits=args.num_bits,
                                          interdim=args.interdim, dropout=0.0)
    dstats_b = distill(target_embed, bin_embed, n_steps=args.distill_steps,
                        device=device)

    # ---- Distill EmByte ----
    print(f"\n=== Distill EmByte (8-byte, interdim={args.interdim}) ===")
    byte_embed = ByteGeneTokenEmbedding(V, D, num_bytes=8, interdim=args.interdim,
                                         dropout=0.0)
    dstats_bt = distill(target_embed, byte_embed, n_steps=args.distill_steps,
                         device=device)

    # ---- Build patched models for PPL eval ----
    def patched_model(new_embed):
        m = AutoModelForCausalLM.from_pretrained(
            args.ckpt_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        # Llama uses embed_tokens for both input and weight-tied output (lm_head).
        # We replace the input side; for the lm_head, we materialise weights via
        # the distilled embedding matrix once at evaluation start.
        m.model.embed_tokens = new_embed.to(next(m.parameters()).device)
        # Re-tie lm_head to the new module's materialised weights
        with torch.no_grad():
            mat = new_embed.to(device)(torch.arange(V, device=device))
            m.lm_head.weight.data = mat.to(m.lm_head.weight.device)
        return m

    print(f"\n=== EmByte (distilled byte FFN) ===")
    bin_embed = bin_embed.cpu(); byte_embed = byte_embed.cpu()
    m_byte = patched_model(byte_embed).to(device)
    t0 = time.time()
    ppl_byte = evaluate_ppl(m_byte, tokenizer, text, device,
                             max_length=args.max_length, stride=args.stride)
    print(f"  PPL = {ppl_byte:.2f}  ({time.time()-t0:.1f}s)")
    byte_params = sum(p.numel() for p in m_byte.parameters())
    del m_byte; torch.cuda.empty_cache()

    print(f"\n=== 001 (distilled binary FFN) ===")
    m_bin = patched_model(bin_embed).to(device)
    t0 = time.time()
    ppl_bin = evaluate_ppl(m_bin, tokenizer, text, device,
                            max_length=args.max_length, stride=args.stride)
    print(f"  PPL = {ppl_bin:.2f}  ({time.time()-t0:.1f}s)")
    bin_params = sum(p.numel() for p in m_bin.parameters())
    del m_bin; torch.cuda.empty_cache()

    summary = {
        'config': vars(args),
        'distillation': {'binary': dstats_b, 'byte': dstats_bt},
        'results': {
            'Subword': {'PPL': ppl_sub, 'params': sub_params, 'mb_fp32': sub_params*4/1e6},
            'EmByte':  {'PPL': ppl_byte, 'params': byte_params, 'mb_fp32': byte_params*4/1e6},
            '001':     {'PPL': ppl_bin, 'params': bin_params, 'mb_fp32': bin_params*4/1e6},
        },
    }
    json.dump(summary, open(os.path.join(args.out_dir, 'summary.json'), 'w'),
              indent=2, default=str)

    print("\n" + "=" * 80)
    print(f"{'method':<10} {'params':>10} {'mb':>8} {'PPL':>9}")
    print('-' * 80)
    for name, r in [('Subword', summary['results']['Subword']),
                    ('EmByte', summary['results']['EmByte']),
                    ('001', summary['results']['001'])]:
        print(f"{name:<10} {fmt_n(r['params']):>10} {r['mb_fp32']:>8.1f} {r['PPL']:>9.2f}")


if __name__ == '__main__':
    main()
