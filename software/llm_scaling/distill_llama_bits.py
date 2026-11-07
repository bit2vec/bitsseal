#!/usr/bin/env python3
"""Llama-3.2-1B embedding swap with Bits encoder.

Pipeline:
  1. Load pretrained Llama-3.2-1B from third_party/llama32_1b/.
  2. Extract input embedding E in R^{V x d}, V=128256, d=2048.
  3. Train a Bits FFN epsilon: {0,1}^64 -> R^{2048} via L2 distillation
     min sum_t || epsilon(b(t)) - E[t] ||_2^2 over all V tokens.
  4. Replace the model's embed_tokens with epsilon AND tie lm_head to it.
  5. Evaluate PPL on a WikiText-103 test slice with both:
       (a) original pretrained embedding (baseline)
       (b) Bits-distilled embedding (ours)
  6. Report PPL gap, embedding compression ratio, training time.

Single-GPU friendly: distillation uses ~5GB peak, eval uses ~6GB.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(THIS, '..')))
from binary_subword_sentiment import BinarySubwordConcateOnehot


def get_bit_codes(vocab_size, num_bits=64):
    """Random fixed bit code per token id, frozen after init."""
    rng = np.random.RandomState(42)
    codes = rng.randint(0, 2, size=(vocab_size, num_bits)).astype(np.float32)
    return torch.from_numpy(codes)


class BitsLlamaEmbedding(nn.Module):
    """Drop-in replacement for nn.Embedding that uses a Bits FFN encoder.

    Identical signature to nn.Embedding(V, d): __call__(input_ids) -> (B, L, d).
    """
    def __init__(self, vocab_size, embed_dim, num_bits=64, interdim='256'):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embed_dim
        # Bits encoder: 64 -> interdim -> embed_dim
        self.encoder = BinarySubwordConcateOnehot(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            num_bits=num_bits,
            interdim=interdim,
            relu_dropout=0.0,
            subword_binary_file='',  # generates fresh
        )

    def forward(self, ids):
        return self.encoder(ids)


def distill(target_E, bits_emb, n_steps=3000, lr=1e-3, batch_size=4096, log_every=200):
    """Train bits_emb to L2-match target_E by sampling token ids."""
    device = next(bits_emb.parameters()).device
    V, d = target_E.shape
    target_E = target_E.detach()
    opt = torch.optim.Adam(bits_emb.parameters(), lr=lr)
    bits_emb.train()
    losses = []
    t0 = time.time()
    for step in range(n_steps):
        idx = torch.randint(0, V, (batch_size,), device=device)
        target = target_E[idx]
        pred = bits_emb(idx)
        loss = F.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if (step + 1) % log_every == 0:
            recent = np.mean(losses[-log_every:])
            print(f'  distill step {step+1}/{n_steps}  L2={recent:.4f}  elapsed={time.time()-t0:.0f}s')
    return losses


@torch.no_grad()
def eval_ppl(model, tokenizer, text, device, max_seq_len=512, batch_size=2):
    """Compute perplexity on `text` chunks."""
    enc = tokenizer(text, return_tensors='pt')['input_ids'][0]
    n_chunks = enc.size(0) // max_seq_len
    if n_chunks == 0:
        return float('nan')
    chunks = enc[:n_chunks * max_seq_len].view(n_chunks, max_seq_len).to(device)
    nll_sum, n_tok = 0.0, 0
    model.eval()
    for i in range(0, n_chunks, batch_size):
        batch = chunks[i:i+batch_size]
        out = model(input_ids=batch, labels=batch)
        # out.loss is mean loss; multiply by token count for sum
        n = batch.numel()
        nll_sum += float(out.loss) * n
        n_tok += n
    return math.exp(nll_sum / n_tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', default='third_party/llama32_1b')
    ap.add_argument('--out_dir', default='results/llm_scaling/llama1b_distill')
    ap.add_argument('--n_distill_steps', type=int, default=3000)
    ap.add_argument('--distill_lr', type=float, default=1e-3)
    ap.add_argument('--distill_batch', type=int, default=4096)
    ap.add_argument('--interdim', type=str, default='256')
    ap.add_argument('--num_bits', type=int, default=64)
    ap.add_argument('--eval_max_seq_len', type=int, default=512)
    ap.add_argument('--eval_batch_size', type=int, default=2)
    ap.add_argument('--eval_max_chars', type=int, default=200000,
                    help='cap eval text length to keep within 1-2 minutes')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  cuda_mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB' if device.type=='cuda' else 'CPU')

    # 1. Load Llama
    print(f'Loading {args.model_dir}...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16)
    model = model.to(device)
    V = model.config.vocab_size
    d = model.config.hidden_size
    print(f'  vocab_size={V}  hidden_size={d}')
    n_emb = V * d
    n_bits_enc = args.num_bits * int(args.interdim) + int(args.interdim) * d + int(args.interdim) + d
    print(f'  pretrained embedding: {n_emb:,} params = {n_emb*2/1e6:.1f} MB (bf16)')
    print(f'  Bits encoder est:     {n_bits_enc:,} params = {n_bits_enc*4/1e6:.2f} MB (fp32)')
    print(f'  compression ratio:    {n_emb / n_bits_enc:.1f}x')

    # 2. Eval baseline PPL on WikiText-103 test slice
    print('Loading WikiText-103 test slice...')
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
    text = '\n'.join(ds['text'])[:args.eval_max_chars]
    print(f'  eval text chars: {len(text):,}')
    print('Eval baseline PPL (pretrained embedding)...')
    t0 = time.time()
    ppl_base = eval_ppl(model, tokenizer, text, device, args.eval_max_seq_len, args.eval_batch_size)
    print(f'  baseline PPL = {ppl_base:.3f}  ({time.time()-t0:.0f}s)')

    # 3. Extract pretrained embedding E
    print('Extracting pretrained embedding...')
    E = model.get_input_embeddings().weight.detach().clone().to(device).float()  # (V, d)
    print(f'  E shape={tuple(E.shape)}  dtype={E.dtype}')

    # 4. Build Bits embedding and distill
    print('Building Bits encoder and distilling...')
    bits_emb = BitsLlamaEmbedding(V, d, num_bits=args.num_bits, interdim=args.interdim).to(device)
    n_params = sum(p.numel() for p in bits_emb.parameters())
    print(f'  Bits encoder params: {n_params:,} ({n_params*4/1e6:.2f} MB fp32)')
    t0 = time.time()
    losses = distill(E, bits_emb,
                     n_steps=args.n_distill_steps,
                     lr=args.distill_lr,
                     batch_size=args.distill_batch)
    distill_time = time.time() - t0
    print(f'  distill time: {distill_time:.0f}s  final L2={losses[-1]:.4f}')

    # 5. Evaluate PPL with Bits embedding swapped in
    print('Eval PPL with Bits-distilled embedding...')
    bits_emb.eval()
    # Replace embedding by hand: monkeypatch the embed_tokens forward
    orig_emb = model.get_input_embeddings()
    # Cache full Bits embedding once (Bits epsilon evaluated on every token id)
    with torch.no_grad():
        all_ids = torch.arange(V, device=device, dtype=torch.long)
        # Process in chunks to avoid one giant matmul that may OOM
        E_bits = []
        chunk = 16384
        for i in range(0, V, chunk):
            E_bits.append(bits_emb(all_ids[i:i+chunk]).to(torch.bfloat16))
        E_bits = torch.cat(E_bits, dim=0)
    print(f'  cached Bits embedding: {tuple(E_bits.shape)}')

    # Patch the model. Llama-3.2-1B uses tie_word_embeddings=True, so embed_tokens
    # and lm_head must point to the same tensor; replacing only embed_tokens leaves
    # lm_head pointing at the original pretrained data and breaks PPL.
    with torch.no_grad():
        new_emb = nn.Embedding.from_pretrained(E_bits, freeze=True).to(device)
        model.set_input_embeddings(new_emb)
        # Force lm_head to share the same weight (preserves tied behavior).
        if model.config.tie_word_embeddings:
            model.lm_head.weight = new_emb.weight
            print('  also retied lm_head.weight to the new Bits embedding')
    t0 = time.time()
    ppl_bits = eval_ppl(model, tokenizer, text, device, args.eval_max_seq_len, args.eval_batch_size)
    print(f'  Bits PPL (tied input+output swap) = {ppl_bits:.3f}  ({time.time()-t0:.0f}s)')

    # 6. Report
    out = {
        'model': args.model_dir,
        'vocab_size': V,
        'hidden_size': d,
        'pretrained_embedding_params': int(n_emb),
        'bits_encoder_params': int(n_params),
        'compression_ratio': float(n_emb / n_params),
        'distill_steps': args.n_distill_steps,
        'distill_final_l2': float(losses[-1]),
        'distill_time_s': float(distill_time),
        'eval_max_chars': args.eval_max_chars,
        'eval_max_seq_len': args.eval_max_seq_len,
        'ppl_baseline': float(ppl_base),
        'ppl_bits_swap': float(ppl_bits),
        'ppl_gap': float(ppl_bits - ppl_base),
        'tied_lm_head': bool(model.config.tie_word_embeddings),
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('\n=== RESULTS ===')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
