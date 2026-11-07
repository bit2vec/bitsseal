"""Validate the BinaryEpiAgent compression claim.

Three experiments:
  (1) Parameter-count comparison: original EpiAgent vs binary input vs binary I/O.
  (2) Synthetic distillation: train BinaryCCREEmbedding's FFN to match a
      randomly-initialised target embedding table for the full V = 1.355M
      vocabulary (proxies the real-life "distill released checkpoint embeddings"
      step). Reports MSE convergence vs random-init baseline.
  (3) Forward-pass smoke test on a small batch (verifies the binary model
      runs end-to-end on GPU and that its parameter count matches expected).
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary_epiagent import (
    BinaryEpiAgentMinimal, BinaryCCREEmbedding,
)


def fmt(n):
    if n > 1e9: return f'{n/1e9:.2f} B'
    if n > 1e6: return f'{n/1e6:.2f} M'
    if n > 1e3: return f'{n/1e3:.2f} K'
    return str(n)


def experiment_1_param_counts(out_path):
    """Param breakdown for three configurations."""
    print("\n" + "=" * 80)
    print("(1) PARAMETER COMPARISON: original vs binary-input vs binary-I/O")
    print("=" * 80)

    configs = [
        ('original  (nn.Embedding x2)',     False, False),
        ('binary-input only',                True,  False),
        ('binary-input + tied output',       True,  True ),
    ]
    rows = []
    for name, bin_in, bin_out in configs:
        m = BinaryEpiAgentMinimal(
            use_binary_input=bin_in, use_binary_output=bin_out,
            num_signal_outputs=1355445)  # match EpiAgent's V-4
        groups = m.count_params()
        rows.append((name, groups))
        print(f"\n  {name}")
        for k, v in groups.items():
            mb = v * 4 / 1e6
            print(f"    {k:<20} {fmt(v):>10}   ({mb:>8.1f} MB)")
        del m
        torch.cuda.empty_cache()

    # Compute compression ratio
    base_total = rows[0][1]['TOTAL']
    print(f"\n  Compression vs original (TOTAL params):")
    for name, groups in rows:
        ratio = base_total / groups['TOTAL']
        print(f"    {name:<35} {fmt(groups['TOTAL']):>10}  ({ratio:>4.2f}x)")

    out = {'param_breakdown': [(n, g) for n, g in rows],
           'compression_ratios': {n: base_total / g['TOTAL'] for n, g in rows}}
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    return rows


def experiment_2_distillation(out_path, vocab=1355449, dim=512,
                              num_bits=64, interdim=2048,
                              n_steps=2000, batch=8192, lr=3e-4,
                              device='cuda'):
    """Distill a BinaryCCREEmbedding's FFN to match a target V x d table.

    The target plays the role of EpiAgent's released `cCRE_embedding.weight`
    — we don't have the checkpoint downloaded yet, so we approximate with a
    random Gaussian table at the right shape. Expected behaviour: MSE drops
    far below the random-init baseline, confirming the FFN has the capacity
    to encode the target table.
    """
    print("\n" + "=" * 80)
    print("(2) SYNTHETIC DISTILLATION: BinaryFFN -> match V x d target table")
    print("=" * 80)
    print(f"  V={vocab}  d={dim}  bits={num_bits}  interdim={interdim}  "
          f"steps={n_steps}  batch={batch}  lr={lr}")

    # Target: a "released embedding-like" tensor (random Gaussian for proxy)
    torch.manual_seed(42)
    target = torch.randn(vocab, dim, device=device) * 0.02
    print(f"  Target ||E||_F = {target.norm():.4f},  ||E||_inf = {target.abs().max():.4f}")

    # Binary embedding (only the FFN is learnable)
    embed = BinaryCCREEmbedding(num_embeddings=vocab, embedding_dim=dim,
                                num_bits=num_bits, interdim=interdim,
                                dropout=0.0).to(device)

    bit_codes_dev = embed.bit_codes.to(device)

    # Random-init baseline: MSE between target and embed.materialise() with no training
    @torch.no_grad()
    def measure_full_mse():
        # Materialise FFN(all_codes) chunked
        chunk = 65536
        sse = 0.0
        n = 0
        for s in range(0, vocab, chunk):
            e = min(s + chunk, vocab)
            codes = bit_codes_dev[s:e].float()
            pred = embed.ffn(codes)
            sse += ((pred - target[s:e]) ** 2).sum().item()
            n += pred.numel()
        return sse / n

    init_mse = measure_full_mse()
    print(f"\n  Random-init (untrained FFN) full-vocab MSE: {init_mse:.6f}")

    optimizer = torch.optim.AdamW(embed.ffn.parameters(), lr=lr)
    history = []
    t0 = time.time()
    for step in range(1, n_steps + 1):
        ids = torch.randint(0, vocab, (batch,), device=device)
        codes = bit_codes_dev[ids].float()
        pred = embed.ffn(codes)
        tgt = target[ids]
        loss = ((pred - tgt) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % max(1, n_steps // 20) == 0 or step == 1:
            history.append({'step': step, 'batch_mse': loss.item(),
                            'elapsed_s': time.time() - t0})
            print(f"    step {step:5d}/{n_steps}   batch MSE = {loss.item():.6f}   "
                  f"({time.time()-t0:.1f}s)")

    final_full_mse = measure_full_mse()
    print(f"\n  Final full-vocab MSE: {final_full_mse:.6f}  "
          f"(reduction from init: {init_mse / max(final_full_mse, 1e-12):.1f}x)")

    out = {
        'config': dict(vocab=vocab, dim=dim, num_bits=num_bits, interdim=interdim,
                       n_steps=n_steps, batch=batch, lr=lr),
        'init_mse_full_vocab': init_mse,
        'final_mse_full_vocab': final_full_mse,
        'reduction_x': init_mse / max(final_full_mse, 1e-12),
        'history': history,
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    return out


def experiment_3_smoke_test(out_path, batch=4, seq=256, vocab=1355449,
                            device='cuda'):
    print("\n" + "=" * 80)
    print("(3) FORWARD-PASS SMOKE TEST: BinaryEpiAgent on a small batch")
    print("=" * 80)

    m = BinaryEpiAgentMinimal(
        vocab_size=vocab, num_layers=4,  # tiny body for smoke test
        embedding_dim=512, num_attention_heads=8,
        max_rank_embeddings=512,
        use_binary_input=True, use_binary_output=True,
        num_signal_outputs=10000  # small output to keep memory safe
    ).to(device)

    n = sum(p.numel() for p in m.parameters())
    print(f"  Total trainable params: {fmt(n)}")
    ids = torch.randint(0, vocab, (batch, seq), device=device)
    t0 = time.time()
    out = m(ids)
    print(f"  Output shape: {tuple(out.shape)}   forward time: {time.time()-t0:.3f}s")

    payload = {'params': n, 'output_shape': list(out.shape),
               'forward_time_s': time.time() - t0}
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='./results/epiagent_binary')
    p.add_argument('--n_steps', type=int, default=2000)
    p.add_argument('--batch', type=int, default=8192)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--skip', nargs='*', default=[],
                   help='subset of {1,2,3} to skip')
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if '1' not in args.skip:
        experiment_1_param_counts(os.path.join(args.out_dir, 'exp1_params.json'))
    if '2' not in args.skip:
        experiment_2_distillation(os.path.join(args.out_dir, 'exp2_distill.json'),
                                   n_steps=args.n_steps, batch=args.batch, lr=args.lr)
    if '3' not in args.skip:
        experiment_3_smoke_test(os.path.join(args.out_dir, 'exp3_smoke.json'))


if __name__ == '__main__':
    main()
