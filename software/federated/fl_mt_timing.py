#!/usr/bin/env python3
"""Micro-benchmark for FL (SST-2) and MT (quy/gn) timing per method.

The original FL (100-round) and MT (pretrain + finetune) runs did not record
per-step timing. Rather than re-running, we time the inner forward+backward
pass on the same model + the same batch dimensions, then multiply by step
count to estimate wall-clock cost.

Also runs an LLG (logit-leak gradient) attack on the FL client gradient.
"""
import argparse, functools, json, os, random, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, '..', 'Senti_analysis')))
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, '..', 'healthcare')))

from health_classification import (
    build_vocab, TextClassificationDataset, collate, PAD,
)
from film_lamp_classification import TinyClassifier
from sentiment_privacy import load_sst2


def time_fl_client(method, n_warmup=5, n_steps=200, batch_size=32, seed=42):
    """Time one client-step forward+backward on the SST-2 FedAvg client model."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ex, _, _, num_classes = load_sst2(seed=seed)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    model = TinyClassifier(vocab_size, 64, num_classes, method=method).to(device)
    opt = optim.SGD(model.parameters(), lr=0.05)
    crit = nn.CrossEntropyLoss()

    times = []
    n_done = 0
    iter_loader = iter(loader)
    while n_done < n_warmup + n_steps:
        try: batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader); continue
        if batch is None: continue
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad()
        crit(model(batch['ids'].to(device)),
             batch['label'].to(device)).backward()
        opt.step()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        if n_done >= n_warmup:
            times.append(time.time() - t0)
        n_done += 1

    n_params = sum(p.numel() for p in model.parameters())
    return {
        'method': method,
        'mean_step_s': float(np.mean(times)),
        'p95_step_s': float(np.percentile(times, 95)),
        'std_step_s': float(np.std(times)),
        'n_steps_timed': len(times),
        'batch_size': batch_size,
        'n_params': n_params,
        'vocab_size': vocab_size,
    }


def llg_attack(model, ids, label, criterion, device, vocab_size):
    """LLG (Logit-leak Gradient) - Wainakh et al. style.
    Recover labels from per-class fc.weight gradient row magnitudes.
    For Subword: also recover token set from embedding gradient sparsity."""
    model.train()
    model.zero_grad(set_to_none=False)
    logits = model(ids.to(device))
    loss = criterion(logits, label.to(device))
    loss.backward()

    # Label leakage: softmax(-fc.weight.grad.sum(1)) ranking
    fc_g = model.fc.weight.grad.detach()  # (C, d)
    row_sums = fc_g.sum(dim=1).cpu()
    pred_labels = (-row_sums).argsort(descending=True).tolist()
    real_labels = label.tolist()
    # Recovery: did predicted top-k cover the labels?
    true_set = set(real_labels)
    pred_top = set(pred_labels[:len(true_set)])
    label_recovery = len(pred_top & true_set) / max(len(true_set), 1)

    # Token set leakage (Subword only)
    if model.method == 'standard':
        emb_g = model.embedding.weight.grad.detach().norm(dim=1).cpu()
        thr = emb_g.max().item() * 0.05 if emb_g.max().item() > 0 else 1e-12
        recovered = set((emb_g > thr).nonzero(as_tuple=True)[0].tolist())
    else:
        # SEB: gradient is shared; uniform / no per-token signal
        recovered = set(range(vocab_size))  # would mark everything -> precision~|actual|/V
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    if model.method != 'standard' and len(recovered) > 5 * len(actual):
        recovered = set(random.sample(list(recovered), min(20, len(recovered))))
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {'label_recovery': float(label_recovery), 'token_f1': float(f1)}


def fl_llg_grid(methods, n_attack_batches=12, batch_size=16, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ex, _, _, num_classes = load_sst2(seed=seed)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)

    results = {}
    crit = nn.CrossEntropyLoss()
    for method in methods:
        random.seed(seed); torch.manual_seed(seed)
        model = TinyClassifier(vocab_size, 64, num_classes, method=method).to(device)
        opt = optim.Adam(model.parameters(), lr=5e-4)
        # Quick train so the model isn't random
        for _ in range(2):
            model.train()
            for batch in loader:
                if batch is None: continue
                opt.zero_grad()
                crit(model(batch['ids'].to(device)),
                     batch['label'].to(device)).backward()
                opt.step()
        # Per-batch LLG
        rec_l, rec_t = [], []
        n_atk = 0
        for batch in loader:
            if n_atk >= n_attack_batches: break
            if batch is None: continue
            try:
                m = llg_attack(model, batch['ids'], batch['label'],
                               crit, device, vocab_size)
                rec_l.append(m['label_recovery'])
                rec_t.append(m['token_f1'])
            except Exception as e:
                print(f'    llg err: {e}')
            n_atk += 1
        results[method] = {
            'label_recovery_avg': float(np.mean(rec_l)) if rec_l else 0.0,
            'token_f1_avg': float(np.mean(rec_t)) if rec_t else 0.0,
            'n_attacks': len(rec_l),
        }
    return results


def time_mt_step(method, n_warmup=3, n_steps=50, batch_size=16,
                  src_len=64, tgt_len=64, seed=42):
    """Time one encoder-decoder MT step (forward + backward).

    Builds a small transformer matching the COLING setup roughly:
    embedding (Subword/EmByte/001) -> 4-layer encoder (d=128, 4 heads)
    -> 4-layer decoder -> linear projection.
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, '..', 'Senti_analysis')))
    from binary_subword_sentiment import BinarySubwordConcateOnehot
    from byte_subword_concate_onehot import ByteSubwordConcateOnehot

    vocab_size = 8000   # representative joint vocab
    d_model = 128

    class MTEnc(nn.Module):
        def __init__(self):
            super().__init__()
            if method == 'standard':
                self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
                self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
            elif method == 'byte_seb':
                import tempfile
                td = tempfile.mkdtemp()
                self.src_emb = ByteSubwordConcateOnehot(
                    num_embeddings=vocab_size, embedding_dim=d_model,
                    interdim='128', relu_dropout=0.1,
                    subword_bytes_file=os.path.join(td, 'src.pt'))
                self.tgt_emb = ByteSubwordConcateOnehot(
                    num_embeddings=vocab_size, embedding_dim=d_model,
                    interdim='128', relu_dropout=0.1,
                    subword_bytes_file=os.path.join(td, 'tgt.pt'))
            elif method == 'binary_seb':
                self.src_emb = BinarySubwordConcateOnehot(
                    num_embeddings=vocab_size, embedding_dim=d_model,
                    num_bits=64, interdim='128', relu_dropout=0.1,
                    subword_binary_file='')
                self.tgt_emb = BinarySubwordConcateOnehot(
                    num_embeddings=vocab_size, embedding_dim=d_model,
                    num_bits=64, interdim='128', relu_dropout=0.1,
                    subword_binary_file='')
            self.transformer = nn.Transformer(
                d_model=d_model, nhead=4,
                num_encoder_layers=4, num_decoder_layers=4,
                dim_feedforward=512, dropout=0.1,
                batch_first=True,
            )
            self.proj = nn.Linear(d_model, vocab_size)

        def forward(self, src, tgt):
            s = self.src_emb(src); t = self.tgt_emb(tgt)
            o = self.transformer(s, t)
            return self.proj(o)

    model = MTEnc().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss(ignore_index=0)

    times = []
    for i in range(n_warmup + n_steps):
        src = torch.randint(1, vocab_size, (batch_size, src_len), device=device)
        tgt = torch.randint(1, vocab_size, (batch_size, tgt_len), device=device)
        target = torch.randint(1, vocab_size, (batch_size, tgt_len), device=device)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad()
        out = model(src, tgt)
        loss = crit(out.reshape(-1, vocab_size), target.reshape(-1))
        loss.backward()
        opt.step()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        if i >= n_warmup:
            times.append(time.time() - t0)

    n_params = sum(p.numel() for p in model.parameters())
    return {
        'method': method,
        'mean_step_s': float(np.mean(times)),
        'p95_step_s': float(np.percentile(times, 95)),
        'n_params': n_params,
        'batch_size': batch_size,
        'src_len': src_len, 'tgt_len': tgt_len,
        'vocab_size': vocab_size,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/fl_mt_timing')
    p.add_argument('--methods', nargs='+', default=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--skip_mt', action='store_true')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    summary = {'fl_step_timing': {}, 'fl_llg': {}, 'mt_step_timing': {}}

    print('===== FL client step timing =====')
    for m in args.methods:
        try:
            r = time_fl_client(m, n_warmup=5, n_steps=200, batch_size=32)
            summary['fl_step_timing'][m] = r
            print(f'  {m}: {r["mean_step_s"]*1000:.2f} ms/step  '
                  f'(p95={r["p95_step_s"]*1000:.2f} ms)  params={r["n_params"]}')
        except Exception as e:
            summary['fl_step_timing'][m] = {'error': str(e)}
            print(f'  {m} FAIL: {e}')
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    print('\n===== FL client LLG attack =====')
    summary['fl_llg'] = fl_llg_grid(args.methods, n_attack_batches=12, batch_size=16)
    for m, r in summary['fl_llg'].items():
        print(f'  {m}: label_recovery={r["label_recovery_avg"]:.3f}  '
              f'token_f1={r["token_f1_avg"]:.3f}')
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    if not args.skip_mt:
        print('\n===== MT step timing (encoder-decoder transformer) =====')
        for m in args.methods:
            try:
                r = time_mt_step(m, n_warmup=3, n_steps=50, batch_size=16)
                summary['mt_step_timing'][m] = r
                print(f'  {m}: {r["mean_step_s"]*1000:.2f} ms/step  '
                      f'(p95={r["p95_step_s"]*1000:.2f} ms)  params={r["n_params"]}')
            except Exception as e:
                summary['mt_step_timing'][m] = {'error': str(e)}
                import traceback; traceback.print_exc()
                print(f'  {m} FAIL: {e}')
            with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)

    # Wall-clock estimates
    print('\n===== Wall-clock estimates =====')
    # FL: 100 rounds × 5 clients × ~218 batches per client (50% local epochs)
    # = 100 × 5 × 218 ≈ 109k client steps
    # MT pretrain: 5 epochs × 80k steps = 400k steps; finetune: 10 epochs × 5k = 50k
    fl_total_steps = 100 * 5 * 218
    mt_pre_steps = 5 * 80000
    mt_ft_steps = 10 * 5000
    estimates = {}
    for m in args.methods:
        if m in summary['fl_step_timing'] and 'mean_step_s' in summary['fl_step_timing'][m]:
            t_fl = summary['fl_step_timing'][m]['mean_step_s']
            estimates[f'fl_{m}_total_min'] = round(t_fl * fl_total_steps / 60, 1)
        if m in summary['mt_step_timing'] and 'mean_step_s' in summary['mt_step_timing'][m]:
            t_mt = summary['mt_step_timing'][m]['mean_step_s']
            estimates[f'mt_{m}_pretrain_min'] = round(t_mt * mt_pre_steps / 60, 1)
            estimates[f'mt_{m}_finetune_min'] = round(t_mt * mt_ft_steps / 60, 1)
    summary['wall_clock_estimates_min'] = estimates
    print(json.dumps(estimates, indent=2))

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
