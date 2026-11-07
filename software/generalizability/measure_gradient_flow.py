#!/usr/bin/env python3
"""Quantitative measurement of gradient flow across the V8/V9 multilingual MT
two-phase protocol. For each encoder (Subword, EmByte, Bits) and each
endangered language pair, we measure:

  (a) Phase-1 vocabulary coverage: |V_indcorpus \ V_phase1|
  (b) Per-parameter gradient norm during fine-tuning, partitioned by
      'seen in Phase 1' vs 'unseen' (Subword only — Bits has no per-token row).
  (c) Cumulative parameter movement ||theta_T - theta_0||_2 per parameter group.
  (d) Effective number of parameters updated per fine-tune step.

Output: results/generalizability_full10/gradient_flow_measurement.json
"""
import argparse, json, os, sys, time, math
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from generalizability_coling import (
    set_seeds, COLINGConfig, TranslationDataset, TransformerNMT,
    collate_byte_batch, _load_americasnlp_local, load_indcorpus_data,
    load_iwslt14_data,
)


def measure_phase1_coverage(method, lang_pair, config, max_pretrain=50000,
                             max_finetune=15000):
    """How many fine-tune-side tokens are absent from the Phase-1 corpus?"""
    pretrain_byte_set = set()
    for hr in config.high_resource_pairs:
        d = load_iwslt14_data(hr, max_samples=max_pretrain)
        for s, t in zip(d['src'], d['tgt']):
            pretrain_byte_set.update(s.encode('utf-8'))
            pretrain_byte_set.update(t.encode('utf-8'))
    ft = _load_americasnlp_local(*lang_pair) or load_indcorpus_data(lang_pair)
    if ft is None:
        return None
    ft_byte_set = set()
    for s, t in zip(ft['train_src'], ft['train_tgt']):
        ft_byte_set.update(s.encode('utf-8'))
        ft_byte_set.update(t.encode('utf-8'))
    overlap = ft_byte_set & pretrain_byte_set
    only_ft = ft_byte_set - pretrain_byte_set
    return {
        'V_phase1_bytes':   len(pretrain_byte_set),
        'V_finetune_bytes': len(ft_byte_set),
        'V_overlap':        len(overlap),
        'V_unseen':         len(only_ft),
        'unseen_ratio':     len(only_ft) / max(len(ft_byte_set), 1),
    }


def measure_grad_flow(method, lang_pair, config, n_grad_batches=20, device=None):
    """Train one fine-tune for `n_grad_batches` steps starting from the saved
    pretrain checkpoint; record per-parameter gradient norms and cumulative
    movement."""
    set_seeds(config.seed)
    src_lang, tgt_lang = lang_pair
    ft = _load_americasnlp_local(src_lang, tgt_lang) or load_indcorpus_data(lang_pair)
    if ft is None: return None

    train_ds = TranslationDataset(ft['train_src'], ft['train_tgt'],
                                   src_lang, tgt_lang,
                                   max_len=config.max_seq_len, tokenization='byte')
    loader = DataLoader(train_ds, batch_size=config.finetune_batch_size,
                         shuffle=True, collate_fn=collate_byte_batch)

    # Build fine-tune model and load Phase-1 checkpoint
    model = TransformerNMT(
        vocab_size=config.byte_vocab_size, d_model=config.hidden_dim,
        nhead=config.num_heads_finetune,
        num_encoder_layers=config.encoder_layers_finetune,
        num_decoder_layers=config.decoder_layers_finetune,
        dim_feedforward=config.ffn_dim, dropout=config.dropout_finetune,
        embedding_type=method, num_bytes=config.num_bytes,
        num_bits=config.num_bits, interdim=config.interdim,
    ).to(device)
    ck = torch.load(os.path.join(config.checkpoint_dir, f'pretrain_{method}.pt'),
                     map_location=device)
    pre = ck['model_state_dict']; md = model.state_dict()
    pre = {k: v for k, v in pre.items() if k in md and v.shape == md[k].shape}
    md.update(pre); model.load_state_dict(md)

    # Snapshot initial parameters for cumulative movement
    init_params = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = optim.Adam(model.parameters(), lr=config.finetune_lr)
    crit = nn.CrossEntropyLoss(ignore_index=0)

    # Group parameters by category
    def categorize(name):
        if 'embedding' in name.lower() or 'embed' in name.lower():
            return 'encoder_embedding'
        if 'binary' in name.lower() or 'byte_seb' in name.lower() or 'ffn' in name.lower():
            return 'encoder_ffn'
        if 'encoder' in name.lower(): return 'transformer_encoder'
        if 'decoder' in name.lower(): return 'transformer_decoder'
        return 'other'

    grad_norm_log = {}  # category -> list of L2 norms over steps
    n_done = 0
    for batch in loader:
        if n_done >= n_grad_batches: break
        try:
            opt.zero_grad()
            src = batch['src'].to(device); tgt = batch['tgt'].to(device)
            tgt_in = tgt[:, :-1]; tgt_out = tgt[:, 1:]
            logits = model(src, tgt_in)
            loss = crit(logits.reshape(-1, logits.size(-1)),
                        tgt_out.reshape(-1))
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is None: continue
                cat = categorize(name)
                grad_norm_log.setdefault(cat, []).append(float(p.grad.norm().item()))
            opt.step()
            n_done += 1
        except Exception as e:
            print(f'  batch err: {e}', flush=True); continue

    # Cumulative parameter movement at end of measurement window
    cum_move = {}
    for name, p in model.named_parameters():
        cat = categorize(name)
        cum_move.setdefault(cat, []).append(
            float((p.detach() - init_params[name]).norm().item()))

    return {
        'n_steps_measured': n_done,
        'grad_norm_per_category_mean': {k: float(np.mean(v)) for k, v in grad_norm_log.items()},
        'grad_norm_per_category_std':  {k: float(np.std(v))  for k, v in grad_norm_log.items()},
        'cum_movement_per_category':   {k: float(np.mean(v)) for k, v in cum_move.items()},
        'effective_params_per_step':   sum(p.numel() for p in model.parameters()
                                            if p.requires_grad),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results/generalizability_full10/gradient_flow_measurement.json')
    p.add_argument('--lang_pairs', nargs='+', default=['quy', 'gn', 'aym'])
    p.add_argument('--methods', nargs='+', default=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--n_grad_batches', type=int, default=20)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)

    config = COLINGConfig(
        encoder_layers_pretrain=6, decoder_layers_pretrain=6,
        encoder_layers_finetune=5, decoder_layers_finetune=5,
        num_heads_pretrain=4, num_heads_finetune=2,
        hidden_dim=512, ffn_dim=1024, dropout=0.2, dropout_finetune=0.4,
        finetune_epochs=1, finetune_lr=1e-4, finetune_batch_size=16,
        checkpoint_dir='checkpoints_generalizability',
    )

    summary = {}
    for src in args.lang_pairs:
        lp = (src, 'es')
        cov = measure_phase1_coverage(None, lp, config)
        summary[src] = {'coverage': cov, 'methods': {}}
        print(f'\n>>> {src}-es: coverage={cov}', flush=True)
        for method in args.methods:
            t0 = time.time()
            print(f'  [{method}] measuring gradient flow...', flush=True)
            try:
                r = measure_grad_flow(method, lp, config,
                                       n_grad_batches=args.n_grad_batches,
                                       device=device)
                r['time_s'] = time.time() - t0
                summary[src]['methods'][method] = r
                print(f'    grad norms: {r["grad_norm_per_category_mean"]}', flush=True)
                print(f'    cum move:   {r["cum_movement_per_category"]}', flush=True)
            except Exception as e:
                summary[src]['methods'][method] = {'error': str(e)}
                print(f'    ERROR: {e}', flush=True)
            with open(args.out, 'w') as f:
                json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
