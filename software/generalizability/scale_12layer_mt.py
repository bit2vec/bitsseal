#!/usr/bin/env python3
"""12-layer MT scaling: does deeper architecture close the +57 ChrF gap?

Trains 12-layer encoder-decoder transformers on quy-es and gn-es endangered
pairs from scratch, no IWSLT14 pretraining. Compares Bits, EmByte, Subword.
The hypothesis: subword's bottleneck is the unseen vocabulary, not the model
capacity, so scaling depth should NOT close the gap; if it does, our claim
is weaker.
"""
import argparse, json, os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from generalizability_coling import (
    set_seeds, COLINGConfig, TranslationDataset, TransformerNMT,
    collate_byte_batch, compute_chrf, compute_bleu, evaluate_model,
    _load_americasnlp_local, load_indcorpus_data,
)


def train_12layer_from_scratch(method, src_lang, tgt_lang, n_layers, n_epochs,
                                hidden_dim, ffn_dim, num_heads, dropout,
                                batch_size, lr, max_seq_len, device, seed,
                                output_dir, max_train_samples=5000):
    set_seeds(seed)
    print(f'\n>>> {method} {src_lang}-{tgt_lang} {n_layers}L from scratch')

    pair_data = _load_americasnlp_local(src_lang, tgt_lang) or load_indcorpus_data((src_lang, tgt_lang))
    if pair_data is None:
        print(f'  data unavailable, skipping')
        return None

    # Cap training to max_train_samples (endangered low-resource regime)
    if max_train_samples and len(pair_data['train_src']) > max_train_samples:
        pair_data['train_src'] = pair_data['train_src'][:max_train_samples]
        pair_data['train_tgt'] = pair_data['train_tgt'][:max_train_samples]

    n_train = len(pair_data['train_src'])
    n_dev   = len(pair_data.get('dev_src', []))
    print(f'  train={n_train} dev={n_dev}')
    if n_train < 500:
        print(f'  too few train samples, skipping')
        return None

    # Build datasets — use dev as test (matches COLING paper convention)
    train_ds = TranslationDataset(pair_data['train_src'], pair_data['train_tgt'],
                                   src_lang, tgt_lang, max_seq_len, tokenization='byte')
    test_ds  = TranslationDataset(pair_data['dev_src'], pair_data['dev_tgt'],
                                   src_lang, tgt_lang, max_seq_len, tokenization='byte')

    coll = lambda b: collate_byte_batch(b, pad_idx=0)
    tr_loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    test_loader = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=coll)

    if method == 'standard':
        vocab_size = 256
    else:
        vocab_size = 256

    model = TransformerNMT(
        vocab_size=vocab_size, d_model=hidden_dim, nhead=num_heads,
        num_encoder_layers=n_layers, num_decoder_layers=n_layers,
        dim_feedforward=ffn_dim, dropout=dropout, embedding_type=method,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  params={n_params:,} ({n_params*4/1024/1024:.1f} MB)')

    opt = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    crit = nn.CrossEntropyLoss(ignore_index=0)

    t0 = time.time()
    best_chrf = 0.0
    for ep in range(n_epochs):
        model.train()
        total_loss = 0; n_tok = 0
        for batch in tr_loader:
            src = batch['src'].to(device); tgt = batch['tgt'].to(device)
            tgt_in  = tgt[:, :-1]; tgt_out = tgt[:, 1:]
            src_pad = (src == 0); tgt_pad = (tgt_in == 0)
            opt.zero_grad()
            logits = model(src, tgt_in, src_padding_mask=src_pad, tgt_padding_mask=tgt_pad)
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            non_pad = (tgt_out != 0).sum().item()
            total_loss += loss.item() * non_pad; n_tok += non_pad
        train_ppl = math.exp(min(total_loss / max(n_tok, 1), 20))

        if (ep + 1) % 5 == 0 or ep == n_epochs - 1:
            _, bleu, chrf = evaluate_model(model, test_loader, crit, device)
            print(f'  ep {ep+1}: ppl={train_ppl:.2f} test_bleu={bleu:.2f} test_chrf={chrf:.2f}')
            if chrf > best_chrf: best_chrf = chrf
        else:
            print(f'  ep {ep+1}: ppl={train_ppl:.2f}')
    train_time = time.time() - t0

    _, final_bleu, final_chrf = evaluate_model(model, test_loader, crit, device)
    return {
        'method': method, 'lang_pair': f'{src_lang}-{tgt_lang}',
        'n_layers': n_layers, 'n_epochs': n_epochs,
        'n_params': n_params,
        'final_bleu': float(final_bleu),
        'final_chrf': float(final_chrf),
        'best_chrf': float(best_chrf),
        'train_time_s': train_time,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/mt_12layer_scaling')
    p.add_argument('--n_epochs', type=int, default=30)
    p.add_argument('--n_layers', nargs='+', type=int, default=[6, 12])
    p.add_argument('--methods', nargs='+', default=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--lang_pairs', nargs='+', default=['quy-es', 'gn-es'])
    p.add_argument('--hidden_dim', type=int, default=256)  # smaller hidden to fit
    p.add_argument('--ffn_dim', type=int, default=1024)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--max_seq_len', type=int, default=400)
    p.add_argument('--max_train_samples', type=int, default=5000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    summary = []
    for n_layers in args.n_layers:
        for pair in args.lang_pairs:
            src, tgt = pair.split('-')
            for method in args.methods:
                try:
                    r = train_12layer_from_scratch(
                        method, src, tgt, n_layers, args.n_epochs,
                        args.hidden_dim, args.ffn_dim, args.num_heads, args.dropout,
                        args.batch_size, args.lr, args.max_seq_len, device, args.seed,
                        args.out_dir, max_train_samples=args.max_train_samples,
                    )
                except Exception as e:
                    import traceback; traceback.print_exc()
                    r = {'method': method, 'lang_pair': pair, 'n_layers': n_layers,
                         'error': str(e)}
                if r is not None:
                    summary.append(r)
                    print(f'  -> ChrF={r.get("final_chrf", -1):.2f} BLEU={r.get("final_bleu", -1):.2f}')
                with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                    json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
