#!/usr/bin/env python3
"""V2 vs V3 modality-specialization comparison on CIViC and TCGA.

Phase A: Train multimodal Bits + Body D classifier on all modalities; save checkpoint.
Phase B (V3): Train each single-modality model from scratch (random init).
Phase C (V2): Load Phase-A checkpoint; fine-tune on each single modality.

For each modality we then have a matched pair (V2, V3) that tells us whether
the multimodal pretraining provides transferable signal for single-modality
deployment, by analogy with the V8/V9 multilingual MT mechanism (joint
pretrain on de+ar -> separate fine-tune per endangered language).

No modality tags. The model infers modality from byte distribution only.

Usage:
    python run_modality_finetune.py --dataset civic --out_dir results/...
    python run_modality_finetune.py --dataset tcga  --out_dir results/...
"""
import argparse, copy, functools, json, os, random, re, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))
sys.path.insert(0, THIS_DIR)

from run_civic_multimodal import PAD, UNK, SEP, collate, load_civic, LABEL_SET as CIVIC_LABELS
from run_civic_multimodal import MODALITIES as CIVIC_MODALITIES
from run_civic_extended import MultiModalClassifier, film_attack
from tcga_loader import load_tcga_pan5

TCGA_MODALITIES_RAW = [('text', None), ('structured', None), ('gene_top', None)]


def tokenize(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower()).split()


def build_separator_vocab(train_examples, modalities, min_freq=2):
    counter = Counter()
    for ex in train_examples:
        for mod, _ in modalities:
            counter.update(tokenize(ex[mod]))
    stoi = {PAD: 0, UNK: 1}
    for mod, _ in modalities:
        stoi[f'{SEP}_{mod}'] = len(stoi)
    for tok, c in counter.most_common():
        if c >= min_freq:
            stoi[tok] = len(stoi)
    return stoi, {'mode': 'separator'}


def encode_with_mask(ex, stoi, vocab_meta, modalities, mask_all_but=None):
    """Concatenate modalities into id sequence. mask_all_but=None => all kept;
    else mask_all_but is the modality NAME that is kept (others -> PAD)."""
    out_ids = []
    for mod, budget in modalities:
        if mask_all_but is not None and mod != mask_all_but:
            tok_ids = [stoi[PAD]] * budget
        else:
            toks = tokenize(ex[mod])
            tok_ids = [stoi.get(t, stoi[UNK]) for t in toks]
            if len(tok_ids) > budget:
                tok_ids = tok_ids[:budget]
            else:
                tok_ids = tok_ids + [stoi[PAD]] * (budget - len(tok_ids))
        out_ids.append(stoi[f'{SEP}_{mod}'])
        out_ids.extend(tok_ids)
    return out_ids


class FlexibleDataset(Dataset):
    def __init__(self, examples, stoi, vocab_meta, modalities,
                 label_field='label', mask_all_but=None):
        self.exs = examples
        self.stoi = stoi
        self.vocab_meta = vocab_meta
        self.mods = modalities
        self.label_field = label_field
        self.mask_all_but = mask_all_but

    def __len__(self):
        return len(self.exs)

    def __getitem__(self, i):
        ex = self.exs[i]
        ids = encode_with_mask(ex, self.stoi, self.vocab_meta, self.mods,
                               mask_all_but=self.mask_all_but)
        return torch.tensor(ids, dtype=torch.long), ex[self.label_field]


def train_eval(model, train_ds, val_ds, test_ds, stoi, device,
               n_epochs=4, batch_size=16, lr=5e-4, run_film=False):
    coll = functools.partial(collate, pad_index=stoi[PAD])
    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    va = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    te = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    t0 = time.time()
    best_val, best_state = 0.0, None
    for ep in range(n_epochs):
        model.train()
        for batch in tr:
            opt.zero_grad()
            crit(model(batch['ids'].to(device)),
                 batch['label'].to(device)).backward()
            opt.step()
        model.eval()
        c = t = 0
        with torch.no_grad():
            for batch in va:
                pred = model(batch['ids'].to(device)).argmax(-1).cpu()
                c += (pred == batch['label']).sum().item()
                t += batch['label'].size(0)
        v = c / max(t, 1)
        if v > best_val:
            best_val = v
            best_state = {k: vv.detach().cpu().clone()
                          for k, vv in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.time() - t0

    model.eval()
    c = t = 0
    with torch.no_grad():
        for batch in te:
            pred = model(batch['ids'].to(device)).argmax(-1).cpu()
            c += (pred == batch['label']).sum().item()
            t += batch['label'].size(0)
    test_acc = c / max(t, 1)

    film_f1 = None
    if run_film:
        model.train()
        f1s = []
        n_atk = 0
        for batch in tr:
            if n_atk >= 16:
                break
            try:
                f1s.append(film_attack(model, batch['ids'], batch['label'],
                                       crit, device))
            except Exception:
                pass
            n_atk += 1
        film_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        'val_acc': float(best_val),
        'test_acc': float(test_acc),
        'train_time_s': train_time,
        'n_params': sum(p.numel() for p in model.parameters()),
        'film_f1': film_f1,
        'best_state': best_state,
    }


def make_model(vocab_size, n_classes, stoi, embedding_dim=64,
               body='D', method='binary_seb', num_bits=64, interdim='128'):
    return MultiModalClassifier(
        vocab_size=vocab_size, embedding_dim=embedding_dim,
        num_classes=n_classes, method=method, body_kind=body,
        num_bits=num_bits, interdim=interdim, pad_idx=stoi[PAD])


def derive_tcga_budgets(examples):
    final = []
    for mod, _ in TCGA_MODALITIES_RAW:
        lens = [len(tokenize(ex[mod])) for ex in examples]
        m = max(lens)
        p95 = int(np.percentile(lens, 95)) or 1
        if mod == 'text':
            budget = min(512, m)
        elif mod == 'gene_top':
            budget = min(2048, m)
        else:
            budget = m if m <= 1.5 * p95 else p95
        final.append((mod, budget))
    return final


def load_dataset(name):
    if name == 'civic':
        train, val, test = load_civic(seed=42)
        modalities = list(CIVIC_MODALITIES)
        labels = list(CIVIC_LABELS)
        for ex_split in (train, val, test):
            for ex in ex_split:
                pass
        return train, val, test, modalities, labels, 'label'
    elif name == 'tcga':
        examples, label_list = load_tcga_pan5(top_k_genes=64)
        label_to_idx = {l: i for i, l in enumerate(label_list)}
        for ex in examples:
            ex['label_idx'] = label_to_idx[ex['label']]
        rng = random.Random(42)
        rng.shuffle(examples)
        modalities = derive_tcga_budgets(examples)
        n = len(examples)
        n_tr = int(0.7 * n)
        n_va = int(0.1 * n)
        return (examples[:n_tr], examples[n_tr:n_tr + n_va],
                examples[n_tr + n_va:], modalities, list(label_list), 'label_idx')
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['civic', 'tcga'], required=True)
    p.add_argument('--out_dir', default=None)
    p.add_argument('--n_epochs_pretrain', type=int, default=6)
    p.add_argument('--n_epochs_finetune', type=int, default=4)
    p.add_argument('--lr_pretrain', type=float, default=5e-4)
    p.add_argument('--lr_finetune', type=float, default=1e-4)
    p.add_argument('--method', default='binary_seb',
                   help='binary_seb=Bits, byte_seb=EmByte, standard=Subword')
    p.add_argument('--body', default='D',
                   help='A=mean, B=BiLSTM, C=cross-attn, D=cross-attn+MLP, E=transformer')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = f'results/multimodality/v2_v3_finetune/{args.dataset}'
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  dataset={args.dataset}  method={args.method}  body={args.body}',
          flush=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    train, val, test, modalities, labels, label_field = load_dataset(args.dataset)
    print(f'  train={len(train)}  val={len(val)}  test={len(test)}  '
          f'#labels={len(labels)}  modalities={[m[0] for m in modalities]}',
          flush=True)

    stoi, vmeta = build_separator_vocab(train, modalities)
    vocab_size = len(stoi)
    print(f'  vocab={vocab_size}', flush=True)

    summary = {'config': vars(args), 'modalities': [m[0] for m in modalities],
               'vocab_size': vocab_size, 'phases': {}}

    # ============ Phase A: train multimodal pretrain ============
    print('\n[Phase A] multimodal pretrain (all modalities active)', flush=True)
    tr_ds = FlexibleDataset(train, stoi, vmeta, modalities,
                             label_field=label_field, mask_all_but=None)
    va_ds = FlexibleDataset(val, stoi, vmeta, modalities,
                             label_field=label_field, mask_all_but=None)
    te_ds = FlexibleDataset(test, stoi, vmeta, modalities,
                             label_field=label_field, mask_all_but=None)

    pretrain_model = make_model(vocab_size, len(labels), stoi,
                                 method=args.method, body=args.body)
    pretrain_model = pretrain_model.to(device)
    pretrain_res = train_eval(pretrain_model, tr_ds, va_ds, te_ds, stoi, device,
                               n_epochs=args.n_epochs_pretrain,
                               lr=args.lr_pretrain, run_film=True)
    pretrain_state = pretrain_res.pop('best_state')
    summary['phases']['A_multimodal_pretrain'] = pretrain_res
    ckpt_path = os.path.join(args.out_dir, 'multimodal_pretrain.pt')
    torch.save({'state_dict': pretrain_state, 'vocab_size': vocab_size,
                'config': vars(args), 'modalities': modalities}, ckpt_path)
    print(f'  test_acc={pretrain_res["test_acc"]:.4f}  '
          f'film={pretrain_res["film_f1"]:.3f}  '
          f'train_time={pretrain_res["train_time_s"]:.1f}s', flush=True)
    print(f'  saved -> {ckpt_path}', flush=True)

    # Save partial after Phase A in case Phase B/C fails
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # ============ Phase B (V3): from-scratch single modality ============
    print('\n[Phase B / V3] train each modality from scratch', flush=True)
    summary['phases']['B_v3_from_scratch'] = {}
    for mod_name, _ in modalities:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        tr_ds_m = FlexibleDataset(train, stoi, vmeta, modalities,
                                   label_field=label_field, mask_all_but=mod_name)
        va_ds_m = FlexibleDataset(val,   stoi, vmeta, modalities,
                                   label_field=label_field, mask_all_but=mod_name)
        te_ds_m = FlexibleDataset(test,  stoi, vmeta, modalities,
                                   label_field=label_field, mask_all_but=mod_name)
        m = make_model(vocab_size, len(labels), stoi,
                       method=args.method, body=args.body).to(device)
        r = train_eval(m, tr_ds_m, va_ds_m, te_ds_m, stoi, device,
                       n_epochs=args.n_epochs_pretrain,
                       lr=args.lr_pretrain, run_film=False)
        r.pop('best_state', None)
        summary['phases']['B_v3_from_scratch'][mod_name] = r
        print(f'  [{mod_name:20s}] test_acc={r["test_acc"]:.4f} '
              f'time={r["train_time_s"]:.1f}s', flush=True)
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    # ============ Phase C (V2): fine-tune from multimodal checkpoint ============
    print('\n[Phase C / V2] fine-tune from multimodal checkpoint per modality',
          flush=True)
    summary['phases']['C_v2_finetune_from_pretrain'] = {}
    for mod_name, _ in modalities:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        tr_ds_m = FlexibleDataset(train, stoi, vmeta, modalities,
                                   label_field=label_field, mask_all_but=mod_name)
        va_ds_m = FlexibleDataset(val,   stoi, vmeta, modalities,
                                   label_field=label_field, mask_all_but=mod_name)
        te_ds_m = FlexibleDataset(test,  stoi, vmeta, modalities,
                                   label_field=label_field, mask_all_but=mod_name)
        m = make_model(vocab_size, len(labels), stoi,
                       method=args.method, body=args.body).to(device)
        m.load_state_dict({k: v.to(device) for k, v in pretrain_state.items()})
        r = train_eval(m, tr_ds_m, va_ds_m, te_ds_m, stoi, device,
                       n_epochs=args.n_epochs_finetune,
                       lr=args.lr_finetune, run_film=False)
        r.pop('best_state', None)
        summary['phases']['C_v2_finetune_from_pretrain'][mod_name] = r
        print(f'  [{mod_name:20s}] test_acc={r["test_acc"]:.4f} '
              f'time={r["train_time_s"]:.1f}s', flush=True)
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    # ============ Final pairwise comparison ============
    print('\n=== Matched-pair comparison (V2 fine-tune vs V3 from-scratch) ===',
          flush=True)
    print(f'{"modality":20s}  {"V3 scratch":10s}  {"V2 finetune":11s}  '
          f'{"delta":8s}', flush=True)
    deltas = []
    for mod_name, _ in modalities:
        v3 = summary['phases']['B_v3_from_scratch'][mod_name]['test_acc']
        v2 = summary['phases']['C_v2_finetune_from_pretrain'][mod_name]['test_acc']
        delta = v2 - v3
        deltas.append(delta)
        print(f'{mod_name:20s}  {v3:10.4f}  {v2:11.4f}  {delta:+.4f}', flush=True)
    summary['mean_delta_v2_minus_v3'] = float(np.mean(deltas))
    summary['multimodal_baseline_test_acc'] = pretrain_res['test_acc']
    print(f'\nmean delta = {np.mean(deltas):+.4f}')
    print(f'multimodal (all modalities active) = {pretrain_res["test_acc"]:.4f}')

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nWrote {args.out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
