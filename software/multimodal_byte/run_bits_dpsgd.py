#!/usr/bin/env python3
"""DP-SGD composition on CIViC body D.

Trains the full multimodal classifier (Bits encoder + body D) under Opacus
DP-SGD at several target_eps levels and reports test accuracy, achieved eps,
and FILM/TAG attack F1 on the DP-trained model.

Compares Bits vs Subword vs EmByte under the same DP-SGD recipe to test the
V5 limitations claim that the smaller Bits encoder permits a less aggressive
DP-SGD setup.
"""
import argparse, csv, functools, json, os, random, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))

from run_civic_extended import MultiModalClassifier
from run_civic_multimodal import (
    PAD, UNK, SEP, CIVIC_TSV, MODALITIES, tokenize, load_civic, build_vocab,
    CIViCMultiModalDataset, collate, film_attack,
)


def tag_attack(model, ids, label, criterion, device, vocab_size):
    model.zero_grad(set_to_none=False)
    logits = model(ids.to(device))
    loss = criterion(logits, label.to(device))
    loss.backward()
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    if model.method == 'standard':
        gnorm = model.embedding.weight.grad.detach().norm(dim=1).cpu()
        k = max(len(actual), 1)
        recovered = set(torch.topk(gnorm, k).indices.tolist())
    else:
        rows = []
        with torch.no_grad():
            for s in range(0, vocab_size, 2048):
                e = min(s + 2048, vocab_size)
                ids_v = torch.arange(s, e, device=device).unsqueeze(0)
                rows.append(model.embedding(ids_v).squeeze(0))
        emb_table = torch.cat(rows, dim=0)
        fc_grad = model.fc.weight.grad.detach()
        g = fc_grad.mean(dim=0).unsqueeze(0)
        sim = (torch.nn.functional.normalize(g, dim=-1) @
               torch.nn.functional.normalize(emb_table, dim=-1).t()).abs().squeeze(0).cpu()
        k = max(len(actual), 1)
        recovered = set(torch.topk(sim, k).indices.tolist())
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


def train_dpsgd(method, eps_target, train_ds, val_ds, test_ds, vocab_size, stoi,
                num_classes, modalities, device, n_epochs, batch_size,
                max_grad_norm, seed):
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    from opacus.validators import ModuleValidator
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    coll = functools.partial(collate, pad_index=stoi[PAD])
    tr   = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=coll)
    val_l= DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=coll)
    test_l= DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)

    model = MultiModalClassifier(vocab_size, 64, num_classes,
                                  method=method, body_kind='D',
                                  pad_idx=stoi[PAD], interdim='128').to(device)
    if eps_target is not None:
        # Replace nn.MultiheadAttention with DPMultiheadAttention etc.
        model = ModuleValidator.fix(model).to(device)
    crit = nn.CrossEntropyLoss()

    if eps_target is None:
        opt = optim.Adam(model.parameters(), lr=5e-4)
        model_p, opt_p, train_loader_p = model, opt, tr
        engine = None
    else:
        opt = optim.Adam(model.parameters(), lr=1e-3)
        engine = PrivacyEngine()
        model_p, opt_p, train_loader_p = engine.make_private_with_epsilon(
            module=model, optimizer=opt, data_loader=tr,
            target_epsilon=eps_target, target_delta=1e-5, epochs=n_epochs,
            max_grad_norm=max_grad_norm,
        )

    t0 = time.time()
    for ep in range(n_epochs):
        model_p.train()
        if engine is not None:
            ctx = BatchMemoryManager(data_loader=train_loader_p,
                                      max_physical_batch_size=batch_size,
                                      optimizer=opt_p)
        else:
            ctx = None
        loader_iter = ctx.__enter__() if ctx is not None else train_loader_p
        try:
            for batch in loader_iter:
                if batch is None: continue
                opt_p.zero_grad()
                logits = model_p(batch['ids'].to(device))
                crit(logits, batch['label'].to(device)).backward()
                opt_p.step()
        finally:
            if ctx is not None: ctx.__exit__(None, None, None)
    train_time = time.time() - t0

    achieved_eps = float(engine.get_epsilon(delta=1e-5)) if engine else float('inf')

    inner = model_p._module if hasattr(model_p, '_module') else model_p
    inner.eval()
    c = t = 0
    with torch.no_grad():
        for batch in test_l:
            if batch is None: continue
            pred = inner(batch['ids'].to(device)).argmax(-1).cpu()
            c += (pred == batch['label']).sum().item(); t += batch['label'].size(0)
    test_acc = c / max(t, 1)

    # Privacy attacks on a clean copy (Opacus hooks interfere with grads)
    fresh = MultiModalClassifier(vocab_size, 64, num_classes,
                                  method=method, body_kind='D',
                                  pad_idx=stoi[PAD], interdim='128').to(device)
    if eps_target is not None:
        fresh = ModuleValidator.fix(fresh).to(device)
    fresh.load_state_dict(inner.state_dict(), strict=False)
    fresh.train()
    film_f1s, tag_f1s = [], []
    n_atk = 0
    for batch in tr:
        if batch is None or n_atk >= 12: break
        try: film_f1s.append(film_attack(fresh, batch['ids'], batch['label'], crit, device))
        except Exception as e: print(f'  film err: {e}')
        try: tag_f1s.append(tag_attack(fresh, batch['ids'], batch['label'], crit, device, vocab_size))
        except Exception as e: print(f'  tag err: {e}')
        n_atk += 1

    return {
        'method': method,
        'target_eps': eps_target,
        'achieved_eps': achieved_eps,
        'test_acc': float(test_acc),
        'train_time_s': train_time,
        'film_f1': float(np.mean(film_f1s)) if film_f1s else 0.0,
        'tag_f1':  float(np.mean(tag_f1s))  if tag_f1s  else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/bits_dpsgd')
    p.add_argument('--n_epochs', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--methods', nargs='+', default=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--epsilons', nargs='+', type=float,
                   default=[1.0, 4.0, 8.0])  # None for non-private
    p.add_argument('--include_nondp', action='store_true', default=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading CIViC...')
    train_ex, val_ex, test_ex = load_civic(seed=42)
    stoi, vocab_meta = build_vocab(train_ex, MODALITIES, boundary='separator')
    vocab_size = len(stoi); num_classes = 5
    print(f'  V={vocab_size}  train={len(train_ex)}/val={len(val_ex)}/test={len(test_ex)}')

    train_ds = CIViCMultiModalDataset(train_ex, stoi, vocab_meta, MODALITIES, ablation='all')
    val_ds   = CIViCMultiModalDataset(val_ex,   stoi, vocab_meta, MODALITIES, ablation='all')
    test_ds  = CIViCMultiModalDataset(test_ex,  stoi, vocab_meta, MODALITIES, ablation='all')

    eps_list = list(args.epsilons)
    if args.include_nondp: eps_list = [None] + eps_list

    summary = []
    for method in args.methods:
        for eps in eps_list:
            tag = f'{method}/eps={eps}'
            print(f'\n>>> {tag}')
            try:
                r = train_dpsgd(method, eps, train_ds, val_ds, test_ds, vocab_size, stoi,
                                num_classes, MODALITIES, device, args.n_epochs,
                                args.batch_size, args.max_grad_norm, seed=42)
                print(f'  acc={r["test_acc"]:.4f}  ach_eps={r["achieved_eps"]:.2f}  '
                      f'FILM={r["film_f1"]:.4f}  TAG={r["tag_f1"]:.4f}')
            except Exception as e:
                print(f'  ERROR: {e}')
                r = {'method': method, 'target_eps': eps, 'error': str(e)}
            summary.append(r)
            with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
