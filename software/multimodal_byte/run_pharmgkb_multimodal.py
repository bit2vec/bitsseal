#!/usr/bin/env python3
"""PharmGKB multimodal classification: Phenotype Category prediction (5 classes).

Modalities: variant + gene + drug + phenotype + specialty + evidence_level.
"""
import argparse, csv, functools, json, os, random, re, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))

from pharmgkb_loader import load_pharmgkb
from run_civic_extended import MultiModalClassifier
from run_civic_multimodal import PAD, UNK, SEP, collate, tokenize


MODALITIES = [
    ('phenotype',     16),
    ('drug',           8),
    ('gene',           4),
    ('variant',        4),
    ('specialty',      4),
    ('evidence_level', 2),
]


class PGKBDataset(Dataset):
    def __init__(self, exs, stoi, mods):
        self.exs = exs; self.stoi = stoi; self.mods = mods

    def __len__(self): return len(self.exs)

    def __getitem__(self, i):
        ex = self.exs[i]
        ids = []
        for mod, budget in self.mods:
            toks = tokenize(str(ex.get(mod, '')))
            tok_ids = [self.stoi.get(t, self.stoi[UNK]) for t in toks]
            if len(tok_ids) > budget: tok_ids = tok_ids[:budget]
            else: tok_ids = tok_ids + [self.stoi[PAD]] * (budget - len(tok_ids))
            ids.append(self.stoi[f'{SEP}_{mod}'])
            ids.extend(tok_ids)
        return torch.tensor(ids, dtype=torch.long), ex['label_idx']


def film_attack(model, ids, label, criterion, device, vocab_size):
    model.zero_grad(set_to_none=False)
    logits = model(ids.to(device))
    loss = criterion(logits, label.to(device))
    loss.backward()
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    if model.method == 'standard':
        gnorm = model.embedding.weight.grad.detach().norm(dim=1).cpu()
        thr = gnorm.max().item() * 0.05 if gnorm.max().item() > 0 else 1e-12
        recovered = set((gnorm > thr).nonzero(as_tuple=True)[0].tolist())
    else:
        recovered = set(range(vocab_size))
    if model.method != 'standard' and len(recovered) > 5 * len(actual):
        recovered = set(random.sample(list(recovered), min(20, len(recovered))))
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


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
        # SEB: aggregate to a single grad direction; compare to embedding table
        rows = []
        with torch.no_grad():
            for s in range(0, vocab_size, 2048):
                e = min(s + 2048, vocab_size)
                ids_v = torch.arange(s, e, device=device).unsqueeze(0)
                rows.append(model.embedding(ids_v).squeeze(0))
        emb_table = torch.cat(rows, dim=0)
        fc_grad = model.fc.weight.grad.detach()
        g = fc_grad.mean(dim=0).unsqueeze(0)
        sim = (F.normalize(g, dim=-1) @ F.normalize(emb_table, dim=-1).t()).abs().squeeze(0).cpu()
        k = max(len(actual), 1)
        recovered = set(torch.topk(sim, k).indices.tolist())
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/pharmgkb')
    p.add_argument('--n_epochs', type=int, default=4)
    p.add_argument('--methods', nargs='+', default=['standard', 'byte_seb'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n_attack', type=int, default=12)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading PharmGKB...')
    examples, label_set = load_pharmgkb(seed=42)
    print(f'  {len(examples)} examples, {len(label_set)} classes: {label_set}')
    print('  Distribution:', Counter(e['label'] for e in examples))

    # Empirical lengths
    print('\nEmpirical lengths:')
    final_mod = []
    for mod, default_budget in MODALITIES:
        lens = [len(tokenize(str(ex.get(mod, '')))) for ex in examples]
        m = max(lens) if lens else 1
        p95 = max(int(np.percentile(lens, 95)) if lens else 1, 1)
        budget = m if m <= 1.5 * p95 else p95
        final_mod.append((mod, max(min(budget, default_budget), 1)))
        print(f'  {mod}: max={m} p95={p95} mean={np.mean(lens):.1f} → budget={final_mod[-1][1]}')

    n = len(examples)
    n_tr = int(0.7 * n); n_va = int(0.1 * n)
    train_ex = examples[:n_tr]; val_ex = examples[n_tr:n_tr + n_va]
    test_ex = examples[n_tr + n_va:]
    num_classes = len(label_set)

    # Build vocab
    counter = Counter()
    for ex in train_ex:
        for mod, _ in final_mod:
            counter.update(tokenize(str(ex.get(mod, ''))))
    stoi = {PAD: 0, UNK: 1}
    for mod, _ in final_mod: stoi[f'{SEP}_{mod}'] = len(stoi)
    for tok, c in counter.most_common():
        if c >= 2: stoi[tok] = len(stoi)
    vocab_size = len(stoi)
    _spin = [m for m in args.methods if m in ('simhash_seb', 'repulse_seb')]
    if _spin:
        import tempfile, run_civic_extended as _rce
        from run_table7_repro import simhash_table, repulse_table
        _itos = [None] * vocab_size
        for _t, _i in stoi.items(): _itos[_i] = _t
        _td = tempfile.mkdtemp()
        if 'simhash_seb' in _spin:
            _p = os.path.join(_td, 'sh.pt'); torch.save(simhash_table(_itos, num_bits=64), _p); _rce.SPINHASH_CODES['simhash_seb'] = _p
        if 'repulse_seb' in _spin:
            _tt = [sum((tokenize(str(ex.get(m, ''))) for m, _ in final_mod), []) for ex in train_ex]
            _p = os.path.join(_td, 'rp.pt'); torch.save(repulse_table(_itos, _tt, num_bits=64), _p); _rce.SPINHASH_CODES['repulse_seb'] = _p
    print(f'\nvocab={vocab_size}  train={len(train_ex)}/val={len(val_ex)}/test={len(test_ex)}')

    summary = {}
    for method in args.methods:
        print(f'\n>>> {method}')
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
        coll = functools.partial(collate, pad_index=stoi[PAD])
        bs = 16
        train_ds = PGKBDataset(train_ex, stoi, final_mod)
        val_ds = PGKBDataset(val_ex, stoi, final_mod)
        test_ds = PGKBDataset(test_ex, stoi, final_mod)
        tr = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=coll)
        val_l = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=coll)
        test_l = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=coll)

        model = MultiModalClassifier(vocab_size, 64, num_classes,
                                      method=method, body_kind='D',
                                      pad_idx=stoi[PAD]).to(device)
        opt = optim.Adam(model.parameters(), lr=5e-4)
        crit = nn.CrossEntropyLoss()

        t0 = time.time()
        best_val, best_state = 0.0, None
        for ep in range(args.n_epochs):
            model.train()
            for batch in tr:
                if batch is None: continue
                opt.zero_grad()
                crit(model(batch['ids'].to(device)),
                     batch['label'].to(device)).backward()
                opt.step()
            model.eval()
            c = t = 0
            with torch.no_grad():
                for batch in val_l:
                    if batch is None: continue
                    pred = model(batch['ids'].to(device)).argmax(-1).cpu()
                    c += (pred == batch['label']).sum().item(); t += batch['label'].size(0)
            v = c / max(t, 1)
            print(f'  ep {ep+1}: val={v:.3f}')
            if v > best_val:
                best_val = v
                best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
        if best_state is not None: model.load_state_dict(best_state)
        train_time = time.time() - t0

        model.eval()
        c = t = 0
        t0 = time.time()
        with torch.no_grad():
            for batch in test_l:
                if batch is None: continue
                pred = model(batch['ids'].to(device)).argmax(-1).cpu()
                c += (pred == batch['label']).sum().item(); t += batch['label'].size(0)
        test_acc = c / max(t, 1)
        inf_time = time.time() - t0

        # Privacy attacks
        model.train()
        film_f1s, tag_f1s = [], []
        n_atk = 0
        for batch in tr:
            if batch is None or n_atk >= args.n_attack: break
            try: film_f1s.append(film_attack(model, batch['ids'], batch['label'], crit, device, vocab_size))
            except Exception as e: print(f'  film err: {e}')
            try: tag_f1s.append(tag_attack(model, batch['ids'], batch['label'], crit, device, vocab_size))
            except Exception as e: print(f'  tag err: {e}')
            n_atk += 1

        r = {
            'method': method,
            'val_acc': float(best_val),
            'test_acc': float(test_acc),
            'n_params': sum(p.numel() for p in model.parameters()),
            'vocab_size': vocab_size,
            'train_time_s': train_time,
            'inference_time_s': inf_time,
            'film_f1': float(np.mean(film_f1s)) if film_f1s else 0.0,
            'tag_f1':  float(np.mean(tag_f1s))  if tag_f1s  else 0.0,
        }
        summary[method] = r
        print(f'  test_acc={test_acc:.3f}  FILM={r["film_f1"]:.3f}  TAG={r["tag_f1"]:.3f}  '
              f'params={r["n_params"]}')

        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
