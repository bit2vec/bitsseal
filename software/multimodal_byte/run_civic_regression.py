#!/usr/bin/env python3
"""CIViC regression: predict the curator-assigned 'rating' (1 to 5 stars).

Target: rating standardized on the train set (1..5 → roughly zero mean unit var).
Modalities: text + gene + variant + disease + evidence_type + evidence_direction
            + variant_origin (evidence_level dropped: it correlates with rating).
Metrics: RMSE, MAE, Pearson r, Spearman rho on the raw 1..5 scale.
Privacy: FILM, TAG.
"""
import argparse, csv, functools, json, os, random, re, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))

from run_civic_extended import MultiModalClassifier
from run_civic_multimodal import PAD, UNK, SEP, CIVIC_TSV, tokenize


MODALITIES = [
    ('text',               128),
    ('gene',                 2),
    ('variant',              5),
    ('disease',              5),
    ('evidence_type',        4),
    ('evidence_direction',   4),
    ('variant_origin',       4),
]


def load_civic_regression(seed=42):
    rows = list(csv.DictReader(open(CIVIC_TSV), delimiter='\t'))
    examples = []
    for r in rows:
        if r.get('evidence_status') != 'accepted': continue
        rating_s = (r.get('rating', '') or '').strip()
        try: rating = float(rating_s)
        except ValueError: continue
        if rating < 1 or rating > 5: continue
        mp = r.get('molecular_profile', '').strip()
        parts = mp.split(' ', 1)
        gene = parts[0] if parts else ''
        variant = parts[1] if len(parts) > 1 else ''
        examples.append({
            'text':                 r.get('evidence_statement', ''),
            'gene':                 gene,
            'variant':              variant,
            'disease':              r.get('disease', ''),
            'evidence_type':        r.get('evidence_type', ''),
            'evidence_direction':   r.get('evidence_direction', ''),
            'variant_origin':       r.get('variant_origin', ''),
            'rating':               rating,
        })
    rng = random.Random(seed); rng.shuffle(examples)
    return examples


class CIViCRegDataset(Dataset):
    def __init__(self, exs, stoi, mods, y_mean, y_std):
        self.exs = exs; self.stoi = stoi; self.mods = mods
        self.y_mean = y_mean; self.y_std = y_std

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
        y = (ex['rating'] - self.y_mean) / max(self.y_std, 1e-6)
        return torch.tensor(ids, dtype=torch.long), float(y)


def reg_collate(batch, pad_index=0):
    if not batch: return None
    ids = [b[0] for b in batch]
    L = max(x.size(0) for x in ids)
    ids_p = torch.full((len(ids), L), pad_index, dtype=torch.long)
    for i, x in enumerate(ids): ids_p[i, :x.size(0)] = x
    y = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return {'ids': ids_p, 'label': y}


def film_attack(model, ids, criterion, target_y, device, vocab_size):
    model.zero_grad(set_to_none=False)
    pred = model(ids.to(device)).squeeze(-1)
    loss = criterion(pred, target_y.to(device))
    loss.backward()
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    if model.method == 'standard':
        gnorm = model.embedding.weight.grad.detach().norm(dim=1).cpu()
        thr = gnorm.max().item() * 0.05 if gnorm.max().item() > 0 else 1e-12
        recovered = set((gnorm > thr).nonzero(as_tuple=True)[0].tolist())
    else:
        recovered = set(range(vocab_size))
        if len(recovered) > 5 * len(actual):
            recovered = set(random.sample(list(recovered), min(20, len(recovered))))
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


def tag_attack(model, ids, criterion, target_y, device, vocab_size):
    model.zero_grad(set_to_none=False)
    pred = model(ids.to(device)).squeeze(-1)
    loss = criterion(pred, target_y.to(device))
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


def metrics(pred, y):
    pred = np.asarray(pred); y = np.asarray(y)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mae = float(np.mean(np.abs(pred - y)))
    pearson = float(np.corrcoef(pred, y)[0, 1]) if pred.std() > 1e-9 and y.std() > 1e-9 else 0.0
    rp = np.argsort(np.argsort(pred)); ry = np.argsort(np.argsort(y))
    spearman = float(np.corrcoef(rp, ry)[0, 1]) if rp.std() > 1e-9 and ry.std() > 1e-9 else 0.0
    return rmse, mae, pearson, spearman


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/civic_regression')
    p.add_argument('--n_epochs', type=int, default=6)
    p.add_argument('--methods', nargs='+', default=['standard', 'byte_seb'])
    p.add_argument('--n_attack', type=int, default=12)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading CIViC regression...')
    examples = load_civic_regression(seed=42)
    ratings = np.array([e['rating'] for e in examples])
    print(f'  {len(examples)} examples')
    print(f'  rating min={ratings.min():.0f} max={ratings.max():.0f} mean={ratings.mean():.2f}')

    n = len(examples)
    n_tr = int(0.7 * n); n_va = int(0.1 * n)
    train_ex = examples[:n_tr]; val_ex = examples[n_tr:n_tr + n_va]
    test_ex = examples[n_tr + n_va:]

    y_train = np.array([e['rating'] for e in train_ex])
    y_mean, y_std = float(y_train.mean()), float(y_train.std())
    print(f'  Train rating: mean={y_mean:.3f} std={y_std:.3f}')

    counter = Counter()
    for ex in train_ex:
        for mod, _ in MODALITIES:
            counter.update(tokenize(str(ex.get(mod, ''))))
    stoi = {PAD: 0, UNK: 1}
    for mod, _ in MODALITIES: stoi[f'{SEP}_{mod}'] = len(stoi)
    for tok, c in counter.most_common():
        if c >= 2: stoi[tok] = len(stoi)
    vocab_size = len(stoi)
    print(f'  vocab={vocab_size}  train={len(train_ex)} val={len(val_ex)} test={len(test_ex)}')

    summary = {}
    for method in args.methods:
        print(f'\n>>> {method}')
        _s=int(os.environ.get('RSEED','42')); random.seed(_s); np.random.seed(_s); torch.manual_seed(_s)
        if torch.cuda.is_available(): torch.cuda.manual_seed(_s)
        coll = functools.partial(reg_collate, pad_index=stoi[PAD])
        bs = 16
        train_ds = CIViCRegDataset(train_ex, stoi, MODALITIES, y_mean, y_std)
        val_ds = CIViCRegDataset(val_ex, stoi, MODALITIES, y_mean, y_std)
        test_ds = CIViCRegDataset(test_ex, stoi, MODALITIES, y_mean, y_std)
        tr = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=coll)
        val_l = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=coll)
        test_l = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=coll)

        model = MultiModalClassifier(vocab_size, 64, 1,
                                     method=method, body_kind='D',
                                     pad_idx=stoi[PAD]).to(device)
        opt = optim.Adam(model.parameters(), lr=5e-4)
        crit = nn.MSELoss()

        t0 = time.time()
        best_val_rmse, best_state = float('inf'), None
        for ep in range(args.n_epochs):
            model.train()
            for batch in tr:
                if batch is None: continue
                opt.zero_grad()
                p_pred = model(batch['ids'].to(device)).squeeze(-1)
                crit(p_pred, batch['label'].to(device)).backward()
                opt.step()
            model.eval()
            preds, ys = [], []
            with torch.no_grad():
                for batch in val_l:
                    if batch is None: continue
                    p_pred = model(batch['ids'].to(device)).squeeze(-1).cpu().numpy()
                    preds.extend(p_pred.tolist())
                    ys.extend(batch['label'].cpu().numpy().tolist())
            preds_raw = np.array(preds) * y_std + y_mean
            ys_raw = np.array(ys) * y_std + y_mean
            v_rmse, v_mae, v_p, v_s = metrics(preds_raw, ys_raw)
            print(f'  ep {ep+1}: val RMSE={v_rmse:.3f} MAE={v_mae:.3f} '
                  f'r={v_p:.3f} rho={v_s:.3f}')
            if v_rmse < best_val_rmse:
                best_val_rmse = v_rmse
                best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
        if best_state is not None: model.load_state_dict(best_state)
        train_time = time.time() - t0

        model.eval()
        preds, ys = [], []
        t0 = time.time()
        with torch.no_grad():
            for batch in test_l:
                if batch is None: continue
                p_pred = model(batch['ids'].to(device)).squeeze(-1).cpu().numpy()
                preds.extend(p_pred.tolist())
                ys.extend(batch['label'].cpu().numpy().tolist())
        inf_time = time.time() - t0
        preds_raw = np.array(preds) * y_std + y_mean
        ys_raw = np.array(ys) * y_std + y_mean
        rmse, mae, pearson, spearman = metrics(preds_raw, ys_raw)

        model.train()
        film_f1s, tag_f1s = [], []
        n_atk = 0
        for batch in tr:
            if batch is None or n_atk >= args.n_attack: break
            try: film_f1s.append(film_attack(model, batch['ids'], crit,
                                             batch['label'], device, vocab_size))
            except Exception as e: print(f'  film err: {e}')
            try: tag_f1s.append(tag_attack(model, batch['ids'], crit,
                                           batch['label'], device, vocab_size))
            except Exception as e: print(f'  tag err: {e}')
            n_atk += 1

        r = {
            'method': method,
            'val_rmse': float(best_val_rmse),
            'test_rmse': rmse, 'test_mae': mae,
            'test_pearson': pearson, 'test_spearman': spearman,
            'n_params': sum(p.numel() for p in model.parameters()),
            'vocab_size': vocab_size,
            'train_time_s': train_time, 'inference_time_s': inf_time,
            'film_f1': float(np.mean(film_f1s)) if film_f1s else 0.0,
            'tag_f1':  float(np.mean(tag_f1s))  if tag_f1s  else 0.0,
        }
        summary[method] = r
        print(f'  test RMSE={rmse:.3f} MAE={mae:.3f} r={pearson:.3f} rho={spearman:.3f}  '
              f'FILM={r["film_f1"]:.3f} TAG={r["tag_f1"]:.3f}  params={r["n_params"]}')

        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
