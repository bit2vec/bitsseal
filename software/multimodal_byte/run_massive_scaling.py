#!/usr/bin/env python3
"""MASSIVE multilingual language-scaling efficiency curve.

The headline NLP demonstration of "Vocabulary Independence Unlocks Efficient
Multimodal Learning": as you add languages, a Subword model's union vocabulary
(hence embedding table) explodes, while Bits stays at one constant ~33K encoder.

For N in a sweep of language counts we train standard vs Bits intent
classifiers on the union of the first N MASSIVE locales and record:
  - union vocab size (=> Subword embedding params/MB)
  - test accuracy (parity)
Bits embedding params are constant.
"""
import argparse, functools, json, os, random, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, THIS_DIR)
from run_civic_multimodal import PAD, UNK, tokenize
from run_civic_extended import MultiModalClassifier

MIRROR = 'mteb/amazon_massive_intent'
# English first (anchor), then a diverse spread of scripts; filled from the
# mirror's config list at runtime so the order is stable.
_PREFERRED = ['en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'ru', 'pl', 'sv',
              'tr', 'fi', 'hu', 'el', 'he', 'ar', 'fa', 'hi', 'bn', 'ta',
              'te', 'kn', 'ml', 'th', 'vi', 'id', 'ms', 'tl', 'ja', 'ko',
              'zh-CN', 'zh-TW', 'sw', 'am', 'is', 'nb', 'da', 'cy', 'ka',
              'hy', 'sq', 'az', 'km', 'my', 'mn', 'jv', 'af', 'sl', 'lv',
              'ur', 'ro']


def _all_locales():
    from datasets import get_dataset_config_names
    cfgs = [c for c in get_dataset_config_names(MIRROR) if c != 'default']
    ordered = [c for c in _PREFERRED if c in cfgs] + [c for c in cfgs if c not in _PREFERRED]
    return ordered


LOCALES = None  # filled in main()
_LABEL_MAP = {}


def load_locales(locales, n_per=None):
    from datasets import load_dataset
    rows = []
    for lc in locales:
        try:
            ds = load_dataset(MIRROR, lc, split='train')
        except Exception as e:
            print(f'  skip {lc}: {str(e)[:60]}'); continue
        n = 0
        for ex in ds:
            lab = ex['label_text']
            if lab not in _LABEL_MAP:
                _LABEL_MAP[lab] = len(_LABEL_MAP)
            rows.append({'text': ex['text'], 'label_idx': _LABEL_MAP[lab]})
            n += 1
            if n_per and n >= n_per: break
    return rows


class TextDS(Dataset):
    def __init__(self, rows, stoi, budget=64):
        self.rows = rows; self.stoi = stoi; self.budget = budget
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        toks = tokenize(self.rows[i]['text'])
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in toks][:self.budget]
        ids += [self.stoi[PAD]] * (self.budget - len(ids))
        return torch.tensor(ids, dtype=torch.long), self.rows[i]['label_idx']


def collate(batch, pad=0):
    ids = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return {'ids': ids, 'label': labels}


def run_one(rows, method, body, emb_dim, n_classes, device, epochs, lr, bs, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed); rng.shuffle(rows)
    n = len(rows); ntr = int(0.8 * n); nva = int(0.1 * n)
    tr, va, te = rows[:ntr], rows[ntr:ntr+nva], rows[ntr+nva:]
    cnt = Counter()
    for r in tr: cnt.update(tokenize(r['text']))
    stoi = {PAD: 0, UNK: 1}
    for t, c in cnt.most_common():
        if c >= 2: stoi[t] = len(stoi)
    V = len(stoi)
    coll = functools.partial(collate, pad=stoi[PAD])
    trL = DataLoader(TextDS(tr, stoi), batch_size=bs, shuffle=True, collate_fn=coll)
    vaL = DataLoader(TextDS(va, stoi), batch_size=bs, collate_fn=coll)
    teL = DataLoader(TextDS(te, stoi), batch_size=bs, collate_fn=coll)
    model = MultiModalClassifier(V, emb_dim, n_classes, method=method,
                                 body_kind=body, pad_idx=stoi[PAD]).to(device)
    opt = optim.Adam(model.parameters(), lr=lr); crit = nn.CrossEntropyLoss()
    def ev(L):
        model.eval(); c = t = 0
        with torch.no_grad():
            for b in L:
                p = model(b['ids'].to(device)).argmax(-1).cpu()
                c += (p == b['label']).sum().item(); t += b['label'].size(0)
        return c / max(t, 1)
    best = 0.0; best_state = None
    for _ in range(epochs):
        model.train()
        for b in trL:
            opt.zero_grad()
            crit(model(b['ids'].to(device)), b['label'].to(device)).backward()
            opt.step()
        v = ev(vaL)
        if v > best: best = v; best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    enc = sum(p.numel() for p in model.embedding.parameters())
    return {'vocab': V, 'test_acc': ev(teL), 'encoder_params': enc,
            'total_params': sum(p.numel() for p in model.parameters())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--lang_counts', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32, 51])
    p.add_argument('--methods', nargs='+', default=['standard', 'binary_seb', 'byte_seb'])
    p.add_argument('--body', default='A')
    p.add_argument('--emb_dim', type=int, default=64)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--n_per', type=int, default=4000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = 60  # MASSIVE intents
    global LOCALES
    LOCALES = _all_locales()
    print(f'{len(LOCALES)} locales available; sweeping {args.lang_counts}')

    summary = {}
    for N in args.lang_counts:
        locs = LOCALES[:N]
        print(f'\n=== {N} languages ({locs[-1]} ...) ===')
        rows = load_locales(locs, n_per=args.n_per)
        print(f'  loaded {len(rows)} utterances')
        for method in args.methods:
            t0 = time.time()
            r = run_one(rows, method, args.body, args.emb_dim, n_classes,
                        device, args.epochs, args.lr, args.bs, args.seed)
            r['n_languages'] = N
            r['subword_gb_d768'] = round(r['vocab'] * 768 * 4 / 1e9, 5)
            summary[f'{N}lang_{method}'] = r
            print(f'  {method}: vocab={r["vocab"]} acc={r["test_acc"]:.4f} '
                  f'enc_params={r["encoder_params"]} [{time.time()-t0:.0f}s]')
            json.dump(summary, open(os.path.join(args.out_dir, 'summary.json'), 'w'), indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
