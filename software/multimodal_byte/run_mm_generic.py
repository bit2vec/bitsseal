#!/usr/bin/env python3
"""Generic multimodal-fusion harness for the EMNLP 'Vocabulary Independence
Unlocks Efficient Multimodal Learning' paper.

Reuses the proven MultiModalClassifier (bodies A/C/D/E x methods
standard/byte_seb/binary_seb) and runs a controlled grid that varies ONLY the
encoder method (vocab built identically per method, same body/lr/seed).

For each (method, body, ablation, seed) it records test accuracy and the
embedding-parameter cost so we can show: Subword embedding table grows with the
union vocabulary while Bits stays at one constant ~33K encoder.

Datasets are pluggable loaders in mm_loaders.py, each returning:
    examples       list[dict]  (modality-name -> text, plus 'label_idx')
    modalities     list[(name, budget)]
    label_names    list[str]
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

from run_civic_multimodal import PAD, UNK, SEP, tokenize, collate
from run_civic_extended import MultiModalClassifier
from mm_loaders import LOADERS


class FusionDataset(Dataset):
    def __init__(self, examples, stoi, modalities, keep_mods):
        self.exs = examples; self.stoi = stoi
        self.mods = modalities; self.keep = keep_mods

    def __len__(self): return len(self.exs)

    def __getitem__(self, i):
        ex = self.exs[i]; ids = []
        for mod, budget in self.mods:
            if mod not in self.keep:
                tok_ids = [self.stoi[PAD]] * budget
            else:
                toks = tokenize(ex.get(mod, ''))
                tok_ids = [self.stoi.get(t, self.stoi[UNK]) for t in toks]
                tok_ids = tok_ids[:budget] + [self.stoi[PAD]] * max(0, budget - len(tok_ids))
            ids.append(self.stoi[f'{SEP}_{mod}'])
            ids.extend(tok_ids)
        return torch.tensor(ids, dtype=torch.long), ex['label_idx']


def build_vocab(train_ex, modalities, min_freq=2):
    counter = Counter()
    for ex in train_ex:
        for mod, _ in modalities:
            counter.update(tokenize(ex.get(mod, '')))
    stoi = {PAD: 0, UNK: 1}
    for mod, _ in modalities:
        stoi[f'{SEP}_{mod}'] = len(stoi)
    for tok, c in counter.most_common():
        if c >= min_freq:
            stoi[tok] = len(stoi)
    return stoi


def evaluate(model, loader, device):
    model.eval(); c = t = 0
    with torch.no_grad():
        for batch in loader:
            pred = model(batch['ids'].to(device)).argmax(-1).cpu()
            c += (pred == batch['label']).sum().item(); t += batch['label'].size(0)
    return c / max(t, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=list(LOADERS.keys()))
    p.add_argument('--out_dir', required=True)
    p.add_argument('--n_epochs', type=int, default=12)
    p.add_argument('--bodies', nargs='+', default=['A', 'D'])
    p.add_argument('--methods', nargs='+', default=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--emb_dim', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--n_target', type=int, default=20000)
    p.add_argument('--modality_scaling', action='store_true',
                   help='also sweep progressive modality subsets (1..K) for the scaling curve')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Loading {args.dataset} ...')
    examples, modalities, label_names = LOADERS[args.dataset](n_target=args.n_target)
    num_classes = len(label_names)
    rng = random.Random(42); rng.shuffle(examples)
    n = len(examples); n_tr = int(0.7 * n); n_va = int(0.1 * n)
    train_ex, val_ex, test_ex = examples[:n_tr], examples[n_tr:n_tr+n_va], examples[n_tr+n_va:]
    stoi = build_vocab(train_ex, modalities)
    vocab_size = len(stoi)
    print(f'examples={n} vocab={vocab_size} classes={num_classes} '
          f'modalities={[m for m,_ in modalities]} tr/va/te={len(train_ex)}/{len(val_ex)}/{len(test_ex)}')

    # ablation sets: 'all', plus progressive subsets if requested
    all_mods = [m for m, _ in modalities]
    ablations = {'all': set(all_mods)}
    if args.modality_scaling:
        for k in range(1, len(all_mods) + 1):
            ablations[f'first{k}'] = set(all_mods[:k])

    summary, meta = {}, {
        'dataset': args.dataset, 'vocab_size': vocab_size, 'emb_dim': args.emb_dim,
        'modalities': all_mods, 'num_classes': num_classes,
        'subword_emb_params': vocab_size * args.emb_dim,
        'n_examples': n,
    }
    coll = functools.partial(collate, pad_index=stoi[PAD])

    for method in args.methods:
        for abl_name, keep in ablations.items():
            for body in args.bodies:
                accs, params = [], None
                for seed in args.seeds:
                    key = f'{method}_{abl_name}_{body}_s{seed}'
                    try:
                        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
                        tr = DataLoader(FusionDataset(train_ex, stoi, modalities, keep),
                                        batch_size=args.batch_size, shuffle=True, collate_fn=coll)
                        va = DataLoader(FusionDataset(val_ex, stoi, modalities, keep),
                                        batch_size=args.batch_size, shuffle=False, collate_fn=coll)
                        te = DataLoader(FusionDataset(test_ex, stoi, modalities, keep),
                                        batch_size=args.batch_size, shuffle=False, collate_fn=coll)
                        model = MultiModalClassifier(vocab_size, args.emb_dim, num_classes,
                                                     method=method, body_kind=body,
                                                     pad_idx=stoi[PAD]).to(device)
                        opt = optim.Adam(model.parameters(), lr=args.lr)
                        crit = nn.CrossEntropyLoss()
                        t0 = time.time(); best_val, best_state = 0.0, None
                        for ep in range(args.n_epochs):
                            model.train()
                            for batch in tr:
                                opt.zero_grad()
                                crit(model(batch['ids'].to(device)),
                                     batch['label'].to(device)).backward()
                                opt.step()
                            v = evaluate(model, va, device)
                            if v > best_val:
                                best_val = v
                                best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
                        if best_state: model.load_state_dict(best_state)
                        acc = evaluate(model, te, device)
                        accs.append(acc)
                        if params is None:
                            params = sum(p.numel() for p in model.parameters())
                            # encoder-only param count (embedding/bits module)
                            enc = sum(p.numel() for p in model.embedding.parameters())
                            meta.setdefault('encoder_params', {})[method] = enc
                        print(f'  {key}: test_acc={acc:.4f} (val {best_val:.4f}) '
                              f'[{time.time()-t0:.1f}s]')
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        accs.append(None)
                good = [a for a in accs if a is not None]
                summary[f'{method}_{abl_name}_{body}'] = {
                    'method': method, 'ablation': abl_name, 'body': body,
                    'seeds': args.seeds, 'test_accs': accs,
                    'mean_acc': float(np.mean(good)) if good else None,
                    'std_acc': float(np.std(good)) if good else None,
                    'n_modalities': len(keep), 'n_params': params,
                }
                json.dump({'meta': meta, 'results': summary},
                          open(os.path.join(args.out_dir, 'summary.json'), 'w'), indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
