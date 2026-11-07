#!/usr/bin/env python3
"""Code-length (n / num_bits) sweep on real CIViC / body D.

Fixes W and sweeps num_bits in {8,16,32,64,128,256}. n must exceed
ceil(log2 V) for injective codes; below the floor codes collide -> accuracy
collapses. Everything else identical to run_bits_width_ablation.py."""
import argparse, functools, json, os, random, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))

from run_civic_extended import MultiModalClassifier
from run_civic_multimodal import (
    PAD, MODALITIES, load_civic, build_vocab, CIViCMultiModalDataset, collate,
)


def evaluate(model, loader, device):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            pred = model(batch['ids'].to(device)).argmax(-1).cpu()
            c += (pred == batch['label']).sum().item()
            t += batch['label'].size(0)
    return c / max(t, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='results/bits_nbits')
    ap.add_argument('--n_epochs', type=int, default=6)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--bits', nargs='+', type=int, default=[8, 16, 32, 64, 128, 256])
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, val_ex, test_ex = load_civic(seed=42)
    stoi, vocab_meta = build_vocab(train_ex, MODALITIES, boundary='separator')
    V = len(stoi)
    num_classes = 5
    floor = math.ceil(math.log2(V))
    print(f'V={V} train={len(train_ex)}/val={len(val_ex)}/test={len(test_ex)} '
          f'W={a.width}  injective floor=ceil(log2 V)={floor} bits')

    coll = functools.partial(collate, pad_index=stoi[PAD])
    train_ds = CIViCMultiModalDataset(train_ex, stoi, vocab_meta, MODALITIES, ablation='all')
    val_ds = CIViCMultiModalDataset(val_ex, stoi, vocab_meta, MODALITIES, ablation='all')
    test_ds = CIViCMultiModalDataset(test_ex, stoi, vocab_meta, MODALITIES, ablation='all')
    tr = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=coll)
    val_l = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=coll)
    test_l = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=coll)

    summary = {}
    for nb in a.bits:
        print(f'\n>>> num_bits={nb} (floor={floor})')
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(a.seed)
        model = MultiModalClassifier(V, 64, num_classes, method='binary_seb', body_kind='D',
                                     pad_idx=stoi[PAD], num_bits=nb,
                                     interdim=str(a.width)).to(device)
        # measure code-collision rate (unique codes / V) to expose the floor
        try:
            codes = model.embedding.subword2binary[:V]
            uniq = len({tuple(r.tolist()) for r in codes})
            collision_rate = 1.0 - uniq / V
        except Exception:
            collision_rate = float('nan')
        opt = optim.Adam(model.parameters(), lr=5e-4)
        crit = nn.CrossEntropyLoss()
        best_val, best_state = 0.0, None
        for ep in range(a.n_epochs):
            model.train()
            for batch in tr:
                if batch is None:
                    continue
                opt.zero_grad()
                crit(model(batch['ids'].to(device)), batch['label'].to(device)).backward()
                opt.step()
            v = evaluate(model, val_l, device)
            print(f'  ep{ep+1} val={v:.4f}')
            if v > best_val:
                best_val = v
                best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        test_acc = evaluate(model, test_l, device)
        summary[nb] = {
            'num_bits': nb, 'val_acc': float(best_val), 'test_acc': float(test_acc),
            'collision_rate': collision_rate, 'below_floor': nb < floor,
        }
        print(f'  test={test_acc:.4f}  collisions={collision_rate:.3f}  below_floor={nb < floor}')
        with open(os.path.join(a.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    print(f'\nWrote {a.out_dir}/summary.json')


if __name__ == '__main__':
    main()
