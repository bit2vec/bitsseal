#!/usr/bin/env python3
"""LLG (label leakage from gradient) on six encoders × SST-2.
Reuses encoder wrappers from run_encoder_privacy_attacks.py.
Output: results/v8_panel_a/encoder_llg/summary.json
"""
import argparse, functools, json, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, THIS_DIR)

from health_classification import (build_vocab, TextClassificationDataset,
                                    collate, PAD)
from sentiment_privacy import load_sst2
from film_lamp_classification import TinyClassifier
from llg_healthcare import extract_labels_llg, asr
from run_encoder_privacy_attacks import (
    HashClassifier, ByteEmbeddingClassifier, ByteSST2Dataset, byte_collate,
    train_classifier,
)


def llg_eval(model, loader, device, num_classes, n_batches=20):
    crit = nn.CrossEntropyLoss()
    model.train()
    asrs = []
    for i, batch in enumerate(loader):
        if batch is None or i >= n_batches: break
        ids = batch['ids'].to(device); lbl = batch['label'].to(device)
        try:
            model.zero_grad(set_to_none=False)
            crit(model(ids), lbl).backward()
            g = model.fc.weight.grad.clone()
            pred = extract_labels_llg(g, ids.size(0), num_classes)
            asrs.append(asr(pred, lbl.cpu().tolist()))
        except Exception as e:
            print(f'  err: {e}', flush=True)
    return float(np.mean(asrs)) if asrs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/v8_panel_a/encoder_llg')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, _, _, num_classes = load_sst2(seed=42)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    print(f'SST-2 train={len(train_ex)} V={vocab_size} C={num_classes}', flush=True)

    summary = {}
    for name, builder in [
        ('Subword',    lambda: TinyClassifier(vocab_size, 64, num_classes, method='standard').to(device)),
        ('EmByte',     lambda: TinyClassifier(vocab_size, 64, num_classes, method='byte_seb').to(device)),
        ('Bits',       lambda: TinyClassifier(vocab_size, 64, num_classes, method='binary_seb').to(device)),
        ('HashEmb',    lambda: HashClassifier(vocab_size, num_classes, B=10000, K=2, d=64).to(device)),
        ('CHARFORMER', lambda: ByteEmbeddingClassifier(num_classes, d=64, family='charformer').to(device)),
        ('ByT5',       lambda: ByteEmbeddingClassifier(num_classes, d=64, family='byt5').to(device)),
        ('MegaByte',   lambda: ByteEmbeddingClassifier(num_classes, d=64, family='megabyte').to(device)),
    ]:
        t0 = time.time()
        print(f'>>> {name}', flush=True)
        try:
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            model = builder()
            if name in ('CHARFORMER', 'ByT5', 'MegaByte'):
                ds = ByteSST2Dataset(train_ex, max_length=128)
                loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=byte_collate)
            else:
                loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=coll)
            model, _ = train_classifier(model, loader, device, n_epochs=1)
            a = llg_eval(model, loader, device, num_classes)
            summary[name] = {'llg_asr': a, 'time_s': time.time() - t0}
            print(f'  llg_asr={a} ({summary[name]["time_s"]:.1f}s)', flush=True)
        except Exception as e:
            summary[name] = {'error': str(e)}
            print(f'  ERROR: {e}', flush=True)
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
