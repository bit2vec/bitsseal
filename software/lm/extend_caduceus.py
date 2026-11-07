#!/usr/bin/env python3
"""Extend Caduceus training from a saved checkpoint for N more epochs.

Loads model.pt + summary.json from a previous run_caduceus_baseline.py output dir,
continues training for --more_epochs epochs, re-evaluates on val/test, overwrites
the same dir's model.pt + summary.json.

Usage:
  python extend_caduceus.py --in_dir results/.../regulatory_pretrained --more_epochs 40
"""
import argparse, functools, json, os, random, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_caduceus_baseline import (
    GB_HF_IDS, load_gb, CharDataset, CaduceusClassifier, collate
)
from transformers import AutoTokenizer, AutoConfig, AutoModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', required=True, help='Directory containing model.pt + summary.json from prior run')
    p.add_argument('--more_epochs', type=int, required=True, help='How many ADDITIONAL epochs to train')
    p.add_argument('--lr', type=float, default=None, help='Override lr (default: keep original)')
    p.add_argument('--out_dir', default=None, help='Where to write new checkpoint (default: same as in_dir)')
    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = args.in_dir
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(os.path.join(args.in_dir, 'model.pt'), map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    print(f'Resuming from {args.in_dir}', flush=True)
    print(f'  Prior: {ckpt["epochs_completed"]} epochs, val_acc={ckpt["best_val_acc"]:.4f}, test_acc={ckpt["test_acc"]:.4f}', flush=True)
    print(f'  Extending by {args.more_epochs} more epochs', flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    # Re-load splits identically (uses same seed)
    train, val, test, n_classes = load_gb(cfg['dataset'], max_len=cfg['max_len'], seed=seed)
    print(f'  train={len(train)} val={len(val)} test={len(test)} n_classes={n_classes}', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'], trust_remote_code=True)
    backbone = AutoModel.from_pretrained(cfg['model_id'], trust_remote_code=True).to(device)
    probe = torch.zeros(1, cfg['max_len'], dtype=torch.long, device=device)
    model = CaduceusClassifier(backbone, n_classes, probe).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    lr = args.lr if args.lr is not None else cfg['lr']
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    try:
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        print('  Optimizer state restored', flush=True)
    except Exception as e:
        print(f'  Optimizer state could not be restored ({e}); using fresh', flush=True)

    coll = functools.partial(collate)
    rc_aug = cfg.get('rc_aug', False)
    tr = DataLoader(CharDataset(train, tokenizer, cfg['max_len'], rc_aug=rc_aug, training=True),
                    batch_size=cfg['batch_size'], shuffle=True, collate_fn=coll,
                    num_workers=4, pin_memory=True, persistent_workers=True)
    va = DataLoader(CharDataset(val, tokenizer, cfg['max_len']),
                    batch_size=cfg['batch_size'], shuffle=False, collate_fn=coll,
                    num_workers=2, pin_memory=True, persistent_workers=True)
    te = DataLoader(CharDataset(test, tokenizer, cfg['max_len']),
                    batch_size=cfg['batch_size'], shuffle=False, collate_fn=coll,
                    num_workers=2, pin_memory=True, persistent_workers=True)
    crit = nn.CrossEntropyLoss()

    def evaluate(loader):
        model.eval(); c, t = 0, 0
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for b in loader:
                ids = b['input_ids'].to(device, non_blocking=True)
                am = b.get('attention_mask')
                if am is not None: am = am.to(device, non_blocking=True)
                logits = model(ids, am)
                pred = logits.argmax(-1).cpu()
                c += (pred == b['label']).sum().item(); t += b['label'].numel()
        return c / max(t, 1)

    best_val = ckpt['best_val_acc']
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    t0 = time.time()
    for ep in range(args.more_epochs):
        model.train()
        for b in tr:
            ids = b['input_ids'].to(device, non_blocking=True)
            am = b.get('attention_mask')
            if am is not None: am = am.to(device, non_blocking=True)
            lab = b['label'].to(device, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(ids, am)
                loss = crit(logits, lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = evaluate(va)
        global_ep = ckpt['epochs_completed'] + ep + 1
        print(f'    epoch {global_ep} (extend {ep+1}/{args.more_epochs}) val_acc={v:.4f}', flush=True)
        if v > best_val:
            best_val = v
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    extend_time = time.time() - t0
    test_acc = evaluate(te)

    total_epochs = ckpt['epochs_completed'] + args.more_epochs
    torch.save({
        'model_state_dict': best_state,
        'optimizer_state_dict': opt.state_dict(),
        'epochs_completed': total_epochs,
        'best_val_acc': float(best_val),
        'test_acc': float(test_acc),
        'config': cfg,
    }, os.path.join(args.out_dir, 'model.pt'))

    # Update summary.json
    summary_path = os.path.join(args.in_dir, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f: summary = json.load(f)
    else:
        summary = {}
    summary.update({
        'val_acc': float(best_val), 'test_acc': float(test_acc),
        'epochs_completed': total_epochs,
        'extend_time_s': extend_time,
        'orig_test_acc': ckpt['test_acc'], 'orig_val_acc': ckpt['best_val_acc'],
    })
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nExtended on {cfg["dataset"]}: total {total_epochs} ep', flush=True)
    print(f'  before: val={ckpt["best_val_acc"]:.4f} test={ckpt["test_acc"]:.4f}', flush=True)
    print(f'  after : val={best_val:.4f} test={test_acc:.4f}', flush=True)
    print(f'  time: {extend_time:.1f}s', flush=True)


if __name__ == '__main__':
    main()
