#!/usr/bin/env python3
"""Extend HyenaDNA training from a saved checkpoint for N more epochs.

Loads model.pt + summary.json from a previous run_hyenadna_baseline.py output dir,
continues training for --more_epochs more epochs, re-evaluates on val/test,
overwrites model.pt + summary.json in the same dir.
"""
import argparse, functools, json, os, random, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_hyenadna_baseline import GB_HF_IDS, load_gb, HyenaDataset, collate
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', required=True)
    p.add_argument('--more_epochs', type=int, required=True)
    p.add_argument('--lr', type=float, default=None, help='Override lr (default keep)')
    p.add_argument('--out_dir', default=None)
    args = p.parse_args()

    if args.out_dir is None: args.out_dir = args.in_dir
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(os.path.join(args.in_dir, 'model.pt'), map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    print(f'Resuming from {args.in_dir}', flush=True)
    print(f'  Prior: {ckpt["epochs_completed"]} ep, val={ckpt["best_val_acc"]:.4f}, test={ckpt["test_acc"]:.4f}', flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    train, val, test, n_classes = load_gb(cfg['dataset'], max_len=cfg['max_len'], seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'], trust_remote_code=True)
    config = AutoConfig.from_pretrained(cfg['model_id'], trust_remote_code=True,
                                        num_labels=n_classes,
                                        problem_type='single_label_classification')
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg['model_id'], config=config, trust_remote_code=True,
        ignore_mismatched_sizes=True).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    lr = args.lr if args.lr is not None else cfg['lr']
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    try:
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        print('  Optimizer state restored', flush=True)
    except Exception as e:
        print(f'  Optimizer state could not be restored ({e}); fresh', flush=True)

    coll = functools.partial(collate)
    rc_aug = cfg.get('rc_aug', False)
    tr = DataLoader(HyenaDataset(train, tokenizer, cfg['max_len'], rc_aug=rc_aug, training=True),
                    batch_size=cfg['batch_size'], shuffle=True, collate_fn=coll)
    va = DataLoader(HyenaDataset(val, tokenizer, cfg['max_len']),
                    batch_size=cfg['batch_size'], shuffle=False, collate_fn=coll)
    te = DataLoader(HyenaDataset(test, tokenizer, cfg['max_len']),
                    batch_size=cfg['batch_size'], shuffle=False, collate_fn=coll)

    def evaluate(loader):
        model.eval(); c, t = 0, 0
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for b in loader:
                ids = b['input_ids'].to(device); lab = b['label'].to(device)
                out = model(input_ids=ids)
                pred = out.logits.argmax(-1)
                c += (pred == lab).sum().item(); t += lab.size(0)
        return c / max(t, 1)

    best_val = ckpt['best_val_acc']
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    t0 = time.time()
    for ep in range(args.more_epochs):
        model.train()
        for b in tr:
            ids = b['input_ids'].to(device); lab = b['label'].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(input_ids=ids, labels=lab)
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = evaluate(va)
        global_ep = ckpt['epochs_completed'] + ep + 1
        print(f'    epoch {global_ep} (extend {ep+1}/{args.more_epochs}) val={v:.4f}', flush=True)
        if v > best_val:
            best_val = v
            best_state = {k: vv.detach().cpu().clone() for k, vv in model.state_dict().items()}

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

    summary_path = os.path.join(args.in_dir, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f: summary = json.load(f)
    else: summary = {}
    summary.update({
        'val_acc': float(best_val), 'test_acc': float(test_acc),
        'epochs_completed': total_epochs, 'extend_time_s': extend_time,
        'orig_test_acc': ckpt['test_acc'], 'orig_val_acc': ckpt['best_val_acc'],
    })
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nExtended on {cfg["dataset"]}: total {total_epochs} ep', flush=True)
    print(f'  before: val={ckpt["best_val_acc"]:.4f} test={ckpt["test_acc"]:.4f}', flush=True)
    print(f'  after : val={best_val:.4f} test={test_acc:.4f}  time: {extend_time:.1f}s', flush=True)


if __name__ == '__main__':
    main()
