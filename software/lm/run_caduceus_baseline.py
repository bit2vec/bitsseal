#!/usr/bin/env python3
"""Caduceus-PS baseline on Genomic Benchmarks.

Loads kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 (bidirectional
Mamba, reverse-complement equivariant, ~7M params) and fine-tunes for sequence
classification. Single-nucleotide tokenization, mean-pool last hidden state,
linear classifier head.

Reports test accuracy + parameter count for direct comparison with our
Hyena+Bits k=14 fine-tune experiment on the same dataset splits.
"""
import argparse, functools, json, os, random, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import datasets
from transformers import AutoTokenizer, AutoModel, AutoConfig


GB_HF_IDS = {
    'gb_human_promoters': 'katarinagresova/Genomic_Benchmarks_human_nontata_promoters',
    'gb_human_enhancers_ensembl': 'katarinagresova/Genomic_Benchmarks_human_enhancers_ensembl',
    'gb_human_ocr_ensembl': 'katarinagresova/Genomic_Benchmarks_human_ocr_ensembl',
    'gb_human_regulatory': 'katarinagresova/Genomic_Benchmarks_human_ensembl_regulatory',
    'gb_demo_human_or_worm': 'katarinagresova/Genomic_Benchmarks_demo_human_or_worm',
}


_RC_TBL = str.maketrans('ACGTN', 'TGCAN')
def revcomp(s): return s.translate(_RC_TBL)[::-1]


def load_gb(name, max_len=512, seed=42):
    ds = datasets.load_dataset(GB_HF_IDS[name])
    full_train = list(ds['train'])
    test = list(ds['test'])
    rng = random.Random(seed); rng.shuffle(full_train)
    n_val = max(1, int(len(full_train) * 0.1))
    val = full_train[:n_val]; train = full_train[n_val:]
    # Detect number of classes
    n_classes = max(r['label'] for r in full_train) + 1

    def proc(rows):
        out = []
        for r in rows:
            seq = r['seq'].upper()
            if len(seq) > max_len: seq = seq[:max_len]
            out.append({'seq': seq, 'label': int(r['label'])})
        return out
    return proc(train), proc(val), proc(test), n_classes


class CharDataset(Dataset):
    """Single-nucleotide tokenization for Caduceus. Maps A/C/G/T/N to fixed IDs
    matching the Caduceus tokenizer's vocab."""
    def __init__(self, examples, tokenizer, max_len=512, rc_aug=False, training=False):
        self.exs = examples; self.tok = tokenizer
        self.max_len = max_len
        self.rc_aug = rc_aug; self.training = training

    def __len__(self): return len(self.exs)

    def __getitem__(self, i):
        ex = self.exs[i]
        seq = ex['seq']
        if self.rc_aug and self.training and random.random() < 0.5:
            seq = revcomp(seq)
        enc = self.tok(seq, padding='max_length', truncation=True,
                       max_length=self.max_len, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0)
                                  if 'attention_mask' in enc else None,
                'label': torch.tensor(ex['label'], dtype=torch.long)}


def collate(batch):
    out = {'input_ids': torch.stack([b['input_ids'] for b in batch]),
           'label': torch.stack([b['label'] for b in batch])}
    if batch[0]['attention_mask'] is not None:
        out['attention_mask'] = torch.stack([b['attention_mask'] for b in batch])
    return out


class CaduceusClassifier(nn.Module):
    """Wrap Caduceus backbone with a mean-pool + linear classifier head.
    Caduceus-PS is bidirectional (concatenates fwd/rev streams) so output dim
    is 2*d_model. We probe the model with a dummy forward to pick the right size."""
    def __init__(self, backbone, n_classes, probe_input_ids):
        super().__init__()
        self.backbone = backbone
        with torch.no_grad():
            out = self.backbone(input_ids=probe_input_ids)
            h = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
        hidden_dim = h.size(-1)
        print(f'  Caduceus actual hidden dim: {hidden_dim}', flush=True)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, input_ids, attention_mask=None):
        out = self.backbone(input_ids=input_ids)
        h = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            h = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            h = h.mean(1)
        return self.classifier(h)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(GB_HF_IDS.keys()),
                   default='gb_human_promoters')
    p.add_argument('--out_dir', default=None)
    p.add_argument('--model_id',
                   default='kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16')
    p.add_argument('--max_len', type=int, default=512)
    p.add_argument('--n_epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--from_pretrained', action='store_true', default=True)
    p.add_argument('--from_scratch', action='store_true')
    p.add_argument('--rc_aug', action='store_true',
                   help='RC augmentation (Caduceus-PS is RC-equivariant so usually not needed)')
    p.add_argument('--schedule', choices=['flat', 'cosine'], default='flat',
                   help='LR schedule: flat constant or cosine with warmup')
    p.add_argument('--warmup_ratio', type=float, default=0.05,
                   help='Fraction of total steps used for warmup (cosine only)')
    args = p.parse_args()

    if args.out_dir is None:
        suffix = 'pretrained' if not args.from_scratch else 'random'
        args.out_dir = f'results/multimodality/caduceus_baseline/{args.dataset}_{suffix}'
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  dataset={args.dataset}  '
          f'init={"pretrained" if not args.from_scratch else "random"}', flush=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

    train, val, test, n_classes = load_gb(args.dataset, max_len=args.max_len,
                                          seed=args.seed)
    print(f'  train={len(train)}  val={len(val)}  test={len(test)}  '
          f'n_classes={n_classes}', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    d_model = getattr(config, 'd_model', getattr(config, 'hidden_size', 256))

    if args.from_scratch:
        backbone = AutoModel.from_config(config, trust_remote_code=True)
    else:
        backbone = AutoModel.from_pretrained(args.model_id, trust_remote_code=True)

    backbone = backbone.to(device)
    probe = torch.zeros(1, args.max_len, dtype=torch.long, device=device)
    model = CaduceusClassifier(backbone, n_classes, probe).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Caduceus+classifier params: total={n_params:,}  trainable={n_trainable:,}',
          flush=True)

    coll = functools.partial(collate)
    tr = DataLoader(CharDataset(train, tokenizer, args.max_len,
                                rc_aug=args.rc_aug, training=True),
                    batch_size=args.batch_size, shuffle=True, collate_fn=coll,
                    num_workers=4, pin_memory=True, persistent_workers=True)
    va = DataLoader(CharDataset(val, tokenizer, args.max_len),
                    batch_size=args.batch_size, shuffle=False, collate_fn=coll,
                    num_workers=2, pin_memory=True, persistent_workers=True)
    te = DataLoader(CharDataset(test, tokenizer, args.max_len),
                    batch_size=args.batch_size, shuffle=False, collate_fn=coll,
                    num_workers=2, pin_memory=True, persistent_workers=True)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    sched = None
    if args.schedule == 'cosine':
        total_steps = max(args.n_epochs * len(tr), 1)
        warmup_steps = max(int(total_steps * args.warmup_ratio), 1)
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            import math as _m
            return 0.5 * (1 + _m.cos(_m.pi * progress))
        sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        print(f'  cosine schedule: total_steps={total_steps}, warmup={warmup_steps}', flush=True)

    def evaluate(loader):
        model.eval(); correct, total = 0, 0
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for b in loader:
                ids = b['input_ids'].to(device, non_blocking=True)
                am = b.get('attention_mask')
                if am is not None: am = am.to(device, non_blocking=True)
                logits = model(ids, am)
                pred = logits.argmax(-1).cpu()
                correct += (pred == b['label']).sum().item()
                total += b['label'].numel()
        return correct / max(total, 1)

    t0 = time.time(); best_val = 0.0; best_state = None
    for ep in range(args.n_epochs):
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
            if sched is not None: sched.step()
        v = evaluate(va)
        if v > best_val:
            best_val = v; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f'    epoch {ep+1}/{args.n_epochs}  val_acc={v:.4f}', flush=True)
    if best_state is not None: model.load_state_dict(best_state)
    train_time = time.time() - t0
    test_acc = evaluate(te)

    # Save best-val state + optimizer for resume / extend training
    ckpt_path = os.path.join(args.out_dir, 'model.pt')
    torch.save({
        'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'optimizer_state_dict': opt.state_dict(),
        'epochs_completed': args.n_epochs,
        'best_val_acc': float(best_val),
        'test_acc': float(test_acc),
        'config': vars(args),
    }, ckpt_path)
    print(f'Saved checkpoint to {ckpt_path}', flush=True)

    summary = {'config': vars(args), 'model_id': args.model_id,
               'init': 'random' if args.from_scratch else 'pretrained',
               'n_params': n_params, 'n_trainable': n_trainable,
               'val_acc': float(best_val), 'test_acc': float(test_acc),
               'train_time_s': train_time, 'epochs_completed': args.n_epochs,
               'splits': {'train': len(train), 'val': len(val), 'test': len(test)},
               'n_classes': n_classes}
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n=== Caduceus-PS on {args.dataset} ===', flush=True)
    print(f'  init: {summary["init"]}  params: {n_params:,}', flush=True)
    print(f'  test_acc: {test_acc:.4f}  val_acc: {best_val:.4f}  time: {train_time:.1f}s', flush=True)
    print(f'Wrote {args.out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
