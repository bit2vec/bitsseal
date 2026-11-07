#!/usr/bin/env python3
"""Caduceus body + Bits encoder swap.

Pretrains a Caduceus-shaped Mamba body from random init with WikiText-103 and
the Bits encoder (vocab-independent FFN over a deterministic V x num_bits map),
then fine-tunes on Genomic Benchmarks at a user-chosen k (default k=22).

Architecture: Caduceus's CaduceusMixerModel body (bidirectional Mamba), but the
embedding layer is replaced with the Bits encoder (BinarySubwordConcateOnehot).
The Caduceus config defines hidden_dim=256 (effective output 512 because bidir
fwd+rev concat) and num_layers=16.

Phase 1: pretrain Caduceus+Bits on WikiText-103 with causal LM objective.
Phase 2: fine-tune at k=22 on gb_human_promoters (or any --dataset).
"""
import argparse, functools, json, os, random, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary_subword_sentiment import BinarySubwordConcateOnehot
from hyena_wikitext_to_dna import (
    WikiTextDataset, kmerize, build_kmer_vocab, load_gb,
    TokenDataset, collate as gb_collate,
)
import datasets
from transformers import GPT2Tokenizer, AutoConfig, AutoModel

CADUCEUS_MODEL_ID = 'kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16'


def _replace_embedding(caduceus_backbone, new_embedding):
    """Swap Caduceus's word_embeddings with our Bits encoder. The Caduceus
    backbone structure: caduceus.backbone.embeddings.word_embeddings (verified
    by inspect during runtime)."""
    # Walk the structure: AutoModel returns Caduceus (top), with .caduceus or
    # direct .backbone attr. Try both.
    bb = caduceus_backbone
    if hasattr(bb, 'caduceus'):
        bb = bb.caduceus
    if hasattr(bb, 'backbone'):
        bb = bb.backbone
    if hasattr(bb, 'embeddings') and hasattr(bb.embeddings, 'word_embeddings'):
        bb.embeddings.word_embeddings = new_embedding
        return
    raise RuntimeError('Could not find word_embeddings in Caduceus backbone')


def build_caduceus_with_bits(vocab_size, num_bits=64, interdim='256',
                              body_init='pretrained'):
    """Build Caduceus body and replace its embedding with the Bits encoder.

    body_init:
      'pretrained' (default) -> load HF Caduceus pretrained weights for the body
      'random'              -> from_config (random init at same architecture)
    """
    config = AutoConfig.from_pretrained(CADUCEUS_MODEL_ID, trust_remote_code=True)
    if body_init == 'pretrained':
        model = AutoModel.from_pretrained(CADUCEUS_MODEL_ID, trust_remote_code=True)
    else:
        config.vocab_size = vocab_size
        model = AutoModel.from_config(config, trust_remote_code=True)
    d_model = getattr(config, 'd_model', getattr(config, 'hidden_size', 256))
    bits_emb = BinarySubwordConcateOnehot(
        num_embeddings=vocab_size, embedding_dim=d_model,
        num_bits=num_bits, interdim=interdim, relu_dropout=0.1,
        subword_binary_file='')
    _replace_embedding(model, bits_emb)
    return model, config, d_model


class CaduceusBitsLM(nn.Module):
    """Wraps Caduceus body + Bits encoder + LM head for pretrain."""
    def __init__(self, backbone, vocab_size, d_out):
        super().__init__()
        self.backbone = backbone
        self.lm_head = nn.Linear(d_out, vocab_size, bias=False)

    def forward(self, input_ids):
        out = self.backbone(input_ids=input_ids)
        h = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
        return self.lm_head(h)


class CaduceusBitsClassifier(nn.Module):
    """Wraps Caduceus+Bits with mean-pool classifier head for fine-tune."""
    def __init__(self, backbone, d_out, n_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(d_out, n_classes)

    def forward(self, input_ids):
        out = self.backbone(input_ids=input_ids)
        h = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
        return self.classifier(h.mean(1))


def pretrain(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    vocab_size = tokenizer.vocab_size
    train_ds = WikiTextDataset(tokenizer, split='train',
                                 seq_length=args.seq_length, stride=args.stride)
    val_ds = WikiTextDataset(tokenizer, split='validation',
                              seq_length=args.seq_length, stride=args.seq_length)

    backbone, config, d_out = build_caduceus_with_bits(vocab_size, args.num_bits, args.interdim)
    model = CaduceusBitsLM(backbone, vocab_size, d_out).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Caduceus+Bits LM: {n_params:,} trainable params, d_out={d_out}, V={vocab_size}', flush=True)

    tr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                     num_workers=2, pin_memory=True)
    va = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                     num_workers=2, pin_memory=True)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    n_per_ep = len(tr)
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                            total_steps=max(args.n_epochs * n_per_ep, 1),
                                            pct_start=0.05)

    best_ppl = float('inf')
    for ep in range(args.n_epochs):
        model.train(); running = 0.0; n = 0; t0 = time.time()
        for step, batch in enumerate(tr):
            ids = batch['input_ids'].to(device, non_blocking=True)
            lab = batch['labels'].to(device, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(ids)
                loss = crit(logits.reshape(-1, logits.size(-1)), lab.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            running += float(loss.item()); n += 1
            if step % max(1, n_per_ep // 5) == 0:
                print(f'  ep {ep+1} step {step}/{n_per_ep} loss={loss.item():.3f}', flush=True)
        model.eval(); v_loss = 0.0; v_n = 0
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for batch in va:
                ids = batch['input_ids'].to(device); lab = batch['labels'].to(device)
                logits = model(ids)
                v_loss += float(crit(logits.reshape(-1, logits.size(-1)),
                                     lab.reshape(-1)).item()) * lab.size(0)
                v_n += lab.size(0)
        v_ppl = float(np.exp(v_loss / max(v_n, 1)))
        print(f'  ep {ep+1}/{args.n_epochs}  train_loss={running/max(n,1):.3f}  val_ppl={v_ppl:.2f}  ({time.time()-t0:.1f}s)', flush=True)
        if v_ppl < best_ppl:
            best_ppl = v_ppl
            torch.save({'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        'config': vars(args), 'vocab_size': vocab_size, 'val_ppl': v_ppl, 'epoch': ep + 1},
                       os.path.join(args.out_dir, 'pretrain.pt'))


def finetune(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

    train, val, test, n_classes = load_gb(args.dataset, k=args.k,
                                          max_len=args.max_len, seed=args.seed)
    stoi = build_kmer_vocab([s['tokens'] for s in train])
    print(f'  train={len(train)} val={len(val)} test={len(test)} vocab={len(stoi)}', flush=True)
    vocab_size = len(stoi)
    tr_ds = TokenDataset(train, stoi, max_len=args.max_len)
    va_ds = TokenDataset(val, stoi, max_len=args.max_len)
    te_ds = TokenDataset(test, stoi, max_len=args.max_len)

    summary = {'config': vars(args), 'vocab_size': vocab_size, 'n_classes': n_classes,
               'splits': {'train': len(train), 'val': len(val), 'test': len(test)},
               'conditions': {}}

    def make_clf():
        backbone, config, d_out = build_caduceus_with_bits(
            vocab_size, args.num_bits, args.interdim,
            body_init=getattr(args, 'body_init', 'pretrained'))
        return CaduceusBitsClassifier(backbone, d_out, n_classes), d_out

    def eval_loader(model, loader):
        model.eval(); c = t = 0
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for batch in loader:
                ids = batch['input_ids'].to(device); lab = batch['label'].to(device)
                pred = model(ids).argmax(-1)
                c += (pred == lab).sum().item(); t += lab.size(0)
        return c / max(t, 1)

    def train_one(model, name):
        tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=gb_collate,
                         num_workers=4, pin_memory=True, persistent_workers=True)
        va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=gb_collate,
                         num_workers=2, pin_memory=True, persistent_workers=True)
        te = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, collate_fn=gb_collate,
                         num_workers=2, pin_memory=True, persistent_workers=True)
        opt = optim.AdamW(model.parameters(), lr=args.lr_finetune, weight_decay=0.01)
        crit = nn.CrossEntropyLoss()
        t0 = time.time(); best_val = 0.0; best_state = None
        for ep in range(args.n_epochs):
            model.train()
            for batch in tr:
                ids = batch['input_ids'].to(device); lab = batch['label'].to(device)
                opt.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(ids)
                    loss = crit(logits, lab)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            v = eval_loader(model, va)
            print(f'    [{name}] epoch {ep+1}/{args.n_epochs}  val={v:.4f}', flush=True)
            if v > best_val:
                best_val = v
                best_state = {k: vv.detach().cpu().clone() for k, vv in model.state_dict().items()}
        if best_state is not None: model.load_state_dict(best_state)
        test_acc = float(eval_loader(model, te))
        ckpt_path = os.path.join(args.out_dir, f'model_{name}.pt')
        torch.save({
            'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'optimizer_state_dict': opt.state_dict(),
            'epochs_completed': args.n_epochs,
            'best_val_acc': float(best_val),
            'test_acc': test_acc,
            'config': vars(args),
            'condition': name,
        }, ckpt_path)
        print(f'  [{name}] saved checkpoint to {ckpt_path}', flush=True)
        return {'val_acc': float(best_val), 'test_acc': test_acc,
                'train_time_s': time.time() - t0,
                'n_params': sum(p.numel() for p in model.parameters()),
                'epochs_completed': args.n_epochs}

    # W: load pretrain
    if args.pretrain_ckpt and os.path.exists(args.pretrain_ckpt):
        print(f'[W] loading pretrain {args.pretrain_ckpt}', flush=True)
        ck = torch.load(args.pretrain_ckpt, map_location='cpu', weights_only=False)
        ps = ck['model_state_dict']
        m, _ = make_clf()
        md = m.state_dict()
        n_t, n_s = 0, 0
        for k, v in ps.items():
            if 'subword2binary' in k or 'subword' in k: n_s += 1; continue
            if k.startswith('lm_head.'): n_s += 1; continue
            if k in md and md[k].shape == v.shape:
                md[k] = v.clone(); n_t += 1
            else: n_s += 1
        m.load_state_dict(md, strict=False)
        print(f'  transferred {n_t}, skipped {n_s}', flush=True)
        rW = train_one(m.to(device), 'W')
        summary['conditions']['W_wikitext_init'] = rW
        print(f'  W: test={rW["test_acc"]:.4f}', flush=True)

    # R: random init
    m, _ = make_clf()
    rR = train_one(m.to(device), 'R')
    summary['conditions']['R_random_init'] = rR
    print(f'  R: test={rR["test_acc"]:.4f}', flush=True)

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['pretrain', 'finetune'], required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--pretrain_ckpt', default=None)
    p.add_argument('--dataset', default='gb_human_promoters')
    p.add_argument('--k', type=int, default=22)
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=6e-4)
    p.add_argument('--lr_finetune', type=float, default=1e-4)
    p.add_argument('--seq_length', type=int, default=256)
    p.add_argument('--stride', type=int, default=128)
    p.add_argument('--max_len', type=int, default=512)
    p.add_argument('--num_bits', type=int, default=64)
    p.add_argument('--interdim', type=str, default='256')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--body_init', choices=['pretrained', 'random'], default='pretrained',
                   help='Caduceus body init: pretrained (HF weights) or random')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.phase == 'pretrain': pretrain(args)
    else: finetune(args)


if __name__ == '__main__':
    main()
