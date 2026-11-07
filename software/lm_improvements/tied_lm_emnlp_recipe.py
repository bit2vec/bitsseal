"""Paper-faithful tied 001 LM with EmByte's exact training recipe.

Adopts the recipe from emnlp2025/mt/byte_subword_embedding_scripts/iwslt/train_bpe.sh:

  --adam-betas '(0.9, 0.98)'         → AdamW(betas=(0.9, 0.98), eps=1e-9)
  --lr-scheduler inverse_sqrt
  --warmup-updates 4000
  --warmup-init-lr 1e-07
  --label-smoothing 0.1               → label_smoothing=0.1 in CE
  --weight-decay 0.0001               → weight_decay=1e-4 (not 1e-2)
  --clip-norm 0.0                     → no grad clip (we keep 1.0 for safety)
  --bw-aggre 'avg'
  --bw-layernum 2
  --bw-interdim 2048                  → FFN hidden 2048 (not 4096)
  --bw-relu-dropout 0.15

Tied via FFN materialisation: W = FFN(all_subword_bits) every forward pass.
Plus the Vaswani √d embedding-scale fix (we discovered this is essential).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LM_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'lm'))
sys.path.insert(0, LM_DIR)

from lm_wikitext_comparison import WikiTextDataset, evaluate            # noqa: E402
from binary_subword_sentiment import BinarySubwordConcateOnehot         # noqa: E402


class TiedFFNLM(nn.Module):
    """Same as PaperTiedLM in tied_lm_full_recipe.py, but supports an
    interdim_override to match EmByte's '2048' rather than our 4096."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads,
                 dropout, num_bits=64, interdim='2048', seq_length=256, mapping_file=''):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embed_scale = math.sqrt(embedding_dim)

        self.embedding = BinarySubwordConcateOnehot(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            num_bits=num_bits,
            interdim=interdim,
            relu_dropout=0.15,             # EmByte: 0.15
            subword_binary_file=mapping_file,
            use_layernorm=False,
            bias_init=0.0,
        )
        self.register_buffer('all_ids', torch.arange(vocab_size, dtype=torch.long))

        self.pos_embedding = nn.Embedding(seq_length + 10, embedding_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=hidden_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.final_ln = nn.LayerNorm(embedding_dim)

        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward(self, input_ids):
        B, T = input_ids.shape
        token_embeds = self.embedding(input_ids) * self.embed_scale
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.drop(token_embeds + self.pos_embedding(positions))
        mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()
        h = self.transformer(x, mask=mask)
        h = self.final_ln(h)
        # Tied via FFN materialisation
        W = self.embedding(self.all_ids)
        return h @ W.t()


class LabelSmoothedCrossEntropy(nn.Module):
    """Fairseq-style label smoothing (uniform over vocab + (1-eps) on target)."""

    def __init__(self, eps=0.1, ignore_index=-100):
        super().__init__()
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        V = log_probs.size(-1)
        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        smooth = -log_probs.mean(dim=-1)
        loss = (1.0 - self.eps) * nll + self.eps * smooth
        if self.ignore_index >= 0:
            mask = (target != self.ignore_index).float()
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1.0)
        return loss.mean()


def make_inv_sqrt_schedule(optimizer, warmup, init_lr, peak_lr):
    init_ratio = init_lr / peak_lr
    def lr_lambda(step):
        if step < warmup:
            return init_ratio + (1.0 - init_ratio) * (step / max(1, warmup))
        return math.sqrt(warmup / step)
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch_grad_accum(loader, model, criterion, optimizer, scheduler,
                           device, accum_steps=4, clip_grad=1.0):
    import tqdm, sys as _sys
    model.train()
    total_loss = 0.0; total_tokens = 0
    optimizer.zero_grad()
    micro = 0
    for batch in tqdm.tqdm(loader, desc='Train', file=_sys.stdout):
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        logits = model(input_ids)
        loss = criterion(logits.view(-1, model.vocab_size), target_ids.view(-1))
        if not torch.isfinite(loss):
            optimizer.zero_grad(); micro = 0; continue
        (loss / accum_steps).backward()
        micro += 1
        if micro == accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step(); scheduler.step()
            optimizer.zero_grad(); micro = 0
        n_tok = target_ids.numel()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
    if micro > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step(); scheduler.step()
        optimizer.zero_grad()
    return total_loss / max(total_tokens, 1), float(np.exp(total_loss / max(total_tokens, 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', required=True)
    p.add_argument('--n_epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--accum_steps', type=int, default=4)
    p.add_argument('--embedding_dim', type=int, default=512)
    p.add_argument('--hidden_dim', type=int, default=2048)
    p.add_argument('--num_layers', type=int, default=6)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.2)         # EmByte default
    p.add_argument('--peak_lr', type=float, default=5e-4)
    p.add_argument('--warmup_steps', type=int, default=4000)
    p.add_argument('--warmup_init_lr', type=float, default=1e-7)
    p.add_argument('--weight_decay', type=float, default=1e-4)   # EmByte
    p.add_argument('--label_smoothing', type=float, default=0.1) # EmByte
    p.add_argument('--num_bits', type=int, default=64)
    p.add_argument('--interdim', type=str, default='2048')        # EmByte
    p.add_argument('--seq_length', type=int, default=256)
    p.add_argument('--stride', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed); torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n===== TIED 001 LM + EmByte recipe :: device={device} =====")
    print(f"  recipe: AdamW β=(0.9, 0.98), eps=1e-9, wd={args.weight_decay}")
    print(f"          inverse-sqrt LR peak={args.peak_lr}, warmup={args.warmup_steps}")
    print(f"          label_smoothing={args.label_smoothing}, dropout={args.dropout}")
    print(f"          interdim={args.interdim}, accum={args.accum_steps}× (eff bs ≈ "
          f"{args.batch_size * args.accum_steps * args.seq_length:,} tokens)")

    tok = GPT2Tokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    vocab_size = len(tok)

    train_ds = WikiTextDataset(tok, 'train', args.seq_length, args.stride)
    valid_ds = WikiTextDataset(tok, 'validation', args.seq_length, args.stride, max_samples=10000)
    test_ds = WikiTextDataset(tok, 'test', args.seq_length, args.stride, max_samples=10000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    mapping_file = os.path.join(args.output_dir, 'binary_seb_mapping.pt')
    model = TiedFFNLM(
        vocab_size=vocab_size, embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_heads=args.num_heads, dropout=args.dropout,
        num_bits=args.num_bits, interdim=args.interdim,
        seq_length=args.seq_length, mapping_file=mapping_file,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_emb = sum(p.numel() for p in model.embedding.parameters())
    print(f"  total params: {n_params:,}  embedding(FFN): {n_emb:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.peak_lr,
                            betas=(0.9, 0.98), eps=1e-9,
                            weight_decay=args.weight_decay)
    scheduler = make_inv_sqrt_schedule(optimizer, args.warmup_steps,
                                       args.warmup_init_lr, args.peak_lr)
    criterion_train = LabelSmoothedCrossEntropy(eps=args.label_smoothing)
    criterion_eval = nn.CrossEntropyLoss()  # plain CE for PPL reporting

    model_path = os.path.join(args.output_dir, 'best_model.pt')
    partial_path = os.path.join(args.output_dir, 'results.json')
    tr_l, tr_p, va_l, va_p, ets = [], [], [], [], []
    best = float('inf'); start_epoch = 0
    if os.path.isfile(model_path) and os.path.isfile(partial_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        prev = json.load(open(partial_path))
        tr_l = prev.get('train_losses', []); tr_p = prev.get('train_ppls', [])
        va_l = prev.get('valid_losses', []); va_p = prev.get('valid_ppls', [])
        ets = prev.get('epoch_times', [0.0] * len(va_p))
        best = min(va_p) if va_p else float('inf')
        start_epoch = len(va_p)
        steps_per_epoch = max(1, len(train_loader) // args.accum_steps)
        for _ in range(start_epoch * steps_per_epoch): scheduler.step()

    start_wall = time.time()
    for epoch in range(start_epoch, args.n_epochs):
        t0 = time.time()
        trl, trp = train_epoch_grad_accum(train_loader, model, criterion_train,
                                          optimizer, scheduler, device,
                                          accum_steps=args.accum_steps)
        vl, vp = evaluate(valid_loader, model, criterion_eval, device)
        dt = time.time() - t0
        ets.append(dt)
        tr_l.append(trl); tr_p.append(trp); va_l.append(vl); va_p.append(vp)
        if vp < best:
            best = vp
            torch.save(model.state_dict(), model_path)
        print(f"Epoch {epoch+1}/{args.n_epochs} ({dt:.1f}s)  "
              f"Train PPL {trp:.2f}  Valid PPL {vp:.2f}  LR {scheduler.get_last_lr()[0]:.2e}")
        json.dump({
            'variant': 'tied_emnlp_recipe',
            'train_losses': tr_l, 'train_ppls': tr_p,
            'valid_losses': va_l, 'valid_ppls': va_p,
            'epoch_times': ets, 'best_valid_ppl': best,
        }, open(partial_path, 'w'), indent=2)

    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    tl, tp = evaluate(test_loader, model, criterion_eval, device)
    summary = {
        'variant': 'tied_emnlp_recipe', 'vocab_size': vocab_size,
        'best_valid_ppl': best, 'test_ppl': tp,
        'n_params': n_params, 'n_embed_params': n_emb,
        'train_ppls': tr_p, 'valid_ppls': va_p, 'epoch_times': ets,
        'config': vars(args),
    }
    json.dump(summary, open(os.path.join(args.output_dir, 'summary.json'), 'w'), indent=2)
    print(f"\n[tied_emnlp_recipe] best valid PPL {best:.2f}   test PPL {tp:.2f}")


if __name__ == '__main__':
    main()
