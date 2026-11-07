"""Geneformer 3-method comparison: Subword (original) vs EmByte vs 001.

Loads V1-10M pretrained Geneformer (BertForMaskedLM), distills its
word_embeddings into a 64-bit code FFN (001) and an 8-byte FFN (EmByte),
then trains a cell-classification head on a synthetic gene-token dataset
that matches Geneformer's input format. Apples-to-apples comparison.

For real Geneformer evaluation, swap make_synthetic_geneformer_dataset for
a loader of one of their published fine-tune corpora (cardiomyopathies,
disease classification etc.). Synthetic mode lets the comparison run today
without further data downloads.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from binary_geneformer import (
    BinaryGeneTokenEmbedding, ByteGeneTokenEmbedding,
    distill_embedding, patch_word_embedding_with_module, count_params,
)

from transformers import BertConfig, BertForMaskedLM, BertModel


def fmt_n(n):
    if n > 1e9: return f'{n/1e9:.2f}B'
    if n > 1e6: return f'{n/1e6:.2f}M'
    if n > 1e3: return f'{n/1e3:.2f}K'
    return str(n)


# -------- synthetic gene-classification task --------
def make_synthetic_geneformer_dataset(num_cells=4000, num_classes=10,
                                       vocab_size=25426, mean_len=512,
                                       seed=42):
    rng = np.random.default_rng(seed)
    markers = [rng.choice(np.arange(4, vocab_size), size=80, replace=False)
               for _ in range(num_classes)]
    cells, labels = [], []
    for _ in range(num_cells):
        c = int(rng.integers(num_classes))
        L = int(np.clip(rng.normal(mean_len, mean_len * 0.2), 64, mean_len * 2))
        n_sig = rng.integers(L // 4, L // 2)
        sig = rng.choice(markers[c], size=n_sig, replace=True)
        bg = rng.integers(4, vocab_size, size=L - n_sig)
        ids = np.concatenate([[2], sig, bg]); rng.shuffle(ids[1:])
        cells.append(ids.tolist()); labels.append(c)
    n_train = int(0.8 * num_cells); n_val = int(0.1 * num_cells)
    return ((cells[:n_train], labels[:n_train]),
            (cells[n_train:n_train+n_val], labels[n_train:n_train+n_val]),
            (cells[n_train+n_val:], labels[n_train+n_val:]))


class GeneSentenceDataset(Dataset):
    def __init__(self, cells, labels, max_len=512, pad_id=0):
        self.cells, self.labels = cells, labels
        self.max_len = max_len; self.pad = pad_id
    def __len__(self): return len(self.cells)
    def __getitem__(self, i):
        ids = self.cells[i][:self.max_len]
        return ids, int(self.labels[i])


def collate(batch, pad_id=0):
    seqs, labels = zip(*batch)
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    am = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.as_tensor(s, dtype=torch.long)
        am[i, :len(s)] = 1
    return out, am, torch.tensor(labels, dtype=torch.long)


# -------- classification wrapper --------
class GeneformerClassifier(nn.Module):
    def __init__(self, bert_model, num_classes, hidden=256):
        super().__init__()
        self.bert = bert_model.bert       # encoder
        self.dropout = nn.Dropout(0.1)
        self.cls = nn.Linear(hidden, num_classes)
    def forward(self, ids, am):
        out = self.bert(input_ids=ids, attention_mask=am)
        h = out.last_hidden_state[:, 0, :]    # CLS
        return self.cls(self.dropout(h))
    def count_params(self):
        word = sum(p.numel() for p in self.bert.embeddings.word_embeddings.parameters())
        pos  = sum(p.numel() for p in self.bert.embeddings.position_embeddings.parameters())
        tt   = sum(p.numel() for p in self.bert.embeddings.token_type_embeddings.parameters())
        body = sum(p.numel() for p in self.bert.encoder.parameters())
        head = sum(p.numel() for p in self.cls.parameters())
        return {'word': word, 'pos': pos, 'token_type': tt, 'body': body, 'cls': head,
                'TOTAL': sum(p.numel() for p in self.parameters())}


def evaluate(model, loader, device):
    from sklearn.metrics import f1_score
    model.eval(); preds, gold = [], []
    with torch.no_grad():
        for ids, am, lab in loader:
            ids = ids.to(device); am = am.to(device); lab = lab.to(device)
            preds.append(model(ids, am).argmax(-1).cpu().numpy())
            gold.append(lab.cpu().numpy())
    preds = np.concatenate(preds); gold = np.concatenate(gold)
    acc = float((preds == gold).mean())
    f1 = float(f1_score(gold, preds, average='macro'))
    return acc, f1


def fine_tune(model, train_loader, val_loader, test_loader, device,
              n_epochs=4, lr=1e-4, label='model'):
    print(f"\n--- fine-tuning {label} ---")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_val = 0; best_state = None
    t0 = time.time()
    for ep in range(1, n_epochs+1):
        model.train()
        for ids, am, lab in train_loader:
            ids = ids.to(device); am = am.to(device); lab = lab.to(device)
            opt.zero_grad(); loss = crit(model(ids, am), lab)
            loss.backward(); opt.step()
        v_acc, v_f1 = evaluate(model, val_loader, device)
        print(f"  ep {ep}/{n_epochs}  val_acc={v_acc:.4f}  val_f1={v_f1:.4f}  ({time.time()-t0:.1f}s)")
        if v_acc > best_val:
            best_val = v_acc
            import copy; best_state = copy.deepcopy(model.state_dict())
    if best_state is not None: model.load_state_dict(best_state)
    t_acc, t_f1 = evaluate(model, test_loader, device)
    n_params = sum(p.numel() for p in model.parameters())
    return {'val_acc': best_val, 'test_acc': t_acc, 'test_f1': t_f1,
            'n_params': n_params, 'time_s': time.time() - t0}


# -------- driver --------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_dir',
                   default='./third_party/Geneformer-snapshot/Geneformer-V1-10M')
    p.add_argument('--out_dir', default='./results/geneformer_binary')
    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--num_cells', type=int, default=4000)
    p.add_argument('--max_len', type=int, default=256)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--n_epochs', type=int, default=4)
    p.add_argument('--distill_steps', type=int, default=2000)
    p.add_argument('--num_bits', type=int, default=64)
    p.add_argument('--interdim', type=int, default=1024)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load original Geneformer V1-10M (BertForMaskedLM)
    print(f"\n--- loading pretrained Geneformer at {args.ckpt_dir} ---")
    cfg = BertConfig.from_pretrained(args.ckpt_dir)
    cfg._attn_implementation = 'eager'
    full = BertForMaskedLM.from_pretrained(args.ckpt_dir, config=cfg)
    target = full.bert.embeddings.word_embeddings.weight.detach().clone()
    V, D = target.shape
    print(f"  word_embeddings: V={V}, d={D}, params={V*D:,}")
    print(f"  total: {fmt_n(sum(p.numel() for p in full.parameters()))}")

    # ---- Step 1: distill binary FFN ----
    print(f"\n--- distilling 64-bit FFN against Geneformer word_embeddings ---")
    bin_embed = BinaryGeneTokenEmbedding(V, D, num_bits=args.num_bits,
                                         interdim=args.interdim, dropout=0.0)
    dstats_bin = distill_embedding(target, bin_embed, n_steps=args.distill_steps,
                                   device=device)
    print(f"  init_mse={dstats_bin['init_mse']:.6f}  final={dstats_bin['final_mse']:.6f}  "
          f"({dstats_bin['reduction_x']:.1f}x)")

    # ---- Step 2: distill byte FFN ----
    print(f"\n--- distilling 8-byte FFN against Geneformer word_embeddings ---")
    byte_embed = ByteGeneTokenEmbedding(V, D, num_bytes=8, interdim=args.interdim,
                                        dropout=0.0)
    dstats_byte = distill_embedding(target, byte_embed, n_steps=args.distill_steps,
                                    device=device)
    print(f"  init_mse={dstats_byte['init_mse']:.6f}  final={dstats_byte['final_mse']:.6f}  "
          f"({dstats_byte['reduction_x']:.1f}x)")

    # ---- Step 3: synthetic dataset ----
    print(f"\n--- building synthetic gene-cell dataset ---")
    train, val, test = make_synthetic_geneformer_dataset(
        num_cells=args.num_cells, num_classes=args.num_classes,
        vocab_size=V, mean_len=args.max_len, seed=args.seed)
    train_ds = GeneSentenceDataset(*train, max_len=args.max_len)
    val_ds   = GeneSentenceDataset(*val,   max_len=args.max_len)
    test_ds  = GeneSentenceDataset(*test,  max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, collate_fn=collate)

    # ---- Step 4: build 3 variants and fine-tune ----
    results = {}

    # Subword (original): use full pretrained model
    print("\n=== Subword (original Geneformer V1-10M) ===")
    full_sub = BertForMaskedLM.from_pretrained(args.ckpt_dir, config=cfg)
    m_sub = GeneformerClassifier(full_sub, args.num_classes, hidden=D)
    print("  params:", {k: fmt_n(v) for k, v in m_sub.count_params().items()})
    res_sub = fine_tune(m_sub, train_loader, val_loader, test_loader, device,
                        n_epochs=args.n_epochs, lr=args.lr, label='Subword')
    results['Subword'] = res_sub
    del m_sub, full_sub; torch.cuda.empty_cache()

    # EmByte: byte FFN replacing word_embeddings
    print("\n=== EmByte (byte FFN, distilled) ===")
    full_byte = BertForMaskedLM.from_pretrained(args.ckpt_dir, config=cfg)
    full_byte.bert.embeddings.word_embeddings = byte_embed
    m_byte = GeneformerClassifier(full_byte, args.num_classes, hidden=D)
    print("  params:", {k: fmt_n(v) for k, v in m_byte.count_params().items()})
    res_byte = fine_tune(m_byte, train_loader, val_loader, test_loader, device,
                         n_epochs=args.n_epochs, lr=args.lr, label='EmByte')
    results['EmByte'] = res_byte
    del m_byte, full_byte; torch.cuda.empty_cache()

    # 001: binary FFN replacing word_embeddings
    print("\n=== 001 (binary FFN, distilled) ===")
    full_bin = BertForMaskedLM.from_pretrained(args.ckpt_dir, config=cfg)
    full_bin.bert.embeddings.word_embeddings = bin_embed
    m_bin = GeneformerClassifier(full_bin, args.num_classes, hidden=D)
    print("  params:", {k: fmt_n(v) for k, v in m_bin.count_params().items()})
    res_bin = fine_tune(m_bin, train_loader, val_loader, test_loader, device,
                        n_epochs=args.n_epochs, lr=args.lr, label='001')
    results['001'] = res_bin
    del m_bin, full_bin; torch.cuda.empty_cache()

    # ---- Save and report ----
    summary = {'results': results, 'distillation': {'binary': dstats_bin, 'byte': dstats_byte},
               'config': vars(args)}
    json.dump(summary, open(os.path.join(args.out_dir, 'summary.json'), 'w'),
              indent=2, default=str)

    print("\n" + "=" * 80)
    print(f"{'method':<10} {'params':>10} {'mb':>8} {'val_acc':>9} {'test_acc':>9} {'test_f1':>9} {'time_s':>8}")
    print('-' * 80)
    for name in ('Subword', 'EmByte', '001'):
        r = results[name]
        print(f"{name:<10} {fmt_n(r['n_params']):>10} {r['n_params']*4/1e6:>8.1f} "
              f"{r['val_acc']:>9.4f} {r['test_acc']:>9.4f} {r['test_f1']:>9.4f} "
              f"{r['time_s']:>8.1f}")


if __name__ == '__main__':
    main()
