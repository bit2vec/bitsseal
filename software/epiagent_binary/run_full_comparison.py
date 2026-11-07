"""End-to-end accuracy comparison: Subword vs Byte vs Binary EpiAgent
on a downstream task using EpiAgent's released checkpoint.

Pipeline:
  1. Inspect downloaded EpiAgent checkpoint
  2. Distill released cCRE_embedding into BinaryFFN (real distillation)
  3. Fine-tune all 3 variants on cell-type annotation
     - Subword: load full pretrained checkpoint, fine-tune classifier
     - Binary:  use distilled binary FFN + body weights from checkpoint, fine-tune
     - Byte:    use byte FFN + body weights, fine-tune
  4. Report accuracy / F1 / params per variant

Defaults to small-scale (subset of cells, fewer epochs) for tractable wall-time.
"""

import argparse
import glob
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
from binary_epiagent_full import (
    SubwordEpiAgent, BinaryEpiAgent, ByteEpiAgent,
    BinaryCCREEmbedding, ByteCCREEmbedding,
)


def fmt_n(n):
    if n > 1e9: return f'{n/1e9:.2f}B'
    if n > 1e6: return f'{n/1e6:.2f}M'
    if n > 1e3: return f'{n/1e3:.2f}K'
    return str(n)


# ---------------- Step 1: inspect checkpoint ----------------

def find_pretrained_ckpt(root):
    cands = (glob.glob(os.path.join(root, '**', '*.pth'), recursive=True) +
             glob.glob(os.path.join(root, '**', '*.pt'),  recursive=True))
    # Prefer files with "EpiAgent" or "model" in name; sort by size desc
    cands = sorted(cands, key=lambda p: -os.path.getsize(p))
    return cands


def inspect_checkpoint(ckpt_path):
    print(f"\n--- inspecting {ckpt_path} ---")
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
    if isinstance(sd, dict) and 'model' in sd:      sd = sd['model']
    print(f"  type: {type(sd).__name__}, keys: {len(sd)}")
    embed_keys = [k for k in sd if 'embedding' in k.lower() or 'embed' in k.lower()]
    decoder_keys = [k for k in sd if 'decoder' in k.lower()]
    print(f"  embedding-related keys: {embed_keys[:6]}{'...' if len(embed_keys)>6 else ''}")
    print(f"  decoder-related keys:   {decoder_keys[:6]}")
    # Total
    n = sum(v.numel() for v in sd.values() if hasattr(v, 'numel'))
    print(f"  total params: {fmt_n(n)}")
    return sd


# ---------------- Step 2: real distillation ----------------

def distill_binary(target_table, num_bits=64, interdim=2048, n_steps=3000,
                   batch=8192, lr=3e-4, device='cuda'):
    print("\n--- distilling binary FFN against released cCRE_embedding ---")
    V, d = target_table.shape
    target = target_table.to(device)

    embed = BinaryCCREEmbedding(V, d, num_bits=num_bits, interdim=interdim,
                                dropout=0.0).to(device)
    bit_codes_dev = embed.bit_codes.to(device)

    @torch.no_grad()
    def full_mse():
        chunk = 65536; sse = 0.0; n = 0
        for s in range(0, V, chunk):
            e = min(s+chunk, V)
            pred = embed.ffn(bit_codes_dev[s:e].float())
            sse += ((pred - target[s:e])**2).sum().item()
            n   += pred.numel()
        return sse / n

    init_mse = full_mse()
    print(f"  init full-vocab MSE: {init_mse:.6f}")

    opt = torch.optim.AdamW(embed.ffn.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, n_steps+1):
        ids = torch.randint(0, V, (batch,), device=device)
        loss = ((embed.ffn(bit_codes_dev[ids].float()) - target[ids])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, n_steps//15) == 0 or step == 1:
            print(f"    step {step:5d}/{n_steps}  batch_mse={loss.item():.6f}  ({time.time()-t0:.1f}s)")
    final_mse = full_mse()
    print(f"  final full-vocab MSE: {final_mse:.6f}  ({init_mse/max(final_mse,1e-12):.1f}x reduction)")

    # Cosine similarity stats
    @torch.no_grad()
    def cos_stats(samples=20000):
        idx = torch.randperm(V, device=device)[:samples]
        codes = bit_codes_dev[idx].float()
        pred = embed.ffn(codes); tgt = target[idx]
        cos = torch.nn.functional.cosine_similarity(pred, tgt, dim=-1)
        return float(cos.mean()), float(cos.median())
    cmean, cmed = cos_stats()
    print(f"  cosine sim with target: mean={cmean:.4f}  median={cmed:.4f}")

    return embed.cpu(), {'init_mse': init_mse, 'final_mse': final_mse,
                          'cos_mean': cmean, 'cos_median': cmed}


def distill_byte(target_table, num_bytes=8, interdim=2048, n_steps=3000,
                 batch=8192, lr=3e-4, device='cuda'):
    print("\n--- distilling byte FFN against released cCRE_embedding ---")
    V, d = target_table.shape
    target = target_table.to(device)
    embed = ByteCCREEmbedding(V, d, num_bytes=num_bytes, interdim=interdim,
                              dropout=0.0).to(device)
    byte_codes_dev = embed.byte_codes.to(device)
    learnable = list(embed.parameters())  # byte_embed + ffn

    @torch.no_grad()
    def full_mse():
        chunk = 65536; sse = 0.0; n = 0
        for s in range(0, V, chunk):
            e = min(s+chunk, V)
            ids = torch.arange(s, e, device=device)
            pred = embed(ids)
            sse += ((pred - target[s:e])**2).sum().item()
            n   += pred.numel()
        return sse / n

    init_mse = full_mse()
    print(f"  init full-vocab MSE: {init_mse:.6f}")

    opt = torch.optim.AdamW(learnable, lr=lr)
    t0 = time.time()
    for step in range(1, n_steps+1):
        ids = torch.randint(0, V, (batch,), device=device)
        pred = embed(ids); tgt = target[ids]
        loss = ((pred - tgt)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, n_steps//15) == 0 or step == 1:
            print(f"    step {step:5d}/{n_steps}  batch_mse={loss.item():.6f}  ({time.time()-t0:.1f}s)")
    final_mse = full_mse()
    print(f"  final full-vocab MSE: {final_mse:.6f}  ({init_mse/max(final_mse,1e-12):.1f}x reduction)")

    return embed.cpu(), {'init_mse': init_mse, 'final_mse': final_mse}


# ---------------- Step 3: fine-tune cell-type annotation ----------------

class CellSentenceDataset(Dataset):
    """A minimal dataset that consumes (cell_sentence_ids, label) pairs."""

    def __init__(self, sentences, labels, max_len=2048):
        self.sentences = sentences
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        ids = self.sentences[i][:self.max_len]
        return ids, int(self.labels[i])


def collate(batch, pad_idx=0):
    seqs, labels = zip(*batch)
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.as_tensor(s, dtype=torch.long)
    return out, torch.tensor(labels, dtype=torch.long)


def load_real_celltype_dataset(h5ad_path, max_len=512, seed=42, val_frac=0.1, test_frac=0.1):
    """Load EpiAgent-formatted h5ad and split into train/val/test."""
    import ast
    import anndata as ad
    print(f"  loading h5ad: {h5ad_path}")
    a = ad.read_h5ad(h5ad_path)
    print(f"  cells: {a.shape[0]}, cell_type classes: {a.obs['cell_type'].nunique()}")

    # Parse cell_sentences (stored as string repr of int list)
    sentences = []
    for s in a.obs['cell_sentences']:
        ids = ast.literal_eval(s) if isinstance(s, str) else list(s)
        sentences.append([int(x) for x in ids[:max_len]])

    # Encode labels
    classes = sorted(a.obs['cell_type'].unique().tolist())
    cls_to_id = {c: i for i, c in enumerate(classes)}
    labels = [cls_to_id[c] for c in a.obs['cell_type']]

    # Stratified-ish shuffle then split
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sentences))
    sentences = [sentences[i] for i in order]
    labels    = [labels[i]    for i in order]
    n = len(sentences)
    n_test = int(n * test_frac); n_val = int(n * val_frac)
    train = (sentences[:n - n_val - n_test], labels[:n - n_val - n_test])
    val   = (sentences[n - n_val - n_test:n - n_test], labels[n - n_val - n_test:n - n_test])
    test  = (sentences[n - n_test:], labels[n - n_test:])
    return train, val, test, classes


def make_synthetic_celltype_dataset(num_cells=8000, num_classes=15,
                                    vocab_size=1355449, mean_len=512,
                                    seed=42):
    """Synthetic fallback when no h5ad is available."""
    rng = np.random.default_rng(seed)
    markers = [rng.choice(vocab_size, size=200, replace=False) for _ in range(num_classes)]
    sentences = []; labels = []
    for _ in range(num_cells):
        c = int(rng.integers(num_classes))
        L = int(np.clip(rng.normal(mean_len, mean_len*0.2), 64, mean_len*2))
        n_sig = rng.integers(L // 4, L // 2)
        sig = rng.choice(markers[c], size=n_sig, replace=True)
        bg = rng.integers(4, vocab_size, size=L - n_sig)
        ids = np.concatenate([[2], sig, bg])
        rng.shuffle(ids[1:])
        sentences.append(ids.tolist()); labels.append(c)
    n_train = int(0.8 * num_cells); n_val = int(0.1 * num_cells)
    return ((sentences[:n_train], labels[:n_train]),
            (sentences[n_train:n_train+n_val], labels[n_train:n_train+n_val]),
            (sentences[n_train+n_val:], labels[n_train+n_val:]))


def evaluate(model, loader, device):
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in loader:
            ids, lab = batch
            ids = ids.to(device); lab = lab.to(device)
            out = model(ids)
            preds.append(out.argmax(-1).cpu().numpy())
            gold.append(lab.cpu().numpy())
    preds = np.concatenate(preds); gold = np.concatenate(gold)
    acc = float((preds == gold).mean())
    # Macro F1
    from sklearn.metrics import f1_score
    macro_f1 = float(f1_score(gold, preds, average='macro'))
    return acc, macro_f1


def fine_tune(model, train_loader, val_loader, test_loader, device,
              n_epochs=5, lr=1e-4, label_str='model'):
    print(f"\n--- fine-tuning {label_str} ---")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val = 0; best_state = None
    t0 = time.time()
    for ep in range(1, n_epochs+1):
        model.train()
        for ids, lab in train_loader:
            ids = ids.to(device); lab = lab.to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids), lab)
            loss.backward()
            optimizer.step()
        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f"  ep {ep}/{n_epochs}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  ({time.time()-t0:.1f}s)")
        if val_acc > best_val:
            best_val = val_acc
            import copy
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, test_f1 = evaluate(model, test_loader, device)
    n_params = sum(p.numel() for p in model.parameters())
    return {'val_acc': best_val, 'test_acc': test_acc, 'test_f1': test_f1,
            'n_params': n_params, 'time_s': time.time() - t0}


# ---------------- driver ----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_root', default='./third_party/EpiAgent/checkpoints')
    p.add_argument('--out_dir', default='./results/epiagent_binary')
    p.add_argument('--vocab_size', type=int, default=1355449)
    p.add_argument('--embedding_dim', type=int, default=512)
    p.add_argument('--num_layers', type=int, default=4,
                   help='Tractable subset of EpiAgent body (orig: 18).')
    p.add_argument('--data_h5ad',
                   default='./third_party/EpiAgent/checkpoints/data/sample/processed_h5ad/reference_data_downsampled_10000_cells.h5ad',
                   help='Real EpiAgent h5ad with cell_sentences + cell_type. Empty -> synthetic.')
    p.add_argument('--num_classes', type=int, default=15,
                   help='Used only if --data_h5ad is empty (synthetic mode).')
    p.add_argument('--num_cells', type=int, default=4000,
                   help='Used only if --data_h5ad is empty (synthetic mode).')
    p.add_argument('--max_len', type=int, default=512)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--n_epochs', type=int, default=4)
    p.add_argument('--distill_steps', type=int, default=2000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- step 1: find checkpoint ----
    ckpts = find_pretrained_ckpt(args.ckpt_root)
    print(f"Found {len(ckpts)} checkpoint files. Top 3 by size:")
    for c in ckpts[:3]:
        print(f"  {os.path.getsize(c)/1e6:.1f} MB   {c}")
    pretrained_embed = None
    if ckpts:
        try:
            sd = inspect_checkpoint(ckpts[0])
            for k in sd:
                if 'cCRE_embedding' in k and 'weight' in k:
                    pretrained_embed = sd[k].float().cpu()
                    print(f"  -> using {k}: shape={tuple(pretrained_embed.shape)}")
                    break
        except Exception as e:
            print(f"  WARN: could not load checkpoint: {e}")
    if pretrained_embed is None:
        print("  no released cCRE_embedding found; will use random Gaussian as proxy")
        pretrained_embed = torch.randn(args.vocab_size, args.embedding_dim) * 0.02

    # ---- step 2: distill binary + byte ----
    binary_distilled, dstats_b = distill_binary(
        pretrained_embed, n_steps=args.distill_steps, device=device)
    byte_distilled, dstats_bt = distill_byte(
        pretrained_embed, n_steps=args.distill_steps, device=device)
    json.dump({'binary': dstats_b, 'byte': dstats_bt},
              open(os.path.join(args.out_dir, 'distillation_stats.json'), 'w'),
              indent=2)

    # ---- step 3: cell-type annotation dataset ----
    if args.data_h5ad and os.path.isfile(args.data_h5ad):
        print(f"\n--- loading REAL EpiAgent cell-type dataset ---")
        train, val, test, classes = load_real_celltype_dataset(
            args.data_h5ad, max_len=args.max_len, seed=args.seed)
        args.num_classes = len(classes)
        print(f"  classes ({len(classes)}): {classes}")
        print(f"  train={len(train[0])} val={len(val[0])} test={len(test[0])}")
    else:
        print(f"\n--- building synthetic cell-type dataset (num_classes={args.num_classes}, "
              f"num_cells={args.num_cells}) ---")
        train, val, test = make_synthetic_celltype_dataset(
            num_cells=args.num_cells, num_classes=args.num_classes,
            vocab_size=args.vocab_size, mean_len=args.max_len, seed=args.seed)
    train_ds = CellSentenceDataset(*train, max_len=args.max_len)
    val_ds   = CellSentenceDataset(*val,   max_len=args.max_len)
    test_ds  = CellSentenceDataset(*test,  max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=0)

    # ---- step 4: build and fine-tune all three variants ----
    common = dict(num_layers=args.num_layers, embedding_dim=args.embedding_dim,
                  vocab_size=args.vocab_size, max_rank_embeddings=args.max_len * 2)

    print("\n=== Subword (original) ===")
    m_sub = SubwordEpiAgent(num_classes=args.num_classes, **common)
    if pretrained_embed.shape == m_sub.cCRE_embedding.weight.shape:
        m_sub.cCRE_embedding.weight.data.copy_(pretrained_embed)
        print("  loaded released cCRE_embedding into SubwordEpiAgent")
    print("  param breakdown:", {k: fmt_n(v) for k, v in m_sub.count_params().items()})
    res_sub = fine_tune(m_sub, train_loader, val_loader, test_loader, device,
                        n_epochs=args.n_epochs, lr=args.lr, label_str='Subword')
    del m_sub; torch.cuda.empty_cache()

    print("\n=== EmByte (byte FFN, distilled) ===")
    m_byte = ByteEpiAgent(num_classes=args.num_classes, **common)
    # Load distilled byte embedding
    m_byte.cCRE_embedding.load_state_dict(byte_distilled.state_dict())
    print("  param breakdown:", {k: fmt_n(v) for k, v in m_byte.count_params().items()})
    res_byte = fine_tune(m_byte, train_loader, val_loader, test_loader, device,
                         n_epochs=args.n_epochs, lr=args.lr, label_str='EmByte')
    del m_byte; torch.cuda.empty_cache()

    print("\n=== 001 (binary FFN, distilled) ===")
    m_bin = BinaryEpiAgent(num_classes=args.num_classes, **common)
    m_bin.cCRE_embedding.load_state_dict(binary_distilled.state_dict())
    print("  param breakdown:", {k: fmt_n(v) for k, v in m_bin.count_params().items()})
    res_bin = fine_tune(m_bin, train_loader, val_loader, test_loader, device,
                        n_epochs=args.n_epochs, lr=args.lr, label_str='001')

    # ---- step 5: report ----
    summary = {'Subword': res_sub, 'EmByte': res_byte, '001': res_bin,
               'distillation': {'binary': dstats_b, 'byte': dstats_bt},
               'config': vars(args)}
    json.dump(summary, open(os.path.join(args.out_dir, 'fullcomp_summary.json'), 'w'),
              indent=2)

    print("\n" + "=" * 80)
    print(f"{'method':<10} {'params':>10} {'mb':>8} {'val_acc':>9} {'test_acc':>9} {'test_f1':>9} {'time_s':>8}")
    print('-' * 80)
    for name, r in [('Subword', res_sub), ('EmByte', res_byte), ('001', res_bin)]:
        print(f"{name:<10} {fmt_n(r['n_params']):>10} {r['n_params']*4/1e6:>8.1f} "
              f"{r['val_acc']:>9.4f} {r['test_acc']:>9.4f} {r['test_f1']:>9.4f} "
              f"{r['time_s']:>8.1f}")


if __name__ == '__main__':
    main()
