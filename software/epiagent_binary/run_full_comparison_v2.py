"""V2: EpiAgent comparison with PRETRAINED body (not random).

Key change vs V1: load body weights from pretrained_EpiAgent.pth into a
custom EpiAgentBody (pre-norm Bert mirror, no flash-attn dependency). All
three variants (Subword / EmByte / 001) use the SAME pretrained body, the
SAME pretrained rank_embedding. They differ only in the cCRE_embedding:
  - Subword: pretrained cCRE_embedding (V x d table)
  - EmByte:  byte FFN distilled to match pretrained cCRE_embedding
  - 001:     binary FFN (interdim=4096) distilled to match pretrained
              cCRE_embedding
"""

import argparse
import ast
import copy
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
from binary_epiagent_full import BinaryCCREEmbedding, ByteCCREEmbedding
from epiagent_pretrained_body import (
    EpiAgentBody, load_pretrained_body, load_pretrained_embeddings,
)


def fmt_n(n):
    if n > 1e9: return f'{n/1e9:.2f}B'
    if n > 1e6: return f'{n/1e6:.2f}M'
    if n > 1e3: return f'{n/1e3:.2f}K'
    return str(n)


# -------- supervised model --------
class _ClassifierHead(nn.Module):
    def __init__(self, embedding_dim, num_classes, dropout=0.25):
        super().__init__()
        self.gelu = nn.GELU(); self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_dim, 256); self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128); self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.fc1(x); x = self.ln1(x); x = self.dropout(x)
        x = self.gelu(self.fc2(x)); x = self.ln2(x)
        return self.fc3(x)


class EpiAgentClassifier(nn.Module):
    """Pretrained body + chosen embedding + classifier head."""

    def __init__(self, ccre_embed: nn.Module, rank_embed: nn.Embedding,
                 body: EpiAgentBody, num_classes: int, embedding_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.cCRE_embedding = ccre_embed
        self.rank_embedding = rank_embed
        self.body = body
        self.classifier = _ClassifierHead(embedding_dim, num_classes, dropout=0.25)

    def forward(self, input_ids):
        x = self.cCRE_embedding(input_ids)                  # [B, T, d]
        T = input_ids.size(1)
        positions = torch.arange(T, device=input_ids.device)
        x = x + self.rank_embedding(positions)[None]
        h = self.body(x, key_padding_mask=(input_ids != 0))
        return self.classifier(h[:, 0, :])

    def count_params(self):
        c = sum(p.numel() for p in self.cCRE_embedding.parameters())
        r = sum(p.numel() for p in self.rank_embedding.parameters())
        b = sum(p.numel() for p in self.body.parameters())
        cls = sum(p.numel() for p in self.classifier.parameters())
        return {'cCRE_embedding': c, 'rank_embedding': r, 'body': b,
                'classifier': cls,
                'TOTAL': sum(p.numel() for p in self.parameters())}


# -------- distillation --------
def distill_module(target, embed_module, n_steps=2000, batch=8192, lr=3e-4,
                   device='cuda'):
    V = target.size(0); target = target.to(device)
    embed_module = embed_module.to(device)
    learnable = [p for p in embed_module.parameters() if p.requires_grad]

    @torch.no_grad()
    def full_mse():
        chunk = 65536; sse = 0; n = 0
        for s in range(0, V, chunk):
            e = min(s+chunk, V)
            ids = torch.arange(s, e, device=device)
            pred = embed_module(ids)
            sse += ((pred - target[s:e])**2).sum().item()
            n   += pred.numel()
        return sse / n

    init = full_mse()
    print(f"  init full-vocab MSE: {init:.6f}")
    opt = torch.optim.AdamW(learnable, lr=lr)
    t0 = time.time()
    for step in range(1, n_steps+1):
        ids = torch.randint(0, V, (batch,), device=device)
        loss = ((embed_module(ids) - target[ids])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, n_steps//10) == 0 or step == 1:
            print(f"    step {step:5d}/{n_steps}  batch_mse={loss.item():.6f}  ({time.time()-t0:.1f}s)")
    final = full_mse()
    print(f"  final full-vocab MSE: {final:.6f}  ({init/max(final,1e-12):.1f}x)")
    return {'init_mse': init, 'final_mse': final}


# -------- data loader --------
def load_real_celltype(h5ad_path, max_len=512, seed=42, val_frac=0.1, test_frac=0.1):
    import anndata as ad
    print(f"  loading h5ad: {h5ad_path}")
    a = ad.read_h5ad(h5ad_path)
    sentences = []
    for s in a.obs['cell_sentences']:
        ids = ast.literal_eval(s) if isinstance(s, str) else list(s)
        sentences.append([int(x) for x in ids[:max_len]])
    classes = sorted(a.obs['cell_type'].unique().tolist())
    cls_to_id = {c: i for i, c in enumerate(classes)}
    labels = [cls_to_id[c] for c in a.obs['cell_type']]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sentences))
    sentences = [sentences[i] for i in order]
    labels = [labels[i] for i in order]
    n = len(sentences)
    n_test = int(n * test_frac); n_val = int(n * val_frac)
    return ((sentences[:n - n_val - n_test], labels[:n - n_val - n_test]),
            (sentences[n - n_val - n_test:n - n_test],
             labels[n - n_val - n_test:n - n_test]),
            (sentences[n - n_test:], labels[n - n_test:]),
            classes)


class CellSentenceDataset(Dataset):
    def __init__(self, ids, labels, max_len): self.ids = ids; self.labels = labels; self.max_len = max_len
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        return self.ids[i][:self.max_len], int(self.labels[i])


def collate(batch, pad=0):
    s, lab = zip(*batch)
    L = max(len(x) for x in s)
    out = torch.full((len(s), L), pad, dtype=torch.long)
    for i, x in enumerate(s):
        out[i, :len(x)] = torch.as_tensor(x, dtype=torch.long)
    return out, torch.tensor(lab, dtype=torch.long)


def evaluate(model, loader, device):
    from sklearn.metrics import f1_score
    model.eval(); preds, gold = [], []
    with torch.no_grad():
        for ids, lab in loader:
            ids = ids.to(device); lab = lab.to(device)
            preds.append(model(ids).argmax(-1).cpu().numpy())
            gold.append(lab.cpu().numpy())
    p = np.concatenate(preds); g = np.concatenate(gold)
    acc = float((p == g).mean())
    f1 = float(f1_score(g, p, average='macro'))
    return acc, f1


def fine_tune(model, train_loader, val_loader, test_loader, device,
              n_epochs=4, lr=1e-4, label='m'):
    print(f"\n--- fine-tuning {label} ---")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_v = 0; best_st = None; t0 = time.time()
    for ep in range(1, n_epochs+1):
        model.train()
        for ids, lab in train_loader:
            ids = ids.to(device); lab = lab.to(device)
            opt.zero_grad(); loss = crit(model(ids), lab); loss.backward(); opt.step()
        v_acc, v_f1 = evaluate(model, val_loader, device)
        print(f"  ep {ep}/{n_epochs}  val_acc={v_acc:.4f}  val_f1={v_f1:.4f}  ({time.time()-t0:.1f}s)")
        if v_acc > best_v:
            best_v = v_acc; best_st = copy.deepcopy(model.state_dict())
    if best_st is not None: model.load_state_dict(best_st)
    t_acc, t_f1 = evaluate(model, test_loader, device)
    return {'val_acc': best_v, 'test_acc': t_acc, 'test_f1': t_f1,
            'n_params': sum(p.numel() for p in model.parameters()),
            'time_s': time.time() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_path',
                   default='./third_party/EpiAgent/checkpoints/model/pretrained_EpiAgent.pth')
    p.add_argument('--data_h5ad',
                   default='./third_party/EpiAgent/checkpoints/data/sample/processed_h5ad/reference_data_downsampled_10000_cells.h5ad')
    p.add_argument('--out_dir', default='./results/epiagent_binary_v2')
    p.add_argument('--vocab_size', type=int, default=1355449)
    p.add_argument('--embedding_dim', type=int, default=512)
    p.add_argument('--num_layers', type=int, default=18)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--max_len', type=int, default=512)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--n_epochs', type=int, default=4)
    p.add_argument('--distill_steps', type=int, default=2000)
    p.add_argument('--num_bits', type=int, default=64)
    p.add_argument('--interdim', type=int, default=4096)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load pretrained embeddings + body weights
    print(f"\n--- loading pretrained EpiAgent ---")
    embeds = load_pretrained_embeddings(args.ckpt_path)
    pretrained_ccre = embeds['cCRE_embedding']
    pretrained_rank = embeds['rank_embedding']
    print(f"  pretrained cCRE_embedding: {tuple(pretrained_ccre.shape)}")
    print(f"  pretrained rank_embedding: {tuple(pretrained_rank.shape)}")

    body_template = EpiAgentBody(hidden_size=args.embedding_dim,
                                  num_layers=args.num_layers,
                                  num_heads=args.num_heads,
                                  intermediate_size=4 * args.embedding_dim)
    res = load_pretrained_body(body_template, args.ckpt_path)
    print(f"  body params loaded: {res['pretrained_body_total']:,}")

    # Distillation targets
    print(f"\n--- distilling 64-bit FFN against pretrained cCRE_embedding ---")
    bin_embed = BinaryCCREEmbedding(args.vocab_size, args.embedding_dim,
                                    num_bits=args.num_bits, interdim=args.interdim,
                                    dropout=0.0)
    dstats_b = distill_module(pretrained_ccre, bin_embed, n_steps=args.distill_steps,
                              device=device)
    bin_embed = bin_embed.cpu()

    print(f"\n--- distilling 8-byte FFN against pretrained cCRE_embedding ---")
    byte_embed = ByteCCREEmbedding(args.vocab_size, args.embedding_dim,
                                   num_bytes=8, interdim=args.interdim, dropout=0.0)
    dstats_bt = distill_module(pretrained_ccre, byte_embed, n_steps=args.distill_steps,
                               device=device)
    byte_embed = byte_embed.cpu()

    # Data
    print(f"\n--- loading real EpiAgent cell-type dataset ---")
    train, val, test, classes = load_real_celltype(args.data_h5ad,
                                                    max_len=args.max_len,
                                                    seed=args.seed)
    print(f"  classes ({len(classes)}): {classes}")
    print(f"  train={len(train[0])} val={len(val[0])} test={len(test[0])}")
    train_ds = CellSentenceDataset(*train, max_len=args.max_len)
    val_ds   = CellSentenceDataset(*val,   max_len=args.max_len)
    test_ds  = CellSentenceDataset(*test,  max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, collate_fn=collate)

    # Helper: build a fresh body + rank with pretrained weights for each variant
    def build_pretrained_body_and_rank():
        body = EpiAgentBody(hidden_size=args.embedding_dim,
                             num_layers=args.num_layers, num_heads=args.num_heads,
                             intermediate_size=4 * args.embedding_dim)
        load_pretrained_body(body, args.ckpt_path)
        rank = nn.Embedding(pretrained_rank.size(0), args.embedding_dim)
        rank.weight.data.copy_(pretrained_rank)
        return body, rank

    results = {}

    print("\n=== Subword (pretrained cCRE_embedding) ===")
    sub_emb = nn.Embedding(args.vocab_size, args.embedding_dim)
    sub_emb.weight.data.copy_(pretrained_ccre)
    body, rank = build_pretrained_body_and_rank()
    m_sub = EpiAgentClassifier(sub_emb, rank, body, num_classes=len(classes))
    print("  params:", {k: fmt_n(v) for k, v in m_sub.count_params().items()})
    res_sub = fine_tune(m_sub, train_loader, val_loader, test_loader, device,
                        n_epochs=args.n_epochs, lr=args.lr, label='Subword')
    results['Subword'] = res_sub
    del m_sub, sub_emb, body, rank; torch.cuda.empty_cache()

    print("\n=== EmByte (byte FFN, distilled) ===")
    body, rank = build_pretrained_body_and_rank()
    m_byte = EpiAgentClassifier(byte_embed, rank, body, num_classes=len(classes))
    print("  params:", {k: fmt_n(v) for k, v in m_byte.count_params().items()})
    res_byte = fine_tune(m_byte, train_loader, val_loader, test_loader, device,
                         n_epochs=args.n_epochs, lr=args.lr, label='EmByte')
    results['EmByte'] = res_byte
    del m_byte, body, rank; torch.cuda.empty_cache()

    print("\n=== 001 (binary FFN, distilled, interdim=4096) ===")
    body, rank = build_pretrained_body_and_rank()
    m_bin = EpiAgentClassifier(bin_embed, rank, body, num_classes=len(classes))
    print("  params:", {k: fmt_n(v) for k, v in m_bin.count_params().items()})
    res_bin = fine_tune(m_bin, train_loader, val_loader, test_loader, device,
                        n_epochs=args.n_epochs, lr=args.lr, label='001')
    results['001'] = res_bin
    del m_bin, body, rank; torch.cuda.empty_cache()

    summary = {'results': results, 'distillation': {'binary': dstats_b, 'byte': dstats_bt},
               'config': vars(args)}
    json.dump(summary, open(os.path.join(args.out_dir, 'summary.json'), 'w'),
              indent=2, default=str)

    print("\n" + "=" * 90)
    print(f"{'method':<10} {'params':>10} {'mb':>8} {'val_acc':>9} {'test_acc':>9} {'test_f1':>9} {'time_s':>8}")
    print('-' * 90)
    for name in ('Subword', 'EmByte', '001'):
        r = results[name]
        print(f"{name:<10} {fmt_n(r['n_params']):>10} {r['n_params']*4/1e6:>8.1f} "
              f"{r['val_acc']:>9.4f} {r['test_acc']:>9.4f} {r['test_f1']:>9.4f} "
              f"{r['time_s']:>8.1f}")


if __name__ == '__main__':
    main()
