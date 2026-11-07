"""scGPT 3-method comparison: replace gene encoder embedding with Binary / Byte FFN.

Loads scGPT-kidney's `encoder.embedding.weight` (60697 x 512), which is the
gene-token embedding equivalent of Geneformer's word_embeddings, distills it
into BinaryGeneTokenEmbedding / ByteGeneTokenEmbedding, then runs synthetic
gene-rank classification on the same 10-class signature task used in Geneformer.

scGPT's body is more complex (value_encoder + transformer body), but for the
embedding-compression claim only the gene encoder matters. We use a simple
HF Bert encoder body initialised randomly to provide a comparable transformer
backbone for all three variants, focusing the comparison on the embedding's
ability to distinguish 60k unique gene IDs.
"""

import argparse
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
from binary_geneformer import BinaryGeneTokenEmbedding, ByteGeneTokenEmbedding


def fmt_n(n):
    if n > 1e9: return f'{n/1e9:.2f}B'
    if n > 1e6: return f'{n/1e6:.2f}M'
    if n > 1e3: return f'{n/1e3:.2f}K'
    return str(n)


# ---- model wrapper ----

class GeneClassifier(nn.Module):
    def __init__(self, gene_embed: nn.Module, embedding_dim, num_classes,
                 num_layers=4, num_heads=8):
        super().__init__()
        self.gene_embed = gene_embed
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=4 * embedding_dim, dropout=0.1,
            batch_first=True, norm_first=False, activation='gelu')
        self.body = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls = nn.Linear(embedding_dim, num_classes)

    def forward(self, ids):
        x = self.gene_embed(ids)
        h = self.body(x, src_key_padding_mask=(ids == 0))
        return self.cls(h[:, 0, :])

    def count_params(self):
        emb = sum(p.numel() for p in self.gene_embed.parameters())
        body = sum(p.numel() for p in self.body.parameters())
        cls = sum(p.numel() for p in self.cls.parameters())
        return {'embed': emb, 'body': body, 'cls': cls,
                'TOTAL': sum(p.numel() for p in self.parameters())}


# ---- distillation ----
def distill_module(target, embed_module, n_steps=2000, batch=8192, lr=3e-4, device='cuda'):
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
    print(f"  init MSE: {init:.6f}")
    opt = torch.optim.AdamW(learnable, lr=lr)
    t0 = time.time()
    for step in range(1, n_steps+1):
        ids = torch.randint(0, V, (batch,), device=device)
        loss = ((embed_module(ids) - target[ids])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, n_steps//10) == 0 or step == 1:
            print(f"    step {step:5d}/{n_steps}  batch_mse={loss.item():.6f}  ({time.time()-t0:.1f}s)")
    final = full_mse()
    print(f"  final MSE: {final:.6f}  ({init/max(final,1e-12):.1f}x)")
    return embed_module.cpu(), {'init_mse': init, 'final_mse': final}


# ---- synthetic data ----
def make_synthetic_dataset(num_cells=4000, num_classes=10, vocab_size=60697,
                            mean_len=512, seed=42):
    rng = np.random.default_rng(seed)
    markers = [rng.choice(np.arange(4, vocab_size), size=80, replace=False)
               for _ in range(num_classes)]
    cells, labels = [], []
    for _ in range(num_cells):
        c = int(rng.integers(num_classes))
        L = int(np.clip(rng.normal(mean_len, mean_len*0.2), 64, mean_len*2))
        n_sig = rng.integers(L//4, L//2)
        sig = rng.choice(markers[c], size=n_sig, replace=True)
        bg = rng.integers(4, vocab_size, size=L - n_sig)
        ids = np.concatenate([[2], sig, bg]); rng.shuffle(ids[1:])
        cells.append(ids.tolist()); labels.append(c)
    n_train = int(0.8 * num_cells); n_val = int(0.1 * num_cells)
    return ((cells[:n_train], labels[:n_train]),
            (cells[n_train:n_train+n_val], labels[n_train:n_train+n_val]),
            (cells[n_train+n_val:], labels[n_train+n_val:]))


class CellDataset(Dataset):
    def __init__(self, cells, labels, max_len): self.cells=cells; self.labels=labels; self.max_len=max_len
    def __len__(self): return len(self.cells)
    def __getitem__(self, i): return self.cells[i][:self.max_len], int(self.labels[i])


def collate(batch, pad=0):
    s, lab = zip(*batch); L = max(len(x) for x in s)
    out = torch.full((len(s), L), pad, dtype=torch.long)
    for i, x in enumerate(s): out[i, :len(x)] = torch.as_tensor(x, dtype=torch.long)
    return out, torch.tensor(lab, dtype=torch.long)


def evaluate(model, loader, device):
    from sklearn.metrics import f1_score
    model.eval(); preds, gold = [], []
    with torch.no_grad():
        for ids, lab in loader:
            ids=ids.to(device); lab=lab.to(device)
            preds.append(model(ids).argmax(-1).cpu().numpy()); gold.append(lab.cpu().numpy())
    p=np.concatenate(preds); g=np.concatenate(gold)
    return float((p==g).mean()), float(f1_score(g, p, average='macro'))


def fine_tune(model, tr, va, te, device, n_epochs=4, lr=1e-4, label='m'):
    print(f"\n--- fine-tuning {label} ---")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr); crit = nn.CrossEntropyLoss()
    best=0; best_st=None; t0=time.time()
    for ep in range(1, n_epochs+1):
        model.train()
        for ids, lab in tr:
            ids=ids.to(device); lab=lab.to(device)
            opt.zero_grad(); loss=crit(model(ids), lab); loss.backward(); opt.step()
        v, vf = evaluate(model, va, device)
        print(f"  ep {ep}/{n_epochs}  val_acc={v:.4f}  val_f1={vf:.4f}  ({time.time()-t0:.1f}s)")
        if v > best: best=v; best_st=copy.deepcopy(model.state_dict())
    if best_st is not None: model.load_state_dict(best_st)
    t_acc, t_f1 = evaluate(model, te, device)
    return {'val_acc': best, 'test_acc': t_acc, 'test_f1': t_f1,
            'n_params': sum(p.numel() for p in model.parameters()),
            'time_s': time.time() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_path',
                   default='./third_party/scGPT-checkpoints/kidney/best_model.pt')
    p.add_argument('--out_dir', default='./results/scgpt_binary')
    p.add_argument('--vocab_size', type=int, default=60697)
    p.add_argument('--embedding_dim', type=int, default=512)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--num_cells', type=int, default=4000)
    p.add_argument('--max_len', type=int, default=256)
    p.add_argument('--batch', type=int, default=16)
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

    # Load pretrained gene-encoder embedding
    print(f"\n--- loading scGPT-kidney gene encoder embedding ---")
    sd = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
    target = sd['encoder.embedding.weight'].float()
    print(f"  encoder.embedding.weight: {tuple(target.shape)}")
    V, D = target.shape

    # Distill binary
    print("\n--- distilling 64-bit FFN ---")
    bin_e = BinaryGeneTokenEmbedding(V, D, num_bits=args.num_bits, interdim=args.interdim, dropout=0.0)
    bin_e, dstats_b = distill_module(target, bin_e, n_steps=args.distill_steps, device=device)

    print("\n--- distilling 8-byte FFN ---")
    byte_e = ByteGeneTokenEmbedding(V, D, num_bytes=8, interdim=args.interdim, dropout=0.0)
    byte_e, dstats_bt = distill_module(target, byte_e, n_steps=args.distill_steps, device=device)

    # Synthetic dataset
    print("\n--- building synthetic gene-cell dataset ---")
    train, val, test = make_synthetic_dataset(num_cells=args.num_cells, num_classes=args.num_classes,
                                                vocab_size=V, mean_len=args.max_len, seed=args.seed)
    train_ds = CellDataset(*train, max_len=args.max_len)
    val_ds   = CellDataset(*val,   max_len=args.max_len)
    test_ds  = CellDataset(*test,  max_len=args.max_len)
    tr = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  collate_fn=collate)
    va = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, collate_fn=collate)
    te = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, collate_fn=collate)

    results = {}

    # Subword
    print("\n=== Subword (pretrained scGPT gene_embedding) ===")
    sub_e = nn.Embedding(V, D)
    sub_e.weight.data.copy_(target)
    m_sub = GeneClassifier(sub_e, D, args.num_classes, num_layers=args.num_layers)
    print("  params:", {k: fmt_n(v) for k, v in m_sub.count_params().items()})
    res_sub = fine_tune(m_sub, tr, va, te, device, n_epochs=args.n_epochs, lr=args.lr, label='Subword')
    results['Subword'] = res_sub
    del m_sub, sub_e; torch.cuda.empty_cache()

    # EmByte
    print("\n=== EmByte (8-byte FFN, distilled) ===")
    m_byte = GeneClassifier(byte_e, D, args.num_classes, num_layers=args.num_layers)
    print("  params:", {k: fmt_n(v) for k, v in m_byte.count_params().items()})
    res_byte = fine_tune(m_byte, tr, va, te, device, n_epochs=args.n_epochs, lr=args.lr, label='EmByte')
    results['EmByte'] = res_byte
    del m_byte; torch.cuda.empty_cache()

    # 001
    print("\n=== 001 (64-bit FFN, distilled, interdim=4096) ===")
    m_bin = GeneClassifier(bin_e, D, args.num_classes, num_layers=args.num_layers)
    print("  params:", {k: fmt_n(v) for k, v in m_bin.count_params().items()})
    res_bin = fine_tune(m_bin, tr, va, te, device, n_epochs=args.n_epochs, lr=args.lr, label='001')
    results['001'] = res_bin

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
