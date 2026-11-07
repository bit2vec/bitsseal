"""CHARFORMER (Tay et al., ICLR 2022) baseline on SST-2.

Uses lucidrains' charformer-pytorch implementation: a learned-block-size
gradient-based subword tokenizer (GBST) plus a small Transformer encoder.
Reports validation accuracy + total parameter count.

Usage:
  python run_charformer_sst2.py --epochs 3 --batch_size 32
"""
import argparse, json, os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

try:
    from charformer_pytorch import GBST
except Exception as e:
    raise SystemExit(
        "charformer-pytorch not installed. "
        "uv pip install charformer-pytorch"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_charformer_sst2.json")
    return p.parse_args()


class ByteDataset(Dataset):
    def __init__(self, pairs, max_len):
        self.pairs = pairs; self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        text, label = self.pairs[i]
        b = list(text.encode("utf-8")[:self.max_len])
        b += [0] * (self.max_len - len(b))
        return torch.tensor(b, dtype=torch.long), torch.tensor(label,
                                                                dtype=torch.long)


class CharformerClassifier(nn.Module):
    def __init__(self, d_model=128, n_head=4, n_layer=2, max_len=512):
        super().__init__()
        # GBST on the raw byte stream (vocab=259 = 256 bytes + 3 special)
        self.gbst = GBST(
            num_tokens=259,
            dim=d_model,
            max_block_size=4,
            downsample_factor=4,
            score_consensus_attn=True,
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        # mask: 1 where pad
        mask = (x == 0)
        # GBST returns (downsampled tokens, downsampled mask) when given mask;
        # Here we pass mask=None and let it process the full byte stream
        e, m = self.gbst(x, mask=~mask)
        # e: (B, L', d) downsampled tokens
        # m: (B, L') downsampled boolean mask (True=valid)
        out = self.encoder(e, src_key_padding_mask=~m)
        # mean pool over valid positions
        m_f = m.unsqueeze(-1).float()
        pooled = (out * m_f).sum(1) / m_f.sum(1).clamp(min=1)
        return self.head(pooled)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_dataset("glue", "sst2")
    train = [(e["sentence"], e["label"]) for e in ds["train"]]
    val = [(e["sentence"], e["label"]) for e in ds["validation"]]
    train_ds = ByteDataset(train, args.max_len)
    val_ds = ByteDataset(val, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    model = CharformerClassifier(args.d_model, args.n_head,
                                  args.n_layer, args.max_len).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[charformer/sst2] params total={n_total:,}", flush=True)

    crit = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=0.01)
    best = 0.0; log = []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += loss.item()
        avg = running / max(1, len(train_loader))

        model.eval(); c = n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                c += (model(x).argmax(-1) == y).sum().item(); n += y.size(0)
        acc = c / max(1, n); best = max(best, acc); et = time.time() - t0
        print(f"[charformer/sst2] ep{ep} loss={avg:.4f} acc={acc:.4f} "
              f"best={best:.4f} t={et:.1f}s", flush=True)
        log.append({"epoch": ep, "loss": avg, "val_acc": acc,
                    "epoch_time_s": et})

    out = {"task": "sst2", "method": "CHARFORMER",
            "best_val_acc": best, "total_params": n_total, "log": log}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[charformer/sst2] DONE wrote {args.out}", flush=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
