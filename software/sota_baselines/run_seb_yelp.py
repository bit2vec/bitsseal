"""Subword / EmByte / Bits comparison cells on Yelp Polarity.

Runs the three encoder variants under matched BiLSTM body + classification
head, so that the Yelp column of the SOTA-baselines sentiment table can be
filled apples-to-apples alongside Hash Embeddings (already run separately).

Usage:
  python run_seb_yelp.py --method subword
  python run_seb_yelp.py --method embyte
  python run_seb_yelp.py --method bits
"""
import argparse, json, os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["subword", "embyte", "bits"],
                    required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--vocab_size", type=int, default=30000)
    p.add_argument("--max_train", type=int, default=100000)
    p.add_argument("--max_test", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    return p.parse_args()


class TokenizerLite:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.tok2id = {"<pad>": 0, "<unk>": 1}

    def fit(self, texts):
        from collections import Counter
        c = Counter()
        for t in texts:
            for w in t.lower().split():
                c[w] += 1
        for w, _ in c.most_common(self.vocab_size - 2):
            self.tok2id.setdefault(w, len(self.tok2id))

    def encode(self, text, max_len):
        ids = [self.tok2id.get(w, 1) for w in text.lower().split()][:max_len]
        ids += [0] * (max_len - len(ids))
        return ids


class TextDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len):
        self.pairs = pairs; self.tokenizer = tokenizer; self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        text, label = self.pairs[i]
        return torch.tensor(self.tokenizer.encode(text, self.max_len),
                            dtype=torch.long), torch.tensor(label,
                                                            dtype=torch.long)


class SubwordEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

    def forward(self, x):
        return self.emb(x)

    def emb_params(self):
        return sum(p.numel() for p in self.emb.parameters())


class EmByteEncoder(nn.Module):
    """EmByte: 8-byte one-hot at width 2048 -> 128 -> emb_dim FFN.

    For a token id t, we read its 8 little-endian bytes, one-hot each into
    R^256, concatenate to R^2048, then apply the 2-layer FFN."""

    def __init__(self, vocab_size, emb_dim, hidden=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.W1 = nn.Linear(2048, hidden)
        self.W2 = nn.Linear(hidden, emb_dim)
        # Precompute byte-level one-hots for the vocabulary
        ohs = torch.zeros(vocab_size, 2048)
        for v in range(vocab_size):
            bytes_ = v.to_bytes(8, "little", signed=False)
            for k, b in enumerate(bytes_):
                ohs[v, 256 * k + b] = 1.0
        self.register_buffer("ohs", ohs)

    def forward(self, x):
        # x: (B, L) ids
        oh = self.ohs[x]  # (B, L, 2048)
        return self.W2(torch.relu(self.W1(oh)))

    def emb_params(self):
        return (sum(p.numel() for p in self.W1.parameters()) +
                sum(p.numel() for p in self.W2.parameters()))


class BitsEncoder(nn.Module):
    """Bits: 64 raw bits at width 64 -> 256 -> emb_dim FFN."""

    def __init__(self, vocab_size, emb_dim, hidden=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.W1 = nn.Linear(64, hidden)
        self.W2 = nn.Linear(hidden, emb_dim)
        # Precompute 64-bit codes for the vocabulary (deterministic LE bits)
        bits = torch.zeros(vocab_size, 64)
        for v in range(vocab_size):
            x = v
            for k in range(64):
                bits[v, k] = (x >> k) & 1
        self.register_buffer("bits", bits)

    def forward(self, x):
        b = self.bits[x]  # (B, L, 64)
        return self.W2(torch.relu(self.W1(b)))

    def emb_params(self):
        return (sum(p.numel() for p in self.W1.parameters()) +
                sum(p.numel() for p in self.W2.parameters()))


class Classifier(nn.Module):
    def __init__(self, encoder, emb_dim, hidden):
        super().__init__()
        self.encoder = encoder
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=1,
                             batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, 2)

    def forward(self, x):
        e = self.encoder(x)
        out, _ = self.lstm(e)
        m = (x != 0).unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_dataset("yelp_polarity")
    train_split = ds["train"].select(range(min(args.max_train,
                                                 len(ds["train"]))))
    test_split = ds["test"].select(range(min(args.max_test,
                                              len(ds["test"]))))
    train = [(e["text"], e["label"]) for e in train_split]
    val = [(e["text"], e["label"]) for e in test_split]

    tokenizer = TokenizerLite(args.vocab_size)
    tokenizer.fit([t for t, _ in train])

    train_ds = TextDataset(train, tokenizer, args.max_len)
    val_ds = TextDataset(val, tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    if args.method == "subword":
        encoder = SubwordEncoder(args.vocab_size, args.emb_dim)
    elif args.method == "embyte":
        encoder = EmByteEncoder(args.vocab_size, args.emb_dim)
    else:
        encoder = BitsEncoder(args.vocab_size, args.emb_dim)
    model = Classifier(encoder, args.emb_dim, args.hidden).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_emb = encoder.emb_params()
    print(f"[seb-yelp/{args.method}] total={n_total:,} embedding={n_emb:,}",
          flush=True)

    crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = 0.0; log = []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            optim.zero_grad(); loss.backward(); optim.step()
            running += loss.item()
        avg = running / max(1, len(train_loader))

        model.eval(); c = n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                c += (model(x).argmax(-1) == y).sum().item(); n += y.size(0)
        acc = c / max(1, n); best = max(best, acc); et = time.time() - t0
        print(f"[seb-yelp/{args.method}] ep{ep} loss={avg:.4f} acc={acc:.4f} "
              f"best={best:.4f} t={et:.1f}s", flush=True)
        log.append({"epoch": ep, "loss": avg, "val_acc": acc,
                    "epoch_time_s": et})

    out = {"task": "yelp", "method": args.method,
            "best_val_acc": best,
            "total_params": n_total, "embedding_params": n_emb,
            "log": log}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[seb-yelp/{args.method}] DONE wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
