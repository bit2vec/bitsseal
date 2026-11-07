"""Hash Embeddings (Svenstrup, Hansen, Winther; NIPS 2017) on SST-2 + IMDB.

Implements the standard hash-embedding layer: K hash functions map each token
to K buckets in a shared pool of size B, with per-token learned importance
weights. The token's embedding is the weighted sum of the K bucket vectors.
This is the closest architectural prior art to Bits ("random discrete code per
token") and is the natural sentiment baseline on SST-2 + IMDB.

Usage:
  python run_hashembeddings_sentiment.py --task sst2 --epochs 3
"""
import argparse, hashlib, json, os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["sst2", "imdb", "yelp"], required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--K", type=int, default=2,
                   help="number of hash functions per token")
    p.add_argument("--B", type=int, default=10000,
                   help="number of buckets in the shared pool")
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--vocab_size", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_hashemb.json")
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


class HashEmbedding(nn.Module):
    """Svenstrup hash embedding: K hash functions, B-bucket pool, learned per
    token importance weights, weighted sum of K bucket vectors."""

    def __init__(self, vocab_size, K, B, emb_dim):
        super().__init__()
        self.vocab_size = vocab_size; self.K = K; self.B = B
        self.pool = nn.Embedding(B, emb_dim, padding_idx=0)
        # K importance weights per token, jointly learned
        self.weights = nn.Embedding(vocab_size, K, padding_idx=0)
        # Precompute the (vocab_size, K) bucket-id table using K independent hashes
        bucket_ids = torch.zeros(vocab_size, K, dtype=torch.long)
        for v in range(vocab_size):
            for k in range(K):
                # Stable per-(token, k) hash via SHA1
                h = hashlib.sha1(f"{v}_{k}".encode()).digest()
                bucket_ids[v, k] = int.from_bytes(h[:8], "big") % B
        self.register_buffer("bucket_ids", bucket_ids)

    def forward(self, ids):
        # ids: (B, L)
        b_ids = self.bucket_ids[ids]                        # (B, L, K)
        bucket_vecs = self.pool(b_ids)                       # (B, L, K, D)
        wts = self.weights(ids).unsqueeze(-1)                # (B, L, K, 1)
        return (bucket_vecs * wts).sum(dim=2)                # (B, L, D)


class Classifier(nn.Module):
    def __init__(self, vocab_size, K, B, emb_dim, hidden):
        super().__init__()
        self.emb = HashEmbedding(vocab_size, K, B, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=1,
                             batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, 2)

    def forward(self, x):
        e = self.emb(x)
        out, _ = self.lstm(e)
        m = (x != 0).unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task == "sst2":
        ds = load_dataset("glue", "sst2")
        train = [(e["sentence"], e["label"]) for e in ds["train"]]
        val = [(e["sentence"], e["label"]) for e in ds["validation"]]
    elif args.task == "imdb":
        ds = load_dataset("imdb")
        train = [(e["text"], e["label"]) for e in ds["train"]]
        val = [(e["text"], e["label"]) for e in ds["test"]]
    else:  # yelp
        ds = load_dataset("yelp_polarity")
        # Cap train at 100K and test at 10K to match our existing Yelp protocol
        train_split = ds["train"].select(range(min(100000, len(ds["train"]))))
        test_split = ds["test"].select(range(min(10000, len(ds["test"]))))
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

    model = Classifier(args.vocab_size, args.K, args.B,
                        args.emb_dim, args.hidden).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_emb = sum(p.numel() for p in model.emb.parameters()
                if p.requires_grad)
    print(f"[hashemb/{args.task}] params total={n_total:,} "
          f"embedding={n_emb:,} K={args.K} B={args.B}", flush=True)

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
        avg_loss = running / max(1, len(train_loader))

        model.eval(); c = n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                c += (model(x).argmax(-1) == y).sum().item(); n += y.size(0)
        acc = c / max(1, n); best = max(best, acc); et = time.time() - t0
        print(f"[hashemb/{args.task}] ep{ep} loss={avg_loss:.4f} "
              f"acc={acc:.4f} best={best:.4f} t={et:.1f}s", flush=True)
        log.append({"epoch": ep, "loss": avg_loss, "val_acc": acc,
                    "epoch_time_s": et})

    out = {
        "task": args.task, "method": "HashEmbeddings",
        "K": args.K, "B": args.B, "best_val_acc": best,
        "total_params": n_total, "embedding_params": n_emb, "log": log,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[hashemb/{args.task}] DONE wrote {args.out}", flush=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
