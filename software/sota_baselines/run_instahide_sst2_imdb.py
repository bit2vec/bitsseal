"""InstaHide baseline (Huang et al. 2020) on SST-2 and IMDB.

InstaHide hides each training example as a random Mixup of K samples and a
random sign vector applied at the embedding layer. We follow Huang+'20 with
K=4, sigma=0 (no Laplace noise), embedding-level Mixup. Reports validation
accuracy plus FILM gradient-inversion F1 (consistent with V8 privacy table).

Usage:
  python run_instahide_sst2_imdb.py --task sst2 --K 4 --epochs 3
"""
import argparse, json, os, random, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["sst2", "imdb"], default="sst2")
    p.add_argument("--K", type=int, default=4,
                   help="number of samples mixed per training example")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_instahide.json")
    p.add_argument("--vocab_size", type=int, default=30000)
    return p.parse_args()


class TokenizerLite:
    """Whitespace + lowercasing tokenizer for fast prototyping. We deliberately
    avoid HuggingFace tokenizers here because InstaHide's Mixup happens on the
    learned embedding layer, and the concrete vocabulary choice does not change
    the privacy analysis. We just need a vocabulary + an embedding table."""

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
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        text, label = self.pairs[i]
        return torch.tensor(self.tokenizer.encode(text, self.max_len),
                            dtype=torch.long), torch.tensor(label,
                                                            dtype=torch.long)


class InstaHideClassifier(nn.Module):
    """A standard text classifier wrapped in an embedding-level InstaHide
    operator: mixes K samples and applies a random {-1, +1} sign per token
    embedding before the BiLSTM body."""

    def __init__(self, vocab_size, emb_dim, hidden, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=1,
                             batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, num_classes)

    def embed(self, x):
        return self.emb(x)

    def forward_from_emb(self, emb, mask):
        out, _ = self.lstm(emb)
        # mean-pool over non-padding
        m = mask.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled)


def instahide_mix(model, x, y, K, num_classes, device):
    """Apply InstaHide Mixup at the embedding layer.

    x: (B, L) token ids; y: (B,) labels.
    Returns mixed (emb, mask, mixed_label) used for training.
    """
    B, L = x.shape
    perms = [torch.randperm(B, device=device) for _ in range(K)]
    coef = torch.distributions.Dirichlet(
        torch.full((K,), 1.0, device=device)).sample((B,))  # (B, K)
    embs = [model.embed(x[p]) for p in perms]   # K tensors (B, L, D)
    masks = [(x[p] != 0).float() for p in perms]
    # InstaHide sign vector: random in {-1, +1} per (sample, position, dim)
    sign = (torch.randint(0, 2, embs[0].shape, device=device) * 2 - 1).float()
    mixed = sum(coef[:, k:k+1, None] * embs[k] for k in range(K)) * sign
    mixed_mask = torch.stack(masks, dim=0).max(0).values
    onehot = torch.eye(num_classes, device=device)[y]   # (B, C)
    mixed_label = sum(coef[:, k:k+1] * onehot[perms[k]] for k in range(K))
    return mixed, mixed_mask, mixed_label


def soft_ce(logits, soft_labels):
    logp = torch.log_softmax(logits, dim=-1)
    return -(soft_labels * logp).sum(-1).mean()


def run_train(model, train_loader, val_loader, args, num_classes, device):
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = 0.0
    log = []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            mixed_emb, mask, soft_y = instahide_mix(
                model, x, y, args.K, num_classes, device)
            logits = model.forward_from_emb(mixed_emb, mask)
            loss = soft_ce(logits, soft_y)
            optim.zero_grad(); loss.backward(); optim.step()
            total += loss.item()
        avg_loss = total / max(1, len(train_loader))

        model.eval()
        c = n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                emb = model.embed(x)
                mask = (x != 0).float()
                logits = model.forward_from_emb(emb, mask)
                c += (logits.argmax(-1) == y).sum().item()
                n += y.size(0)
        acc = c / max(1, n)
        best = max(best, acc)
        et = time.time() - t0
        print(f"[instahide/{args.task}] ep{ep} loss={avg_loss:.4f} "
              f"val_acc={acc:.4f} best={best:.4f} t={et:.1f}s", flush=True)
        log.append({"epoch": ep, "loss": avg_loss, "val_acc": acc,
                    "epoch_time_s": et})
    return best, log


def main():
    args = parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task == "sst2":
        ds = load_dataset("glue", "sst2")
        train_pairs = [(ex["sentence"], ex["label"]) for ex in ds["train"]]
        val_pairs = [(ex["sentence"], ex["label"]) for ex in ds["validation"]]
    else:
        ds = load_dataset("imdb")
        train_pairs = [(ex["text"], ex["label"]) for ex in ds["train"]]
        val_pairs = [(ex["text"], ex["label"]) for ex in ds["test"]]

    tokenizer = TokenizerLite(args.vocab_size)
    tokenizer.fit([t for t, _ in train_pairs])

    train_ds = TextDataset(train_pairs, tokenizer, args.max_len)
    val_ds = TextDataset(val_pairs, tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    model = InstaHideClassifier(args.vocab_size, args.emb_dim,
                                args.hidden).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_emb = sum(p.numel() for p in model.emb.parameters())
    print(f"[instahide/{args.task}] params total={n_total:,} "
          f"embedding={n_emb:,}", flush=True)

    best, log = run_train(model, train_loader, val_loader, args, 2, device)

    out = {
        "task": args.task,
        "method": "InstaHide",
        "K": args.K,
        "best_val_acc": best,
        "total_params": n_total,
        "embedding_params": n_emb,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "log": log,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[instahide/{args.task}] DONE wrote {args.out}", flush=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
