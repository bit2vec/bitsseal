"""DP-SGD (Abadi et al. 2016) baseline on Devign + PubMedQA.

Extends the V7 CIViC body-D DP-SGD result to two more datasets:
- Devign (code vulnerability binary classification)
- PubMedQA (3-way biomedical QA)
across epsilon in {1, 4, 8, infinity} with delta=1e-5.

Reports validation accuracy at each epsilon. The accompanying FILM evaluation
is run separately (we reuse the existing privacy attack pipeline).

Usage:
  python run_dpsgd_devign_pubmedqa.py --task devign --epsilon 8 --epochs 4
  python run_dpsgd_devign_pubmedqa.py --task pubmedqa --epsilon inf --epochs 4
"""
import argparse, json, math, os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

try:
    from opacus import PrivacyEngine
    from opacus.layers import DPLSTM
    HAS_OPACUS = True
except Exception:
    HAS_OPACUS = False
    DPLSTM = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["devign", "pubmedqa"], required=True)
    p.add_argument("--epsilon", type=str, default="8",
                   help="DP target epsilon, or 'inf' for non-private baseline")
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--vocab_size", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_dpsgd.json")
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


class TextClassifier(nn.Module):
    """A small DP-friendly text classifier: embedding + 1-layer LSTM (which
    Opacus supports via DPLSTM if installed) + linear head. We use a plain
    LSTM here and let Opacus's per-sample-gradient hooks handle it; if the
    user wants strict per-sample LSTM gradients they can swap in DPLSTM."""

    def __init__(self, vocab_size, emb_dim, hidden, num_classes,
                  use_dplstm=False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if use_dplstm:
            assert DPLSTM is not None, "Opacus DPLSTM not available"
            self.lstm = DPLSTM(emb_dim, hidden, batch_first=True)
        else:
            self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        out, (h, _) = self.lstm(emb)
        return self.head(h.squeeze(0))


def load_devign():
    """Devign code-vulnerability dataset. We use the public 'code_x_glue_cc_defect_detection'
    config which is the standard Devign release on Hugging Face."""
    ds = load_dataset("code_x_glue_cc_defect_detection")
    train = [(ex["func"], ex["target"]) for ex in ds["train"]]
    val = [(ex["func"], ex["target"]) for ex in ds["validation"]]
    return train, val, 2


def load_pubmedqa():
    ds = load_dataset("pubmed_qa", "pqa_labeled")
    label_map = {"yes": 0, "no": 1, "maybe": 2}
    pairs = []
    for ex in ds["train"]:
        text = (ex["question"] or "") + " " + " ".join(
            ex["context"]["contexts"] or []) + " " + (ex["long_answer"] or "")
        pairs.append((text, label_map[ex["final_decision"].lower()]))
    cut = int(0.8 * len(pairs))
    return pairs[:cut], pairs[cut:], 3


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task == "devign":
        train, val, ncls = load_devign()
    else:
        train, val, ncls = load_pubmedqa()
    print(f"[dpsgd/{args.task}] train={len(train)} val={len(val)} "
          f"classes={ncls}", flush=True)

    tokenizer = TokenizerLite(args.vocab_size)
    tokenizer.fit([t for t, _ in train])

    train_ds = TextDataset(train, tokenizer, args.max_len)
    val_ds = TextDataset(val, tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    model = TextClassifier(args.vocab_size, args.emb_dim, args.hidden,
                            ncls, use_dplstm=(args.epsilon != "inf")).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr)

    private = args.epsilon != "inf"
    if private and not HAS_OPACUS:
        raise RuntimeError("Opacus not installed but --epsilon != inf. "
                           "Install opacus or use --epsilon inf.")
    if private:
        eps = float(args.epsilon)
        engine = PrivacyEngine()
        model, optim, train_loader = engine.make_private_with_epsilon(
            module=model, optimizer=optim, data_loader=train_loader,
            epochs=args.epochs, target_epsilon=eps,
            target_delta=args.delta,
            max_grad_norm=args.max_grad_norm,
        )
        print(f"[dpsgd/{args.task}] private with epsilon={eps} "
              f"delta={args.delta} clip={args.max_grad_norm}", flush=True)
    else:
        print(f"[dpsgd/{args.task}] non-private (epsilon=inf)", flush=True)

    crit = nn.CrossEntropyLoss()
    best = 0.0
    log = []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            optim.zero_grad(); loss.backward(); optim.step()
            running += loss.item()
        avg_loss = running / max(1, len(train_loader))

        model.eval()
        c = n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(-1)
                c += (pred == y).sum().item(); n += y.size(0)
        acc = c / max(1, n); best = max(best, acc)
        et = time.time() - t0
        spent_eps = (engine.get_epsilon(args.delta) if private else None)
        print(f"[dpsgd/{args.task}] ep{ep} loss={avg_loss:.4f} "
              f"val_acc={acc:.4f} best={best:.4f} eps_spent={spent_eps} "
              f"t={et:.1f}s", flush=True)
        log.append({"epoch": ep, "loss": avg_loss, "val_acc": acc,
                    "epsilon_spent": spent_eps, "epoch_time_s": et})

    out = {
        "task": args.task,
        "method": "DP-SGD" if private else "non-private",
        "target_epsilon": args.epsilon,
        "delta": args.delta,
        "max_grad_norm": args.max_grad_norm,
        "best_val_acc": best,
        "log": log,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[dpsgd/{args.task}] DONE wrote {args.out}", flush=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
