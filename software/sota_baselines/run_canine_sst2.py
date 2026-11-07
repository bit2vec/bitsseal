"""CANINE byte-level encoder baseline on SST-2.

Reports test accuracy + total parameter count + embedding parameter count.
Compared in V8 against Bits 64-d, EmByte 2048-d, Subword for the SST-2 sentiment task.

Usage:
  python run_canine_sst2.py --model_name google/canine-s --epochs 3 --batch_size 16
"""
import argparse, json, os, time, sys
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    CanineTokenizer,
    CanineForSequenceClassification,
    get_linear_schedule_with_warmup,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="google/canine-s")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_canine_sst2.json")
    p.add_argument("--limit_train", type=int, default=0,
                   help="0 = full train set; >0 truncates for quick smoke tests")
    return p.parse_args()


def collate(batch, tokenizer, max_len, device):
    sentences = [b["sentence"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    enc = tokenizer(sentences, padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
    return enc.to(device), labels.to(device)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[canine-sst2] device={device} model={args.model_name}", flush=True)

    raw = load_dataset("glue", "sst2")
    train, val = raw["train"], raw["validation"]
    if args.limit_train and args.limit_train < len(train):
        train = train.select(range(args.limit_train))

    tokenizer = CanineTokenizer.from_pretrained(args.model_name)
    model = CanineForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    ).to(device)

    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_emb = sum(p.numel() for n, p in model.named_parameters()
                       if "embeddings" in n or "char_embeddings" in n)
    print(f"[canine-sst2] total_params={n_params_total:,} "
          f"embedding_params={n_params_emb:,}", flush=True)

    def loader(split, shuffle):
        return DataLoader(split, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=lambda b: collate(b, tokenizer,
                                                       args.max_len, device))

    train_loader = loader(train, True)
    val_loader = loader(val, False)

    total_steps = len(train_loader) * args.epochs
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val = 0.0
    log = []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for step, (enc, lab) in enumerate(train_loader):
            out = model(**enc, labels=lab)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad()
            running += out.loss.item()
            if step % 100 == 0:
                print(f"[canine-sst2] ep{ep} step{step}/{len(train_loader)} "
                      f"loss={out.loss.item():.4f}", flush=True)
        avg_loss = running / max(1, len(train_loader))
        train_t = time.time() - t0

        # Eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for enc, lab in val_loader:
                pred = model(**enc).logits.argmax(-1)
                correct += (pred == lab).sum().item()
                total += lab.size(0)
        val_acc = correct / max(1, total)
        best_val = max(best_val, val_acc)
        print(f"[canine-sst2] ep{ep} train_loss={avg_loss:.4f} "
              f"val_acc={val_acc:.4f} best={best_val:.4f} "
              f"epoch_time_s={train_t:.1f}", flush=True)
        log.append({"epoch": ep, "train_loss": avg_loss,
                    "val_acc": val_acc, "epoch_time_s": train_t})

    result = {
        "task": "sst2",
        "method": "CANINE",
        "model_name": args.model_name,
        "best_val_acc": best_val,
        "total_params": n_params_total,
        "embedding_params": n_params_emb,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "log": log,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[canine-sst2] DONE wrote {args.out}", flush=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
