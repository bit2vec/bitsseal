"""MegaByte (Yu et al., NeurIPS 2023) byte-level autoregressive LM baseline.

Trains a small MegaByte on a chosen text corpus and reports bits-per-byte (BPB)
on a held-out split. Two corpora supported in the paper:
  - devign  : Devign C-code function bodies (programming-language AR LM)
  - pubmedqa: PubMedQA contexts field (biomedical text AR LM)

Usage:
  python run_megabyte_arlm.py --corpus devign   --steps 4000
  python run_megabyte_arlm.py --corpus pubmedqa --steps 4000
"""
import argparse, json, math, os, random, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

try:
    from MEGABYTE_pytorch import MEGABYTE
except Exception as e:
    raise SystemExit(
        "MEGABYTE-pytorch not installed. "
        "uv pip install MEGABYTE-pytorch"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", choices=["devign", "pubmedqa"], required=True)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=1024,
                   help="byte sequence length per example "
                        "(must equal product of patch_size)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dim", type=str, default="256,128",
                   help="hidden dims per stage (comma-separated)")
    p.add_argument("--depth", type=str, default="4,4")
    p.add_argument("--patch_size", type=str, default="4,8")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_megabyte.json")
    p.add_argument("--max_train_samples", type=int, default=20000)
    p.add_argument("--max_eval_samples", type=int, default=1000)
    return p.parse_args()


def load_corpus(name, max_train, max_eval):
    if name == "devign":
        ds = load_dataset("code_x_glue_cc_defect_detection")
        train_text = [ex["func"] for ex in ds["train"]
                      if ex["func"]][:max_train]
        eval_text = [ex["func"] for ex in ds["validation"]
                     if ex["func"]][:max_eval]
    else:
        ds = load_dataset("pubmed_qa", "pqa_labeled")
        rows = list(ds["train"])
        random.shuffle(rows)
        cut = int(0.9 * len(rows))
        train_rows, eval_rows = rows[:cut], rows[cut:]

        def to_text(ex):
            ctxs = ex.get("context", {}).get("contexts") or []
            return " ".join(ctxs)
        train_text = [t for t in (to_text(r) for r in train_rows[:max_train])
                       if t]
        eval_text = [t for t in (to_text(r) for r in eval_rows[:max_eval])
                      if t]
    return train_text, eval_text


class ByteSeqDataset(Dataset):
    def __init__(self, texts, seq_len):
        self.seq_len = seq_len
        self.examples = []
        for t in texts:
            b = list(t.encode("utf-8", errors="ignore"))
            for i in range(0, max(1, len(b) - seq_len + 1), seq_len):
                chunk = b[i:i + seq_len]
                if len(chunk) == seq_len:
                    self.examples.append(chunk)
        if not self.examples:
            raise RuntimeError("No examples after byte-chunking; "
                                "increase corpus or shrink seq_len.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            loss = model(x, return_loss=True)
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
    avg_loss = total_loss / max(1, total_tokens)
    bpb = avg_loss / math.log(2)
    ppl = math.exp(avg_loss)
    return {"bpb": bpb, "ppl": ppl, "tokens": total_tokens}


def main():
    args = parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dim = tuple(int(x) for x in args.dim.split(","))
    depth = tuple(int(x) for x in args.depth.split(","))
    patch = tuple(int(x) for x in args.patch_size.split(","))
    assert len(dim) == len(depth) == len(patch)
    prod = 1
    for p in patch:
        prod *= p
    assert args.seq_len % prod == 0, \
        f"seq_len ({args.seq_len}) must be divisible by prod(patch_size)={prod}"

    print(f"[megabyte/{args.corpus}] dim={dim} depth={depth} "
          f"patch={patch} seq_len={args.seq_len}", flush=True)

    train_text, eval_text = load_corpus(args.corpus, args.max_train_samples,
                                          args.max_eval_samples)
    print(f"[megabyte/{args.corpus}] train_texts={len(train_text)} "
          f"eval_texts={len(eval_text)}", flush=True)

    train_ds = ByteSeqDataset(train_text, args.seq_len)
    eval_ds = ByteSeqDataset(eval_text, args.seq_len)
    print(f"[megabyte/{args.corpus}] train_chunks={len(train_ds)} "
          f"eval_chunks={len(eval_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=2)

    # MEGABYTE-pytorch expects max_seq_len as a tuple of per-stage capacities
    # whose product equals seq_len. For a 2-stage hierarchy patch_size=(p1, p2),
    # the capacities are (seq_len // p1, p1) so that prec_dims align.
    max_seq = (args.seq_len // patch[0], patch[0])
    model = MEGABYTE(
        num_tokens=259,         # 256 bytes + a few specials
        dim=dim,
        depth=depth,
        max_seq_len=max_seq,
        flash_attn=False,
    ).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[megabyte/{args.corpus}] params total={n_total:,}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=0.01)
    log = []
    it = iter(train_loader)
    step = 0
    t0 = time.time()
    while step < args.steps:
        try:
            x = next(it)
        except StopIteration:
            it = iter(train_loader); x = next(it)
        x = x.to(device)
        loss = model(x, return_loss=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step(); optim.zero_grad()
        step += 1
        if step % 100 == 0 or step == 1:
            print(f"[megabyte/{args.corpus}] step={step}/{args.steps} "
                  f"loss={loss.item():.4f} t={time.time()-t0:.0f}s",
                  flush=True)
            log.append({"step": step, "loss": float(loss.item())})

    metrics = evaluate(model, eval_loader, device)
    out = {
        "corpus": args.corpus, "method": "MegaByte",
        "total_params": n_total,
        "dim": dim, "depth": depth, "patch_size": patch,
        "seq_len": args.seq_len, "steps": args.steps,
        "metrics": metrics, "log_tail": log[-50:],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[megabyte/{args.corpus}] DONE BPB={metrics['bpb']:.4f} "
          f"PPL={metrics['ppl']:.2f}", flush=True)
    print(f"[megabyte/{args.corpus}] wrote {args.out}")


if __name__ == "__main__":
    main()
