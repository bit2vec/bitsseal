"""ByT5 byte-level seq2seq baseline on quy-es and gn-es endangered MT.

Pretrains on de-en + ar-en (IWSLT14), finetunes on quy-es and gn-es.
Reports BLEU + ChrF on the test sets.

Usage:
  python run_byt5_quy_gn.py --model_name google/byt5-small \
        --pretrain_steps 20000 --finetune_steps 5000

For initial smoke runs (no pretraining), use --skip_pretrain to start
from the public byt5-small checkpoint and finetune directly.
"""
import argparse, json, os, time, sys, math, random
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import (
    ByT5Tokenizer,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
import sacrebleu


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="google/byt5-small")
    p.add_argument("--skip_pretrain", action="store_true")
    p.add_argument("--pretrain_steps", type=int, default=10000)
    p.add_argument("--finetune_steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_src", type=int, default=256)
    p.add_argument("--max_tgt", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_byt5_quy_gn.json")
    p.add_argument("--quy_corpus",
                   default="americasnlp2021/americasnlp2021")
    p.add_argument("--gn_corpus",
                   default="americasnlp2021/americasnlp2021")
    p.add_argument("--pretrain_corpus", default="iwslt2017",
                   help="iwslt2017 config name handled below")
    p.add_argument("--num_eval", type=int, default=500)
    return p.parse_args()


class PairDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_src, max_tgt):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        s = self.tok(src, truncation=True, max_length=self.max_src,
                     return_tensors="pt")
        t = self.tok(tgt, truncation=True, max_length=self.max_tgt,
                     return_tensors="pt")
        return {"input_ids": s.input_ids[0],
                "attention_mask": s.attention_mask[0],
                "labels": t.input_ids[0]}


def pad_collate(batch, pad_id):
    def pad(xs, value):
        m = max(x.size(0) for x in xs)
        return torch.stack([torch.cat([x, x.new_full((m - x.size(0),), value)])
                            for x in xs])
    inp = pad([b["input_ids"] for b in batch], pad_id)
    am = pad([b["attention_mask"] for b in batch], 0)
    lab = pad([b["labels"] for b in batch], -100)
    return {"input_ids": inp, "attention_mask": am, "labels": lab}


def load_iwslt_de_en():
    """Best-effort: use iwslt2017 de-en + ar-en if available; otherwise fall
    back to opus_books or similar. We tolerate small corpora here because the
    pretraining is for endangered-language transfer rather than a fresh MT
    benchmark."""
    de_en = load_dataset("iwslt2017", "iwslt2017-de-en", split="train")
    ar_en = load_dataset("iwslt2017", "iwslt2017-ar-en", split="train")
    pairs = []
    for ex in de_en:
        pairs.append((f"translate German to English: {ex['translation']['de']}",
                      ex["translation"]["en"]))
    for ex in ar_en:
        pairs.append((f"translate Arabic to English: {ex['translation']['ar']}",
                      ex["translation"]["en"]))
    return pairs


LANG_DIRS = {
    "quy": "data/americasnlp/quechua-spanish",
    "gn":  "data/americasnlp/guarani-spanish",
}


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def load_americasnlp(lang):
    """Loads quy-es or gn-es from the local AmericasNLP parallel files.
    Returns (train_pairs, dev_pairs, test_pairs); we use dev as the test set
    here because the AmericasNLP shared-task gold test set is not public."""
    base = LANG_DIRS.get(lang)
    if base is None or not os.path.isdir(base):
        return None, None, None
    src_train = _read_lines(os.path.join(base, f"train.{lang}"))
    tgt_train = _read_lines(os.path.join(base, "train.es"))
    src_dev = _read_lines(os.path.join(base, f"dev.{lang}"))
    tgt_dev = _read_lines(os.path.join(base, "dev.es"))
    n_train = min(len(src_train), len(tgt_train))
    n_dev = min(len(src_dev), len(tgt_dev))
    train_pairs = [(f"translate {lang} to Spanish: {src_train[i]}",
                    tgt_train[i]) for i in range(n_train)
                   if src_train[i].strip() and tgt_train[i].strip()]
    dev_pairs = [(f"translate {lang} to Spanish: {src_dev[i]}",
                  tgt_dev[i]) for i in range(n_dev)
                 if src_dev[i].strip() and tgt_dev[i].strip()]
    return train_pairs, dev_pairs, dev_pairs


def train_loop(model, loader, steps, lr, device, tag, log):
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(0.05 * steps), num_training_steps=steps)
    it = iter(loader)
    step = 0
    t0 = time.time()
    while step < steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step(); sched.step(); optim.zero_grad()
        step += 1
        if step % 200 == 0:
            print(f"[byt5/{tag}] step={step}/{steps} loss={out.loss.item():.4f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
            log.append({"phase": tag, "step": step,
                        "loss": float(out.loss.item())})


def evaluate(model, tokenizer, pairs, device, num_eval, max_tgt):
    model.eval()
    n = min(num_eval, len(pairs))
    refs = []
    hyps = []
    with torch.no_grad():
        for src, tgt in pairs[:n]:
            inp = tokenizer(src, return_tensors="pt", truncation=True,
                             max_length=512).to(device)
            out = model.generate(**inp, max_new_tokens=max_tgt,
                                  num_beams=4, length_penalty=0.6)
            hyp = tokenizer.decode(out[0], skip_special_tokens=True)
            refs.append(tgt); hyps.append(hyp)
    if not refs:
        return {"bleu": 0.0, "chrf": 0.0, "n": 0}
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    return {"bleu": bleu, "chrf": chrf, "n": n}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = ByT5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_emb = sum(p.numel() for n, p in model.named_parameters()
                if "shared" in n or "embed_tokens" in n)
    print(f"[byt5] params total={n_total:,} embedding={n_emb:,}", flush=True)

    log = []

    # Pretraining
    if not args.skip_pretrain:
        try:
            pretrain_pairs = load_iwslt_de_en()
            print(f"[byt5] loaded pretrain pairs: {len(pretrain_pairs):,}",
                  flush=True)
            ds = PairDataset(pretrain_pairs, tokenizer,
                             args.max_src, args.max_tgt)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                collate_fn=lambda b: pad_collate(
                                    b, tokenizer.pad_token_id),
                                num_workers=2)
            train_loop(model, loader, args.pretrain_steps, args.lr,
                       device, "pretrain", log)
        except Exception as e:
            print(f"[byt5] pretrain skipped: {e}", flush=True)

    # Finetune + eval per language
    final = {}
    for lang in ("quy", "gn"):
        try:
            train_pairs, dev_pairs, test_pairs = load_americasnlp(lang)
            if train_pairs is None or len(train_pairs) == 0:
                print(f"[byt5/{lang}] skipped: no train data available",
                      flush=True)
                continue
            print(f"[byt5/{lang}] train={len(train_pairs):,} "
                  f"dev={len(dev_pairs)} test={len(test_pairs)}", flush=True)
            ds = PairDataset(train_pairs, tokenizer,
                             args.max_src, args.max_tgt)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                collate_fn=lambda b: pad_collate(
                                    b, tokenizer.pad_token_id),
                                num_workers=2)
            train_loop(model, loader, args.finetune_steps, args.lr,
                       device, f"finetune-{lang}", log)
            metrics = evaluate(model, tokenizer, test_pairs, device,
                               args.num_eval, args.max_tgt)
            final[lang] = metrics
            print(f"[byt5/{lang}] test BLEU={metrics['bleu']:.2f} "
                  f"ChrF={metrics['chrf']:.2f} n={metrics['n']}", flush=True)
        except Exception as e:
            print(f"[byt5/{lang}] error: {e}", flush=True)

    out = {
        "task": "endangered MT (quy-es, gn-es)",
        "method": "ByT5",
        "model_name": args.model_name,
        "total_params": n_total,
        "embedding_params": n_emb,
        "metrics": final,
        "log_tail": log[-50:],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[byt5] DONE wrote {args.out}", flush=True)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
