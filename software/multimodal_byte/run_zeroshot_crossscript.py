#!/usr/bin/env python
"""Zero-shot cross-script demo: train sentiment on ENGLISH only, evaluate on
never-seen scripts (Arabic, Hindi) with ZERO new embedding parameters.

Subword: vocab built on EN train -> unseen-script tokens collapse to <unk>.
SpinHash: every token -> UTF-8 bytes -> SimHash spin code -> one shared FFN,
which generalizes to codes of tokens it never saw -> +0 embedding params.

Reports per-language accuracy (subword vs spin-simhash vs spin-random) and the
structural coverage stat (% of test tokens representable by each encoder).

Data: cardiffnlp/tweet_sentiment_multilingual (Barbieri et al. 2022), 3-class.
"""
import argparse, hashlib, json, os, time, re
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PARQUET = "hf://datasets/cardiffnlp/tweet_sentiment_multilingual@refs%2Fconvert%2Fparquet/{lang}/{split}/0000.parquet"
TOK = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize(s):
    return TOK.findall(str(s).lower())

def load(lang, split):
    df = pd.read_parquet(PARQUET.format(lang=lang, split=split))
    return [tokenize(t) for t in df["text"].tolist()], df["label"].astype(int).tolist()

# ---- SimHash spin code: pure function of token bytes (vocab-independent) ----
def simhash_table(itos, num_bits=64, F_=2048, seed=0):
    rng = np.random.RandomState(seed); R = rng.randn(F_, num_bits).astype("float32")
    tab = torch.zeros(len(itos), num_bits, dtype=torch.long)
    for i, tok in enumerate(itos):
        s = "  " + str(tok) + "  "; feat = np.zeros(F_, dtype="float32")
        for j in range(len(s) - 2):
            feat[int(hashlib.md5(s[j:j+3].encode("utf-8", "ignore")).hexdigest(), 16) % F_] += 1.0
        tab[i] = torch.tensor((feat @ R > 0).astype("int64"))
    return tab

def random_table(itos, num_bits=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (len(itos), num_bits), generator=g)

# ---- models ----
class SpinEncoder(nn.Module):  # code -> phi one-hot lift -> shared FFN (EmByte/SpinHash MLP-512)
    def __init__(self, code_table, d=256, interdim=512):
        super().__init__()
        self.register_buffer("codes", code_table.long())     # [V, J]
        J = code_table.shape[1]
        self.ffn = nn.Sequential(nn.Linear(2 * J, interdim), nn.ReLU(), nn.Linear(interdim, d))
    def forward(self, ids):
        c = self.codes[ids]                                   # [B,L,J] in {0,1}
        phi = torch.stack([1 - c, c], -1).flatten(-2).float() # [B,L,2J]
        return self.ffn(phi)
    def emb_params(self):
        return sum(p.numel() for p in self.ffn.parameters())

class SubwordEncoder(nn.Module):
    def __init__(self, V, d=256):
        super().__init__(); self.emb = nn.Embedding(V, d, padding_idx=0)
    def forward(self, ids): return self.emb(ids)
    def emb_params(self): return self.emb.weight.numel()

class Clf(nn.Module):
    def __init__(self, enc, d=256, h=256, n_cls=3):
        super().__init__(); self.enc = enc
        self.lstm = nn.LSTM(d, h, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * h, n_cls)
    def forward(self, ids):
        x = self.enc(ids); o, _ = self.lstm(x)
        mask = (ids != 0).unsqueeze(-1).float()
        pooled = (o * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled)

def encode(tok_lists, stoi, max_len, unk, allow_oov):
    """allow_oov=False -> OOV maps to <unk> (subword); True -> must be in stoi (spin, full vocab)."""
    out = []
    for toks in tok_lists:
        ids = [stoi.get(t, unk) for t in toks][:max_len]
        ids += [0] * (max_len - len(ids))
        out.append(ids)
    return torch.tensor(out, dtype=torch.long)

def run_epochs(model, Xtr, ytr, dev, epochs, lr, bs):
    opt = torch.optim.Adam(model.parameters(), lr=lr); lossf = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); loss = lossf(model(xb.to(dev)), yb.to(dev)); loss.backward(); opt.step()

@torch.no_grad()
def acc(model, X, y, dev, bs=256):
    model.eval(); correct = 0
    for i in range(0, len(X), bs):
        p = model(X[i:i+bs].to(dev)).argmax(1).cpu()
        correct += (p == y[i:i+bs]).sum().item()
    return correct / len(X)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_lang", default="english")
    ap.add_argument("--test_langs", nargs="+", default=["english", "arabic", "hindi"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", default="results/multimodal_byte/zeroshot_crossscript/summary.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)

    tr_tok, tr_y = load(args.train_lang, "train")
    tests = {l: load(l, "test") for l in args.test_langs}

    # EN-train vocab (subword); <pad>=0 <unk>=1
    from collections import Counter
    cnt = Counter(t for toks in tr_tok for t in toks)
    en_vocab = [w for w, c in cnt.items() if c >= args.min_freq]
    sub_itos = ["<pad>", "<unk>"] + en_vocab
    sub_stoi = {w: i for i, w in enumerate(sub_itos)}; UNK = 1

    # spin vocab = EN-train UNION all test tokens (codes are pure byte fns; train still only sees EN)
    all_test_toks = {t for l in tests for toks in tests[l][0] for t in toks}
    spin_itos = ["<pad>", "<unk>"] + sorted(set(en_vocab) | all_test_toks)
    spin_stoi = {w: i for i, w in enumerate(spin_itos)}

    # coverage stat: fraction of test tokens present in EN-train vocab.
    # "native": tokens containing a non-ASCII char (script-native content words) ->
    # the honest linguistic contrast, where EN-vocab coverage collapses to ~0.
    cov = {}
    en_set = set(en_vocab)
    is_native = lambda t: any(ord(ch) > 127 for ch in t)
    for l, (toks, _) in tests.items():
        flat = [t for s in toks for t in s]
        nat = [t for t in flat if is_native(t)]
        cov[l] = {"n_tokens": len(flat),
                  "subword_covered": sum(t in en_set for t in flat) / max(1, len(flat)),
                  "n_native": len(nat),
                  "subword_covered_native": (sum(t in en_set for t in nat) / max(1, len(nat))),
                  "spin_covered": 1.0}
    print("coverage:", json.dumps(cov, indent=2), flush=True)

    n_cls = max(max(tr_y), *[max(tests[l][1]) for l in tests]) + 1
    maj = {l: max(np.bincount(tests[l][1])) / len(tests[l][1]) for l in tests}

    # precompute encodings
    Xtr_sub = encode(tr_tok, sub_stoi, args.max_len, UNK, False)
    Xtr_spin = encode(tr_tok, spin_stoi, args.max_len, UNK, True)
    ytr = torch.tensor(tr_y)
    Xte_sub = {l: encode(tests[l][0], sub_stoi, args.max_len, UNK, False) for l in tests}
    Xte_spin = {l: encode(tests[l][0], spin_stoi, args.max_len, UNK, True) for l in tests}
    yte = {l: torch.tensor(tests[l][1]) for l in tests}

    methods = {
        "subword": lambda s: SubwordEncoder(len(sub_itos)),
        "spin_simhash": lambda s: SpinEncoder(simhash_table(spin_itos, seed=s)),
        "spin_random": lambda s: SpinEncoder(random_table(spin_itos, seed=s)),
    }
    results = {"coverage": cov, "majority": maj, "n_cls": int(n_cls), "per_method": {}}
    for name, build in methods.items():
        accs = {l: [] for l in tests}; emb_p = None
        for s in args.seeds:
            torch.manual_seed(s); np.random.seed(s)
            enc = build(s); emb_p = enc.emb_params()
            model = Clf(enc, n_cls=int(n_cls)).to(dev)
            Xtr = Xtr_sub if name == "subword" else Xtr_spin
            t0 = time.time(); run_epochs(model, Xtr, ytr, dev, args.epochs, args.lr, args.bs)
            for l in tests:
                Xte = Xte_sub[l] if name == "subword" else Xte_spin[l]
                accs[l].append(acc(model, Xte, yte[l], dev))
            print(f"[{name}] seed{s} {time.time()-t0:.0f}s " +
                  " ".join(f"{l}={np.mean(accs[l]):.3f}" for l in tests), flush=True)
        results["per_method"][name] = {
            "emb_params": int(emb_p),
            "acc": {l: {"mean": float(np.mean(accs[l])), "std": float(np.std(accs[l]))} for l in tests}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print("WROTE", args.out, flush=True)
    print(json.dumps(results["per_method"], indent=2))

if __name__ == "__main__":
    main()
