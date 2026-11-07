#!/usr/bin/env python3
"""num_bits ablation for the SpinHash plot: Spin-Random code of width B through
the SAME MLP-512 encoder + BiLSTM body (run_table7 spec), 3 seeds per B, so we
get real mean+/-std bands (no fabricated error margins)."""
import argparse, os, json, time
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_table7_repro import basic_english, Model, DS, coll, BinarySubwordConcateOnehot


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--dataset', default='imdb')
    p.add_argument('--bits', nargs='+', type=int,
                   default=[8, 16, 24, 32, 48, 64, 96, 128, 192, 256])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--d', type=int, default=256)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--min_freq', type=int, default=5)
    p.add_argument('--n_train', type=int, default=25000)
    p.add_argument('--max_len', type=int, default=256)
    args = p.parse_args(); os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from datasets import load_dataset
    HF = {'imdb': ('stanfordnlp/imdb', 'text', 'label', 2, 'test'),
          'sst2': ('stanfordnlp/sst2', 'sentence', 'label', 2, 'validation')}
    hf, tf, lf, NC, tesp = HF[args.dataset]
    tr_raw = load_dataset(hf, split='train').shuffle(seed=args.seeds[0])
    te_raw = load_dataset(hf, split=tesp)
    tr = [(basic_english(r[tf]), int(r[lf])) for r in tr_raw.select(range(min(args.n_train, len(tr_raw))))]
    te = [(basic_english(r[tf]), int(r[lf])) for r in te_raw.select(range(len(te_raw)))]
    cnt = Counter(w for toks, _ in tr for w in toks)
    itos = ['<pad>', '<unk>'] + [w for w, c in cnt.most_common() if c >= args.min_freq]
    stoi = {w: i for i, w in enumerate(itos)}; V = len(itos)
    trL = DataLoader(DS(tr, stoi, args.max_len), batch_size=args.bs, shuffle=True, collate_fn=coll)
    teL = DataLoader(DS(te, stoi, args.max_len), batch_size=args.bs, collate_fn=coll)
    print(f'[{args.dataset}] train={len(tr)} test={len(te)} vocab={V}', flush=True)
    out = os.path.join(args.out_dir, f'numbits_{args.dataset}.json')
    res = json.load(open(out)) if os.path.isfile(out) else {}
    tmp = os.path.join(args.out_dir, 'maps'); os.makedirs(tmp, exist_ok=True)

    for B in args.bits:
        if str(B) in res:
            continue
        accs = []
        for seed in args.seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            rng = np.random.RandomState(10_000 + seed)
            tab = torch.tensor(rng.randint(0, 2, size=(V, B)), dtype=torch.long)
            tf_ = os.path.join(tmp, f'b{B}_s{seed}.pt'); torch.save(tab, tf_)
            emb = BinarySubwordConcateOnehot(num_embeddings=V, embedding_dim=args.d,
                                             num_bits=B, interdim='512', subword_binary_file=tf_)
            model = Model(emb, d=args.d, nc=NC).to(dev)
            opt = optim.Adam(model.parameters(), lr=args.lr); crit = nn.CrossEntropyLoss()
            t0 = time.time(); best = 0.0
            for ep in range(args.epochs):
                model.train()
                for b in trL:
                    opt.zero_grad(); crit(model(b['ids'].to(dev)), b['y'].to(dev)).backward(); opt.step()
                model.eval(); c = n = 0
                with torch.no_grad():
                    for b in teL:
                        pr = model(b['ids'].to(dev)).argmax(-1).cpu()
                        c += (pr == b['y']).sum().item(); n += b['y'].size(0)
                best = max(best, c / n)
            accs.append(best)
            print(f'  B={B:4d} s{seed}: acc={best:.4f} [{time.time()-t0:.0f}s]', flush=True)
        res[str(B)] = {'mean': round(float(np.mean(accs)), 4),
                       'std': round(float(np.std(accs)), 4),
                       'accs': [round(a, 4) for a in accs]}
        json.dump(res, open(out, 'w'), indent=2)
        print(f'  -> B={B}: {np.mean(accs):.4f}+/-{np.std(accs):.4f}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
