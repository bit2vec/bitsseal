#!/usr/bin/env python3
"""Mean-pool (body A) vs cross-attn+MLP (body D) ablation on CIViC, for
subword / EmByte(byte_seb) / SpinHash-SimHash. Verified numbers for a dedicated
pooling-ablation table (tqyj suggestion). Final-epoch acc over N seeds.
"""
import sys, os, random, json, argparse
import numpy as np, torch, torch.nn as nn, torch.optim as optim
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(THIS, 'paper_modules'))
sys.path.insert(0, os.path.join(THIS, '..', 'Senti_analysis'))
from run_civic_extended import MultiModalClassifier
from run_clinical_sig_simhash import setup_civic, simhash_table
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_eval(method, body, V, nc, pad, trL, teL, seed, epochs, sh_tab):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    base = 'binary_seb' if method == 'simhash-spin' else method
    model = MultiModalClassifier(V, 64, nc, method=base, body_kind=body,
                                 num_bits=64, interdim='512', pad_idx=pad).to(DEV)
    if method == 'simhash-spin':
        with torch.no_grad():
            model.embedding.subword2binary.copy_(sh_tab.to(model.embedding.subword2binary.device))
    opt = optim.Adam(model.parameters(), lr=5e-4); crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for b in trL:
            if b is None: continue
            opt.zero_grad(); crit(model(b['ids'].to(DEV)), b['label'].to(DEV)).backward(); opt.step()
    model.eval(); c = t = 0
    with torch.no_grad():
        for b in teL:
            if b is None: continue
            p = model(b['ids'].to(DEV)).argmax(-1).cpu()
            c += (p == b['label']).sum().item(); t += b['label'].size(0)
    return c / max(t, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--out_dir', default='results/multimodal_byte/meanpool_ablation')
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    trL, teL, V, nc, pad, itos = setup_civic(); sh = simhash_table(itos)
    print(f"CIViC vocab={V} classes={nc} test={len(teL.dataset)}", flush=True)
    out = {}
    for method in ['standard', 'byte_seb', 'simhash-spin']:
        out[method] = {}
        for body in ['A', 'D']:
            accs = [train_eval(method, body, V, nc, pad, trL, teL, s, a.epochs, sh) for s in a.seeds]
            out[method][body] = {'mean': round(float(np.mean(accs)), 4), 'std': round(float(np.std(accs)), 4)}
            print(f"  {method:13s} body {body}: {np.mean(accs):.4f}+/-{np.std(accs):.4f}", flush=True)
        d = out[method]['D']['mean'] - out[method]['A']['mean']
        out[method]['drop_meanpool'] = round(d, 4)
        print(f"  => {method}: attn(D) {out[method]['D']['mean']} - mean(A) {out[method]['A']['mean']} = drop {d:+.4f}", flush=True)
        json.dump(out, open(os.path.join(a.out_dir, 'meanpool.json'), 'w'), indent=2)
    print("done", flush=True)


if __name__ == '__main__':
    main()
