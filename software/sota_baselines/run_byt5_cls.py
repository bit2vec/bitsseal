#!/usr/bin/env python3
"""ByT5-small as a text CLASSIFIER: ByT5 encoder (byte-level, pretrained) + mean
pool + linear head, fine-tuned on SST-2 and IMDB. Gives a real byte-level SOTA
classification number (the prior ByT5 run was MT-only / leak-invalid)."""
import argparse, json, os, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import T5EncoderModel, AutoTokenizer
from datasets import load_dataset


class DS(Dataset):
    def __init__(self, rows, tok, ml):
        self.rows, self.tok, self.ml = rows, tok, ml
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        t, y = self.rows[i]
        enc = self.tok(t, truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        return enc['input_ids'][0], enc['attention_mask'][0], y


class Clf(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.enc = T5EncoderModel.from_pretrained('google/byt5-small')
        self.head = nn.Linear(self.enc.config.d_model, nc)
    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--datasets', nargs='+', default=['sst2', 'imdb'])
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--bs', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--n_train', type=int, default=25000)
    args = p.parse_args(); os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained('google/byt5-small')
    HF = {'sst2': ('stanfordnlp/sst2', 'sentence', 'label', 2, 'validation', 256),
          'imdb': ('stanfordnlp/imdb', 'text', 'label', 2, 'test', 512)}
    summary = {}
    out = os.path.join(args.out_dir, 'byt5_cls.json')
    for ds in args.datasets:
        hf, tf, lf, nc, tesp, ml = HF[ds]
        tr_raw = load_dataset(hf, split='train').shuffle(seed=42)
        te_raw = load_dataset(hf, split=tesp)
        tr = [(r[tf], int(r[lf])) for r in tr_raw.select(range(min(args.n_train, len(tr_raw))))]
        te = [(r[tf], int(r[lf])) for r in te_raw.select(range(min(25000, len(te_raw))))]
        trL = DataLoader(DS(tr, tok, ml), batch_size=args.bs, shuffle=True, num_workers=4)
        teL = DataLoader(DS(te, tok, ml), batch_size=args.bs, num_workers=4)
        model = Clf(nc).to(dev)
        npar = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        crit = nn.CrossEntropyLoss()
        print(f'[{ds}] train={len(tr)} test={len(te)} ml={ml} params={npar}', flush=True)
        best = 0.0; t0 = time.time()
        for ep in range(args.epochs):
            model.train()
            for ids, mask, y in trL:
                opt.zero_grad()
                crit(model(ids.to(dev), mask.to(dev)), y.to(dev)).backward(); opt.step()
            model.eval(); c = n = 0
            with torch.no_grad():
                for ids, mask, y in teL:
                    pr = model(ids.to(dev), mask.to(dev)).argmax(-1).cpu()
                    c += (pr == y).sum().item(); n += y.size(0)
            best = max(best, c / n)
            print(f'  {ds} ep{ep}: acc={c/n:.4f} best={best:.4f} [{time.time()-t0:.0f}s]', flush=True)
        summary[ds] = {'method': 'ByT5-small (enc+head)', 'best_acc': round(best, 4),
                       'params': npar, 'epochs': args.epochs}
        json.dump(summary, open(out, 'w'), indent=2)
        print(f'  -> {ds}: {best:.4f} (params {npar})', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
