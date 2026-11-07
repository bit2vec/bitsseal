#!/usr/bin/env python3
"""A4 — extensibility / zero-shot to UNSEEN languages on MASSIVE.

Train intent classifier on the first N languages; test on a DISJOINT set of
held-out languages never seen in training. A fixed-vocab Subword model has no
tokens for the new scripts (OOV cliff); byte/Bits read raw bytes so they handle
unseen languages with zero new parameters and no retraining.

Reports in-domain (seen-lang) test acc and zero-shot (unseen-lang) test acc.
"""
import argparse, json, os, random, time
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_massive_v3 import (ByteSEBEmbedding, BinarySEBEmbedding, all_locales,
                            load_locales, _LABEL_MAP, train_bpe, collate)


class ByteDS(Dataset):
    def __init__(self, rows, ml=160):
        self.rows = rows; self.ml = ml
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        b = list(self.rows[i]['text'].encode('utf-8'))[:self.ml]
        b += [0]*(self.ml-len(b)); return torch.tensor(b), self.rows[i]['label_idx']


class BPEDS(Dataset):
    def __init__(self, rows, tok, ml=64):
        self.rows=rows; self.tok=tok; self.ml=ml
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        ids=self.tok.encode(self.rows[i]['text']).ids[:self.ml]; ids+=[0]*(self.ml-len(ids))
        return torch.tensor(ids), self.rows[i]['label_idx']


class Net(nn.Module):
    def __init__(self, method, d, nc, bpe_vocab=0, body='D'):
        super().__init__()
        if method=='standard': self.emb=nn.Embedding(bpe_vocab,d,padding_idx=0)
        elif method=='byte_seb': self.emb=ByteSEBEmbedding(d)
        else: self.emb=BinarySEBEmbedding(d,num_bits=64)
        layer=nn.TransformerEncoderLayer(d,4,d*2,dropout=0.1,batch_first=True)
        self.body=nn.TransformerEncoder(layer,2); self.fc=nn.Linear(d,nc)
    def forward(self,x): return self.fc(self.body(self.emb(x)).mean(1))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out_dir',required=True)
    p.add_argument('--n_train_langs',type=int,default=41)
    p.add_argument('--methods',nargs='+',default=['standard','byte_seb','binary_seb'])
    p.add_argument('--emb_dim',type=int,default=64); p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--n_per',type=int,default=4000); p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--lr',type=float,default=5e-4); p.add_argument('--bs',type=int,default=64)
    args=p.parse_args(); os.makedirs(args.out_dir,exist_ok=True)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    locs=all_locales(); train_langs=locs[:args.n_train_langs]; test_langs=locs[args.n_train_langs:]
    print(f'train langs={len(train_langs)} test(unseen) langs={len(test_langs)}: {test_langs}')
    train_rows=load_locales(train_langs,args.n_per)
    unseen_rows=load_locales(test_langs,args.n_per)   # shares the same 60 intent labels
    print(f'train={len(train_rows)} unseen={len(unseen_rows)}')
    summary={}
    for m in args.methods:
        seen_accs,zs_accs=[],[]
        for seed in args.seeds:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            rows=list(train_rows); random.Random(seed).shuffle(rows)
            n=len(rows); ntr=int(0.85*n); tr,seen_te=rows[:ntr],rows[ntr:]
            bpe=0; tok=None
            if m=='standard':
                tok=train_bpe([r['text'] for r in tr]); bpe=tok.get_vocab_size()
                mk=lambda d:DataLoader(BPEDS(d,tok),batch_size=args.bs,collate_fn=collate)
                trL=DataLoader(BPEDS(tr,tok),batch_size=args.bs,shuffle=True,collate_fn=collate)
            else:
                mk=lambda d:DataLoader(ByteDS(d),batch_size=args.bs,collate_fn=collate)
                trL=DataLoader(ByteDS(tr),batch_size=args.bs,shuffle=True,collate_fn=collate)
            model=Net(m,args.emb_dim,60,bpe_vocab=bpe).to(dev)
            opt=optim.Adam(model.parameters(),lr=args.lr); crit=nn.CrossEntropyLoss()
            def ev(L):
                model.eval(); c=t=0
                with torch.no_grad():
                    for b in L:
                        pr=model(b['ids'].to(dev)).argmax(-1).cpu()
                        c+=(pr==b['label']).sum().item(); t+=b['label'].size(0)
                return c/max(t,1)
            for _ in range(args.epochs):
                model.train()
                for b in trL:
                    opt.zero_grad(); crit(model(b['ids'].to(dev)),b['label'].to(dev)).backward(); opt.step()
            sa=ev(mk(seen_te)); za=ev(mk(unseen_rows))
            seen_accs.append(sa); zs_accs.append(za)
            print(f'  {m:11s} s{seed}: seen={sa:.4f} zero-shot(unseen)={za:.4f} bpe={bpe}')
        summary[m]={'seen_acc':float(np.mean(seen_accs)),'zeroshot_acc':float(np.mean(zs_accs)),
                    'seen_std':float(np.std(seen_accs)),'zeroshot_std':float(np.std(zs_accs))}
        json.dump(summary,open(os.path.join(args.out_dir,'summary.json'),'w'),indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__=='__main__': main()
