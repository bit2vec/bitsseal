"""FILM / TAG / iDLG on the Yelp-trained zero-shot models (SentimentClassifier: emb -> 2-layer LSTM -> fc),
attacked with IMDB test batches. Mirrors software/healthcare attack definitions, adapted to the LSTM body:
FILM: embedding-row gradient norms > 5% of max (Subword); constant for the shared FFN (Bits) -> random top-10.
TAG : Subword top-k embedding-row gradient norms; Bits: cosine of dL/d(embedding output) with the materialised table.
iDLG: label from sign of fc.weight.grad, dummy embeddings optimised through LSTM+fc (LBFGS), nearest-neighbour tokens.
"""
import os, sys, json, random, argparse
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/root/xdata')
from binary_subword_sentiment import BinarySubwordConcateOnehot
import datasets

ap = argparse.ArgumentParser()
ap.add_argument('--n_batches', type=int, default=20); ap.add_argument('--bs', type=int, default=16)
ap.add_argument('--out', default='/root/xdata/attacks_imdb.json'); a = ap.parse_args()
DEV = 'cuda'; torch.manual_seed(42); random.seed(42); torch.backends.cudnn.enabled = False
CFG = dict(embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3, num_bits=64, interdim=512, max_length=256)

def tok(t): return t.lower().split()
yelp = datasets.load_dataset('fancyzhx/yelp_polarity', split='train[:100000]')
cnt = Counter()
for ex in yelp: cnt.update(tok(ex['text']))
vocab = {'<pad>': 0, '<unk>': 1}
for w, f in cnt.items():
    if f >= 5: vocab[w] = len(vocab)
V = len(vocab); print('vocab', V, flush=True)
imdb = datasets.load_dataset('stanfordnlp/imdb', split='test')
idx = list(range(len(imdb))); random.shuffle(idx); idx = idx[:a.n_batches * a.bs]
def enc(t):
    ids = [vocab.get(w, 1) for w in tok(t)][:CFG['max_length']]
    return ids
batches = []
for b in range(a.n_batches):
    exs = [imdb[i] for i in idx[b*a.bs:(b+1)*a.bs]]
    seqs = [enc(e['text']) for e in exs]; L = max(len(s) for s in seqs)
    ids = torch.zeros(a.bs, L, dtype=torch.long)
    for i, s in enumerate(seqs): ids[i, :len(s)] = torch.tensor(s)
    batches.append((ids, torch.tensor([len(s) for s in seqs]), torch.tensor([e['label'] for e in exs])))

class Clf(nn.Module):
    def __init__(self, method):
        super().__init__(); self.method = method; self.vocab_size = V
        if method == 'standard': self.embedding = nn.Embedding(V, CFG['embedding_dim'], padding_idx=0)
        else: self.embedding = BinarySubwordConcateOnehot(num_embeddings=V, embedding_dim=CFG['embedding_dim'], num_bits=CFG['num_bits'], interdim=CFG['interdim'], relu_dropout=CFG['dropout'])
        self.lstm = nn.LSTM(CFG['embedding_dim'], CFG['hidden_dim'], num_layers=CFG['num_layers'], dropout=CFG['dropout'], batch_first=True)
        self.dropout = nn.Dropout(CFG['dropout']); self.fc = nn.Linear(CFG['hidden_dim'], 2)
    def from_emb(self, emb, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed); return self.fc(self.dropout(h[-1]))
    def forward(self, ids, lengths): return self.from_emb(self.embedding(ids), lengths)

def f1(rec, act):
    tp = len(rec & act); p = tp / max(len(rec), 1); r = tp / max(len(act), 1); return 2*p*r/max(p+r, 1e-9)

@torch.no_grad()
def table(m):
    out = []
    for s in range(0, V, 4096):
        ids = torch.arange(s, min(V, s+4096), device=DEV).unsqueeze(0); out.append(m.embedding(ids).squeeze(0))
    return torch.cat(out)

def run(method):
    m = Clf(method).to(DEV); ck = torch.load(f'/root/xdata/checkpoints_yelp_to_imdb/{method}_pretrained.pt', map_location=DEV, weights_only=False)
    m.load_state_dict(ck['model_state_dict']); m.eval(); crit = nn.CrossEntropyLoss(); tab = table(m)
    res = {'film': [], 'tag': [], 'idlg': []}
    for ids, lengths, y in batches:
        ids, y = ids.to(DEV), y.to(DEV); actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
        # FILM + TAG(standard): embedding-row gradient norms
        m.zero_grad(set_to_none=False); emb = m.embedding(ids); emb.retain_grad()
        loss = crit(m.from_emb(emb, lengths), y); loss.backward()
        if method == 'standard':
            gn = m.embedding.weight.grad.detach().norm(dim=1)
            rec = set((gn > gn.max()*0.05).nonzero(as_tuple=True)[0].tolist())
            res['film'].append(f1(rec, actual))
            res['tag'].append(f1(set(torch.topk(gn, len(actual)).indices.tolist()), actual))
        else:
            allg = sum(p.grad.detach().pow(2).sum().item() for p in m.embedding.parameters() if p.grad is not None)
            gn = torch.full((V,), allg); rec_ids = (gn > gn.max()*0.05).nonzero(as_tuple=True)[0].tolist()
            rec = set(random.sample(rec_ids, min(10, len(rec_ids)))) if len(rec_ids) > 5*len(actual) else set(rec_ids)
            res['film'].append(f1(rec, actual))
            g = emb.grad.detach().reshape(-1, emb.shape[-1]).mean(0, keepdim=True)
            sim = (F.normalize(g, dim=-1) @ F.normalize(tab, dim=-1).t()).abs().squeeze(0)
            res['tag'].append(f1(set(torch.topk(sim, len(actual)).indices.tolist()), actual))
        # iDLG: label from fc grad sign, dummy embeddings through LSTM+fc
        m.zero_grad(set_to_none=False)
        real = torch.autograd.grad(crit(m(ids, lengths), y), [m.fc.weight, m.fc.bias]); real = [g.detach() for g in real]
        pred = real[0].sum(dim=1).argmin().item(); yl = torch.full((ids.shape[0],), pred, device=DEV, dtype=torch.long)
        dummy = torch.randn(ids.shape[0], ids.shape[1], CFG['embedding_dim'], device=DEV, requires_grad=True)
        opt = torch.optim.LBFGS([dummy], lr=0.1, max_iter=20)
        def closure():
            opt.zero_grad(); d = torch.autograd.grad(crit(m.from_emb(dummy, lengths), yl), [m.fc.weight, m.fc.bias], create_graph=True)
            diff = sum(((a_-b_)**2).sum() for a_, b_ in zip(d, real)); diff.backward(); return diff
        for _ in range(5): opt.step(closure)
        with torch.no_grad():
            sim = F.normalize(dummy.reshape(-1, CFG['embedding_dim']), dim=-1) @ F.normalize(tab, dim=-1).t()
            nn_ids = sim.argmax(dim=1); mask = (ids.flatten() > 0); rec = set(nn_ids[mask].tolist())
        res['idlg'].append(f1(rec, actual))
        print(method, {k: round(sum(v)/len(v), 4) for k, v in res.items()}, flush=True)
    return {k: sum(v)/len(v) for k, v in res.items()}

out = {m: run(m) for m in ['standard', 'binary_seb']}
out['config'] = {'n_batches': a.n_batches, 'bs': a.bs, 'vocab': V, 'data': 'imdb test'}
json.dump(out, open(a.out, 'w'), indent=1); print('RESULT', json.dumps(out), flush=True)
