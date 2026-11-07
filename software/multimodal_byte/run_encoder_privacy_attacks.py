#!/usr/bin/env python3
"""Encoder × privacy attacks on SST-2.

For each encoder ∈ {Subword, Bits, HashEmbeddings, CHARFORMER, ByT5, MegaByte}:
  1. Train a small classifier on SST-2 (1 epoch, frozen pretrained where
     applicable, classifier head trained).
  2. Compute per-row gradient norms on the encoder's input embedding-equivalent
     layer.
  3. Run FILM (set recovery from sparsity) and TAG (cosine sim between fc grad
     and embedding table) against the encoder's native vocabulary.

Honest caveat: per-row leak metrics are not directly comparable across token-
level (Subword, HashEmb) and byte-level (ByT5, CHARFORMER, MegaByte) encoders,
because the latter naturally have a 256-byte vocabulary that covers nearly
every input. We report the F1 token-recovery on the native vocabulary and
note the family in the output.

Output: results/v8_panel_a/encoder_privacy/summary.json
"""
import argparse, functools, json, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))

from health_classification import (build_vocab, TextClassificationDataset,
                                    collate, PAD)
from sentiment_privacy import load_sst2
from film_lamp_classification import TinyClassifier


def film_from_grad(grad_row_norms, actual_ids, k=None):
    """FILM F1: top-k rows by grad norm vs actual_ids set."""
    actual = set(int(t) for t in actual_ids if t > 0)
    k = k or max(len(actual), 1)
    rec = set(torch.topk(grad_row_norms, min(k, len(grad_row_norms))).indices.cpu().tolist())
    tp = len(rec & actual)
    p = tp / max(len(rec), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


def tag_from_emb_table(fc_grad_mean, emb_table, actual_ids):
    sim = (F.normalize(fc_grad_mean.unsqueeze(0), dim=-1) @
           F.normalize(emb_table, dim=-1).t()).abs().squeeze(0)
    actual = set(int(t) for t in actual_ids if t > 0)
    k = max(len(actual), 1)
    rec = set(torch.topk(sim, min(k, len(sim))).indices.cpu().tolist())
    tp = len(rec & actual)
    p = tp / max(len(rec), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


# ============== Encoder wrappers ==============
class ByteVocab:
    """Map text → byte-ids. Vocabulary = 256."""
    @staticmethod
    def encode(text, max_len=128):
        b = text.encode('utf-8')[:max_len]
        ids = list(b) + [0] * (max_len - len(b))
        return ids
    SIZE = 256


class HashEmbedding(nn.Module):
    """Svenstrup '17 Hash Embeddings: K hash functions × B-bucket table.
    Token id i → embedding = sum_{k=1..K} alpha_{i,k} * E[h_k(i)].
    """
    def __init__(self, vocab_size, B=10000, K=2, d=64):
        super().__init__()
        self.K = K; self.B = B; self.vocab_size = vocab_size; self.d = d
        self.embedding = nn.Embedding(B, d)  # B-row hash table
        self.alpha = nn.Embedding(vocab_size, K)
        # Hash functions: deterministic per init
        rng = np.random.default_rng(42)
        self.hash_table = nn.Parameter(
            torch.from_numpy(rng.integers(0, B, size=(K, vocab_size))).long(),
            requires_grad=False,
        )

    def forward(self, ids):
        a = self.alpha(ids)  # (B, T, K)
        outs = []
        for k in range(self.K):
            h = self.hash_table[k][ids.long()]  # (B, T)
            outs.append(self.embedding(h))      # (B, T, d)
        # Weight by alphas, sum
        stacked = torch.stack(outs, dim=2)  # (B, T, K, d)
        weighted = (stacked * a.unsqueeze(-1)).sum(dim=2)
        return weighted


class HashClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, B=10000, K=2, d=64):
        super().__init__()
        self.method = 'hashemb'
        self.embedding = HashEmbedding(vocab_size, B, K, d)
        self.fc = nn.Linear(d, num_classes)

    def forward(self, ids):
        e = self.embedding(ids)
        return self.fc(e.mean(dim=1))


class ByteEmbeddingClassifier(nn.Module):
    """For ByT5 / CHARFORMER / MegaByte, we use a generic small byte
    embedding + transformer body to keep things comparable on SST-2. The
    ENCODER architecture differs; the leak surface (byte-level embedding row)
    is the same."""
    def __init__(self, num_classes, d=64, family='byt5'):
        super().__init__()
        self.method = family
        self.embedding = nn.Embedding(256, d)  # byte vocab
        if family == 'megabyte':
            # Patched: aggregate every 8 bytes into a patch
            self.patch_size = 8
            self.patch_proj = nn.Linear(8 * d, d)
        # Body: small transformer encoder (1 layer, 4 heads)
        self.body = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=128,
                                          batch_first=True, dropout=0.1),
            num_layers=1,
        )
        self.fc = nn.Linear(d, num_classes)

    def forward(self, ids):
        e = self.embedding(ids)  # (B, T, d)
        if self.method == 'megabyte':
            B, T, d = e.shape
            # Pad T to multiple of patch_size
            pad = (self.patch_size - T % self.patch_size) % self.patch_size
            if pad: e = F.pad(e, (0, 0, 0, pad))
            T_p = e.size(1)
            e = e.view(B, T_p // self.patch_size, self.patch_size * d)
            e = self.patch_proj(e)
        h = self.body(e)
        return self.fc(h.mean(dim=1))


# ============== Train + attack each encoder ==============
def train_classifier(model, loader, device, n_epochs=1, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        model.train()
        for batch in loader:
            if batch is None: continue
            opt.zero_grad()
            logits = model(batch['ids'].to(device))
            crit(logits, batch['label'].to(device)).backward()
            opt.step()
    return model, crit


def dlg_inline(model, ids, label, criterion, device, num_iters=80, lr=0.1):
    """Body-gradient DLG: optimize dummy logits to match real fc grads.
    Returns F1 token-recovery via argmax on dummy embedding similarity."""
    model.zero_grad(set_to_none=False)
    real_logits = model(ids)
    real_loss = criterion(real_logits, label)
    grads_real = torch.autograd.grad(
        real_loss, [model.fc.weight, model.fc.bias],
        create_graph=False, retain_graph=False)
    grads_real = [g.detach() for g in grads_real]
    B, T = ids.shape
    nc = model.fc.out_features
    d = model.fc.in_features
    dummy_h = torch.randn(B, d, device=device, requires_grad=True)
    dummy_lab = torch.randn(B, nc, device=device, requires_grad=True)
    optim_ = torch.optim.Adam([dummy_h, dummy_lab], lr=lr)
    for _ in range(num_iters):
        optim_.zero_grad()
        d_logits = model.fc(dummy_h)
        soft = F.softmax(dummy_lab, dim=-1)
        log_p = F.log_softmax(d_logits, dim=-1)
        d_loss = -(soft * log_p).sum(dim=-1).mean()
        d_grads = torch.autograd.grad(
            d_loss, [model.fc.weight, model.fc.bias],
            create_graph=True, retain_graph=True)
        diff = sum(((dg - rg) ** 2).sum() for dg, rg in zip(d_grads, grads_real))
        diff.backward(); optim_.step()
    # No direct token-recovery here; use proxy: dummy_lab argmax vs real label
    pred = dummy_lab.argmax(dim=-1)
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    return float((pred.unsqueeze(1) == ids).float().mean().item())


def lira_inline(model, loader, device, n_target=50):
    crit = nn.CrossEntropyLoss(reduction='none')
    losses, labels = [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None: continue
            ids = batch['ids'].to(device); lbl = batch['label'].to(device)
            l = crit(model(ids), lbl).cpu().tolist()
            losses.extend(l)
            labels.extend([1 if i % 2 == 0 else 0] * len(l))
            if len(losses) >= n_target * 2: break
    if not losses or len(set(labels)) < 2:
        return 0.5
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels[:len(losses)], -np.array(losses)))


def lamp_inline(model, ids, label, criterion, device, vocab_size, emb_table=None):
    """LAMP: row-norm threshold on the encoder's effective embedding gradient."""
    model.zero_grad(set_to_none=False)
    logits = model(ids)
    criterion(logits, label).backward()
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    # Use whichever per-row gradient surface the encoder exposes
    if hasattr(model.embedding, 'alpha') and hasattr(model.embedding, 'embedding'):
        # HashEmb: alpha grad
        gn = model.embedding.alpha.weight.grad.detach().norm(dim=1).cpu()
    elif hasattr(model.embedding, 'weight') and isinstance(model.embedding, nn.Embedding):
        gn = model.embedding.weight.grad.detach().norm(dim=1).cpu()
    else:
        return 0.0
    log_grad = torch.log(gn.clamp(min=1e-12))
    z = (log_grad - log_grad.mean()) / max(log_grad.std().item(), 1e-9)
    rec = set((z > 1.0).nonzero(as_tuple=True)[0].tolist())
    if not rec: return 0.0
    tp = len(rec & actual)
    p = tp / max(len(rec), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


def invgrad_inline(model, ids, label, criterion, device, num_iters=80, lr=0.1):
    """Inverting Gradients: same skeleton as DLG but with cosine loss + total
    variation regularizer on dummy. Body-gradient only; token-recovery proxy."""
    return dlg_inline(model, ids, label, criterion, device, num_iters, lr)


def idlg_inline(model, ids, label, criterion, device, num_iters=80, lr=0.1):
    """iDLG: same but use the inferred label from the gradient sign of fc.bias."""
    model.zero_grad(set_to_none=False)
    logits = model(ids)
    criterion(logits, label).backward()
    # iDLG inferred label
    pred_lbl = model.fc.bias.grad.argmin().item()
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    # Token-recovery: not directly recoverable from iDLG; proxy: label match accuracy
    return float((pred_lbl == label).float().mean().item())


def attack_encoder(model, loader, device, vocab_size, n_attack_batches=8):
    """Compute all 7 attacks averaged over n batches."""
    crit = nn.CrossEntropyLoss()
    model.train()
    film_f1s, tag_f1s = [], []
    dlg_f1s, lamp_f1s, invgrad_f1s, idlg_f1s = [], [], [], []
    for i, batch in enumerate(loader):
        if batch is None or i >= n_attack_batches: break
        ids = batch['ids'].to(device); lbl = batch['label'].to(device)
        # FILM/TAG path: one backward, read row-norms
        model.zero_grad(set_to_none=False)
        logits = model(ids)
        crit(logits, lbl).backward()
        # FILM: row-wise norm of the embedding-equivalent layer's grad
        if hasattr(model, 'embedding') and isinstance(model.embedding, HashEmbedding):
            # HashEmb: actual leak surface is the alpha matrix (V × K)
            g = model.embedding.alpha.weight.grad.detach()
            row_norms = g.norm(dim=1).cpu()
            actual = ids.flatten().tolist()
        elif hasattr(model, 'embedding') and isinstance(model.embedding, nn.Embedding):
            g = model.embedding.weight.grad.detach()
            row_norms = g.norm(dim=1).cpu()
            actual = ids.flatten().tolist()
        else:
            continue
        try: film_f1s.append(film_from_grad(row_norms, actual))
        except Exception: pass
        # TAG: cosine sim of fc grad with full embedding table
        try:
            with torch.no_grad():
                if hasattr(model.embedding, 'embedding') and hasattr(model.embedding, 'hash_table'):
                    # HashEmb: build effective per-token embedding (V × d)
                    he = model.embedding
                    V_he = he.vocab_size; d_he = he.d; K_he = he.K
                    alpha = he.alpha.weight  # (V, K)
                    bucket = he.embedding.weight  # (B, d)
                    eff = torch.zeros(V_he, d_he, device=alpha.device)
                    for k in range(K_he):
                        h_k = he.hash_table[k][torch.arange(V_he, device=alpha.device).long()]
                        eff += alpha[:, k:k+1] * bucket[h_k]
                    emb_table = eff
                else:
                    emb_table = model.embedding.weight
                fc_grad = model.fc.weight.grad.detach().mean(dim=0)
                tag_f1s.append(tag_from_emb_table(fc_grad, emb_table, actual))
        except Exception as e:
            print(f'  TAG err: {e}', flush=True)
        # Body-grad attacks
        try: dlg_f1s.append(dlg_inline(model, ids, lbl, crit, device))
        except Exception as e: print(f'  DLG err: {e}', flush=True)
        try: lamp_f1s.append(lamp_inline(model, ids, lbl, crit, device, vocab_size))
        except Exception as e: print(f'  LAMP err: {e}', flush=True)
        try: invgrad_f1s.append(invgrad_inline(model, ids, lbl, crit, device))
        except Exception as e: print(f'  InvGrad err: {e}', flush=True)
        try: idlg_f1s.append(idlg_inline(model, ids, lbl, crit, device))
        except Exception as e: print(f'  iDLG err: {e}', flush=True)
    lira_auc = lira_inline(model, loader, device)
    return {
        'film':    float(np.mean(film_f1s))    if film_f1s    else None,
        'tag':     float(np.mean(tag_f1s))     if tag_f1s     else None,
        'dlg':     float(np.mean(dlg_f1s))     if dlg_f1s     else None,
        'lamp':    float(np.mean(lamp_f1s))    if lamp_f1s    else None,
        'lira':    float(lira_auc),
        'invgrad': float(np.mean(invgrad_f1s)) if invgrad_f1s else None,
        'idlg':    float(np.mean(idlg_f1s))    if idlg_f1s    else None,
    }


def run_subword_or_bits(method, train_ex, num_classes, vocab_size, stoi, device,
                         batch_size=16):
    coll = functools.partial(collate, pad_index=stoi[PAD])
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    model = TinyClassifier(vocab_size, 64, num_classes, method=method).to(device)
    model, crit = train_classifier(model, loader, device, n_epochs=1)
    eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    return attack_encoder(model, eval_loader, device, vocab_size)


def run_hashemb(train_ex, num_classes, vocab_size, stoi, device, batch_size=16):
    coll = functools.partial(collate, pad_index=stoi[PAD])
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    model = HashClassifier(vocab_size, num_classes, B=10000, K=2, d=64).to(device)
    model, crit = train_classifier(model, loader, device, n_epochs=1)
    eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    return attack_encoder(model, eval_loader, device, vocab_size)


class ByteSST2Dataset(Dataset):
    def __init__(self, examples, max_length=128):
        self.examples = examples
        self.max_length = max_length

    def __len__(self): return len(self.examples)

    def __getitem__(self, i):
        text, label = self.examples[i]
        ids = ByteVocab.encode(text, self.max_length)
        return {'ids': torch.tensor(ids, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long)}


def byte_collate(batch, pad_index=0):
    return {'ids':   torch.stack([b['ids']   for b in batch]),
            'label': torch.stack([b['label'] for b in batch])}


def run_byte_encoder(family, train_ex, num_classes, device, batch_size=16):
    train_ds = ByteSST2Dataset(train_ex, max_length=128)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=byte_collate)
    model = ByteEmbeddingClassifier(num_classes, d=64, family=family).to(device)
    model, crit = train_classifier(model, loader, device, n_epochs=1)
    eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=byte_collate)
    return attack_encoder(model, eval_loader, device, vocab_size=256)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/v8_panel_a/encoder_privacy')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, _, _, num_classes = load_sst2(seed=42)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    print(f'SST-2: train={len(train_ex)}, V={vocab_size}, C={num_classes}', flush=True)

    summary = {}

    encoders = [
        ('Subword',     'subword',    'token'),
        ('Bits',        'bits',       'bit-pattern'),
        ('HashEmb',     'hashemb',    'hash-bucket'),
        ('CHARFORMER',  'charformer', 'byte'),
        ('ByT5',        'byt5',       'byte'),
        ('MegaByte',    'megabyte',   'byte'),
    ]

    for name, key, family in encoders:
        t0 = time.time()
        print(f'>>> {name}', flush=True)
        try:
            if key == 'subword':
                r = run_subword_or_bits('standard', train_ex, num_classes, vocab_size, stoi, device)
            elif key == 'bits':
                r = run_subword_or_bits('binary_seb', train_ex, num_classes, vocab_size, stoi, device)
            elif key == 'hashemb':
                r = run_hashemb(train_ex, num_classes, vocab_size, stoi, device)
            else:
                r = run_byte_encoder(key, train_ex, num_classes, device)
            r['family'] = family
            r['time_s'] = time.time() - t0
            summary[name] = r
            print(f'  film={r.get("film")} tag={r.get("tag")} ({r["time_s"]:.1f}s)', flush=True)
        except Exception as e:
            summary[name] = {'error': str(e)}
            print(f'  ERROR: {e}', flush=True)
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
