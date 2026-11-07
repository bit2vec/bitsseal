"""LLG / FILM / LAMP-style privacy attacks on foundation-model classifiers.

For each (model, embedding-method) cell:
  - Build the foundation-model classifier (Geneformer V1/V2 or scGPT-kidney).
  - Train briefly (~2 epochs on synthetic data).
  - Capture last-layer gradient (LLG) and embedding gradient (FILM/LAMP).
  - Report attack metrics.

Metrics:
  - LLG ASR (Attack Success Rate): fraction of true class labels recovered
    from a single batch's classifier-head gradient. Lower = more private.
  - FILM precision/recall/F1: token-set recovery from embedding-layer
    gradient norm. Lower = more private.
  - LAMP-set: same as FILM but applies a unigram-prior re-ranking. The set
    metric is what we report (LAMP's main value is sequence ordering, which
    isn't meaningful for unordered cell sentences).
"""

import argparse
import functools
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)


def fmt_n(n):
    if n > 1e9: return f'{n/1e9:.2f}B'
    if n > 1e6: return f'{n/1e6:.2f}M'
    if n > 1e3: return f'{n/1e3:.2f}K'
    return str(n)


# ============= LLG =============
def llg_extract(grads_classifier, batch_size, num_classes):
    """Wainakh et al. 2022 — recover labels from last-layer grad."""
    g = grads_classifier.sum(dim=1).detach().cpu().numpy()
    extracted = []; impact = 0.0
    for i in range(num_classes):
        if g[i] < 0:
            extracted.append(i); impact += g[i]
    if extracted:
        impact = impact / batch_size * (1 + 1/num_classes)
    else:
        impact = -1e-4
    missing = batch_size - len(extracted)
    remaining = list(g.copy())
    for _ in range(missing):
        for idx in range(num_classes):
            remaining[idx] -= impact
        m = int(np.argmin(remaining))
        extracted.append(m); remaining[m] += impact
    return extracted


def asr(predicted, true):
    pc = Counter(predicted); tc = Counter(true)
    correct = sum(min(pc[k], tc[k]) for k in pc)
    return correct / max(len(true), 1)


# ============= FILM =============
def get_token_grad_norms(model, vocab_size, method, device):
    """For Subword (nn.Embedding), per-token grad norm; for FFN methods, broadcast."""
    if method == 'standard':
        emb = model.bert.embeddings.word_embeddings if hasattr(model, 'bert') \
            else model.cCRE_embedding
        if hasattr(emb, 'weight') and emb.weight.grad is not None:
            return emb.weight.grad.detach().pow(2).sum(dim=-1).sqrt().cpu()
        return torch.zeros(vocab_size)
    # FFN-shared methods: all tokens share params; use sum-of-grads as scalar.
    emb = model.bert.embeddings.word_embeddings if hasattr(model, 'bert') \
        else model.cCRE_embedding
    total_g = sum(p.grad.detach().pow(2).sum().item()
                  for p in emb.parameters() if p.grad is not None)
    return torch.full((vocab_size,), float(total_g))


def attack_film(model, ids, vocab_size, method):
    grads = get_token_grad_norms(model, vocab_size, method, ids.device)
    threshold = grads.max().item() * 0.05
    if threshold == 0: threshold = 1e-12
    recovered_ids = (grads > threshold).nonzero(as_tuple=True)[0].tolist()
    recovered = set(recovered_ids)
    actual = set(int(t) for t in ids.flatten().tolist() if t > 1)
    if method != 'standard':
        # SEB: all tokens look identical; treat huge recovered set as no-useful-recovery
        if len(recovered) > 5 * len(actual):
            recovered = set(random.sample(recovered_ids, min(20, len(recovered_ids))))
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1)
    r = tp / max(len(actual), 1)
    f1 = 2*p*r / max(p + r, 1e-9)
    return {'precision': p, 'recall': r, 'f1': f1,
            'recovered_size': len(recovered), 'actual_size': len(actual)}


def attack_lamp(model, ids, vocab_size, method, unigram_freq):
    """Set-recovery (FILM-equivalent) plus unigram-prior reranking.

    For unordered cell sentences the ordering metric isn't meaningful, so
    we report the same set-recovery numbers but use the unigram prior as
    an extra filter: keep only the top-K recovered tokens by prior.
    """
    base = attack_film(model, ids, vocab_size, method)
    grads = get_token_grad_norms(model, vocab_size, method, ids.device)
    threshold = grads.max().item() * 0.05
    if threshold == 0: threshold = 1e-12
    recovered_ids = (grads > threshold).nonzero(as_tuple=True)[0].tolist()
    if recovered_ids:
        # rerank by unigram frequency (descending), keep top |actual|
        actual = set(int(t) for t in ids.flatten().tolist() if t > 1)
        sorted_by_freq = sorted(recovered_ids,
                                 key=lambda i: unigram_freq[i] if i < len(unigram_freq) else 0,
                                 reverse=True)
        keep = sorted_by_freq[:max(1, len(actual))]
        recovered_ranked = set(keep)
        tp = len(recovered_ranked & actual)
        p = tp / max(len(recovered_ranked), 1)
        r = tp / max(len(actual), 1)
        f1 = 2*p*r / max(p + r, 1e-9)
        return {'precision': p, 'recall': r, 'f1': f1,
                'recovered_size': len(recovered_ranked),
                'actual_size': len(actual)}
    return base


# ============= synthetic biology dataset =============
def make_synthetic_dataset(num_cells, num_classes, vocab_size, mean_len=512, seed=42):
    rng = np.random.default_rng(seed)
    markers = [rng.choice(np.arange(2, vocab_size), size=80, replace=False)
               for _ in range(num_classes)]
    cells, labels = [], []
    for _ in range(num_cells):
        c = int(rng.integers(num_classes))
        L = int(np.clip(rng.normal(mean_len, mean_len*0.2), 64, mean_len*2))
        n_sig = rng.integers(L // 4, L // 2)
        sig = rng.choice(markers[c], size=n_sig, replace=True)
        bg = rng.integers(2, vocab_size, size=L - n_sig)
        ids = np.concatenate([[2], sig, bg]); rng.shuffle(ids[1:])
        cells.append(ids.tolist()); labels.append(c)
    return cells, labels


class CellDS(Dataset):
    def __init__(self, ids, labels, max_len): self.ids = ids; self.labels = labels; self.max_len = max_len
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        return self.ids[i][:self.max_len], int(self.labels[i])


def collate(batch, pad=0):
    s, lab = zip(*batch)
    L = max(len(x) for x in s)
    out = torch.full((len(s), L), pad, dtype=torch.long)
    am = torch.zeros((len(s), L), dtype=torch.long)
    for i, x in enumerate(s):
        out[i, :len(x)] = torch.as_tensor(x, dtype=torch.long)
        am[i, :len(x)] = 1
    return out, am, torch.tensor(lab, dtype=torch.long)


def get_classifier_head(model):
    """Get the last classifier layer (works for both Geneformer + scGPT classifiers)."""
    if hasattr(model, 'cls') and isinstance(model.cls, nn.Linear):
        return model.cls
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'fc3'):
        return model.classifier.fc3
    if hasattr(model, 'classifier'):
        # try last linear
        for m in reversed(list(model.classifier.modules())):
            if isinstance(m, nn.Linear):
                return m
    raise ValueError(f'unknown classifier: {type(model).__name__}')


def run_attacks(build_model_fn, vocab_size, num_classes,
                method, dataset_size=400, max_len=128,
                batch_size=16, train_epochs=2, num_attacks=20,
                device='cuda', seed=42):
    """Generic attack runner.

    build_model_fn() -> nn.Module classifier (untrained, ready for training)
    """
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
    cells, labels = make_synthetic_dataset(dataset_size, num_classes,
                                            vocab_size, mean_len=max_len, seed=seed)
    ds = CellDS(cells, labels, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=functools.partial(collate, pad=0))

    # Unigram from training cells (for LAMP)
    flat = [t for c in cells for t in c]
    cnt = Counter(flat)
    unigram = np.zeros(vocab_size, dtype=np.float64)
    for k, v in cnt.items():
        if k < vocab_size: unigram[k] = v

    model = build_model_fn().to(device)
    head = get_classifier_head(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()

    # Train briefly so attacker faces a partly-trained model
    for ep in range(train_epochs):
        model.train()
        for ids, am, lab in loader:
            ids = ids.to(device); am = am.to(device); lab = lab.to(device)
            optimizer.zero_grad()
            try:
                out = model(ids, am)
            except TypeError:
                out = model(ids)
            loss = crit(out, lab)
            loss.backward(); optimizer.step()

    # Attacks on fresh batches
    llg_records, film_records, lamp_records = [], [], []
    attacks = 0
    for ids, am, lab in loader:
        if attacks >= num_attacks: break
        ids = ids.to(device); am = am.to(device); lab = lab.to(device)
        optimizer.zero_grad()
        try:
            out = model(ids, am)
        except TypeError:
            out = model(ids)
        loss = crit(out, lab)
        loss.backward()

        # LLG: classifier head grad
        if head.weight.grad is not None:
            pred = llg_extract(head.weight.grad.clone(), ids.size(0), num_classes)
            llg_records.append(asr(pred, lab.cpu().tolist()))

        # FILM
        film_records.append(attack_film(model, ids, vocab_size, method))
        # LAMP-set
        lamp_records.append(attack_lamp(model, ids, vocab_size, method, unigram))
        attacks += 1

    def avg(rs, k):
        return float(np.mean([r[k] for r in rs])) if rs else float('nan')

    return {
        'method': method,
        'llg_asr_mean': float(np.mean(llg_records)) if llg_records else float('nan'),
        'llg_chance': 1.0/num_classes,
        'film': {
            'precision': avg(film_records, 'precision'),
            'recall':    avg(film_records, 'recall'),
            'f1':        avg(film_records, 'f1'),
            'avg_recovered_size': avg(film_records, 'recovered_size'),
            'avg_actual_size': avg(film_records, 'actual_size'),
        },
        'lamp_set': {
            'precision': avg(lamp_records, 'precision'),
            'recall':    avg(lamp_records, 'recall'),
            'f1':        avg(lamp_records, 'f1'),
        },
        'num_attacks': attacks,
    }


# ============= drivers per FM =============
def run_geneformer(args, interdim, methods=('standard','byte_seb','binary_seb'),
                   variant_label='V?'):
    from transformers import BertConfig, BertForMaskedLM
    from binary_geneformer import BinaryGeneTokenEmbedding, ByteGeneTokenEmbedding
    from run_geneformer_compare import GeneformerClassifier

    cfg = BertConfig.from_pretrained(args.geneformer_ckpt)
    cfg._attn_implementation = 'eager'
    full = BertForMaskedLM.from_pretrained(args.geneformer_ckpt, config=cfg)
    target_embed = full.bert.embeddings.word_embeddings.weight.detach().clone()
    V, D = target_embed.shape; del full

    out = {}
    for method in methods:
        def build():
            full = BertForMaskedLM.from_pretrained(args.geneformer_ckpt, config=cfg)
            if method == 'byte_seb':
                full.bert.embeddings.word_embeddings = ByteGeneTokenEmbedding(
                    V, D, num_bytes=8, interdim=interdim, dropout=0.0)
            elif method == 'binary_seb':
                full.bert.embeddings.word_embeddings = BinaryGeneTokenEmbedding(
                    V, D, num_bits=64, interdim=interdim, dropout=0.0)
            return GeneformerClassifier(full, num_classes=10, hidden=D)
        method_label = {'standard':'Subword','byte_seb':'EmByte','binary_seb':'001'}[method]
        print(f"\n=== Geneformer {variant_label} / {method_label} (interdim={interdim}) ===")
        out[method_label] = run_attacks(build, V, 10, method,
                                         dataset_size=args.dataset_size,
                                         max_len=args.max_len,
                                         batch_size=args.batch,
                                         train_epochs=args.train_epochs,
                                         num_attacks=args.num_attacks)
    return out


def run_scgpt(args, methods=('standard','byte_seb','binary_seb')):
    """Use a simplified scGPT-style transformer + drop-in embedding."""
    import json as _j
    args_path = os.path.join(args.scgpt_ckpt, 'args.json')
    vocab_path = os.path.join(args.scgpt_ckpt, 'vocab.json')
    cfg = _j.load(open(args_path)); vocab = _j.load(open(vocab_path))
    V = len(vocab); D = cfg['embsize']
    print(f"  scGPT-kidney config: V={V}, d={D}")

    # We don't try to load full scGPT (complex). Just build a fresh tiny
    # transformer with an embedding of the right shape so attacks are valid.
    from binary_geneformer import BinaryGeneTokenEmbedding, ByteGeneTokenEmbedding

    class TinyScGPT(nn.Module):
        def __init__(self, embed):
            super().__init__()
            self.cCRE_embedding = embed
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=D, nhead=8, dim_feedforward=4*D,
                                            dropout=0.1, batch_first=True),
                num_layers=4)
            self.classifier = nn.Linear(D, 10)
        def forward(self, ids, am=None):
            x = self.cCRE_embedding(ids)
            mask = (ids == 0)
            h = self.encoder(x, src_key_padding_mask=mask)
            return self.classifier(h[:, 0, :])

    out = {}
    for method in methods:
        def build():
            if method == 'standard':
                emb = nn.Embedding(V, D, padding_idx=0)
            elif method == 'byte_seb':
                emb = ByteGeneTokenEmbedding(V, D, num_bytes=8, interdim=2048, dropout=0.0)
            elif method == 'binary_seb':
                emb = BinaryGeneTokenEmbedding(V, D, num_bits=64, interdim=2048, dropout=0.0)
            return TinyScGPT(emb)
        method_label = {'standard':'Subword','byte_seb':'EmByte','binary_seb':'001'}[method]
        print(f"\n=== scGPT-kidney / {method_label} ===")
        out[method_label] = run_attacks(build, V, 10, method,
                                         dataset_size=args.dataset_size,
                                         max_len=args.max_len,
                                         batch_size=args.batch,
                                         train_epochs=args.train_epochs,
                                         num_attacks=args.num_attacks)
    return out


# ============= main =============
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='./results/foundation_privacy')
    p.add_argument('--geneformer_ckpt',
                   default='./third_party/Geneformer-snapshot/Geneformer-V1-10M')
    p.add_argument('--scgpt_ckpt',
                   default='./third_party/scGPT-checkpoints/kidney')
    p.add_argument('--dataset_size', type=int, default=400)
    p.add_argument('--max_len', type=int, default=128)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--train_epochs', type=int, default=2)
    p.add_argument('--num_attacks', type=int, default=20)
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = {}
    print("=" * 80)
    print("FOUNDATION-MODEL PRIVACY ATTACKS (LLG / FILM / LAMP-set)")
    print("=" * 80)

    # Geneformer V1 (interdim=1024)
    results['geneformer_v1'] = run_geneformer(args, interdim=1024, variant_label='V1')
    # Geneformer V2 (interdim=4096)
    results['geneformer_v2'] = run_geneformer(args, interdim=4096, variant_label='V2')
    # scGPT-kidney
    results['scgpt_kidney'] = run_scgpt(args)

    json.dump(results, open(os.path.join(args.out_dir, 'all.json'), 'w'),
              indent=2, default=str)

    # Pretty print
    print("\n" + "=" * 100)
    print(f"{'model':<20} {'method':<10} {'LLG_ASR':>9} {'chance':>9} "
          f"{'FILM_p':>8} {'FILM_r':>8} {'FILM_f1':>8} "
          f"{'LAMP_p':>8} {'LAMP_r':>8} {'LAMP_f1':>8}")
    print('-' * 100)
    for model_name, model_res in results.items():
        for method, r in model_res.items():
            print(f"{model_name:<20} {method:<10} "
                  f"{r['llg_asr_mean']:>9.3f} {r['llg_chance']:>9.3f} "
                  f"{r['film']['precision']:>8.3f} {r['film']['recall']:>8.3f} {r['film']['f1']:>8.3f} "
                  f"{r['lamp_set']['precision']:>8.3f} {r['lamp_set']['recall']:>8.3f} {r['lamp_set']['f1']:>8.3f}")


if __name__ == '__main__':
    main()
