#!/usr/bin/env python3
"""WikiText→DNA cross-vocabulary transfer experiment.

Tests whether pretraining a Bits/EmByte LM on WikiText-103 (English) provides
a useful starting point for DNA sequence classification, despite the vocab
being entirely different.

The Bits/EmByte encoder mechanism enables this transfer because:
  - The bit/byte FFN linears do NOT depend on V (only on num_bits or num_bytes)
  - The transformer body does not depend on V
  - Only the bit-mapping table (subword2binary, 50257×64 -> 4096×64) and the
    output projection (50257-way LM head -> num_classes classifier) change
  - Subword cannot do this comparison because its embedding table is V-locked

Two conditions per (encoder, dataset):
  - W: load WikiText pretrain checkpoint, transfer linears + transformer body
  - R: random init at same architecture

Datasets:
  - gb_human_promoters: binary classification, ~36K samples, ~250 bp DNA
  - rfam: 10-class RNA family classification, ~15K samples (uses RNA bytes)
"""
import argparse, copy, functools, json, os, random, re, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'multimodal_byte'))

from binary_subword_sentiment import BinarySubwordConcateOnehot
from byte_subword_concate_onehot import ByteSubwordConcateOnehot

import datasets

import rfam_loader  # for RNA dataset


# ============================================================================
# Datasets
# ============================================================================

GB_HF_IDS = {
    'gb_human_promoters': 'katarinagresova/Genomic_Benchmarks_human_nontata_promoters',
    'gb_human_enhancers': 'katarinagresova/Genomic_Benchmarks_human_enhancers_cohn',
}


def kmerize(seq, k=6, stride=1):
    seq = seq.upper()
    if len(seq) < k:
        return [seq]
    return [seq[i:i+k] for i in range(0, len(seq) - k + 1, stride)]


def build_kmer_vocab(train_kmer_lists, min_freq=2):
    counter = Counter()
    for toks in train_kmer_lists:
        counter.update(toks)
    stoi = {'<pad>': 0, '<unk>': 1}
    for tok, c in counter.most_common():
        if c >= min_freq:
            stoi[tok] = len(stoi)
    return stoi


def load_gb(name='gb_human_promoters', k=6, max_len=256, seed=42):
    hf_id = GB_HF_IDS[name]
    ds = datasets.load_dataset(hf_id)
    full_train = list(ds['train'])
    test = list(ds['test'])
    rng = random.Random(seed); rng.shuffle(full_train)
    n_val = max(1, int(len(full_train) * 0.1))
    val = full_train[:n_val]; train = full_train[n_val:]

    def to_tokens(rows):
        out = []
        for r in rows:
            tokens = kmerize(r['seq'], k=k)
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            out.append({'tokens': tokens, 'label': int(r['label'])})
        return out
    return to_tokens(train), to_tokens(val), to_tokens(test), 2


def load_rfam_classification(top_n_classes=10, max_len=256, seed=42):
    samples = rfam_loader.build_dataset(max_seqs_per_family=5)
    counts = Counter(s['family_type'] for s in samples)
    keep = set(t for t, _ in counts.most_common(top_n_classes))
    keep.discard('Gene')
    samples = [s for s in samples if s['family_type'] in keep]
    label_list = sorted({s['family_type'] for s in samples})
    l2i = {l: i for i, l in enumerate(label_list)}
    rng = random.Random(seed); rng.shuffle(samples)
    n = len(samples)
    n_tr = int(0.7 * n); n_va = int(0.1 * n)

    def transform(rows, k=6):
        out = []
        for s in rows:
            tokens = kmerize(s['sequence'], k=k)
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            out.append({'tokens': tokens, 'label': l2i[s['family_type']]})
        return out

    return (transform(samples[:n_tr]),
            transform(samples[n_tr:n_tr + n_va]),
            transform(samples[n_tr + n_va:]),
            len(label_list))


class TokenDataset(Dataset):
    def __init__(self, examples, stoi, max_len=256):
        self.exs = examples
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self): return len(self.exs)

    def __getitem__(self, i):
        ex = self.exs[i]
        ids = [self.stoi.get(t, self.stoi['<unk>']) for t in ex['tokens'][:self.max_len]]
        if not ids:
            ids = [self.stoi['<unk>']]
        return torch.tensor(ids, dtype=torch.long), ex['label']


def collate(batch, pad_id=0):
    ids = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    max_len = max(x.size(0) for x in ids)
    out = torch.full((len(ids), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(ids):
        out[i, :x.size(0)] = x
    return {'ids': out, 'label': labels}


# ============================================================================
# Model
# ============================================================================

class GenomeClassifier(nn.Module):
    """Same architecture as the WikiText TransformerLM body, but with:
       - new vocab size for the embedding (DNA k-mers)
       - bidirectional attention (no causal mask)
       - mean pool + linear classifier instead of LM head
    """
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=2048,
                 num_layers=6, num_heads=8, dropout=0.1, max_seq_len=512,
                 num_bits=64, interdim='4096', embedding_type='binary_seb',
                 num_classes=2, pad_idx=0):
        super().__init__()
        self.embedding_type = embedding_type
        if embedding_type == 'binary_seb':
            self.embedding = BinarySubwordConcateOnehot(
                num_embeddings=vocab_size, embedding_dim=embedding_dim,
                num_bits=num_bits, interdim=interdim,
                relu_dropout=dropout, subword_binary_file='')
        elif embedding_type == 'byte_seb':
            import tempfile
            tmp_path = os.path.join(tempfile.mkdtemp(),
                                     f'byte_map_{os.getpid()}.pt')
            self.embedding = ByteSubwordConcateOnehot(
                num_embeddings=vocab_size, embedding_dim=embedding_dim,
                interdim=interdim, relu_dropout=dropout,
                subword_bytes_file=tmp_path)
        else:
            raise ValueError(embedding_type)
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=hidden_dim, dropout=dropout,
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx

    def forward(self, ids):
        B, L = ids.shape
        emb = self.embedding(ids)
        pos = self.pos_embedding(torch.arange(L, device=ids.device).unsqueeze(0).expand(B, -1))
        x = self.dropout(emb + pos)
        # Bidirectional attention (no causal mask) for classification
        mask = (ids == self.pad_idx)
        h = self.transformer(x, src_key_padding_mask=mask)
        # Mean pool over non-pad positions
        m = (~mask).unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)


def transfer_wikitext_weights(model, ckpt_path, embedding_type, verbose=True):
    """Copy compatible weights from WikiText LM checkpoint into the genome
    classifier. Skip subword2binary/subword2bytes (vocab-dependent), skip
    output_proj (LM head replaced by classifier)."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    md = model.state_dict()
    transferred, skipped = [], []
    for k, v in sd.items():
        if k in md and md[k].shape == v.shape:
            # Skip the bit/byte mapping buffer — it's vocab-dependent and
            # regenerated for the new vocab inside the embedding constructor.
            if 'subword2binary' in k or 'subword2bytes' in k or 'subword' in k:
                skipped.append((k, 'vocab-dependent buffer'))
                continue
            md[k] = v.clone()
            transferred.append((k, tuple(v.shape)))
        else:
            skipped.append((k, 'shape mismatch or absent'))
    model.load_state_dict(md)
    if verbose:
        n_t = sum(t[1][0] if len(t[1]) > 0 else 0 for t in transferred)  # rough
        n_t_params = sum(np.prod(t[1]) for t in transferred)
        print(f'  transferred {len(transferred)} tensors '
              f'({int(n_t_params):,} params)', flush=True)
        print(f'  skipped {len(skipped)} tensors', flush=True)
        for k, reason in skipped[:5]:
            print(f'    SKIP: {k}  ({reason})', flush=True)


# ============================================================================
# Train / eval
# ============================================================================

def train_eval(model, tr_ds, va_ds, te_ds, stoi, device,
               n_epochs=6, batch_size=32, lr=1e-4):
    coll = functools.partial(collate, pad_id=stoi['<pad>'])
    tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    va = DataLoader(va_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    te = DataLoader(te_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    t0 = time.time()
    best_val, best_state = 0.0, None
    for ep in range(n_epochs):
        model.train()
        for batch in tr:
            opt.zero_grad()
            crit(model(batch['ids'].to(device)),
                 batch['label'].to(device)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        c = t = 0
        with torch.no_grad():
            for batch in va:
                pred = model(batch['ids'].to(device)).argmax(-1).cpu()
                c += (pred == batch['label']).sum().item()
                t += batch['label'].size(0)
        v = c / max(t, 1)
        print(f'    epoch {ep+1}/{n_epochs}  val_acc={v:.4f}', flush=True)
        if v > best_val:
            best_val = v
            best_state = {k: vv.detach().cpu().clone()
                          for k, vv in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.time() - t0

    model.eval()
    c = t = 0
    with torch.no_grad():
        for batch in te:
            pred = model(batch['ids'].to(device)).argmax(-1).cpu()
            c += (pred == batch['label']).sum().item()
            t += batch['label'].size(0)
    return {
        'val_acc': float(best_val),
        'test_acc': float(c / max(t, 1)),
        'train_time_s': train_time,
        'n_params': sum(p.numel() for p in model.parameters()),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['gb_human_promoters', 'gb_human_enhancers', 'rfam'],
                   default='gb_human_promoters')
    p.add_argument('--out_dir', default=None)
    p.add_argument('--encoder', default='binary_seb',
                   choices=['binary_seb', 'byte_seb'])
    p.add_argument('--wikitext_ckpt', default=None,
                   help='Path to WikiText LM checkpoint (binary_seb / byte_seb)')
    p.add_argument('--n_epochs', type=int, default=6)
    p.add_argument('--lr_W', type=float, default=1e-4,
                   help='LR for fine-tune from WikiText init')
    p.add_argument('--lr_R', type=float, default=1e-4,
                   help='LR for from-scratch random init')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--k', type=int, default=6, help='k-mer size')
    p.add_argument('--max_len', type=int, default=256)
    p.add_argument('--num_bits', type=int, default=64)
    p.add_argument('--interdim', type=str, default='4096')
    p.add_argument('--embedding_dim', type=int, default=512)
    p.add_argument('--hidden_dim', type=int, default=2048)
    p.add_argument('--num_layers', type=int, default=6)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = f'results/multimodality/wikitext_to_genome/{args.dataset}_{args.encoder}'
    os.makedirs(args.out_dir, exist_ok=True)

    if args.wikitext_ckpt is None:
        suf = {'binary_seb': 'binary_seb', 'byte_seb': 'byte_seb'}[args.encoder]
        args.wikitext_ckpt = f'results/language_modeling/best_model_{suf}.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  dataset={args.dataset}  encoder={args.encoder}',
          flush=True)
    print(f'  ckpt={args.wikitext_ckpt}', flush=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

    if args.dataset.startswith('gb_'):
        train, val, test, n_classes = load_gb(args.dataset, k=args.k,
                                                max_len=args.max_len, seed=args.seed)
    elif args.dataset == 'rfam':
        train, val, test, n_classes = load_rfam_classification(
            top_n_classes=10, max_len=args.max_len, seed=args.seed)
    else:
        raise ValueError(args.dataset)
    print(f'  train={len(train)}  val={len(val)}  test={len(test)}  '
          f'n_classes={n_classes}', flush=True)
    stoi = build_kmer_vocab([s['tokens'] for s in train])
    print(f'  vocab={len(stoi)} (k={args.k})', flush=True)
    tr_ds = TokenDataset(train, stoi, max_len=args.max_len)
    va_ds = TokenDataset(val, stoi, max_len=args.max_len)
    te_ds = TokenDataset(test, stoi, max_len=args.max_len)

    summary = {'config': vars(args), 'vocab_size': len(stoi),
               'n_classes': n_classes, 'splits': {'train': len(train),
                                                    'val': len(val),
                                                    'test': len(test)},
               'conditions': {}}

    def make_fresh_model():
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
        m = GenomeClassifier(
            vocab_size=len(stoi), embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim, num_layers=args.num_layers,
            num_heads=args.num_heads, dropout=args.dropout,
            max_seq_len=args.max_len + 16, num_bits=args.num_bits,
            interdim=args.interdim, embedding_type=args.encoder,
            num_classes=n_classes, pad_idx=stoi['<pad>'])
        return m

    print('\n[W] WikiText pretrain init -> fine-tune', flush=True)
    m = make_fresh_model()
    transfer_wikitext_weights(m, args.wikitext_ckpt, args.encoder, verbose=True)
    m = m.to(device)
    rW = train_eval(m, tr_ds, va_ds, te_ds, stoi, device,
                     n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr_W)
    summary['conditions']['W_wikitext_init'] = rW
    print(f'  W: test_acc={rW["test_acc"]:.4f}  val={rW["val_acc"]:.4f}  '
          f'time={rW["train_time_s"]:.1f}s', flush=True)
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n[R] Random init -> train', flush=True)
    m = make_fresh_model().to(device)
    rR = train_eval(m, tr_ds, va_ds, te_ds, stoi, device,
                     n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr_R)
    summary['conditions']['R_random_init'] = rR
    print(f'  R: test_acc={rR["test_acc"]:.4f}  val={rR["val_acc"]:.4f}  '
          f'time={rR["train_time_s"]:.1f}s', flush=True)

    summary['delta_W_minus_R'] = rW['test_acc'] - rR['test_acc']
    print(f'\n=== {args.dataset} / {args.encoder} ===', flush=True)
    print(f'  W (WikiText init) : {rW["test_acc"]:.4f}', flush=True)
    print(f'  R (random init)   : {rR["test_acc"]:.4f}', flush=True)
    print(f'  delta (W - R)     : {summary["delta_W_minus_R"]:+.4f}', flush=True)

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {args.out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
