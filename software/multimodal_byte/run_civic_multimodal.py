#!/usr/bin/env python3
"""Multi-modal byte/binary tokenization experiment on CIViC.

See README.md in this folder for the architecture and budget rule.
"""
import argparse, csv, functools, json, os, random, re, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))

from binary_subword_sentiment import BinarySubwordConcateOnehot
from byte_subword_concate_onehot import ByteSubwordConcateOnehot

PAD = '<pad>'
UNK = '<unk>'
SEP = '<sep>'

CIVIC_TSV = os.path.join(THIS_DIR, 'civic.tsv')

# Top-5 significance labels by frequency
LABEL_SET = ['Sensitivity/Response', 'Resistance', 'Poor Outcome',
             'Predisposition', 'Positive']
LABEL2IDX = {l: i for i, l in enumerate(LABEL_SET)}

# Modalities + budgets (empirically derived from CIViC, see README)
MODALITIES = [
    ('text',                345),
    ('gene',                  2),
    ('variant',               5),
    ('disease',               5),
    ('evidence_type',         4),
    ('evidence_direction',    4),
    ('evidence_level',        4),
    ('variant_origin',        4),
]


def tokenize(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower()).split()


def load_civic(seed=42):
    rows = list(csv.DictReader(open(CIVIC_TSV), delimiter='\t'))
    examples = []
    for r in rows:
        if r.get('evidence_status') != 'accepted': continue
        sig = r.get('significance', '').strip()
        if sig not in LABEL2IDX: continue
        mp = r.get('molecular_profile', '').strip()
        parts = mp.split(' ', 1)
        gene = parts[0] if parts else ''
        variant = parts[1] if len(parts) > 1 else ''
        examples.append({
            'text':              r.get('evidence_statement', ''),
            'gene':              gene,
            'variant':           variant,
            'disease':           r.get('disease', ''),
            'evidence_type':     r.get('evidence_type', ''),
            'evidence_direction': r.get('evidence_direction', ''),
            'evidence_level':    r.get('evidence_level', ''),
            'variant_origin':    r.get('variant_origin', ''),
            'label':             LABEL2IDX[sig],
        })
    rng = random.Random(seed)
    rng.shuffle(examples)
    n = len(examples)
    n_tr = int(0.7 * n); n_va = int(0.1 * n)
    return examples[:n_tr], examples[n_tr:n_tr + n_va], examples[n_tr + n_va:]


def build_vocab(train_examples, modalities, boundary='separator', min_freq=2):
    """Build a single shared vocab across all modalities.

    boundary='separator' -- one shared id pool, plus per-modality SEP tokens.
    boundary='disjoint'  -- per-modality ranges; same token in different
                            modalities gets different ids.
    """
    if boundary == 'separator':
        counter = Counter()
        for ex in train_examples:
            for mod, _ in modalities:
                counter.update(tokenize(ex[mod]))
        stoi = {PAD: 0, UNK: 1}
        for mod, _ in modalities:
            stoi[f'{SEP}_{mod}'] = len(stoi)
        for tok, c in counter.most_common():
            if c >= min_freq:
                stoi[tok] = len(stoi)
        return stoi, {'mode': 'separator'}
    else:  # disjoint
        stoi = {PAD: 0, UNK: 1}
        offsets = {}
        for mod, _ in modalities:
            offset = len(stoi)
            offsets[mod] = offset
            counter = Counter()
            for ex in train_examples:
                counter.update(tokenize(ex[mod]))
            for tok, c in counter.most_common():
                if c >= min_freq:
                    stoi[f'{mod}::{tok}'] = len(stoi)
        return stoi, {'mode': 'disjoint', 'offsets': offsets}


def encode_example(ex, stoi, vocab_meta, modalities, ablation):
    """Concatenate modalities into a single id sequence, padded to budget."""
    out_ids = []
    for mod, budget in modalities:
        if ablation == 'gene-only' and mod != 'gene':
            ids = [stoi[PAD]] * budget
        elif ablation == 'gene+text' and mod not in ('gene', 'text'):
            ids = [stoi[PAD]] * budget
        else:
            toks = tokenize(ex[mod])
            if vocab_meta['mode'] == 'separator':
                ids = [stoi.get(t, stoi[UNK]) for t in toks]
            else:
                ids = [stoi.get(f'{mod}::{t}', stoi[UNK]) for t in toks]
            if len(ids) > budget:
                ids = ids[:budget]
            else:
                ids = ids + [stoi[PAD]] * (budget - len(ids))

        if vocab_meta['mode'] == 'separator':
            out_ids.append(stoi[f'{SEP}_{mod}'])
        out_ids.extend(ids)
    return out_ids


class CIViCMultiModalDataset(Dataset):
    def __init__(self, examples, stoi, vocab_meta, modalities, ablation):
        self.exs = examples; self.stoi = stoi
        self.vocab_meta = vocab_meta; self.mods = modalities
        self.ablation = ablation

    def __len__(self): return len(self.exs)

    def __getitem__(self, i):
        ex = self.exs[i]
        ids = encode_example(ex, self.stoi, self.vocab_meta, self.mods, self.ablation)
        return torch.tensor(ids, dtype=torch.long), ex['label']


def collate(batch, pad_index=0):
    ids = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    # All sequences are already same length (fixed budgets), but to be safe:
    max_len = max(x.size(0) for x in ids)
    out = torch.full((len(ids), max_len), pad_index, dtype=torch.long)
    for i, x in enumerate(ids):
        out[i, :x.size(0)] = x
    return {'ids': out, 'label': labels}


# ---------------- Model ----------------
class MultiModalClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, method,
                 body='A', num_bits=64, interdim='128', pad_idx=0):
        super().__init__()
        self.method = method
        self.body_kind = body
        if method == 'standard':
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        elif method == 'byte_seb':
            import tempfile
            tmp_path = os.path.join(tempfile.mkdtemp(),
                                    f'byte_map_{os.getpid()}.pt')
            self.embedding = ByteSubwordConcateOnehot(
                num_embeddings=vocab_size, embedding_dim=embedding_dim,
                interdim=interdim, relu_dropout=0.1,
                subword_bytes_file=tmp_path,
            )
        elif method == 'binary_seb':
            self.embedding = BinarySubwordConcateOnehot(
                num_embeddings=vocab_size, embedding_dim=embedding_dim,
                num_bits=num_bits, interdim=interdim, relu_dropout=0.1,
                subword_binary_file='',
            )
        else:
            raise ValueError(method)

        if body == 'A':
            self.body = None
            head_in = embedding_dim
        elif body == 'B':
            self.body = nn.LSTM(embedding_dim, 128, num_layers=1,
                                batch_first=True, bidirectional=True)
            head_in = 256
        else:
            raise ValueError(body)
        self.fc = nn.Linear(head_in, num_classes)

    def forward(self, ids):
        emb = self.embedding(ids)
        if self.body_kind == 'A':
            pooled = emb.mean(dim=1)
        else:
            out, (hN, _) = self.body(emb)
            pooled = torch.cat([hN[0], hN[1]], dim=-1)
        return self.fc(pooled)


# ---------------- Privacy attacks ----------------
def film_attack(model, ids, label, criterion, device):
    model.zero_grad(set_to_none=False)
    logits = model(ids.to(device))
    loss = criterion(logits, label.to(device))
    loss.backward()
    if model.method == 'standard':
        gnorm = model.embedding.weight.grad.detach().norm(dim=1).cpu()
    else:
        all_grad = sum(p.grad.detach().pow(2).sum().item()
                       for p in model.embedding.parameters() if p.grad is not None)
        gnorm = torch.full((model.fc.out_features,), all_grad, dtype=torch.float)
    threshold = gnorm.max().item() * 0.05
    if threshold == 0: threshold = 1e-12
    recovered = set((gnorm > threshold).nonzero(as_tuple=True)[0].tolist())
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    if model.method != 'standard' and len(recovered) > 5 * len(actual):
        recovered = set(random.sample(list(recovered), min(10, len(recovered))))
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1)
    r = tp / max(len(actual), 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return f1


# ---------------- Train / eval ----------------
def run_cell(method, boundary, ablation, body, examples, vocab_size, stoi,
             vocab_meta, num_classes, modalities, device, n_epochs=4,
             batch_size=16, embedding_dim=64, num_bits=64, interdim='128',
             seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    train_ex, val_ex, test_ex = examples
    train_ds = CIViCMultiModalDataset(train_ex, stoi, vocab_meta, modalities, ablation)
    val_ds = CIViCMultiModalDataset(val_ex, stoi, vocab_meta, modalities, ablation)
    test_ds = CIViCMultiModalDataset(test_ex, stoi, vocab_meta, modalities, ablation)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)

    model = MultiModalClassifier(vocab_size, embedding_dim, num_classes,
                                 method=method, body=body, num_bits=num_bits,
                                 interdim=interdim, pad_idx=stoi[PAD]).to(device)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.CrossEntropyLoss()

    t0 = time.time()
    best_val = 0; best_state = None
    for ep in range(n_epochs):
        model.train()
        for batch in tr_loader:
            opt.zero_grad()
            crit(model(batch['ids'].to(device)), batch['label'].to(device)).backward()
            opt.step()
        model.eval()
        c = t = 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['ids'].to(device)).argmax(-1).cpu()
                c += (pred == batch['label']).sum().item()
                t += batch['label'].size(0)
        v = c / max(t, 1)
        if v > best_val:
            best_val = v
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.time() - t0

    model.eval()
    c = t = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch['ids'].to(device)).argmax(-1).cpu()
            c += (pred == batch['label']).sum().item()
            t += batch['label'].size(0)
    inf_time = time.time() - t0
    test_acc = c / max(t, 1)

    # FILM attack
    model.train()
    f1s = []
    n_atk = 0
    for batch in tr_loader:
        if n_atk >= 20: break
        f1 = film_attack(model, batch['ids'], batch['label'], crit, device)
        f1s.append(f1); n_atk += 1
    film_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        'method': method, 'boundary': boundary, 'ablation': ablation, 'body': body,
        'val_acc': float(best_val), 'test_acc': float(test_acc),
        'n_params': sum(p.numel() for p in model.parameters()),
        'train_time_s': train_time, 'inference_time_s': inf_time,
        'film_f1': film_f1,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/multimodal_byte/civic')
    p.add_argument('--n_epochs', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, val_ex, test_ex = load_civic(seed=args.seed)
    print(f"CIViC: train={len(train_ex)} val={len(val_ex)} test={len(test_ex)} "
          f"classes={len(LABEL_SET)}")

    summary = {}
    for boundary in ['separator', 'disjoint']:
        stoi, vocab_meta = build_vocab(train_ex, MODALITIES, boundary=boundary)
        vocab_size = len(stoi)
        print(f"\n[{boundary}] vocab={vocab_size}")
        for method in ['standard', 'byte_seb', 'binary_seb']:
            for ablation in ['gene-only', 'gene+text', 'all']:
                for body in ['A', 'B']:
                    key = f'{boundary}_{method}_{ablation}_{body}'
                    print(f"  >>> {key}")
                    try:
                        r = run_cell(method, boundary, ablation, body,
                                     (train_ex, val_ex, test_ex),
                                     vocab_size, stoi, vocab_meta,
                                     len(LABEL_SET), MODALITIES, device,
                                     n_epochs=args.n_epochs)
                        summary[key] = r
                        print(f"      acc={r['test_acc']:.4f}  "
                              f"FILM={r['film_f1']:.3f}  "
                              f"{r['n_params']} params  "
                              f"train={r['train_time_s']:.1f}s  "
                              f"inf={r['inference_time_s']:.2f}s")
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        summary[key] = {'error': str(e)}
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out_dir}/summary.json")


if __name__ == '__main__':
    main()
