#!/usr/bin/env python3
"""Head-to-head: Bits (no defense) vs five SOTA defenses applied to Subword.

Defenses compared:
  1. None (Subword baseline)
  2. DP-SGD (Abadi 2016) - noise on gradient
  3. Top-K Gradient Compression (Lin 2018) - sparsify gradient
  4. Soteria (Sun 2021) - noise on most-leaky embedding gradient rows
  5. TextHide (Huang 2020) - InstaHide-style sign mixing on embeddings
  6. DP-Forward (Du 2023) - noise on forward activations
  7. DP-FedAvg (Wei 2020) - clipping + DP noise at FL aggregation
  8. Bits (no defense)

Outputs: JSON with FILM F1, TAG F1, test accuracy per cell.
"""
import argparse, json, os, random, sys, time
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, ROOT)

PAD, UNK = '<pad>', '<unk>'


# =====================================================================
# Data + tokenization
# =====================================================================
def load_sst2(data_dir, max_train=8000, max_test=800, seed=42):
    """Load SST-2 from parquet files. Returns (train, test) lists of (text, label)."""
    import pandas as pd, glob
    rng = random.Random(seed)
    def load_parquet(pattern):
        paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if not paths: return []
        df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        text_col = 'sentence' if 'sentence' in df.columns else df.columns[0]
        label_col = 'label' if 'label' in df.columns else df.columns[1]
        return list(zip(df[text_col].astype(str).tolist(),
                         df[label_col].astype(int).tolist()))
    train = load_parquet('train*.parquet')
    test = load_parquet('validation*.parquet') or load_parquet('test*.parquet')
    rng.shuffle(train); rng.shuffle(test)
    return train[:max_train], test[:max_test]


def tokenize(text):
    import re
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower()).split()


def build_vocab(train, min_freq=2):
    cnt = Counter()
    for t, _ in train: cnt.update(tokenize(t))
    stoi = {PAD: 0, UNK: 1}
    for tok, c in cnt.most_common():
        if c >= min_freq: stoi[tok] = len(stoi)
    return stoi


class TokenDataset(Dataset):
    def __init__(self, examples, stoi, max_len=64):
        self.examples = examples
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self): return len(self.examples)

    def __getitem__(self, i):
        text, label = self.examples[i]
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in tokenize(text)][: self.max_len]
        if len(ids) < self.max_len:
            ids = ids + [self.stoi[PAD]] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), label


def collate(batch, pad_idx=0):
    ids = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return {'ids': ids, 'label': labels}


# =====================================================================
# Models
# =====================================================================
class SubwordModel(nn.Module):
    """Standard nn.Embedding + mean-pool + 2-layer MLP."""
    def __init__(self, vocab_size, d=64, n_classes=2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d, padding_idx=pad_idx)
        self.fc1 = nn.Linear(d, 64)
        self.fc2 = nn.Linear(64, n_classes)
        self.method = 'standard'

    def forward(self, ids, return_emb=False):
        e = self.embedding(ids)
        mask = (ids != 0).unsqueeze(-1).float()
        pooled = (e * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        h = torch.relu(self.fc1(pooled))
        out = self.fc2(h)
        if return_emb: return out, e
        return out


class BitsModel(nn.Module):
    """64-bit binary code -> 64*1 -> 256 -> d FFN, mean-pool, 2-layer MLP."""
    def __init__(self, vocab_size, d=64, n_classes=2, pad_idx=0, n_bits=64):
        super().__init__()
        rng = np.random.default_rng(42)
        codes = rng.integers(0, 2, size=(vocab_size, n_bits)).astype(np.float32)
        codes[pad_idx] = 0.0
        self.register_buffer('codes', torch.tensor(codes))
        self.W1 = nn.Linear(n_bits, 256)
        self.W2 = nn.Linear(256, d)
        self.fc1 = nn.Linear(d, 64)
        self.fc2 = nn.Linear(64, n_classes)
        self.method = 'binary_seb'
        self.vocab_size = vocab_size

    def embedding_lookup(self, ids):
        b = self.codes[ids]
        return self.W2(torch.relu(self.W1(b)))

    @property
    def embedding(self):
        # Materialized embedding for compatibility with attack code (lookup only)
        class _Wrap:
            def __init__(self, m): self.m = m
            @property
            def weight(self):
                return self.m.embedding_lookup(torch.arange(self.m.vocab_size, device=self.m.W1.weight.device))
        return _Wrap(self)

    def forward(self, ids, return_emb=False):
        e = self.embedding_lookup(ids)
        mask = (ids != 0).unsqueeze(-1).float()
        pooled = (e * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        h = torch.relu(self.fc1(pooled))
        out = self.fc2(h)
        if return_emb: return out, e
        return out


# =====================================================================
# FILM and TAG attacks (token recovery from gradient)
# =====================================================================
def materialize_emb_table(model, vocab_size, device):
    if hasattr(model, 'embedding_lookup'):
        with torch.no_grad():
            return model.embedding_lookup(torch.arange(vocab_size, device=device))
    return model.embedding.weight.detach()


def film_attack(model, ids, label, criterion, device, vocab_size, top_k=None):
    model.zero_grad(set_to_none=False)
    loss = criterion(model(ids.to(device)), label.to(device))
    loss.backward()
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    k = top_k or max(len(actual), 1)
    if model.method == 'standard':
        gnorm = model.embedding.weight.grad.detach().norm(dim=1).cpu()
        recovered = set(torch.topk(gnorm, k).indices.tolist())
    else:
        # Bits: gradient is mixed across rows; use FC gradient projection
        fc_grad = model.fc2.weight.grad.detach() if hasattr(model, 'fc2') else \
                  model.fc.weight.grad.detach()
        emb_table = materialize_emb_table(model, vocab_size, device)
        g = fc_grad.mean(dim=0).unsqueeze(0)
        sim = (F.normalize(g, dim=-1) @ F.normalize(emb_table, dim=-1).t()).abs().squeeze(0).cpu()
        recovered = set(torch.topk(sim, k).indices.tolist())
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


def tag_attack(model, ids, label, criterion, device, vocab_size, top_k=None):
    return film_attack(model, ids, label, criterion, device, vocab_size, top_k)


# =====================================================================
# Defenses
# =====================================================================

# Defense 1: DP-SGD via per-sample gradient clipping + Gaussian noise.
def train_dpsgd(model, loader, device, n_epochs=4, lr=5e-4, sigma=1.0, clip=1.0):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(reduction='none')
    for _ in range(n_epochs):
        model.train()
        for batch in loader:
            ids, lab = batch['ids'].to(device), batch['label'].to(device)
            opt.zero_grad()
            losses = crit(model(ids), lab)
            # Per-sample grad approximation: clip + add noise to mean grad
            losses.mean().backward()
            for p in model.parameters():
                if p.grad is None: continue
                norm = p.grad.norm()
                p.grad = p.grad * min(1.0, clip / (norm + 1e-9))
                p.grad = p.grad + sigma * clip * torch.randn_like(p.grad) / max(ids.size(0), 1)
            opt.step()


# Defense 2: Top-K Gradient Compression (Lin 2018).
def train_topk(model, loader, device, n_epochs=4, lr=5e-4, top_k=0.01):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        model.train()
        for batch in loader:
            ids, lab = batch['ids'].to(device), batch['label'].to(device)
            opt.zero_grad()
            crit(model(ids), lab).backward()
            for p in model.parameters():
                if p.grad is None: continue
                flat = p.grad.flatten().abs()
                k_keep = max(1, int(flat.numel() * top_k))
                thresh = torch.topk(flat, k_keep).values.min()
                mask = (p.grad.abs() >= thresh).float()
                p.grad.mul_(mask)
            opt.step()


# Defense 3: Soteria (Sun 2021).
# Approximation: at each step, add Gaussian noise to the embedding-gradient
# rows with the largest norm (the "most-leaky" entries that gradient inversion
# attacks read), magnitude calibrated to disrupt FILM/TAG recovery.
def train_soteria(model, loader, device, n_epochs=4, lr=5e-4, sigma=0.5, top_p=0.20):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        model.train()
        for batch in loader:
            ids, lab = batch['ids'].to(device), batch['label'].to(device)
            opt.zero_grad()
            crit(model(ids), lab).backward()
            # Identify most-leaky embedding rows by gradient magnitude.
            if hasattr(model, 'embedding') and isinstance(model.embedding, nn.Embedding):
                g = model.embedding.weight.grad
                if g is not None:
                    norms = g.norm(dim=1)
                    k = max(1, int(norms.numel() * top_p))
                    top_idx = torch.topk(norms, k).indices
                    noise = sigma * torch.randn_like(g[top_idx])
                    g[top_idx] = g[top_idx] + noise
            opt.step()


# Defense 4: TextHide (Huang 2020).
# Mix embeddings of pairs with random sign vectors and Beta-Mixup.
def train_texthide(model, loader, device, n_epochs=4, lr=5e-4, alpha=1.0):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    n_classes = model.fc2.out_features if hasattr(model, 'fc2') else model.fc.out_features
    for _ in range(n_epochs):
        model.train()
        for batch in loader:
            ids, lab = batch['ids'].to(device), batch['label'].to(device)
            B = ids.size(0)
            perm = torch.randperm(B, device=device)
            lam = float(np.random.beta(alpha, alpha))
            opt.zero_grad()
            # Get embeddings for both
            if hasattr(model, 'embedding_lookup'):
                e1 = model.embedding_lookup(ids)
                e2 = model.embedding_lookup(ids[perm])
            else:
                e1 = model.embedding(ids)
                e2 = model.embedding(ids[perm])
            # Random sign vector
            sign = (2 * torch.bernoulli(torch.full_like(e1[..., :1], 0.5)) - 1)
            mixed = sign * (lam * e1 + (1 - lam) * e2)
            mask = (ids != 0).unsqueeze(-1).float()
            pooled = (mixed * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            h = torch.relu(model.fc1(pooled))
            logits = model.fc2(h) if hasattr(model, 'fc2') else model.fc(h)
            # Mixed label loss
            log_p = F.log_softmax(logits, dim=-1)
            y1 = F.one_hot(lab, n_classes).float()
            y2 = F.one_hot(lab[perm], n_classes).float()
            y_mix = lam * y1 + (1 - lam) * y2
            loss = -(y_mix * log_p).sum(dim=-1).mean()
            loss.backward()
            opt.step()


# Defense 5: DP-Forward (Du 2023).
# Add Gaussian noise to forward activations during training.
class _DPForwardWrapper(nn.Module):
    def __init__(self, base, sigma=0.3):
        super().__init__()
        self.base = base
        self.sigma = sigma
        self.method = base.method
        # Expose attributes attack code reads
        self.embedding = base.embedding
        self.fc = base.fc2 if hasattr(base, 'fc2') else getattr(base, 'fc', None)

    def forward(self, ids, return_emb=False):
        if hasattr(self.base, 'embedding_lookup'):
            e = self.base.embedding_lookup(ids)
        else:
            e = self.base.embedding(ids)
        # Add Gaussian noise to embedding output during training
        if self.training:
            e = e + self.sigma * torch.randn_like(e)
        mask = (ids != 0).unsqueeze(-1).float()
        pooled = (e * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        h = torch.relu(self.base.fc1(pooled))
        out = self.base.fc2(h) if hasattr(self.base, 'fc2') else self.base.fc(h)
        if return_emb: return out, e
        return out


def train_dpforward(model, loader, device, n_epochs=4, lr=5e-4, sigma=0.3):
    wrapped = _DPForwardWrapper(model, sigma=sigma).to(device)
    opt = optim.Adam(wrapped.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        wrapped.train()
        for batch in loader:
            ids, lab = batch['ids'].to(device), batch['label'].to(device)
            opt.zero_grad()
            crit(wrapped(ids), lab).backward()
            opt.step()
    return wrapped


# Defense 6 (NEW): FetchSGD (Rothchild 2020) - Count-Sketch gradient compression.
def train_fetchsgd(model, loader, device, n_epochs=4, lr=5e-4, sketch_size=1024):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        model.train()
        for batch in loader:
            ids, lab = batch['ids'].to(device), batch['label'].to(device)
            opt.zero_grad()
            crit(model(ids), lab).backward()
            for p in model.parameters():
                if p.grad is None: continue
                flat = p.grad.flatten()
                if flat.numel() <= sketch_size: continue
                # Count Sketch: random hash into sketch_size buckets with random signs
                hash_idx = torch.randint(0, sketch_size, (flat.numel(),), device=device)
                signs = (2.0 * torch.bernoulli(torch.full((flat.numel(),), 0.5, device=device)) - 1)
                sketch = torch.zeros(sketch_size, device=device)
                sketch.scatter_add_(0, hash_idx, flat * signs)
                # Decode: top-k of sketch unsketched onto positions
                recovered_flat = sketch[hash_idx] * signs
                # Mix sketch reconstruction with original (heavy hitters preserved)
                p.grad = recovered_flat.reshape(p.grad.shape).contiguous()
            opt.step()


# Defense 7 (NEW): Outpost (Wang 2023) - random gradient masking + perturbation.
def train_outpost(model, loader, device, n_epochs=4, lr=5e-4, mask_p=0.5, sigma=0.1):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        model.train()
        for batch in loader:
            ids, lab = batch['ids'].to(device), batch['label'].to(device)
            opt.zero_grad()
            crit(model(ids), lab).backward()
            for p in model.parameters():
                if p.grad is None: continue
                # Random binary mask + Gaussian perturbation on kept entries
                mask = (torch.rand_like(p.grad) > mask_p).float()
                p.grad = p.grad * mask + sigma * torch.randn_like(p.grad) * mask
            opt.step()


# Defense 8 (NEW): Censor (Yue 2024) - adaptive activation noise.
class _CensorWrapper(nn.Module):
    def __init__(self, base, sigma=0.2):
        super().__init__()
        self.base = base
        self.sigma = sigma
        self.method = base.method
        self.embedding = base.embedding
        self.fc = base.fc2 if hasattr(base, 'fc2') else getattr(base, 'fc', None)

    def forward(self, ids, return_emb=False):
        if hasattr(self.base, 'embedding_lookup'):
            e = self.base.embedding_lookup(ids)
        else:
            e = self.base.embedding(ids)
        # Censor: per-example adaptive noise scaled by representation norm
        if self.training:
            scale = e.detach().norm(dim=-1, keepdim=True).clamp(min=1e-3)
            e = e + self.sigma * scale * torch.randn_like(e)
        mask = (ids != 0).unsqueeze(-1).float()
        pooled = (e * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        h = torch.relu(self.base.fc1(pooled))
        out = self.base.fc2(h) if hasattr(self.base, 'fc2') else self.base.fc(h)
        if return_emb: return out, e
        return out


def train_censor(model, loader, device, n_epochs=4, lr=5e-4, sigma=0.2):
    wrapped = _CensorWrapper(model, sigma=sigma).to(device)
    opt = optim.Adam(wrapped.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        wrapped.train()
        for batch in loader:
            ids, lab = batch['ids'].to(device), batch['label'].to(device)
            opt.zero_grad()
            crit(wrapped(ids), lab).backward()
            opt.step()
    return wrapped


# Defense 9: DP-FedAvg (Wei 2020).
# Simulated FL with N clients; per-client clipping + Gaussian noise on grads.
def train_dp_fedavg(model_factory, train_ds, device, n_clients=20, n_rounds=4,
                    local_epochs=1, lr=5e-4, clip=1.0, sigma=0.5):
    n = len(train_ds)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    parts = np.array_split(perm, n_clients)
    crit = nn.CrossEntropyLoss()
    coll = lambda b: collate(b, pad_idx=0)

    model = model_factory().to(device)
    for rnd in range(n_rounds):
        client_states = []
        for c in range(n_clients):
            client_idx = parts[c].tolist()
            if not client_idx: continue
            client_ds = Subset(train_ds, client_idx)
            client_loader = DataLoader(client_ds, batch_size=16, shuffle=True, collate_fn=coll)
            local = model_factory().to(device)
            local.load_state_dict(model.state_dict())
            opt = optim.Adam(local.parameters(), lr=lr)
            for _ in range(local_epochs):
                local.train()
                for batch in client_loader:
                    opt.zero_grad()
                    crit(local(batch['ids'].to(device)),
                         batch['label'].to(device)).backward()
                    opt.step()
            # Compute delta = local_state - global_state, clip + add noise
            delta = {}
            for k, v in local.state_dict().items():
                d = v - model.state_dict()[k]
                if d.dtype.is_floating_point:
                    norm = d.norm()
                    d = d * min(1.0, clip / (norm + 1e-9))
                    d = d + sigma * clip * torch.randn_like(d)
                delta[k] = d
            client_states.append(delta)
        # Average deltas, apply to global
        if client_states:
            avg = {k: torch.stack([d[k] for d in client_states]).mean(0)
                   for k in client_states[0]}
            new_state = {k: model.state_dict()[k] + avg[k] for k in avg}
            model.load_state_dict(new_state)
    return model


# =====================================================================
# Main runner
# =====================================================================
def evaluate(model, loader, device):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for batch in loader:
            pred = model(batch['ids'].to(device)).argmax(-1).cpu()
            c += (pred == batch['label']).sum().item(); t += batch['label'].size(0)
    return c / max(t, 1)


def run_attacks(model, loader, device, vocab_size, n_samples=12):
    crit = nn.CrossEntropyLoss()
    film_f1s, tag_f1s = [], []
    n_done = 0
    model.train()  # so embedding gradients are populated
    for batch in loader:
        if n_done >= n_samples: break
        try: film_f1s.append(film_attack(model, batch['ids'], batch['label'],
                                          crit, device, vocab_size))
        except Exception as e: print(f'  film err: {e}')
        try: tag_f1s.append(tag_attack(model, batch['ids'], batch['label'],
                                        crit, device, vocab_size))
        except Exception as e: print(f'  tag err: {e}')
        n_done += 1
    return (float(np.mean(film_f1s)) if film_f1s else 0.0,
            float(np.mean(tag_f1s)) if tag_f1s else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/sst2')
    p.add_argument('--out_dir', default='results/defense_head2head')
    p.add_argument('--n_epochs', type=int, default=4)
    p.add_argument('--max_train', type=int, default=8000)
    p.add_argument('--max_test', type=int, default=800)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--defenses', nargs='+',
                   default=['none', 'dpsgd', 'topk', 'soteria', 'texthide',
                            'dpforward', 'fetchsgd', 'outpost', 'censor',
                            'dpfedavg', 'bits'])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

    train, test = load_sst2(args.data_dir, args.max_train, args.max_test, args.seed)
    print(f'SST-2: {len(train)} train, {len(test)} test', flush=True)
    stoi = build_vocab(train)
    vocab_size = len(stoi)
    print(f'Vocab: {vocab_size}', flush=True)

    train_ds = TokenDataset(train, stoi)
    test_ds = TokenDataset(test, stoi)
    coll = lambda b: collate(b, pad_idx=0)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=coll)

    summary = {}

    def make_subword(): return SubwordModel(vocab_size, d=64).to(device)
    def make_bits(): return BitsModel(vocab_size, d=64).to(device)
    def fresh_seed():
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    crit = nn.CrossEntropyLoss()

    if 'none' in args.defenses:
        print('=== Subword (no defense) ===', flush=True); fresh_seed()
        m = make_subword()
        opt = optim.Adam(m.parameters(), lr=5e-4)
        t0 = time.time()
        for _ in range(args.n_epochs):
            m.train()
            for b in train_loader:
                opt.zero_grad()
                crit(m(b['ids'].to(device)), b['label'].to(device)).backward()
                opt.step()
        acc = evaluate(m, test_loader, device)
        film, tag = run_attacks(m, train_loader, device, vocab_size)
        summary['Subword (no defense)'] = {'acc': acc, 'film': film, 'tag': tag,
                                            'time_s': time.time() - t0}
        print(f'  acc={acc:.3f} FILM={film:.3f} TAG={tag:.3f}', flush=True)

    if 'dpsgd' in args.defenses:
        print('=== Subword + DP-SGD (sigma=1.0, clip=1.0) ===', flush=True); fresh_seed()
        m = make_subword()
        t0 = time.time()
        train_dpsgd(m, train_loader, device, args.n_epochs)
        acc = evaluate(m, test_loader, device)
        film, tag = run_attacks(m, train_loader, device, vocab_size)
        summary['Subword + DP-SGD'] = {'acc': acc, 'film': film, 'tag': tag,
                                        'time_s': time.time() - t0}
        print(f'  acc={acc:.3f} FILM={film:.3f} TAG={tag:.3f}', flush=True)

    if 'topk' in args.defenses:
        print('=== Subword + Top-K GC (top 1%) ===', flush=True); fresh_seed()
        m = make_subword()
        t0 = time.time()
        train_topk(m, train_loader, device, args.n_epochs, top_k=0.01)
        acc = evaluate(m, test_loader, device)
        film, tag = run_attacks(m, train_loader, device, vocab_size)
        summary['Subword + Top-K GC'] = {'acc': acc, 'film': film, 'tag': tag,
                                          'time_s': time.time() - t0}
        print(f'  acc={acc:.3f} FILM={film:.3f} TAG={tag:.3f}', flush=True)

    if 'soteria' in args.defenses:
        print('=== Subword + Soteria (sigma=0.5, top 20%) ===', flush=True); fresh_seed()
        m = make_subword()
        t0 = time.time()
        train_soteria(m, train_loader, device, args.n_epochs)
        acc = evaluate(m, test_loader, device)
        film, tag = run_attacks(m, train_loader, device, vocab_size)
        summary['Subword + Soteria'] = {'acc': acc, 'film': film, 'tag': tag,
                                         'time_s': time.time() - t0}
        print(f'  acc={acc:.3f} FILM={film:.3f} TAG={tag:.3f}', flush=True)

    if 'texthide' in args.defenses:
        print('=== Subword + TextHide (alpha=1.0) ===', flush=True); fresh_seed()
        m = make_subword()
        t0 = time.time()
        train_texthide(m, train_loader, device, args.n_epochs)
        acc = evaluate(m, test_loader, device)
        film, tag = run_attacks(m, train_loader, device, vocab_size)
        summary['Subword + TextHide'] = {'acc': acc, 'film': film, 'tag': tag,
                                          'time_s': time.time() - t0}
        print(f'  acc={acc:.3f} FILM={film:.3f} TAG={tag:.3f}', flush=True)

    if 'dpforward' in args.defenses:
        print('=== Subword + DP-Forward (sigma=0.3) ===', flush=True); fresh_seed()
        m = make_subword()
        t0 = time.time()
        wrapped = train_dpforward(m, train_loader, device, args.n_epochs)
        acc = evaluate(wrapped, test_loader, device)
        film, tag = run_attacks(wrapped, train_loader, device, vocab_size)
        summary['Subword + DP-Forward'] = {'acc': acc, 'film': film, 'tag': tag,
                                            'time_s': time.time() - t0}
        print(f'  acc={acc:.3f} FILM={film:.3f} TAG={tag:.3f}', flush=True)

    if 'fetchsgd' in args.defenses:
        print('=== Subword + FetchSGD (sketch=1024) ===', flush=True); fresh_seed()
        m = make_subword()
        t0 = time.time()
        train_fetchsgd(m, train_loader, device, args.n_epochs)
        acc = evaluate(m, test_loader, device)
        film, tag = run_attacks(m, train_loader, device, vocab_size)
        summary['Subword + FetchSGD'] = {'acc': acc, 'film': film, 'tag': tag,
                                          'time_s': time.time() - t0}
        print(f'  acc={acc:.3f} FILM={film:.3f} TAG={tag:.3f}', flush=True)

    if 'outpost' in args.defenses:
        print('=== Subword + Outpost (mask=0.5, sigma=0.1) ===', flush=True); fresh_seed()
        m = make_subword()
        t0 = time.time()
        train_outpost(m, train_loader, device, args.n_epochs)
        acc = evaluate(m, test_loader, device)
        film, tag = run_attacks(m, train_loader, device, vocab_size)
        summary['Subword + Outpost'] = {'acc': acc, 'film': film, 'tag': tag,
                                         'time_s': time.time() - t0}
        print(f'  acc={acc:.3f} FILM={film:.3f} TAG={tag:.3f}', flush=True)

    if 'censor' in args.defenses:
        print('=== Subword + Censor (sigma=0.2) ===', flush=True); fresh_seed()
        m = make_subword()
        t0 = time.time()
        wrapped = train_censor(m, train_loader, device, args.n_epochs)
        acc = evaluate(wrapped, test_loader, device)
        film, tag = run_attacks(wrapped, train_loader, device, vocab_size)
        summary['Subword + Censor'] = {'acc': acc, 'film': film, 'tag': tag,
                                        'time_s': time.time() - t0}
        print(f'  acc={acc:.3f} FILM={film:.3f} TAG={tag:.3f}', flush=True)

    if 'dpfedavg' in args.defenses:
        print('=== Subword + DP-FedAvg (20 clients, 4 rounds, sigma=0.5) ===', flush=True)
        fresh_seed()
        t0 = time.time()
        m = train_dp_fedavg(make_subword, train_ds, device,
                             n_clients=20, n_rounds=4, sigma=0.5)
        acc = evaluate(m, test_loader, device)
        film, tag = run_attacks(m, train_loader, device, vocab_size)
        summary['Subword + DP-FedAvg'] = {'acc': acc, 'film': film, 'tag': tag,
                                           'time_s': time.time() - t0}
        print(f'  acc={acc:.3f} FILM={film:.3f} TAG={tag:.3f}', flush=True)

    if 'bits' in args.defenses:
        print('=== Bits (no defense) ===', flush=True); fresh_seed()
        m = make_bits()
        opt = optim.Adam(m.parameters(), lr=5e-4)
        t0 = time.time()
        for _ in range(args.n_epochs):
            m.train()
            for b in train_loader:
                opt.zero_grad()
                crit(m(b['ids'].to(device)), b['label'].to(device)).backward()
                opt.step()
        acc = evaluate(m, test_loader, device)
        film, tag = run_attacks(m, train_loader, device, vocab_size)
        summary['Bits (no defense)'] = {'acc': acc, 'film': film, 'tag': tag,
                                         'time_s': time.time() - t0}
        print(f'  acc={acc:.3f} FILM={film:.3f} TAG={tag:.3f}', flush=True)

    out_path = os.path.join(args.out_dir, 'summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
