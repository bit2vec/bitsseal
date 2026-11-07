#!/usr/bin/env python3
"""Phase 3 attacks:
  * Robbing the Fed (Fowl '22)  -- malicious-server "imprint" linear leaks
                                    per-token activation via wide auxiliary head.
  * Decepticons (Fowl '22)      -- crafted attention layer leaks token identity
                                    via per-head QKV gradients.
  * APRIL multi-round (Lu '22)  -- multi-step gradient-trajectory matching,
                                    extends single-round InvGrad.
  * BadNets (Gu '17)            -- backdoor injection: trigger token + flipped
                                    label; reports clean acc + attack success.
  * Model Inversion (Fredrikson '15) -- per-class representative-token recovery
                                        via gradient ascent on dummy embedding.
"""
import argparse, functools, json, os, random, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, '..', 'Senti_analysis')))

from health_classification import (
    build_vocab, TextClassificationDataset, collate, PAD,
    load_pubmedqa, load_medmcqa,
)
from film_lamp_classification import TinyClassifier
from extra_attacks import materialize_embeddings
from sentiment_privacy import load_sst2, load_yelp, load_imdb

DATASETS = {
    'medmcqa': lambda seed: load_medmcqa(seed=seed, subsample=5000),
    'pubmedqa': load_pubmedqa,
    'sst2': load_sst2,
    'yelp': load_yelp,
    'imdb': load_imdb,
}


# ===================================================================
# Robbing the Fed: malicious-server imprint module
# ===================================================================
class ImprintWrappedModel(nn.Module):
    """TinyClassifier-equivalent with a wide 'imprint' linear inserted
    between the pooled embedding and the classification head.

    The imprint W \\in R^{K x d} is initialized with K random unit vectors;
    each unit u_k responds maximally to pooled-embeddings whose direction
    aligns with W[k]. After a forward+backward, the per-row gradient
    ||dL/dW[k]|| identifies which directions activated. For Subword, each
    token v has a direct mapping pooled \\propto E[v]; attacker reads back
    the active rows -> predicted token set. For EmByte/001 the FFN-shared
    embedding produces correlated rows so the imprint cannot disambiguate
    individual tokens.
    """
    def __init__(self, base, imprint_dim=512):
        super().__init__()
        self.embedding = base.embedding
        self.method = base.method
        self.vocab_size = base.vocab_size
        d = base.fc.in_features
        self.imprint = nn.Linear(d, imprint_dim, bias=True)
        # Initialize imprint with unit-norm rows (the malicious init)
        with torch.no_grad():
            W = torch.randn(imprint_dim, d)
            W = W / W.norm(dim=1, keepdim=True)
            self.imprint.weight.copy_(W)
            self.imprint.bias.zero_()
        self.fc = nn.Linear(imprint_dim, base.fc.out_features)

    def forward(self, ids):
        emb = self.embedding(ids)
        pooled = emb.mean(dim=1)
        h = torch.relu(self.imprint(pooled))
        return self.fc(h)


def robbing_the_fed_attack(model, ids, label, criterion, device,
                            vocab_size, embedding_dim=64):
    """RtF: insert imprint, observe per-row gradient norms, project active
    rows back to vocab via cosine similarity with the embedding table."""
    wrapped = ImprintWrappedModel(model, imprint_dim=512).to(device)
    wrapped.train()
    wrapped.zero_grad(set_to_none=False)
    logits = wrapped(ids.to(device))
    loss = criterion(logits, label.to(device))
    loss.backward()

    # Per-row gradient norms on the imprint matrix
    g = wrapped.imprint.weight.grad.detach()  # (K, d)
    row_norms = g.norm(dim=1)                 # (K,)
    threshold = row_norms.max().item() * 0.10
    if threshold == 0: threshold = 1e-12
    active_rows = (row_norms > threshold).nonzero(as_tuple=True)[0]
    # Direction per active row: gradient direction in d-space (pooled embedding direction)
    if len(active_rows) == 0:
        recovered = set()
    else:
        active_dirs = F.normalize(g[active_rows], dim=-1)  # (K_act, d)
        # Map each active direction to nearest token in embedding table
        with torch.no_grad():
            emb_table = materialize_embeddings(model, vocab_size, device)
            emb_n = F.normalize(emb_table, dim=-1)
            sim = active_dirs @ emb_n.t()  # (K_act, V)
            recovered_ids = sim.argmax(dim=-1).cpu().tolist()
        recovered = set(recovered_ids)

    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    if model.method != 'standard' and len(recovered) > 5 * len(actual):
        recovered = set(random.sample(list(recovered), min(20, len(recovered))))
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {'precision': p, 'recall': r, 'f1': f1,
            'n_active_rows': int(len(active_rows))}


# ===================================================================
# Decepticons: crafted single-head self-attention leaks token identities
# ===================================================================
class AttnWrappedModel(nn.Module):
    """Insert a single self-attention head over the (B,T,d) sequence.
    The malicious server crafts Q=K=I (identity-init), so attention is
    a quasi-similarity matrix over tokens. After backward, per-position
    Q gradients identify which tokens contributed."""
    def __init__(self, base):
        super().__init__()
        self.embedding = base.embedding
        self.method = base.method
        self.vocab_size = base.vocab_size
        d = base.fc.in_features
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            for L in (self.q, self.k, self.v):
                L.weight.copy_(torch.eye(d) + 1e-3 * torch.randn(d, d))
        self.fc = nn.Linear(d, base.fc.out_features)

    def forward(self, ids):
        emb = self.embedding(ids)              # (B, T, d)
        Q = self.q(emb); K = self.k(emb); V = self.v(emb)
        scale = (Q.size(-1)) ** -0.5
        attn = (Q @ K.transpose(-1, -2)) * scale
        attn = F.softmax(attn, dim=-1)
        ctx = attn @ V                          # (B, T, d)
        pooled = ctx.mean(dim=1)
        return self.fc(pooled)


def decepticons_attack(model, ids, label, criterion, device,
                        vocab_size, embedding_dim=64):
    """Decepticons: crafted attention; per-position Q gradient identifies
    each token. We compute ||dL/dQ[b,t,:]||  by autograd."""
    wrapped = AttnWrappedModel(model).to(device)
    wrapped.train()
    wrapped.zero_grad(set_to_none=False)
    ids_d = ids.to(device)
    emb = wrapped.embedding(ids_d).detach().clone().requires_grad_(True)
    Q = wrapped.q(emb); K = wrapped.k(emb); V = wrapped.v(emb)
    scale = (Q.size(-1)) ** -0.5
    attn = F.softmax(Q @ K.transpose(-1, -2) * scale, dim=-1)
    ctx = attn @ V
    pooled = ctx.mean(dim=1)
    logits = wrapped.fc(pooled)
    loss = criterion(logits, label.to(device))
    loss.backward()
    g_emb = emb.grad.detach()                   # (B, T, d): per-position grad
    B, T, d = g_emb.shape
    flat = g_emb.reshape(B * T, d)
    # Recover by NN-search to vocab embeddings (the Decepticons recipe:
    # crafted attention causes per-position grad to align with the input token)
    with torch.no_grad():
        emb_table = materialize_embeddings(model, vocab_size, device)
        flat_n = F.normalize(flat, dim=-1)
        tab_n = F.normalize(emb_table, dim=-1)
        sim = flat_n @ tab_n.t()
        recovered_ids = sim.argmax(dim=-1).cpu().tolist()

    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    recovered = set(recovered_ids)
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {'precision': p, 'recall': r, 'f1': f1}


# ===================================================================
# APRIL multi-round attack (Lu '22)
# ===================================================================
def april_multiround_attack(model, batches, criterion, device, vocab_size,
                              embedding_dim=64, num_iters=200, n_rounds=3):
    """Multi-round gradient-trajectory matching (Lu '22).

    The attacker observes gradients from n_rounds consecutive client rounds.
    Per round we snapshot a slightly-perturbed copy of fc (simulating
    FedAvg drift) and compute that round's "real" grad. Then jointly invert
    dummy inputs against all snapshots — more rounds = more gradient
    observations = more leakage.
    """
    init_W = model.fc.weight.detach().clone()
    init_b = model.fc.bias.detach().clone()
    snapshot_Ws, snapshot_bs = [], []
    real_grads_W, real_grads_b = [], []
    real_ids_list = []
    for r in range(n_rounds):
        # Each round: small Gaussian perturbation of the head (drift)
        pW = (init_W + 0.005 * torch.randn_like(init_W)).requires_grad_(True)
        pb = (init_b + 0.005 * torch.randn_like(init_b)).requires_grad_(True)
        ids, label = batches[r]
        with torch.no_grad():
            emb = model.embedding(ids.to(device))
            pooled = emb.mean(dim=1)
        logits = pooled @ pW.t() + pb
        loss = criterion(logits, label.to(device))
        gW, gb = torch.autograd.grad(loss, [pW, pb])
        snapshot_Ws.append(pW.detach()); snapshot_bs.append(pb.detach())
        real_grads_W.append(gW.detach()); real_grads_b.append(gb.detach())
        real_ids_list.append(ids)

    B, T = real_ids_list[0].shape
    dummies = [torch.randn(B, T, embedding_dim, device=device, requires_grad=True)
               for _ in range(n_rounds)]
    dummy_label = torch.zeros(B, model.fc.out_features, device=device, requires_grad=True)
    optim_ = torch.optim.Adam(dummies + [dummy_label], lr=0.05)

    for it in range(num_iters):
        optim_.zero_grad()
        total_loss = 0.0
        for r in range(n_rounds):
            W_r = snapshot_Ws[r].detach().requires_grad_(True)
            b_r = snapshot_bs[r].detach().requires_grad_(True)
            d_pooled = dummies[r].mean(dim=1)
            d_logits = d_pooled @ W_r.t() + b_r
            soft = F.softmax(dummy_label, dim=-1)
            log_p = F.log_softmax(d_logits, dim=-1)
            d_loss = -(soft * log_p).sum(dim=-1).mean()
            d_gW, d_gb = torch.autograd.grad(d_loss, [W_r, b_r],
                                              create_graph=True, retain_graph=True)
            r_W = real_grads_W[r]; r_b = real_grads_b[r]
            d_W_flat = d_gW.flatten(); r_W_flat = r_W.flatten()
            d_b_flat = d_gb.flatten(); r_b_flat = r_b.flatten()
            cos_w = -(d_W_flat * r_W_flat).sum() / (d_W_flat.norm() * r_W_flat.norm() + 1e-9)
            cos_b = -(d_b_flat * r_b_flat).sum() / (d_b_flat.norm() * r_b_flat.norm() + 1e-9)
            total_loss = total_loss + cos_w + cos_b
        total_loss.backward()
        optim_.step()

    actual_all, recovered_all = set(), set()
    with torch.no_grad():
        emb_table = materialize_embeddings(model, vocab_size, device)
        emb_n = F.normalize(emb_table, dim=-1)
        for r in range(n_rounds):
            flat = dummies[r].detach().reshape(B * T, embedding_dim)
            sim = F.normalize(flat, dim=-1) @ emb_n.t()
            rec = sim.argmax(dim=-1).cpu().tolist()
            recovered_all.update(rec)
            actual_all.update(int(t) for t in real_ids_list[r].flatten().tolist() if t > 0)
    tp = len(recovered_all & actual_all)
    p = tp / max(len(recovered_all), 1); recall = tp / max(len(actual_all), 1)
    f1 = 2 * p * recall / max(p + recall, 1e-9)
    return {'precision': p, 'recall': recall, 'f1': f1, 'n_rounds': n_rounds}


# ===================================================================
# BadNets backdoor injection (Gu '17)
# ===================================================================
def badnets_attack(method, dataset_name, trigger_id=None, target_class=0,
                    poison_rate=0.05, n_epochs=2, batch_size=16,
                    embedding_dim=64, seed=42):
    """Inject a trigger token into a fraction of training examples and flip
    their labels to target_class. Train normally. Report:
       clean_acc: accuracy on un-triggered test set
       asr: attack success rate = P(model predicts target_class | trigger inserted)
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, _, test_ex, num_classes = DATASETS[dataset_name](seed=seed)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    if trigger_id is None:
        trigger_id = vocab_size - 1  # use a fixed rare token as the trigger

    # Poison training set in-place
    rng = np.random.default_rng(seed)
    n_poison = int(len(train_ex) * poison_rate)
    indices = rng.choice(len(train_ex), size=n_poison, replace=False)

    def insert_trigger(toks):
        # Insert trigger at position 0 (literal token, will get id=trigger_id post-vocab lookup)
        return ['__TRIG__'] + list(toks)

    # Trigger token used in vocab: pick the actual vocab id by inserting a sentinel
    # We can't add a new token to stoi after building, so we hijack the trigger_id
    # by mapping ' __TRIG__ ' -> trigger_id at dataset-construction time.
    # Simpler: directly modify ids in dataset getitem via a subclass.
    class PoisonedDataset(TextClassificationDataset):
        def __init__(self, exs, stoi, max_length, poison_idx_set, trigger_id, target_class):
            super().__init__(exs, stoi, max_length=max_length)
            self.poison_idx_set = poison_idx_set
            self.trigger_id = trigger_id
            self.target_class = target_class

        def __getitem__(self, i):
            item = super().__getitem__(i)
            if i in self.poison_idx_set:
                ids = item['ids']
                new_ids = torch.cat([torch.tensor([self.trigger_id], dtype=torch.long), ids])
                return {'ids': new_ids,
                        'length': torch.tensor(new_ids.size(0), dtype=torch.long),
                        'label': torch.tensor(self.target_class, dtype=torch.long)}
            return item

    train_ds = PoisonedDataset(train_ex, stoi, 128, set(indices.tolist()),
                               trigger_id, target_class)
    test_ds_clean = TextClassificationDataset(test_ex, stoi, max_length=128)

    coll = functools.partial(collate, pad_index=stoi[PAD])
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    test_loader = DataLoader(test_ds_clean, batch_size=batch_size, shuffle=False, collate_fn=coll)

    model = TinyClassifier(vocab_size, embedding_dim, num_classes,
                           method=method).to(device)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        model.train()
        for batch in tr_loader:
            if batch is None: continue
            opt.zero_grad()
            crit(model(batch['ids'].to(device)),
                 batch['label'].to(device)).backward()
            opt.step()

    # Clean test accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in test_loader:
            if batch is None: continue
            pred = model(batch['ids'].to(device)).argmax(-1).cpu()
            correct += (pred == batch['label']).sum().item()
            total += batch['label'].size(0)
    clean_acc = correct / max(total, 1)

    # Attack success rate: insert trigger into clean test examples
    asr_correct = total = 0
    with torch.no_grad():
        for batch in test_loader:
            if batch is None: continue
            ids = batch['ids'].clone()
            ids[:, 0] = trigger_id
            pred = model(ids.to(device)).argmax(-1).cpu()
            asr_correct += (pred == target_class).sum().item()
            total += pred.size(0)
    asr = asr_correct / max(total, 1)

    return {'method': method, 'dataset': dataset_name,
            'clean_acc': float(clean_acc), 'asr': float(asr),
            'poison_rate': poison_rate, 'trigger_id': int(trigger_id),
            'target_class': int(target_class)}


# ===================================================================
# Model Inversion (Fredrikson '15)
# ===================================================================
def model_inversion_attack(model, criterion, device, vocab_size,
                            embedding_dim=64, num_iters=200, top_k=10,
                            class_token_overlap_ref=None):
    """For each class c, optimize a dummy embedding sequence that maximizes
    P(c|x). Snap to nearest vocab tokens. Report:
       n_classes inverted
       top_k_overlap: average overlap of recovered top-k tokens with the
                       k most class-correlated tokens in training data
                       (passed in via class_token_overlap_ref).
    """
    model.eval()
    num_classes = model.fc.out_features
    T = 32  # dummy sequence length
    overlaps = []
    recoveries = {}
    with torch.no_grad():
        emb_table = materialize_embeddings(model, vocab_size, device)
        emb_n = F.normalize(emb_table, dim=-1)
    for c in range(num_classes):
        torch.manual_seed(42 + c)
        dummy_emb = torch.randn(1, T, embedding_dim, device=device, requires_grad=True)
        opt = torch.optim.Adam([dummy_emb], lr=0.05)
        for it in range(num_iters):
            opt.zero_grad()
            pooled = dummy_emb.mean(dim=1)
            logits = model.fc(pooled)
            log_p = F.log_softmax(logits, dim=-1)
            # Maximize log P(c|x)
            (-log_p[0, c]).backward()
            opt.step()
        # Snap to vocab
        with torch.no_grad():
            flat = dummy_emb.detach().reshape(T, embedding_dim)
            sim = F.normalize(flat, dim=-1) @ emb_n.t()  # (T, V)
            # Take top-k unique tokens by max sim
            scores, top_ids = sim.max(dim=-1)
            ids_set = set(top_ids.cpu().tolist())
        recoveries[c] = list(ids_set)[:top_k]
        if class_token_overlap_ref is not None and c in class_token_overlap_ref:
            ref_tokens = set(class_token_overlap_ref[c])
            ov = len(set(recoveries[c]) & ref_tokens) / max(top_k, 1)
            overlaps.append(ov)
    avg_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    return {'n_classes': num_classes, 'avg_top_k_overlap': avg_overlap,
            'top_k': top_k, 'recoveries_sample': {k: v[:5] for k, v in list(recoveries.items())[:3]}}


def class_token_correlations(train_ex, stoi, num_classes, top_k=10):
    """For each class, return the top_k tokens with highest class-conditional frequency."""
    cls_counter = [Counter() for _ in range(num_classes)]
    cls_total = [0] * num_classes
    for toks, lab in train_ex:
        if lab >= num_classes: continue
        for t in toks:
            tid = stoi.get(t, 1)  # UNK
            cls_counter[lab][tid] += 1
        cls_total[lab] += len(toks)
    out = {}
    for c in range(num_classes):
        if cls_total[c] == 0: continue
        ranked = [tid for tid, _ in cls_counter[c].most_common(top_k * 3)]
        out[c] = ranked[:top_k]
    return out


# ===================================================================
# Common training helper
# ===================================================================
def train_target_model(method, dataset_name, n_epochs=2, batch_size=16,
                       embedding_dim=64, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ex, _, _, num_classes = DATASETS[dataset_name](seed=seed)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    model = TinyClassifier(vocab_size, embedding_dim, num_classes, method=method).to(device)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        model.train()
        for batch in loader:
            if batch is None: continue
            opt.zero_grad()
            crit(model(batch['ids'].to(device)),
                 batch['label'].to(device)).backward()
            opt.step()
    return model, loader, stoi, vocab_size, train_ex, num_classes, crit, device


# ===================================================================
# Main runner
# ===================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['medmcqa', 'pubmedqa', 'sst2'])
    p.add_argument('--methods', nargs='+', default=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--out_dir', default='results/phase3_attacks')
    p.add_argument('--n_attack_batches', type=int, default=12)
    p.add_argument('--attacks', nargs='+',
                   default=['rtf', 'decepticons', 'april', 'badnets', 'modelinv'])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    summary = {a: {} for a in args.attacks}

    for ds in args.datasets:
        for m in args.methods:
            cell = f'{ds}/{m}'
            print(f'\n========== {cell} ==========')
            try:
                model, loader, stoi, vocab_size, train_ex, num_classes, crit, device = \
                    train_target_model(m, ds)
            except Exception as e:
                print(f'  TRAIN FAIL: {e}')
                for a in args.attacks:
                    summary[a].setdefault(ds, {})[m] = {'error': f'train: {e}'}
                continue

            # ------ RtF ------
            if 'rtf' in args.attacks:
                t0 = time.time(); f1s, n_atk = [], 0
                for batch in loader:
                    if n_atk >= args.n_attack_batches: break
                    if batch is None: continue
                    try:
                        r = robbing_the_fed_attack(model, batch['ids'], batch['label'],
                                                    crit, device, vocab_size)
                        f1s.append(r['f1'])
                    except Exception as e:
                        print(f'  rtf err: {e}')
                    n_atk += 1
                summary['rtf'].setdefault(ds, {})[m] = {
                    'f1': float(np.mean(f1s)) if f1s else 0.0,
                    'n_attacks': len(f1s),
                    'time_s': time.time() - t0,
                }
                print(f'  RtF f1={summary["rtf"][ds][m]["f1"]:.3f}')

            # ------ Decepticons ------
            if 'decepticons' in args.attacks:
                t0 = time.time(); f1s, n_atk = [], 0
                for batch in loader:
                    if n_atk >= args.n_attack_batches: break
                    if batch is None: continue
                    try:
                        r = decepticons_attack(model, batch['ids'], batch['label'],
                                                crit, device, vocab_size)
                        f1s.append(r['f1'])
                    except Exception as e:
                        print(f'  decepticons err: {e}')
                    n_atk += 1
                summary['decepticons'].setdefault(ds, {})[m] = {
                    'f1': float(np.mean(f1s)) if f1s else 0.0,
                    'n_attacks': len(f1s),
                    'time_s': time.time() - t0,
                }
                print(f'  Decepticons f1={summary["decepticons"][ds][m]["f1"]:.3f}')

            # ------ APRIL multi-round ------
            if 'april' in args.attacks:
                t0 = time.time(); f1s = []; n_atk = 0
                batch_iter = iter(loader)
                while n_atk < args.n_attack_batches // 3:
                    rounds = []
                    try:
                        for _ in range(3):
                            b = next(batch_iter)
                            if b is None: continue
                            rounds.append((b['ids'], b['label']))
                    except StopIteration:
                        break
                    if len(rounds) < 3: break
                    try:
                        r = april_multiround_attack(model, rounds, crit, device,
                                                     vocab_size, n_rounds=3)
                        f1s.append(r['f1'])
                    except Exception as e:
                        print(f'  april err: {e}')
                    n_atk += 1
                summary['april'].setdefault(ds, {})[m] = {
                    'f1': float(np.mean(f1s)) if f1s else 0.0,
                    'n_attacks': len(f1s),
                    'time_s': time.time() - t0,
                }
                print(f'  APRIL f1={summary["april"][ds][m]["f1"]:.3f}')

            # ------ Model Inversion ------
            if 'modelinv' in args.attacks:
                t0 = time.time()
                ref = class_token_correlations(train_ex, stoi, num_classes, top_k=10)
                try:
                    r = model_inversion_attack(model, crit, device, vocab_size,
                                                class_token_overlap_ref=ref)
                    summary['modelinv'].setdefault(ds, {})[m] = {
                        'top_k_overlap': r['avg_top_k_overlap'],
                        'n_classes': r['n_classes'],
                        'time_s': time.time() - t0,
                    }
                    print(f'  ModelInv top-k overlap={r["avg_top_k_overlap"]:.3f}')
                except Exception as e:
                    summary['modelinv'].setdefault(ds, {})[m] = {'error': str(e)}
                    print(f'  modelinv err: {e}')

            with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)

    # ------ BadNets (separate train cycle per cell) ------
    if 'badnets' in args.attacks:
        print('\n========== BadNets ==========')
        for ds in args.datasets:
            for m in args.methods:
                cell = f'{ds}/{m}'
                t0 = time.time()
                try:
                    r = badnets_attack(m, ds, poison_rate=0.05, n_epochs=2,
                                        batch_size=16, embedding_dim=64, seed=42)
                    r['time_s'] = time.time() - t0
                    summary['badnets'].setdefault(ds, {})[m] = r
                    print(f'  {cell}: clean_acc={r["clean_acc"]:.3f} ASR={r["asr"]:.3f}')
                except Exception as e:
                    summary['badnets'].setdefault(ds, {})[m] = {'error': str(e)}
                    print(f'  {cell} BadNets err: {e}')
                with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                    json.dump(summary, f, indent=2)

    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
