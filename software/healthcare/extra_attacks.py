#!/usr/bin/env python3
"""Tier-1 + Tier-2 extra privacy attacks beyond FILM/LAMP/LLG.

Implements:
  * DLG  (Zhu '19) -- iterative gradient matching
  * TAG  (Deng '21) -- cosine similarity between grad rows and embedding table
  * MIA  (reference-model LiRA-lite) -- loss-ratio membership inference
  * EmbInv -- embedding inversion: train inverter from final hidden state to token set

Datasets: medmcqa, pubmedqa, education, yelp, imdb, sst2.
Methods : standard, byte_seb, binary_seb.
"""
import argparse, functools, json, math, os, random, sys, time
from collections import Counter

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, '..', 'Senti_analysis')))

from health_classification import (
    build_vocab, tokenize, TextClassificationDataset, collate, PAD,
    load_ade, load_pubmedqa, load_medmcqa,
)
from domain_classification import load_education
from film_lamp_classification import TinyClassifier


def load_yelp(seed=42, n_train=10000, n_eval=1000):
    raw = datasets.load_dataset('yelp_polarity', split=f'train[:{n_train}]')
    test = datasets.load_dataset('yelp_polarity', split=f'test[:{n_eval}]')
    return ([(r['text'], int(r['label'])) for r in raw],
            [(r['text'], int(r['label'])) for r in test][:n_eval // 2],
            [(r['text'], int(r['label'])) for r in test][n_eval // 2:], 2)


def load_imdb(seed=42, n_train=10000, n_eval=1000):
    raw = datasets.load_dataset('imdb', split=f'train[:{n_train}]')
    test = datasets.load_dataset('imdb', split=f'test[:{n_eval}]')
    return ([(r['text'], int(r['label'])) for r in raw],
            [(r['text'], int(r['label'])) for r in test][:n_eval // 2],
            [(r['text'], int(r['label'])) for r in test][n_eval // 2:], 2)


def load_sst2(seed=42):
    raw = datasets.load_dataset('glue', 'sst2')
    train = [(r['sentence'], int(r['label'])) for r in raw['train']][:10000]
    val = list(raw['validation'])
    return (train,
            [(r['sentence'], int(r['label'])) for r in val[:len(val)//2]],
            [(r['sentence'], int(r['label'])) for r in val[len(val)//2:]], 2)


DATASETS = {
    'medmcqa': lambda seed: load_medmcqa(seed=seed, subsample=5000),
    'pubmedqa': load_pubmedqa,
    'education': load_education,
    'yelp': load_yelp,
    'imdb': load_imdb,
    'sst2': load_sst2,
}


def build_target_model(method, vocab_size, embedding_dim, num_classes,
                      num_bits=64, interdim='128'):
    return TinyClassifier(vocab_size, embedding_dim, num_classes,
                          method=method, num_bits=num_bits, interdim=interdim)


def materialize_embeddings(model, vocab_size, device, chunk=1024):
    """Build the V x d effective embedding table for any of the three methods."""
    rows = []
    model.eval()
    with torch.no_grad():
        for s in range(0, vocab_size, chunk):
            e = min(s + chunk, vocab_size)
            ids = torch.arange(s, e, device=device).unsqueeze(0)
            emb = model.embedding(ids).squeeze(0)  # (chunk, d)
            rows.append(emb)
    return torch.cat(rows, dim=0)  # (V, d)


# ----------------- DLG -----------------
def dlg_attack(model, ids, label, criterion, device, num_iters=200, lr=0.1,
               vocab_size=None, embedding_dim=64):
    """Iterative gradient matching to recover input embeddings, then map to vocab.

    For all three methods, we optimize a dummy embedding tensor (B, T, d) that
    is fed directly into the model body (bypassing the embedding layer). The
    body is everything *after* model.embedding. We compare gradients through
    the body parameters only (this is the threat model: attacker sees the body
    gradient regardless of embedding choice -- so 001/EmByte's protection
    must come from the body gradient itself being less informative).
    """
    model.zero_grad(set_to_none=False)
    B, T = ids.shape
    real_logits = model.embedding(ids.to(device)) if False else None
    # Step 1: get real gradient on the body parameters only.
    # The body in TinyClassifier is `model.fc` (linear head) -- model.embedding -> mean -> fc.
    real_emb = model.embedding(ids.to(device))  # (B, T, d)
    real_pooled = real_emb.mean(dim=1)
    real_logits = model.fc(real_pooled)
    real_loss = criterion(real_logits, label.to(device))
    grads_real = torch.autograd.grad(real_loss, [model.fc.weight, model.fc.bias],
                                     create_graph=False, retain_graph=False)
    grads_real = [g.detach() for g in grads_real]

    # Step 2: dummy embeddings + dummy label
    dummy_emb = torch.randn(B, T, embedding_dim, device=device, requires_grad=True)
    dummy_logits = torch.randn(B, model.fc.out_features, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([dummy_emb, dummy_logits], lr=lr, max_iter=20)

    def closure():
        optimizer.zero_grad()
        d_pooled = dummy_emb.mean(dim=1)
        d_pred = model.fc(d_pooled)
        d_loss = criterion(d_pred, dummy_logits.softmax(dim=-1).argmax(dim=-1))
        # Use soft labels (KL) for differentiable
        log_p = F.log_softmax(d_pred, dim=-1)
        soft = F.softmax(dummy_logits, dim=-1)
        d_loss = -(soft * log_p).sum(dim=-1).mean()
        d_grads = torch.autograd.grad(d_loss, [model.fc.weight, model.fc.bias],
                                      create_graph=True, retain_graph=True)
        diff = sum(((dg - rg) ** 2).sum() for dg, rg in zip(d_grads, grads_real))
        diff.backward()
        return diff

    for _ in range(num_iters // 20):
        try:
            optimizer.step(closure)
        except Exception:
            break

    # Step 3: project recovered dummy_emb back to vocab via NN search in embed table
    with torch.no_grad():
        emb_table = materialize_embeddings(model, vocab_size, device)  # (V, d)
        # Cosine similarity per (B,T) -> argmax over V
        flat = dummy_emb.detach().reshape(B * T, embedding_dim)
        flat_n = F.normalize(flat, dim=-1)
        tab_n = F.normalize(emb_table, dim=-1)
        sim = flat_n @ tab_n.t()  # (B*T, V)
        recovered_ids = sim.argmax(dim=-1).cpu().tolist()

    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    recovered = set(recovered_ids)
    tp = len(recovered & actual)
    precision = tp / max(len(recovered), 1)
    recall = tp / max(len(actual), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'recovered_size': len(recovered), 'actual_size': len(actual)}


# ----------------- TAG -----------------
def tag_attack(model, ids, label, criterion, device, vocab_size,
               top_k=None):
    """Text-AGnostic gradient attack (Deng '21).

    For Subword, the embedding-layer parameter `embedding.weight` (V x d)
    has nonzero gradient rows only at token ids that appear in the batch.
    The attack: for each row v in [V], if ||g_emb[v]|| is non-trivial, mark
    v as recovered. (This is essentially TAG-without-cosine because the
    cosine adds no extra signal when the only nonzero rows are batch tokens.)

    For SEB methods the embedding is an FFN -- the token-by-token signal
    is mixed across all FFN parameters, so per-token recovery should fail.
    We mirror FILM's threat-model handling: extract per-vocab "gradient
    proxy" by re-querying the FFN with each token id and measuring how
    much the body's input changes.
    """
    model.zero_grad(set_to_none=False)
    ids_dev = ids.to(device)
    real_emb = model.embedding(ids_dev)  # (B, T, d)
    pooled = real_emb.mean(dim=1)
    logits = model.fc(pooled)
    loss = criterion(logits, label.to(device))
    loss.backward()

    if model.method == 'standard':
        # Per-vocab gradient row magnitude
        gnorm = model.embedding.weight.grad.detach().norm(dim=1).cpu()  # (V,)
        actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
        avg_size = max(len(actual), 1)
        k = top_k or avg_size
        recovered = set(torch.topk(gnorm, k).indices.tolist())
    else:
        # No per-vocab gradient row exists. Fairly: fall back to projecting the
        # body's input gradient onto each vocab embedding via cosine similarity.
        # That is, the attacker has fc.weight.grad -- (num_classes, d) -- and the
        # materialised embedding table; pick tokens whose embedding has highest
        # cosine similarity with the gradient direction.
        fc_grad = model.fc.weight.grad.detach()  # (C, d)
        emb_table = materialize_embeddings(model, vocab_size, device)  # (V, d)
        # average gradient across class rows
        g = fc_grad.mean(dim=0).unsqueeze(0)  # (1, d)
        g_n = F.normalize(g, dim=-1)
        tab_n = F.normalize(emb_table, dim=-1)
        sim = (g_n @ tab_n.t()).abs().squeeze(0).cpu()  # (V,)
        actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
        k = top_k or max(len(actual), 1)
        recovered = set(torch.topk(sim, k).indices.tolist())

    tp = len(recovered & actual)
    precision = tp / max(len(recovered), 1)
    recall = tp / max(len(actual), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'recovered_size': len(recovered), 'actual_size': len(actual)}


# ----------------- MIA (reference-model) -----------------
def train_simple(model, loader, device, epochs, lr=5e-4):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for batch in loader:
            if batch is None: continue
            opt.zero_grad()
            crit(model(batch['ids'].to(device)), batch['label'].to(device)).backward()
            opt.step()
    return model


@torch.no_grad()
def per_example_loss(model, examples_ds, device, batch_size=32, stoi=None):
    crit = nn.CrossEntropyLoss(reduction='none')
    losses = []
    coll = functools.partial(collate, pad_index=stoi[PAD])
    loader = DataLoader(examples_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    for batch in loader:
        if batch is None: continue
        logits = model(batch['ids'].to(device))
        l = crit(logits, batch['label'].to(device))
        losses.extend(l.cpu().tolist())
    return np.array(losses)


def mia_attack(method, dataset_name, vocab_size, embedding_dim, num_classes,
               train_ex, val_ex, device, epochs=2, batch_size=16,
               num_bits=64, interdim='128'):
    """LiRA-lite: train two models on disjoint halves; use loss-ratio for MIA.

    Members of model_A = first half of train_ex; non-members = second half.
    Reference model = trained on second half.
    Score(x) = loss_B(x) - loss_A(x). Higher = more likely a member of A.
    """
    n = len(train_ex)
    rng = random.Random(0)
    idx = list(range(n)); rng.shuffle(idx)
    half = n // 2
    A_ex = [train_ex[i] for i in idx[:half]]
    B_ex = [train_ex[i] for i in idx[half:2*half]]
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)

    A_ds = TextClassificationDataset(A_ex, stoi, max_length=128)
    B_ds = TextClassificationDataset(B_ex, stoi, max_length=128)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    loader_A = DataLoader(A_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    loader_B = DataLoader(B_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)

    model_A = build_target_model(method, vocab_size, embedding_dim, num_classes,
                                num_bits, interdim).to(device)
    model_B = build_target_model(method, vocab_size, embedding_dim, num_classes,
                                num_bits, interdim).to(device)
    train_simple(model_A, loader_A, device, epochs)
    train_simple(model_B, loader_B, device, epochs)

    # Score every example with model_A loss and model_B loss
    all_losses_A = per_example_loss(model_A, A_ds, device, stoi=stoi)
    nonmem_losses_A = per_example_loss(model_A, B_ds, device, stoi=stoi)
    all_losses_B = per_example_loss(model_B, A_ds, device, stoi=stoi)
    nonmem_losses_B = per_example_loss(model_B, B_ds, device, stoi=stoi)
    # Membership scores for A: ref_loss - target_loss
    mem_scores_A = all_losses_B - all_losses_A
    nonmem_scores_A = nonmem_losses_B - nonmem_losses_A
    y = np.concatenate([np.ones_like(mem_scores_A), np.zeros_like(nonmem_scores_A)])
    s = np.concatenate([mem_scores_A, nonmem_scores_A])
    auc = float(roc_auc_score(y, s))
    return {'auc': auc, 'n_member': int(len(mem_scores_A)),
            'n_nonmember': int(len(nonmem_scores_A))}


# ----------------- Embedding inversion -----------------
class InverterNet(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, vocab_size)

    def forward(self, h):
        return self.fc2(F.relu(self.fc1(h)))


def emb_inversion_attack(method, dataset_name, vocab_size, embedding_dim,
                         num_classes, train_ex, val_ex, device, epochs=2,
                         batch_size=16, num_bits=64, interdim='128'):
    """Train an inverter: hidden_state (post-embedding mean-pool) -> bag-of-tokens.

    The threat model: attacker sees the model's mean-pooled hidden state for
    each input and tries to recover the token set.

    A small inverter network (h -> sigmoid over V) is trained on (h, set) pairs
    from the train split, then evaluated on the val split.
    """
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
    val_ds = TextClassificationDataset(val_ex, stoi, max_length=128)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)

    target = build_target_model(method, vocab_size, embedding_dim, num_classes,
                                num_bits, interdim).to(device)
    train_simple(target, train_loader, device, epochs=1)
    target.eval()

    inverter = InverterNet(embedding_dim, vocab_size).to(device)
    opt = optim.Adam(inverter.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    # Train inverter
    for _ in range(epochs):
        for batch in train_loader:
            if batch is None: continue
            ids = batch['ids'].to(device)
            with torch.no_grad():
                h = target.embedding(ids).mean(dim=1)  # (B, d)
            # Build bag-of-tokens target
            B = ids.size(0)
            y = torch.zeros(B, vocab_size, device=device)
            for b in range(B):
                for t in ids[b].tolist():
                    if t > 0: y[b, t] = 1.0
            pred = inverter(h)
            loss = bce(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()

    # Evaluate on val: top-k recovery
    inverter.eval()
    f1s = []
    with torch.no_grad():
        for batch in val_loader:
            if batch is None: continue
            ids = batch['ids'].to(device)
            B = ids.size(0)
            actual_sets = [set(int(t) for t in ids[b].tolist() if t > 0) for b in range(B)]
            h = target.embedding(ids).mean(dim=1)
            logit = inverter(h)
            # Take top-k = avg actual size
            avg_k = int(np.mean([len(s) for s in actual_sets])) or 1
            topk = logit.topk(avg_k, dim=-1).indices.cpu().tolist()
            for b in range(B):
                recovered = set(topk[b])
                tp = len(recovered & actual_sets[b])
                if not actual_sets[b]: continue
                p = tp / max(len(recovered), 1)
                r = tp / len(actual_sets[b])
                f1 = 2 * p * r / max(p + r, 1e-9)
                f1s.append(f1)
    return {'f1': float(np.mean(f1s)) if f1s else 0.0,
            'num_eval_batches': len(val_ds) // batch_size}


# ----------------- runner -----------------
def run_one(dataset, method, attack, out_dir, batch_size=16, num_attack_batches=20,
            seed=42, embedding_dim=64, train_epochs=2):
    print(f"\n===== {attack.upper()} :: {dataset} / {method} =====")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, val_ex, _, num_classes = DATASETS[dataset](seed=seed) if dataset == 'medmcqa' else DATASETS[dataset](seed=seed)
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    print(f"  vocab={vocab_size} classes={num_classes} train={len(train_ex)}")

    if attack == 'mia':
        result = mia_attack(method, dataset, vocab_size, embedding_dim,
                            num_classes, train_ex, val_ex, device,
                            epochs=train_epochs, batch_size=batch_size)
        print(f"  AUC={result['auc']:.3f}")
    elif attack == 'embinv':
        result = emb_inversion_attack(method, dataset, vocab_size, embedding_dim,
                                      num_classes, train_ex, val_ex, device,
                                      epochs=train_epochs, batch_size=batch_size)
        print(f"  inversion F1={result['f1']:.3f}")
    else:
        # DLG / TAG: build model, lightly train, then attack
        train_ds = TextClassificationDataset(train_ex, stoi, max_length=128)
        coll = functools.partial(collate, pad_index=stoi[PAD])
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
        model = build_target_model(method, vocab_size, embedding_dim, num_classes).to(device)
        opt = optim.Adam(model.parameters(), lr=5e-4)
        crit = nn.CrossEntropyLoss()
        for _ in range(train_epochs):
            for batch in loader:
                if batch is None: continue
                opt.zero_grad()
                crit(model(batch['ids'].to(device)), batch['label'].to(device)).backward()
                opt.step()

        metrics_list = []; done = 0
        for batch in loader:
            if batch is None or done >= num_attack_batches: break
            if attack == 'dlg':
                m = dlg_attack(model, batch['ids'], batch['label'], crit, device,
                               vocab_size=vocab_size, embedding_dim=embedding_dim)
            else:  # tag
                m = tag_attack(model, batch['ids'], batch['label'], crit, device,
                               vocab_size=vocab_size)
            metrics_list.append(m); done += 1
        result = {k: float(np.mean([m[k] for m in metrics_list]))
                  for k in metrics_list[0]}
        result['n_attacks'] = done
        print(f"  precision={result['precision']:.3f} recall={result['recall']:.3f} f1={result['f1']:.3f}")

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'{dataset}_{method}_{attack}.json')
    with open(out_file, 'w') as f:
        json.dump({'dataset': dataset, 'method': method, 'attack': attack,
                   'metrics': result, 'vocab_size': vocab_size,
                   'num_classes': num_classes}, f, indent=2)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+',
                   default=['medmcqa', 'pubmedqa', 'education', 'yelp', 'imdb', 'sst2'])
    p.add_argument('--methods', nargs='+',
                   default=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--attacks', nargs='+',
                   default=['dlg', 'tag', 'mia', 'embinv'])
    p.add_argument('--out_dir', default='results/extra_privacy')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_attack_batches', type=int, default=20)
    args = p.parse_args()

    summary = {}
    for ds in args.datasets:
        summary[ds] = {}
        for m in args.methods:
            summary[ds][m] = {}
            for atk in args.attacks:
                t0 = time.time()
                try:
                    r = run_one(ds, m, atk, args.out_dir,
                                batch_size=args.batch_size,
                                num_attack_batches=args.num_attack_batches)
                    summary[ds][m][atk] = r
                except Exception as e:
                    print(f"  FAILED: {e}")
                    summary[ds][m][atk] = {'error': str(e)}
                summary[ds][m][f'{atk}_time_s'] = time.time() - t0
            os.makedirs(args.out_dir, exist_ok=True)
            with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out_dir}/summary.json")


if __name__ == '__main__':
    main()
