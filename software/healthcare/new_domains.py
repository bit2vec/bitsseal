#!/usr/bin/env python3
"""Run 001 / EmByte / Subword on three new domains:
  #2 Sleep-EDF EEG sleep-stage classification (5-class)
  #4 Devign / CodeXGLUE code vulnerability detection (binary)
  #5 UCI Diabetes 130-Hospitals readmission (binary)

Same TinyClassifier scaffold as the medical/sentiment pipeline. Reports
test accuracy, params/MB, training time, inference time, and runs FILM
+ TAG + MIA attacks for the privacy story.
"""
import argparse, functools, json, os, random, re, sys, time, urllib.request
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, '..', 'Senti_analysis')))
from health_classification import (
    build_vocab, TextClassificationDataset, collate, PAD,
)
from film_lamp_classification import TinyClassifier, get_token_grad_norms
from extra_attacks import tag_attack, mia_attack, materialize_embeddings


# ============================================================
# Loaders
# ============================================================
SLEEP_DIR = '/tmp/sleep_edf'
SLEEP_BASE = 'https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/'
SLEEP_SUBJECTS = ['SC4001', 'SC4002', 'SC4011', 'SC4012', 'SC4021',
                  'SC4022', 'SC4031', 'SC4032', 'SC4041', 'SC4042']
PSG_SUFFIX = ['E0', 'E0', 'E0', 'E0', 'E0', 'E0', 'E0', 'E0', 'E0', 'E0']
HYP_SUFFIX = ['EC', 'EC', 'EH', 'EH', 'EH', 'EJ', 'EH', 'EJ', 'EC', 'EH']

STAGE_MAP = {
    'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
    'Sleep stage 3': 3, 'Sleep stage 4': 3,  # merge N3+N4
    'Sleep stage R': 4, 'Sleep stage ?': -1,
    'Movement time': -1,
}


def _download_sleep_subject(subj_id, psg_suffix, hyp_suffix):
    psg = os.path.join(SLEEP_DIR, f'{subj_id}{psg_suffix}-PSG.edf')
    hyp = os.path.join(SLEEP_DIR, f'{subj_id}{hyp_suffix}-Hypnogram.edf')
    os.makedirs(SLEEP_DIR, exist_ok=True)
    if not os.path.exists(psg) or os.path.getsize(psg) < 1e6:
        try:
            urllib.request.urlretrieve(f'{SLEEP_BASE}{subj_id}{psg_suffix}-PSG.edf', psg)
        except Exception as e:
            print(f"  download {subj_id} PSG fail: {e}"); return None, None
    if not os.path.exists(hyp) or os.path.getsize(hyp) < 1e3:
        try:
            urllib.request.urlretrieve(f'{SLEEP_BASE}{subj_id}{hyp_suffix}-Hypnogram.edf', hyp)
        except Exception as e:
            print(f"  download {subj_id} hyp fail: {e}"); return None, None
    return psg, hyp


def _quantize_epoch(signal_30s, fs, n_tokens=64, n_levels=256):
    """Split a 30 s epoch into n_tokens windows; per-window mean is rescaled
    to 0..n_levels-1 to produce a token sequence."""
    import numpy as np
    n_samples = len(signal_30s)
    win = max(1, n_samples // n_tokens)
    feats = []
    for i in range(n_tokens):
        s = i * win; e = min(s + win, n_samples)
        feats.append(np.sqrt(np.mean(signal_30s[s:e].astype(np.float64) ** 2)))
    feats = np.array(feats)
    if feats.max() == feats.min():
        return [0] * n_tokens
    norm = (feats - feats.min()) / (feats.max() - feats.min())
    quant = np.clip((norm * (n_levels - 1)).astype(int), 0, n_levels - 1)
    return quant.tolist()


def load_sleep_edf(seed=42, n_subjects=10, n_tokens=64, n_levels=256):
    import pyedflib
    examples = []
    for i in range(min(n_subjects, len(SLEEP_SUBJECTS))):
        psg, hyp = _download_sleep_subject(SLEEP_SUBJECTS[i],
                                           PSG_SUFFIX[i], HYP_SUFFIX[i])
        if psg is None: continue
        try:
            f = pyedflib.EdfReader(psg)
        except Exception as e:
            print(f"  reader fail {SLEEP_SUBJECTS[i]}: {e}"); continue
        ch_names = [f.getLabel(c) for c in range(f.signals_in_file)]
        # Pick first EEG channel (usually 'EEG Fpz-Cz' index 0)
        eeg_ch = 0
        fs = int(f.getSampleFrequency(eeg_ch))
        sig = f.readSignal(eeg_ch)
        f.close()

        # Hypnogram annotations: list of (onset_s, duration_s, label_str)
        h = pyedflib.EdfReader(hyp)
        anns = h.readAnnotations()
        h.close()
        onsets, durations, labels = anns

        for onset, dur, lab in zip(onsets, durations, labels):
            stage = STAGE_MAP.get(lab.strip(), -1)
            if stage < 0: continue
            n_epochs_in_stage = int(round(dur / 30.0))
            for k in range(n_epochs_in_stage):
                start = int((onset + k * 30.0) * fs)
                end = start + 30 * fs
                if end > len(sig): break
                seg = sig[start:end]
                if len(seg) < 30 * fs * 0.9: continue
                tokens = _quantize_epoch(seg, fs, n_tokens=n_tokens,
                                         n_levels=n_levels)
                # Use space-joined token ints as the "text"
                text = ' '.join(f't{t}' for t in tokens)
                examples.append((text, stage))
        print(f"  {SLEEP_SUBJECTS[i]}: cumulative {len(examples)} epochs")
    rng = random.Random(seed); rng.shuffle(examples)
    n = len(examples)
    n_tr = int(0.7 * n); n_va = int(0.1 * n)
    return (examples[:n_tr], examples[n_tr:n_tr + n_va],
            examples[n_tr + n_va:], 5)


def load_devign(seed=42):
    import datasets as hfds
    train = hfds.load_dataset('code_x_glue_cc_defect_detection', split='train')
    val = hfds.load_dataset('code_x_glue_cc_defect_detection', split='validation')
    test = hfds.load_dataset('code_x_glue_cc_defect_detection', split='test')
    def to_ex(ds):
        return [(r['func'], int(r['target'])) for r in ds]
    return to_ex(train), to_ex(val), to_ex(test), 2


def load_diabetes(seed=42, max_samples=20000):
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=296)
    X = d.data.features; y = d.data.targets
    rng = random.Random(seed)
    idx = list(range(len(X))); rng.shuffle(idx)
    idx = idx[:max_samples]
    X = X.iloc[idx].reset_index(drop=True)
    y = y.iloc[idx].reset_index(drop=True)
    target_col = list(y.columns)[0]
    # Binary: readmitted within 30 days vs not (rare-event task)
    y_bin = (y[target_col] == '<30').astype(int).tolist()

    rows = []
    cols = list(X.columns)
    for i in range(len(X)):
        toks = []
        for c in cols:
            v = X.iloc[i][c]
            if v != v: continue  # NaN
            sv = str(v)[:30].replace(' ', '_')
            toks.append(f'{c}={sv}')
        rows.append((' '.join(toks), y_bin[i]))
    rng.shuffle(rows)
    n = len(rows); n_tr = int(0.7 * n); n_va = int(0.1 * n)
    return (rows[:n_tr], rows[n_tr:n_tr + n_va],
            rows[n_tr + n_va:], 2)


# ============================================================
# Train / eval
# ============================================================
DOMAINS = {'sleep_edf': load_sleep_edf, 'devign': load_devign, 'diabetes': load_diabetes}


def train_eval(method, dataset_name, out_dir, n_epochs=4, batch_size=16,
               embedding_dim=64, max_length=128, num_bits=64, interdim='128',
               seed=42, train_truncate=10000):
    print(f"\n=== {dataset_name} / {method} ===")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ex, val_ex, test_ex, num_classes = DOMAINS[dataset_name](seed=seed)
    if len(train_ex) > train_truncate:
        rng = random.Random(seed); rng.shuffle(train_ex)
        train_ex = train_ex[:train_truncate]
    print(f"  train={len(train_ex)} val={len(val_ex)} test={len(test_ex)} classes={num_classes}")
    stoi, _ = build_vocab([t for t, _ in train_ex], min_freq=2)
    vocab_size = len(stoi)
    print(f"  vocab={vocab_size}")

    train_ds = TextClassificationDataset(train_ex, stoi, max_length=max_length)
    val_ds = TextClassificationDataset(val_ex, stoi, max_length=max_length)
    test_ds = TextClassificationDataset(test_ex, stoi, max_length=max_length)
    coll = functools.partial(collate, pad_index=stoi[PAD])
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)

    model = TinyClassifier(vocab_size, embedding_dim, num_classes,
                           method=method, num_bits=num_bits, interdim=interdim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        for batch in tr_loader:
            if batch is None: continue
            optimizer.zero_grad()
            criterion(model(batch['ids'].to(device)), batch['label'].to(device)).backward()
            optimizer.step()
    train_time = time.time() - t0

    # eval (test inference time)
    model.eval()
    correct = total = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            if batch is None: continue
            logits = model(batch['ids'].to(device))
            pred = logits.argmax(dim=-1).cpu()
            correct += (pred == batch['label']).sum().item()
            total += batch['label'].size(0)
    inf_time = time.time() - t0
    test_acc = correct / max(total, 1)
    print(f"  test_acc={test_acc:.4f} train={train_time:.1f}s inf={inf_time:.2f}s")

    # FILM attack (token recovery via embedding gradient)
    film_metrics = []
    model.train()
    n_atk = 0
    for batch in tr_loader:
        if batch is None or n_atk >= 20: break
        ids = batch['ids']; label = batch['label']
        gnorms = get_token_grad_norms(model, ids, label, criterion, device)
        threshold = gnorms.max().item() * 0.05
        if threshold == 0: threshold = 1e-12
        recovered_ids = (gnorms > threshold).nonzero(as_tuple=True)[0].tolist()
        actual = set(t for t in ids.flatten().tolist() if t > 0)
        recovered = set(recovered_ids)
        if method != 'standard' and len(recovered) > 5 * len(actual):
            recovered = set(random.sample(recovered_ids, min(10, len(recovered_ids))))
        tp = len(recovered & actual)
        prec = tp / max(len(recovered), 1)
        rec = tp / max(len(actual), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        film_metrics.append({'precision': prec, 'recall': rec, 'f1': f1})
        n_atk += 1
    film_avg = {k: float(np.mean([m[k] for m in film_metrics])) for k in film_metrics[0]}

    # TAG attack
    tag_metrics = []; n_atk = 0
    for batch in tr_loader:
        if batch is None or n_atk >= 20: break
        ids = batch['ids']; label = batch['label']
        m = tag_attack(model, ids, label, criterion, device, vocab_size=vocab_size)
        tag_metrics.append(m); n_atk += 1
    tag_avg = {k: float(np.mean([m[k] for m in tag_metrics])) for k in tag_metrics[0]}

    # MIA AUC (uses extra_attacks helper)
    mia = mia_attack(method, dataset_name, vocab_size, embedding_dim, num_classes,
                     train_ex, val_ex, device, epochs=2, batch_size=batch_size,
                     num_bits=num_bits, interdim=interdim)

    result = {
        'dataset': dataset_name, 'method': method, 'num_classes': num_classes,
        'vocab_size': vocab_size, 'n_train': len(train_ex), 'n_test': len(test_ex),
        'test_acc': float(test_acc), 'n_params': int(n_params),
        'mb_fp32': n_params * 4 / 1e6,
        'train_time_s': train_time, 'inference_time_s': inf_time,
        'film': film_avg, 'tag': tag_avg, 'mia': mia,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{dataset_name}_{method}.json'), 'w') as f:
        json.dump(result, f, indent=2)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+',
                   default=['sleep_edf', 'devign', 'diabetes'])
    p.add_argument('--methods', nargs='+',
                   default=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--out_dir', default='results/new_domains')
    p.add_argument('--n_epochs', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=16)
    args = p.parse_args()

    summary = {}
    for ds in args.datasets:
        summary[ds] = {}
        for m in args.methods:
            try:
                summary[ds][m] = train_eval(m, ds, args.out_dir,
                                            n_epochs=args.n_epochs,
                                            batch_size=args.batch_size)
            except Exception as e:
                import traceback; traceback.print_exc()
                summary[ds][m] = {'error': str(e)}
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {args.out_dir}/summary.json")


if __name__ == '__main__':
    main()
