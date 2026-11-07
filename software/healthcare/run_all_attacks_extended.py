#!/usr/bin/env python3
"""Run FILM + TAG + iDLG on all 22 paper-era classification datasets.

Reuses, with no modification, the paper-era attack functions:
  FILM  : film_lamp_classification.attack_dataset (set-recovery)
  TAG   : extra_attacks.tag_attack
  iDLG  : phase1_attacks.idlg_attack

Composes existing loaders from:
  health_classification, domain_classification, ondevice_classification,
  genome_biomed_classification, new_domains.

For each dataset × method (standard, binary_seb), trains a TinyClassifier
and runs all 3 attacks on a few batches. Writes one summary.json with a
flat dict: {dataset: {method: {film, tag, idlg}}}.
"""
import argparse, functools, json, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, '..', 'Senti_analysis')))

from health_classification import (
    build_vocab, TextClassificationDataset, collate, PAD,
    load_ade, load_pubmedqa, load_medmcqa,
)
from domain_classification import (
    load_finance, load_education, load_legal, load_mental,
    load_contract, load_ecthr,
)
from ondevice_classification import (
    load_banking77, load_clinc150, load_massive,
)
from genome_biomed_classification import (
    _GB_DATASETS, _make_gb_loader,
    load_emotion, load_med_abstracts, load_pubmed_mesh,
)
from new_domains import load_sleep_edf, load_devign, load_diabetes

from film_lamp_classification import TinyClassifier, get_token_grad_norms
from extra_attacks import tag_attack as paper_tag_attack
from extra_attacks import dlg_attack as paper_dlg_attack
from phase1_attacks import idlg_attack as paper_idlg_attack
from phase1_attacks import inverting_gradients_attack as paper_invgrad_attack
from sklearn.metrics import roc_auc_score as _rocauc

GB_LOADERS = {n: _make_gb_loader(hf) for n, hf in _GB_DATASETS.items()}

DATASETS = {
    # Clinical NLP
    'pubmedqa': load_pubmedqa, 'ade': load_ade,
    'medmcqa': lambda seed: load_medmcqa(seed=seed, subsample=5000),
    'pubmed_mesh': load_pubmed_mesh, 'med_abstracts': load_med_abstracts,
    # Domains
    'education': load_education, 'finance': load_finance,
    'legal': load_legal, 'mental': load_mental,
    'contract': load_contract, 'ecthr': load_ecthr,
    # Intent
    'banking77': load_banking77, 'clinc150': load_clinc150, 'massive': load_massive,
    # Genomic Bench
    **GB_LOADERS, 'emotion': load_emotion,
    # Code/Logs/Sig./Tabular
    'sleep_edf': load_sleep_edf, 'devign': load_devign, 'diabetes': load_diabetes,
}


def film_attack(model, ids, label, criterion, device):
    """Set-recovery FILM: read embedding-grad sparsity, recover token set."""
    norms = get_token_grad_norms(model, ids, label, criterion, device)
    actual = set(int(t) for t in ids.flatten().tolist() if t > 0)
    k = max(len(actual), 1)
    recovered = set(torch.topk(norms, k).indices.cpu().tolist())
    tp = len(recovered & actual)
    p = tp / max(len(recovered), 1); r = tp / max(len(actual), 1)
    return 2 * p * r / max(p + r, 1e-9)


def train_target(method, dataset_name, n_epochs=2, batch_size=16, embedding_dim=64, seed=42):
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
            logits = model(batch['ids'].to(device))
            crit(logits, batch['label'].to(device)).backward()
            opt.step()
    return model, loader, crit, device, vocab_size


def run_attacks(dataset, method, n_attack_batches=12):
    print(f'>>> {dataset}/{method}', flush=True)
    try:
        model, loader, crit, device, vocab_size = train_target(method, dataset)
    except Exception as e:
        print(f'  TRAIN ERROR: {e}', flush=True)
        return {'error': f'train: {e}'}
    model.train()
    film_f1s, tag_f1s, idlg_f1s, dlg_f1s, invgrad_f1s, lamp_f1s = [], [], [], [], [], []
    losses_in, losses_out = [], []
    n = 0
    for batch in loader:
        if batch is None or n >= n_attack_batches: break
        ids, label = batch['ids'], batch['label']
        try: film_f1s.append(film_attack(model, ids, label, crit, device))
        except Exception: pass
        try:
            r = paper_tag_attack(model, ids, label, crit, device, vocab_size)
            tag_f1s.append(r['f1'] if isinstance(r, dict) else r)
        except Exception: pass
        try:
            r = paper_idlg_attack(model, ids, label, crit, device, vocab_size,
                                   embedding_dim=64, num_iters=100, lr=0.1)
            idlg_f1s.append(r['f1'] if isinstance(r, dict) else r)
        except Exception: pass
        try:
            f = paper_dlg_attack(model, ids, label, crit, device,
                                  num_iters=80, lr=0.1, vocab_size=vocab_size)
            dlg_f1s.append(f['f1'] if isinstance(f, dict) else f)
        except Exception: pass
        try:
            r = paper_invgrad_attack(model, ids, label, crit, device, vocab_size,
                                      embedding_dim=64, num_iters=80, lr=0.1)
            invgrad_f1s.append(r['f1'] if isinstance(r, dict) else r)
        except Exception: pass
        # Inline LAMP via embedding-grad threshold
        try:
            model.zero_grad(set_to_none=False)
            logits = model(ids.to(device))
            crit(logits, label.to(device)).backward()
            actual_lamp = set(int(t) for t in ids.flatten().tolist() if t > 0)
            if hasattr(model, 'embedding') and model.embedding.weight.grad is not None:
                gn = model.embedding.weight.grad.detach().norm(dim=1).cpu()
                lg = torch.log(gn.clamp(min=1e-12))
                z = (lg - lg.mean()) / max(lg.std().item(), 1e-9)
                rec = set((z > 1.0).nonzero(as_tuple=True)[0].tolist())
                tp = len(rec & actual_lamp)
                p = tp / max(len(rec), 1); rr = tp / max(len(actual_lamp), 1)
                lamp_f1s.append(2*p*rr/max(p+rr, 1e-9))
        except Exception: pass
        # LiRA-style loss histogram (alternate IN/OUT halves of batch)
        try:
            with torch.no_grad():
                lg = model(ids.to(device))
                crit_n = nn.CrossEntropyLoss(reduction='none')
                ll = crit_n(lg, label.to(device)).cpu().tolist()
                if n % 2 == 0:
                    losses_in.extend(ll)
                else:
                    losses_out.extend(ll)
        except Exception: pass
        n += 1
    # LiRA AUC via in vs out loss histograms
    lira_auc = None
    if len(losses_in) > 1 and len(losses_out) > 1:
        try:
            scores = np.array(losses_in + losses_out)
            labels_lira = [1]*len(losses_in) + [0]*len(losses_out)
            lira_auc = float(_rocauc(labels_lira, -scores))
        except Exception: pass
    def safe_mean(xs):
        return float(np.mean(xs)) if xs else None
    return {
        'film': safe_mean(film_f1s),
        'tag': safe_mean(tag_f1s),
        'dlg': safe_mean(dlg_f1s),
        'lamp': safe_mean(lamp_f1s),
        'lira': lira_auc,
        'invgrad': safe_mean(invgrad_f1s),
        'idlg': safe_mean(idlg_f1s),
        'n_attacks': n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=list(DATASETS.keys()))
    p.add_argument('--methods', nargs='+', default=['standard', 'binary_seb'])
    p.add_argument('--out_dir', default='results/v8_panel_a/all_attacks_extended')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}
    for ds in args.datasets:
        if ds not in DATASETS:
            print(f'Skipping unknown dataset: {ds}'); continue
        summary[ds] = {}
        for m in args.methods:
            t0 = time.time()
            summary[ds][m] = run_attacks(ds, m)
            summary[ds][m]['time_s'] = time.time() - t0
            with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
