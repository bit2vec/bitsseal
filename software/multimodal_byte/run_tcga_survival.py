#!/usr/bin/env python3
"""TCGA survival prediction (binary mortality) with multi-modal Bits/EmByte/Subword.

Predicts: did the patient die during follow-up (binary OS_STATUS)?
- label 1 = DECEASED at last follow-up
- label 0 = LIVING at last follow-up

CRITICAL leakage prevention:
  Excludes OS_STATUS, OS_MONTHS, DFS_STATUS, DFS_MONTHS, PFS_STATUS, PFS_MONTHS,
  DSS_STATUS, DSS_MONTHS, DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS, and
  PERSON_NEOPLASM_CANCER_STATUS from features. Only keeps patient demographics,
  tumor stage, grade, gene expression, mutation panel, etc.

Reuses architecture: MultiModalClassifier (Body D), embedding_dim=64, FFN=128.
Identical setup to the cancer-type classification runner; only the label changes.
"""
import argparse, copy, functools, json, os, random, re, sys, time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'healthcare'))
sys.path.insert(0, THIS_DIR)

from run_civic_multimodal import PAD, UNK, SEP, collate
from run_civic_extended import MultiModalClassifier, film_attack
import tcga_loader

# Survival fields to EXCLUDE from input features (would leak the label)
LEAKAGE_FIELDS = {
    'OS_STATUS', 'OS_MONTHS',
    'DSS_STATUS', 'DSS_MONTHS',
    'PFS_STATUS', 'PFS_MONTHS',
    'DFS_STATUS', 'DFS_MONTHS',
    'DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS',
    'PERSON_NEOPLASM_CANCER_STATUS',
}


def patients_to_survival_examples(study_label, study_id, top_k_genes=64,
                                    min_followup_months=0):
    """Build (text, structured, gene_top, label) where label = 1 if DECEASED."""
    clinical = tcga_loader.fetch_clinical(study_id)
    by_patient = defaultdict(dict)
    for d in clinical:
        by_patient[d['patientId']][d['clinicalAttributeId']] = d['value']
    mrna, _ = tcga_loader.fetch_mrna(study_id)

    # Filter feature attributes to remove leakage fields
    text_attrs = [a for a in tcga_loader.CLINICAL_TEXT_ATTRS
                  if a not in LEAKAGE_FIELDS]
    struct_attrs = [a for a in tcga_loader.CLINICAL_STRUCT_ATTRS
                    if a not in LEAKAGE_FIELDS]

    examples = []
    for pid, attrs in by_patient.items():
        os_status = attrs.get('OS_STATUS', '').strip()
        os_months = attrs.get('OS_MONTHS', '').strip()
        if not os_status or not os_months:
            continue
        try:
            months = float(os_months)
        except ValueError:
            continue
        # Optionally drop short-followup censored patients
        if 'LIVING' in os_status and months < min_followup_months:
            continue

        label = 1 if 'DECEASED' in os_status else 0

        # Build text from non-leakage attrs only
        text_parts = []
        for a in text_attrs:
            v = attrs.get(a, '').strip()
            if v: text_parts.append(f'{a.lower()}:{v}')
        text = ' '.join(text_parts) if text_parts else f'study:{study_label}'

        struct_parts = []
        for a in struct_attrs:
            v = attrs.get(a, '').strip()
            if v:
                try:
                    fv = float(v)
                    if a == 'AGE': v_bin = f'{int(fv//10)*10}'
                    elif 'SCORE' in a: v_bin = f'{round(fv, 1)}'
                    else: v_bin = f'{fv:.2f}'
                    struct_parts.append(f'{a.lower()}={v_bin}')
                except ValueError:
                    struct_parts.append(f'{a.lower()}={v.replace(" ", "_")[:30]}')
        structured = ' '.join(struct_parts)

        if pid in mrna:
            ranked = sorted(((g, v) for g, v in mrna[pid].items() if v == v),
                             key=lambda x: x[1], reverse=True)[:top_k_genes]
            gene_top = ' '.join(g for g, _ in ranked)
        else:
            gene_top = ''

        examples.append({
            'text':       text,
            'structured': structured,
            'gene_top':   gene_top,
            'os_months':  months,
            'os_status':  os_status,
            'label':      label,
            'study':      study_label,
        })
    return examples


def load_tcga_survival(top_k_genes=64, min_followup_months=0):
    all_examples = []
    for label, study in tcga_loader.STUDIES.items():
        ex = patients_to_survival_examples(label, study, top_k_genes=top_k_genes,
                                            min_followup_months=min_followup_months)
        n_dead = sum(1 for e in ex if e['label'] == 1)
        n_alive = sum(1 for e in ex if e['label'] == 0)
        print(f'  {label}: {len(ex)} patients  alive={n_alive}  deceased={n_dead}')
        all_examples.extend(ex)
    return all_examples


def tokenize(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower()).split()


def build_vocab(train_examples, modalities, min_freq=2):
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
    return stoi


def encode_example(ex, stoi, modalities, mask_all_but=None):
    out = []
    for mod, budget in modalities:
        if mask_all_but is not None and mod not in mask_all_but:
            tok_ids = [stoi[PAD]] * budget
        else:
            toks = tokenize(ex[mod])
            tok_ids = [stoi.get(t, stoi[UNK]) for t in toks]
            if len(tok_ids) > budget:
                tok_ids = tok_ids[:budget]
            else:
                tok_ids = tok_ids + [stoi[PAD]] * (budget - len(tok_ids))
        out.append(stoi[f'{SEP}_{mod}'])
        out.extend(tok_ids)
    return out


class SurvivalDataset(Dataset):
    def __init__(self, examples, stoi, modalities, mask_all_but=None):
        self.exs = examples; self.stoi = stoi
        self.mods = modalities; self.mask = mask_all_but
    def __len__(self): return len(self.exs)
    def __getitem__(self, i):
        ex = self.exs[i]
        ids = encode_example(ex, self.stoi, self.mods, mask_all_but=self.mask)
        return torch.tensor(ids, dtype=torch.long), ex['label']


def derive_budgets(examples, modalities_raw):
    final = []
    for mod in modalities_raw:
        lens = [len(tokenize(ex[mod])) for ex in examples]
        m = max(lens) if lens else 1
        p95 = int(np.percentile(lens, 95)) if lens else 1
        if mod == 'text': budget = min(512, m)
        elif mod == 'gene_top': budget = min(2048, m)
        else: budget = m if m <= 1.5 * p95 else max(p95, 1)
        final.append((mod, budget))
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/multimodality/tcga_survival')
    p.add_argument('--n_epochs', type=int, default=8)
    p.add_argument('--bodies', nargs='+', default=['A', 'D'])
    p.add_argument('--methods', nargs='+',
                   default=['standard', 'byte_seb', 'binary_seb'])
    p.add_argument('--ablations', nargs='+',
                   default=['all'])  # ['gene-only','gene+text','all'] also possible
    p.add_argument('--top_k_genes', type=int, default=64)
    p.add_argument('--min_followup_months', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)

    print('Loading TCGA pan5 with survival labels...', flush=True)
    examples = load_tcga_survival(top_k_genes=args.top_k_genes,
                                    min_followup_months=args.min_followup_months)
    n_total = len(examples)
    n_dead = sum(1 for e in examples if e['label'] == 1)
    print(f'\nTotal: {n_total} patients, {n_dead} deceased '
          f'({100*n_dead/n_total:.1f}% mortality)', flush=True)

    rng = random.Random(args.seed); rng.shuffle(examples)
    n_tr = int(0.7 * n_total); n_va = int(0.1 * n_total)
    train_ex = examples[:n_tr]
    val_ex = examples[n_tr:n_tr + n_va]
    test_ex = examples[n_tr + n_va:]
    print(f'Split: train={len(train_ex)} val={len(val_ex)} test={len(test_ex)}',
          flush=True)

    modalities_raw = ['text', 'structured', 'gene_top']
    modalities = derive_budgets(examples, modalities_raw)
    print(f'Modalities + budgets: {modalities}', flush=True)

    stoi = build_vocab(train_ex, modalities)
    vocab_size = len(stoi)
    print(f'Vocab size: {vocab_size}', flush=True)

    summary = {'config': vars(args), 'vocab_size': vocab_size,
               'modalities': [m[0] for m in modalities],
               'splits': {'train': len(train_ex), 'val': len(val_ex),
                           'test': len(test_ex)},
               'mortality_rate': n_dead / n_total,
               'cells': {}}

    for method in args.methods:
        for ablation in args.ablations:
            for body in args.bodies:
                key = f'{method}_{ablation}_{body}'
                print(f'\n>>> {key}', flush=True)
                random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
                if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
                try:
                    mask_all_but = None
                    if ablation == 'gene-only': mask_all_but = {'gene_top'}
                    elif ablation == 'gene+text': mask_all_but = {'gene_top', 'text'}
                    elif ablation == 'text-only': mask_all_but = {'text'}

                    tr_ds = SurvivalDataset(train_ex, stoi, modalities, mask_all_but)
                    va_ds = SurvivalDataset(val_ex, stoi, modalities, mask_all_but)
                    te_ds = SurvivalDataset(test_ex, stoi, modalities, mask_all_but)
                    coll = functools.partial(collate, pad_index=stoi[PAD])
                    bs = 16
                    tr = DataLoader(tr_ds, batch_size=bs, shuffle=True, collate_fn=coll)
                    va = DataLoader(va_ds, batch_size=bs, shuffle=False, collate_fn=coll)
                    te = DataLoader(te_ds, batch_size=bs, shuffle=False, collate_fn=coll)

                    model = MultiModalClassifier(
                        vocab_size=vocab_size, embedding_dim=64, num_classes=2,
                        method=method, body_kind=body, num_bits=64,
                        interdim='128', pad_idx=stoi[PAD]).to(device)
                    opt = optim.Adam(model.parameters(), lr=5e-4)
                    crit = nn.CrossEntropyLoss()

                    t0 = time.time()
                    best_val, best_state = 0.0, None
                    for ep in range(args.n_epochs):
                        model.train()
                        for batch in tr:
                            opt.zero_grad()
                            crit(model(batch['ids'].to(device)),
                                 batch['label'].to(device)).backward()
                            opt.step()
                        model.eval()
                        c = t = 0
                        with torch.no_grad():
                            for batch in va:
                                pred = model(batch['ids'].to(device)).argmax(-1).cpu()
                                c += (pred == batch['label']).sum().item()
                                t += batch['label'].size(0)
                        v = c / max(t, 1)
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
                    test_acc = c / max(t, 1)

                    # FILM attack
                    model.train()
                    f1s = []; n_atk = 0
                    for batch in tr:
                        if n_atk >= 12: break
                        try: f1s.append(film_attack(model, batch['ids'], batch['label'], crit, device))
                        except: pass
                        n_atk += 1
                    film_f1 = float(np.mean(f1s)) if f1s else 0.0

                    r = {
                        'method': method, 'ablation': ablation, 'body': body,
                        'val_acc': float(best_val), 'test_acc': float(test_acc),
                        'film_f1': film_f1,
                        'n_params': sum(p.numel() for p in model.parameters()),
                        'train_time_s': train_time,
                    }
                    summary['cells'][key] = r
                    print(f'  acc={test_acc:.4f}  film={film_f1:.3f}  '
                          f'params={r["n_params"]:,}  time={train_time:.1f}s', flush=True)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    summary['cells'][key] = {'error': str(e)}
                with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
                    json.dump(summary, f, indent=2)

    # Pretty-print final table
    print('\n=== TCGA Survival (binary mortality) results ===', flush=True)
    print(f'{"method":12s} {"ablation":10s} {"body":4s} {"acc":7s} {"film":7s} {"params":12s}',
          flush=True)
    for k, r in summary['cells'].items():
        if 'test_acc' in r:
            print(f'{r["method"]:12s} {r["ablation"]:10s} {r["body"]:4s} '
                  f'{r["test_acc"]:.4f}  {r["film_f1"]:.3f}  {r["n_params"]:>10,}',
                  flush=True)

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nWrote {args.out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
