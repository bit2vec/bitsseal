#!/usr/bin/env python3
"""Unified master tables across all 001 / EmByte / Subword experiments."""

import json
import os

ROOT = '.'
HC_BASE = os.path.join(ROOT, 'results/healthcare')
DM_BASE = os.path.join(ROOT, 'results/domains')
LLG_BASE = os.path.join(ROOT, 'results/healthcare_privacy')
FILM_BASE = os.path.join(ROOT, 'results/healthcare_film')
LAMP_BASE = os.path.join(ROOT, 'results/healthcare_lamp')

DATASETS = [
    ('ade',       'ade_corpus_v2',           HC_BASE, 2,   'Healthcare'),
    ('pubmedqa',  'pubmed_qa',               HC_BASE, 3,   'Healthcare'),
    ('medmcqa',   'medmcqa',                 HC_BASE, 4,   'Healthcare'),
    ('finance',   'twitter_fin_news',        DM_BASE, 3,   'Finance'),
    ('education', 'openbookqa',              DM_BASE, 4,   'Education'),
    ('legal',     'lex_glue/ledgar',         DM_BASE, 100, 'Legal'),
    ('mental',    'go_emotions',             DM_BASE, 28,  'Mental health'),
    ('contract',  'lex_glue/unfair_tos',     DM_BASE, 2,   'Legal/contract'),
    ('ecthr',     'lex_glue/ecthr_a',        DM_BASE, 10,  'Legal/human rights'),
]
METHODS = ['standard', 'byte_seb', 'binary_seb']


def load_or_none(path):
    if os.path.exists(path):
        return json.load(open(path))
    return None


def acc_table():
    print("\n" + "="*120)
    print("TABLE 1: Classification accuracy + model size + training time")
    print("="*120)
    print(f"{'Domain':<18} {'Dataset':<25} {'Cls':<5} {'Method':<11} "
          f"{'Test':<7} {'Val':<7} {'Params':<11} {'Embed':<10} {'Embed MB':<9} {'Time(s)':<8}")
    print('-'*120)
    for short, full, base, cls, domain in DATASETS:
        for m in METHODS:
            p = os.path.join(base, short, f'{short}_{m}.json')
            d = load_or_none(p)
            if d:
                emb_mb = d.get('embed_params', 0) * 4 / 1024 / 1024
                test = d.get('test_acc', 0)
                val = d.get('best_val_acc', 0)
                print(f"{domain:<18} {full:<25} {cls:<5} {m:<11} "
                      f"{test:<7.4f} {val:<7.4f} {d.get('total_params',0):<11,d} "
                      f"{d.get('embed_params',0):<10,d} {emb_mb:<9.3f} "
                      f"{d.get('total_time_s',0):<8.0f}")


def llg_table():
    print("\n" + "="*100)
    print("TABLE 2: LLG (Label Leakage from Gradients) — Attack Success Rate (lower = more private)")
    print("="*100)
    print(f"{'Dataset':<25} {'Cls':<5} {'Chance':<7} "
          f"{'Subword ASR':<12} {'EmByte ASR':<12} {'001 ASR':<10}  Best")
    print('-'*100)
    for short, full, _, cls, _ in DATASETS:
        if short in ('contract', 'ecthr'):
            continue  # not in LLG runs
        row = {}
        chance = 0.0
        for m in METHODS:
            p = os.path.join(LLG_BASE, f'{short}_{m}_llg.json')
            d = load_or_none(p)
            if d:
                row[m] = d['overall_mean_asr']
                chance = d['chance_level']
        if not row: continue
        best = min(row, key=row.get)
        best_label = {'standard': 'Subword', 'byte_seb': 'EmByte', 'binary_seb': '001'}[best]
        print(f"{full:<25} {cls:<5} {chance:<7.3f} "
              f"{row.get('standard',0):<12.3f} {row.get('byte_seb',0):<12.3f} "
              f"{row.get('binary_seb',0):<10.3f}  {best_label}")


def film_lamp_table(base, name):
    print("\n" + "="*120)
    print(f"TABLE 3{name[0]}: {name} — gradient-based input recovery (lower recall = more private)")
    print("="*120)
    print(f"{'Dataset':<25} "
          f"{'Subword P':<10} {'Subword R':<10} {'EmByte P':<10} {'EmByte R':<10} "
          f"{'001 P':<8} {'001 R':<8}")
    print('-'*120)
    for short, full, _, cls, _ in DATASETS:
        if short in ('contract', 'ecthr'):
            continue
        row = {}
        for m in METHODS:
            p = os.path.join(base, f'{short}_{m}_{name.lower()}.json')
            d = load_or_none(p)
            if d:
                row[m] = d.get('metrics', d)
        if not row: continue
        s = row.get('standard', {}); b = row.get('byte_seb', {}); bi = row.get('binary_seb', {})
        print(f"{full:<25} "
              f"{s.get('precision',0):<10.3f} {s.get('recall',0):<10.3f} "
              f"{b.get('precision',0):<10.3f} {b.get('recall',0):<10.3f} "
              f"{bi.get('precision',0):<8.3f} {bi.get('recall',0):<8.3f}")


def compression_summary():
    print("\n" + "="*100)
    print("TABLE 4: Embedding compression summary (001 vs Subword/EmByte)")
    print("="*100)
    print(f"{'Dataset':<25} {'Cls':<5} {'Vocab':<7} {'Sub Emb #':<11} {'001 Emb #':<11} "
          f"{'vs Sub':<10} {'vs EmB':<10}")
    print('-'*100)
    for short, full, base, cls, _ in DATASETS:
        sd = load_or_none(os.path.join(base, short, f'{short}_standard.json'))
        bd = load_or_none(os.path.join(base, short, f'{short}_byte_seb.json'))
        bid = load_or_none(os.path.join(base, short, f'{short}_binary_seb.json'))
        if not sd or not bid: continue
        s_emb = sd['embed_params']; bi_emb = bid['embed_params']
        b_emb = bd['embed_params'] if bd else 1
        vocab = sd.get('vocab_size', '-')
        print(f"{full:<25} {cls:<5} {vocab:<7} "
              f"{s_emb:<11,d} {bi_emb:<11,d} "
              f"{s_emb/bi_emb:<10.1f}× {b_emb/bi_emb:<10.1f}×")


def lm_table():
    print("\n" + "="*70)
    print("TABLE 5: Language Modeling on WikiText-103 (valid PPL, lower is better)")
    print("="*70)
    p_byte = os.path.join(ROOT, 'results/language_modeling/results_byte_seb.json')
    p_bin  = os.path.join(ROOT, 'results/language_modeling/results_binary_seb.json')
    print(f"{'Method':<15} {'Final PPL':<11} {'Test PPL':<10} {'Total #':<11}")
    # Subword (parsed from log)
    log = open(os.path.join(ROOT, 'results/language_modeling/log.txt')).read() \
          if os.path.exists(os.path.join(ROOT, 'results/language_modeling/log.txt')) else ''
    import re
    ppls = [float(m.group(1)) for m in re.finditer(r'Valid PPL:\s*([\d.]+)', log)]
    if ppls:
        print(f"{'Subword':<15} {min(ppls):<11.2f} {'n/a':<10} {'~46M':<11}")
    for name, key in [('EmByte', p_byte), ('001 (binary)', p_bin)]:
        d = load_or_none(key)
        if d:
            print(f"{name:<15} {d.get('best_valid_ppl', 0):<11.2f} "
                  f"{d.get('test_ppl', 0):<10.2f} "
                  f"{d.get('model_memory', {}).get('total_params', 0)/1e6:.1f}M")


if __name__ == '__main__':
    acc_table()
    compression_summary()
    llg_table()
    film_lamp_table(FILM_BASE, 'FILM')
    film_lamp_table(LAMP_BASE, 'LAMP')
    lm_table()
