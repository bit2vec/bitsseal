#!/usr/bin/env python3
"""Bits cross-modal smoke tests.

Uses BinarySubwordConcateOnehot exactly as published in the original Bits paper.
The ONLY thing that differs per task is the upstream subword tokenizer:
  - L2G smoke      : GPT-2 BPE (text) for pretrain, DNABERT-2 BPE (DNA) for fine-tune.
  - CodonBERT smoke: codon tokenizer (V=64) for both pretrain and fine-tune (mRNA).
  - CaLM smoke     : codon (V=64) + AA (V=20) cross-modal pretrain; fine-tune on protein property.

The downstream pipeline is identical: subword_id -> stochastic_mapping -> bit_onehot -> concat -> FFN -> dense.

Smoke = small (~100K pretrain, ~1 epoch, ~30-90 min on one GPU). Confirms the
pipeline works end-to-end and gives a signal number to decide full experiments.
"""
import argparse, functools, json, os, random, sys, time, math
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import datasets
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, GPT2Tokenizer, AutoTokenizer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
# binary_subword_sentiment.py lives in different places on different machines:
# local repo: software/Senti_analysis/, alien: software/lm/../Senti_analysis/, vast: software/
for cand in [os.path.join(ROOT, 'software', 'Senti_analysis'),
             os.path.join(ROOT, 'software'),
             os.path.join(THIS_DIR, '..')]:
    if os.path.isfile(os.path.join(cand, 'binary_subword_sentiment.py')):
        sys.path.insert(0, cand)
        break

# === ORIGINAL PAPER MODULE, UNTOUCHED ===
from binary_subword_sentiment import BinarySubwordConcateOnehot

HYENA_MODEL_ID = 'LongSafari/hyenadna-tiny-1k-seqlen-hf'


# ============================================================================
# Tokenizers (the subword step that feeds Bits)
# ============================================================================

CODON_VOCAB = {f"{a}{b}{c}": idx + 2  # 0=pad, 1=unk; codons 2..65
               for idx, (a, b, c) in enumerate(
                   (x, y, z) for x in 'ACGT' for y in 'ACGT' for z in 'ACGT')}
CODON_VOCAB_SIZE = 64 + 2  # 66 with pad/unk
AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'  # 20 standard AAs
AA_VOCAB = {a: idx + 2 for idx, a in enumerate(AA_LETTERS)}  # 0=pad, 1=unk
AA_VOCAB_SIZE = 20 + 2


def codon_tokenize(seq, max_len=512):
    """DNA/mRNA -> codon IDs (V=66). Frame 0 only. Skip codons with non-ACGT."""
    seq = seq.upper().replace('U', 'T')
    out = []
    for i in range(0, min(len(seq) - 2, max_len * 3), 3):
        c = seq[i:i+3]
        out.append(CODON_VOCAB.get(c, 1))  # 1=unk
        if len(out) >= max_len:
            break
    return out


def aa_tokenize(seq, max_len=512):
    """Protein -> AA IDs (V=22)."""
    seq = seq.upper()
    out = [AA_VOCAB.get(a, 1) for a in seq[:max_len]]
    return out


def get_dnabert2_tokenizer():
    """DNABERT-2 BPE tokenizer (V~4096), the published learned-BPE on DNA."""
    return AutoTokenizer.from_pretrained('zhihan1996/DNABERT-2-117M',
                                          trust_remote_code=True)


# ============================================================================
# Model builders (Hyena-tiny body + Bits encoder)
# ============================================================================

def build_bits_hyena_causal_lm(vocab_size, num_bits=64, interdim='256', dropout=0.1):
    config = AutoConfig.from_pretrained(HYENA_MODEL_ID, trust_remote_code=True)
    config.vocab_size = vocab_size
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    enc = BinarySubwordConcateOnehot(
        num_embeddings=vocab_size, embedding_dim=config.d_model,
        num_bits=num_bits, interdim=interdim, relu_dropout=dropout,
        subword_binary_file='')
    model.set_input_embeddings(enc)
    return model, config


def build_bits_hyena_classifier(vocab_size, num_classes, num_bits=64, interdim='256',
                                 dropout=0.1, problem_type='single_label_classification'):
    config = AutoConfig.from_pretrained(HYENA_MODEL_ID, trust_remote_code=True)
    config.vocab_size = vocab_size
    config.num_labels = num_classes
    config.problem_type = problem_type
    config.pad_token_id = 0
    model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True)
    enc = BinarySubwordConcateOnehot(
        num_embeddings=vocab_size, embedding_dim=config.d_model,
        num_bits=num_bits, interdim=interdim, relu_dropout=dropout,
        subword_binary_file='')
    model.set_input_embeddings(enc)
    return model, config


def transfer_pretrain_to_classifier(classifier_model, pretrain_state):
    md = classifier_model.state_dict()
    n_t = 0
    for k, v in pretrain_state.items():
        if 'subword2binary' in k or k.startswith('lm_head.'):
            continue
        if k in md and md[k].shape == v.shape:
            md[k] = v.clone()
            n_t += 1
    classifier_model.load_state_dict(md, strict=False)
    print(f'  transferred {n_t} tensors from pretrain', flush=True)


# ============================================================================
# Generic pretrain / finetune loops
# ============================================================================

class SeqDataset(Dataset):
    def __init__(self, token_lists, max_len, pad_id=0):
        self.x = token_lists
        self.max_len = max_len
        self.pad = pad_id
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        ids = self.x[i][:self.max_len]
        ids = ids + [self.pad] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class LabeledDataset(Dataset):
    def __init__(self, token_lists, labels, max_len, pad_id=0, label_dtype=torch.long):
        self.x = token_lists
        self.y = labels
        self.max_len = max_len
        self.pad = pad_id
        self.label_dtype = label_dtype
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        ids = self.x[i][:self.max_len]
        ids = ids + [self.pad] * (self.max_len - len(ids))
        return {'input_ids': torch.tensor(ids, dtype=torch.long),
                'label': torch.tensor(self.y[i], dtype=self.label_dtype)}


def causal_pretrain(model, tokens, args, device):
    ds = SeqDataset(tokens, max_len=args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    t0 = time.time()
    step = 0
    losses = []
    for ep in range(args.n_epochs):
        for batch in loader:
            ids = batch.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(input_ids=ids, labels=ids)
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(out.loss.item())
            step += 1
            if step % 100 == 0:
                ppl = math.exp(min(20, np.mean(losses[-100:])))
                print(f'  step {step:5d}  loss={np.mean(losses[-100:]):.4f}  ppl={ppl:.2f}', flush=True)
    print(f'  pretrain done in {time.time()-t0:.1f}s, final loss={np.mean(losses[-100:]):.4f}', flush=True)
    return {'final_loss': float(np.mean(losses[-100:])), 'steps': step}


def evaluate_classifier(model, loader, device, regression=False):
    model.eval()
    preds, labs = [], []
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for batch in loader:
            ids = batch['input_ids'].to(device)
            lab = batch['label'].to(device)
            out = model(input_ids=ids)
            if regression:
                preds.extend(out.logits.float().squeeze(-1).cpu().numpy().tolist())
                labs.extend(lab.float().cpu().numpy().tolist())
            else:
                preds.extend(out.logits.argmax(-1).cpu().numpy().tolist())
                labs.extend(lab.cpu().numpy().tolist())
    if regression:
        preds, labs = np.array(preds), np.array(labs)
        try:
            from scipy.stats import spearmanr
            sp, _ = spearmanr(preds, labs)
        except Exception:
            sp = float('nan')
        mse = float(np.mean((preds - labs) ** 2))
        return {'spearman': float(sp) if not math.isnan(float(sp)) else 0.0,
                'mse': mse, 'n': len(labs)}
    else:
        c = sum(int(p == l) for p, l in zip(preds, labs))
        return {'acc': c / max(len(labs), 1), 'n': len(labs)}


def supervised_finetune(model, train_pairs, val_pairs, test_pairs, args, device, regression=False):
    pad = 0
    label_dtype = torch.float if regression else torch.long
    tr = DataLoader(LabeledDataset([t for t, l in train_pairs], [l for t, l in train_pairs],
                                    args.max_len, pad, label_dtype),
                    batch_size=args.batch_size, shuffle=True)
    va = DataLoader(LabeledDataset([t for t, l in val_pairs], [l for t, l in val_pairs],
                                    args.max_len, pad, label_dtype),
                    batch_size=args.batch_size, shuffle=False)
    te = DataLoader(LabeledDataset([t for t, l in test_pairs], [l for t, l in test_pairs],
                                    args.max_len, pad, label_dtype),
                    batch_size=args.batch_size, shuffle=False)
    opt = optim.AdamW(model.parameters(), lr=args.ft_lr, weight_decay=0.01)
    t0 = time.time()
    best_val = -1e9
    best_state = None
    for ep in range(args.ft_epochs):
        model.train()
        for batch in tr:
            ids = batch['input_ids'].to(device)
            lab = batch['label'].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if regression:
                    out = model(input_ids=ids)
                    loss = F.mse_loss(out.logits.float().squeeze(-1), lab.float())
                else:
                    out = model(input_ids=ids, labels=lab)
                    loss = out.loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        v = evaluate_classifier(model, va, device, regression=regression)
        score = v.get('spearman', v.get('acc', 0))
        print(f'  epoch {ep+1}/{args.ft_epochs}  val={score:.4f}  {v}', flush=True)
        if score > best_val:
            best_val = score
            best_state = {k: vv.detach().cpu().clone() for k, vv in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test_res = evaluate_classifier(model, te, device, regression=regression)
    print(f'  test={test_res}  time={time.time()-t0:.1f}s', flush=True)
    return {'best_val': best_val, 'test': test_res, 'train_time_s': time.time() - t0}


# ============================================================================
# Task-specific loaders
# ============================================================================

def load_l2g_smoke(args):
    """L2G smoke: pretrain on WikiText (GPT-2 BPE), finetune on nt_promoter_all (DNABERT-2 BPE).
    Same Bits encoder for both — vocab_size is the UNION of GPT-2 and DNABERT-2 BPE."""
    print('loading WikiText sample...', flush=True)
    wt = datasets.load_dataset('wikitext', 'wikitext-103-v1', split='train')
    gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
    pre_tokens = []
    for row in wt.select(range(min(len(wt), args.pretrain_size))):
        t = row['text'].strip()
        if t:
            ids = gpt2.encode(t)[:args.max_len]
            if len(ids) >= 16:
                pre_tokens.append(ids)
    print(f'  WikiText pretrain corpus: {len(pre_tokens)} sequences', flush=True)

    print('loading nt_promoter_all (DNABERT-2 BPE)...', flush=True)
    nt = datasets.load_dataset('InstaDeepAI/nucleotide_transformer_downstream_tasks')
    nt = {sp: nt[sp].filter(lambda r: r['task'] == 'promoter_all') for sp in nt}
    dnabert2 = get_dnabert2_tokenizer()
    def tok_dna(row):
        return dnabert2.encode(row['sequence'][:1024], add_special_tokens=False)[:args.max_len]
    full_train = list(nt['train'])
    test = list(nt['test'])
    random.shuffle(full_train)
    nv = max(1, int(len(full_train) * 0.1))
    val, train = full_train[:nv], full_train[nv:]
    train_pairs = [(tok_dna(r), int(r['label'])) for r in train[:args.finetune_size]]
    val_pairs   = [(tok_dna(r), int(r['label'])) for r in val[:args.finetune_size // 10]]
    test_pairs  = [(tok_dna(r), int(r['label'])) for r in test]
    # Vocab union: GPT-2 (50257) + DNABERT-2 (~4096). Use max+offset to keep disjoint.
    gpt2_v = gpt2.vocab_size
    dnabert2_v = dnabert2.vocab_size
    # Re-offset DNA tokens to live above text tokens
    train_pairs = [([t + gpt2_v for t in ids], l) for ids, l in train_pairs]
    val_pairs   = [([t + gpt2_v for t in ids], l) for ids, l in val_pairs]
    test_pairs  = [([t + gpt2_v for t in ids], l) for ids, l in test_pairs]
    union_v = gpt2_v + dnabert2_v
    n_classes = 2
    return {'pre_tokens': pre_tokens, 'train': train_pairs, 'val': val_pairs, 'test': test_pairs,
            'vocab_size': union_v, 'n_classes': n_classes, 'regression': False}


def load_codonbert_smoke(args):
    """CodonBERT smoke: pretrain on mRNA codons (V=66), finetune on mRNA stability (CodonBERT GitHub data)."""
    # Pretrain corpus: use NT promoter_all coding-frame DNA as a stand-in mRNA corpus for smoke.
    # (Real CodonBERT corpus is 10M mRNA; we just need a smoke signal.)
    print('loading mRNA-like pretrain corpus...', flush=True)
    nt = datasets.load_dataset('InstaDeepAI/nucleotide_transformer_downstream_tasks')
    src = nt['train'].filter(lambda r: r['task'] == 'splice_sites_all')
    pre_tokens = []
    for row in src.select(range(min(len(src), args.pretrain_size))):
        ids = codon_tokenize(row['sequence'], max_len=args.max_len)
        if len(ids) >= 16:
            pre_tokens.append(ids)
    print(f'  pretrain codon corpus: {len(pre_tokens)} sequences', flush=True)

    # Fine-tune: nt_H3K4me3 (mRNA stability proxy, binary classification). Real CodonBERT data
    # would need git clone of Sanofi-Public/CodonBERT; for smoke we use a closest binary task.
    print('loading H3K4me3 as mRNA-stability proxy...', flush=True)
    ft = nt['train'].filter(lambda r: r['task'] == 'H3K4me3')
    fte = nt['test'].filter(lambda r: r['task'] == 'H3K4me3')
    train_rows = list(ft)
    test_rows = list(fte)
    random.shuffle(train_rows)
    nv = max(1, int(len(train_rows) * 0.1))
    val_rows, train_rows = train_rows[:nv], train_rows[nv:]
    def to_pair(r):
        return (codon_tokenize(r['sequence'], max_len=args.max_len), int(r['label']))
    train_pairs = [to_pair(r) for r in train_rows[:args.finetune_size]]
    val_pairs   = [to_pair(r) for r in val_rows[:args.finetune_size // 10]]
    test_pairs  = [to_pair(r) for r in test_rows]
    return {'pre_tokens': pre_tokens, 'train': train_pairs, 'val': val_pairs, 'test': test_pairs,
            'vocab_size': CODON_VOCAB_SIZE, 'n_classes': 2, 'regression': False}


def load_calm_smoke(args):
    """CaLM smoke: cross-modal pretrain on codon (mRNA) + AA (protein), finetune on protein property.
    Bits encoder sees a UNION vocab: codons 0..65 + AAs (offset by 66) = 66..87. V_total=88.
    """
    # Pretrain corpus: mix codon-tokenized DNA + AA-tokenized protein.
    print('building cross-modal pretrain corpus...', flush=True)
    nt = datasets.load_dataset('InstaDeepAI/nucleotide_transformer_downstream_tasks')
    src = nt['train'].filter(lambda r: r['task'] == 'splice_sites_all')
    codon_offset = 0
    aa_offset = CODON_VOCAB_SIZE  # AA IDs live above codon IDs in the union
    pre_tokens = []
    for row in src.select(range(min(len(src), args.pretrain_size // 2))):
        ids = codon_tokenize(row['sequence'], max_len=args.max_len)
        if len(ids) >= 16:
            pre_tokens.append([t + codon_offset for t in ids])
    # Protein corpus: use UniRef50 sample via HF
    try:
        print('loading UniRef50 sample...', flush=True)
        ur = datasets.load_dataset('agemagician/uniref50', split='train', streaming=True)
        prot_n = 0
        for row in ur:
            if prot_n >= args.pretrain_size // 2:
                break
            ids = aa_tokenize(row['text'], max_len=args.max_len)
            if len(ids) >= 16:
                pre_tokens.append([t + aa_offset for t in ids])
                prot_n += 1
    except Exception as e:
        print(f'  WARNING: UniRef50 unavailable ({e}); using codon-only pretrain', flush=True)
    print(f'  cross-modal pretrain: {len(pre_tokens)} sequences', flush=True)

    # Fine-tune: a protein task. For smoke use a small downstream — we'll use UniRef50 random pair
    # for now, but ideally a TAPE task. Try HF 'Synthyra/protein_property_stability' as fallback.
    print('loading protein property task...', flush=True)
    try:
        tape = datasets.load_dataset('Synthyra/TAPE-stability')
        train_rows = list(tape['train'])
        test_rows = list(tape['test'])
        seq_key = 'primary' if 'primary' in train_rows[0] else 'sequence'
        label_key = 'stability_score' if 'stability_score' in train_rows[0] else 'label'
        regression = isinstance(train_rows[0][label_key], float)
    except Exception:
        # Fallback: use H3 as a protein-property proxy (NT chromatin task on AA-tokenized DNA-derived seq)
        print('  TAPE unavailable; using H3 with AA-tokenized translation as proxy', flush=True)
        h3_tr = nt['train'].filter(lambda r: r['task'] == 'H3')
        h3_te = nt['test'].filter(lambda r: r['task'] == 'H3')
        train_rows = list(h3_tr)
        test_rows = list(h3_te)
        seq_key, label_key = 'sequence', 'label'
        regression = False
    random.shuffle(train_rows)
    nv = max(1, int(len(train_rows) * 0.1))
    val_rows, train_rows = train_rows[:nv], train_rows[nv:]
    def to_pair(r):
        s = r[seq_key]
        # Best-effort: if looks like AA, use aa_tokenize; if looks like DNA, codon-tokenize
        is_dna = set(s.upper()[:50]).issubset(set('ACGTN'))
        ids = (codon_tokenize(s, max_len=args.max_len) if is_dna
               else [t + aa_offset for t in aa_tokenize(s, max_len=args.max_len)])
        if is_dna:
            ids = [t + codon_offset for t in ids]
        y = r[label_key]
        if regression:
            return (ids, float(y))
        return (ids, int(y))
    train_pairs = [to_pair(r) for r in train_rows[:args.finetune_size]]
    val_pairs   = [to_pair(r) for r in val_rows[:args.finetune_size // 10]]
    test_pairs  = [to_pair(r) for r in test_rows]
    vocab_size = CODON_VOCAB_SIZE + AA_VOCAB_SIZE
    n_classes = 1 if regression else (max(p[1] for p in train_pairs) + 1)
    return {'pre_tokens': pre_tokens, 'train': train_pairs, 'val': val_pairs, 'test': test_pairs,
            'vocab_size': vocab_size, 'n_classes': n_classes, 'regression': regression}


TASK_LOADERS = {'l2g': load_l2g_smoke, 'codonbert': load_codonbert_smoke, 'calm': load_calm_smoke}


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=list(TASK_LOADERS.keys()), required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--pretrain_size', type=int, default=20000,
                   help='Number of pretrain sequences (smoke = 20K)')
    p.add_argument('--finetune_size', type=int, default=5000)
    p.add_argument('--max_len', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--ft_epochs', type=int, default=5)
    p.add_argument('--ft_lr', type=float, default=1e-4)
    p.add_argument('--num_bits', type=int, default=64)
    p.add_argument('--interdim', default='256')
    p.add_argument('--skip_pretrain', action='store_true',
                   help='Skip pretrain, fine-tune from random init')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'task={args.task}  device={device}  out={args.out_dir}', flush=True)

    data = TASK_LOADERS[args.task](args)
    V = data['vocab_size']
    n_classes = data['n_classes']
    print(f'  V_union={V}  n_classes={n_classes}  regression={data["regression"]}', flush=True)
    print(f'  n_pretrain={len(data["pre_tokens"])}  train={len(data["train"])}  '
          f'val={len(data["val"])}  test={len(data["test"])}', flush=True)

    # ----- Phase 1: pretrain (causal LM) -----
    pre_state = None
    if not args.skip_pretrain and len(data['pre_tokens']) >= 100:
        print('=== Pretrain (causal LM) ===', flush=True)
        pre_model, _ = build_bits_hyena_causal_lm(
            V, num_bits=args.num_bits, interdim=args.interdim)
        pre_model = pre_model.to(device)
        pre_stats = causal_pretrain(pre_model, data['pre_tokens'], args, device)
        pre_state = {k: v.detach().cpu().clone() for k, v in pre_model.state_dict().items()}
        del pre_model
        torch.cuda.empty_cache()
    else:
        print('=== Skipping pretrain ===', flush=True)
        pre_stats = None

    # ----- Phase 2: fine-tune -----
    print('=== Fine-tune ===', flush=True)
    problem_type = 'regression' if data['regression'] else 'single_label_classification'
    cls_model, _ = build_bits_hyena_classifier(
        V, n_classes if not data['regression'] else 1,
        num_bits=args.num_bits, interdim=args.interdim,
        problem_type=problem_type)
    if pre_state is not None:
        transfer_pretrain_to_classifier(cls_model, pre_state)
    cls_model = cls_model.to(device)
    ft_stats = supervised_finetune(cls_model, data['train'], data['val'], data['test'],
                                    args, device, regression=data['regression'])

    summary = {
        'task': args.task, 'config': vars(args),
        'vocab_size': V, 'n_classes': n_classes,
        'regression': data['regression'],
        'n_pretrain_seqs': len(data['pre_tokens']),
        'n_train': len(data['train']), 'n_val': len(data['val']), 'n_test': len(data['test']),
        'pretrain_stats': pre_stats,
        'finetune_stats': ft_stats,
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved {args.out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
