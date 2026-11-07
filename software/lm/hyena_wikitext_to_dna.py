#!/usr/bin/env python3
"""HyenaDNA-tiny architecture + Bits/EmByte/Subword encoder + WikiText pretrain
→ DNA fine-tune comparison.

We take the HyenaDNA-tiny-1k-seqlen-hf architecture (2 Hyena layers, d_model=128,
~436K params) and replace its native single-nucleotide nn.Embedding(5, 128) with
our V-independent Bits or EmByte encoder, OR keep a standard nn.Embedding for
the comparison baseline.

Phase 1 (pretrain): causal LM next-token prediction on full WikiText-103 with
GPT-2 BPE tokenizer (V=50,257), output_dim=128.

Phase 2 (fine-tune): swap vocab to DNA k=6 mer (V≈4,096), load Phase-1 weights
(skip vocab-dependent buffers), fine-tune as classifier on GB human_promoters.

Compare W (WikiText init) vs R (random init).

Usage:
    # pretrain
    python hyena_wikitext_to_dna.py --phase pretrain --encoder binary_seb \\
        --out_dir results/hyena_wikitext/binary_seb

    # fine-tune from pretrained
    python hyena_wikitext_to_dna.py --phase finetune --encoder binary_seb \\
        --pretrain_ckpt results/hyena_wikitext/binary_seb/pretrain.pt

    # fine-tune from random
    python hyena_wikitext_to_dna.py --phase finetune --encoder binary_seb \\
        --pretrain_ckpt none
"""
import argparse, copy, functools, json, os, random, re, sys, time
from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import datasets
from transformers import GPT2Tokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'software', 'Senti_analysis'))

from binary_subword_sentiment import BinarySubwordConcateOnehot
from byte_subword_concate_onehot import ByteSubwordConcateOnehot
from bits_variants import StructuredBinarySubwordConcateOnehot, HashEmbeddingEncoder


HYENA_MODEL_ID = os.environ.get('HYENA_MODEL_ID', 'LongSafari/hyenadna-tiny-1k-seqlen-hf')

GB_HF_IDS = {
    'gb_human_promoters':
        'katarinagresova/Genomic_Benchmarks_human_nontata_promoters',
    'gb_human_enhancers':
        'katarinagresova/Genomic_Benchmarks_human_enhancers_cohn',
    'gb_human_enhancers_ensembl':
        'katarinagresova/Genomic_Benchmarks_human_enhancers_ensembl',
    'gb_human_ocr_ensembl':
        'katarinagresova/Genomic_Benchmarks_human_ocr_ensembl',
    'gb_human_regulatory':
        'katarinagresova/Genomic_Benchmarks_human_ensembl_regulatory',
    'gb_demo_human_or_worm':
        'katarinagresova/Genomic_Benchmarks_demo_human_or_worm',
    'gb_demo_coding_vs_intergenomic':
        'katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs',
    'gb_dummy_mouse_enhancers_ensembl':
        'katarinagresova/Genomic_Benchmarks_dummy_mouse_enhancers_ensembl',
}

# Nucleotide Transformer downstream tasks (Dalla-Torre et al., NeurIPS 2023).
# All 18 tasks live as configs in one HF repo, so the value is a (repo, config) tuple.
_NT_REPO = 'InstaDeepAI/nucleotide_transformer_downstream_tasks'
NT_HF_IDS = {f'nt_{c}': (_NT_REPO, c) for c in [
    'H3', 'H3K14ac', 'H3K36me3', 'H3K4me1', 'H3K4me2', 'H3K4me3',
    'H3K79me3', 'H3K9ac', 'H4', 'H4ac',
    'enhancers', 'enhancers_types',
    'promoter_all', 'promoter_no_tata', 'promoter_tata',
    'splice_sites_donors', 'splice_sites_acceptors', 'splice_sites_all',
]}
GB_HF_IDS.update(NT_HF_IDS)


# ============================================================================
# Embedding swap utilities
# ============================================================================

class _ID(nn.Module):
    """Identity wrapper so we can drop in our encoder where word_embeddings was."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, ids):
        return self.encoder(ids)


def _simhash_codes_for_vocab(itos, num_bits=64, F_=2048, seed=0):
    """SimHash code per token via bag-of-byte-trigrams + fixed Gaussian hyperplanes.
    Returns LongTensor [V, num_bits] in {0,1}. Matches the SpinHash paper Eq. 3."""
    rng = np.random.RandomState(seed)
    R = rng.randn(F_, num_bits).astype('float32')
    V = len(itos)
    feats = np.zeros((V, F_), dtype='float32')
    for i, tok in enumerate(itos):
        s = ('  ' + str(tok) + '  ').encode('utf-8', errors='replace')
        for j in range(len(s) - 2):
            tri = s[j] * 65536 + s[j+1] * 256 + s[j+2]
            feats[i, tri % F_] += 1.0
    scores = feats @ R                            # [V, num_bits]
    codes = (scores >= 0).astype('int64')
    return torch.from_numpy(codes)


def make_encoder(encoder_type, vocab_size, d_model, num_bits=64, num_bytes=8,
                 interdim='256', dropout=0.1, itos=None):
    if encoder_type == 'standard':
        return nn.Embedding(vocab_size, d_model)
    if encoder_type == 'binary_seb':
        return BinarySubwordConcateOnehot(
            num_embeddings=vocab_size, embedding_dim=d_model,
            num_bits=num_bits, interdim=interdim, relu_dropout=dropout,
            subword_binary_file='')
    if encoder_type == 'simhash_seb':
        assert itos is not None, 'simhash_seb requires itos (vocabulary list)'
        import tempfile
        tmp = os.path.join(tempfile.mkdtemp(), f'simhash_codes_{os.getpid()}.pt')
        codes = _simhash_codes_for_vocab(itos, num_bits=num_bits)
        assert codes.shape == (vocab_size, num_bits), f'{codes.shape} vs ({vocab_size},{num_bits})'
        torch.save(codes, tmp)
        return BinarySubwordConcateOnehot(
            num_embeddings=vocab_size, embedding_dim=d_model,
            num_bits=num_bits, interdim=interdim, relu_dropout=dropout,
            subword_binary_file=tmp)
    if encoder_type == 'byte_seb':
        import tempfile
        tmp = os.path.join(tempfile.mkdtemp(), f'byte_map_{os.getpid()}.pt')
        return ByteSubwordConcateOnehot(
            num_embeddings=vocab_size, embedding_dim=d_model,
            interdim=interdim, relu_dropout=dropout, subword_bytes_file=tmp)
    if encoder_type == 'binary_struct':
        return StructuredBinarySubwordConcateOnehot(
            num_embeddings=vocab_size, embedding_dim=d_model,
            num_bits=num_bits, interdim=interdim, relu_dropout=dropout,
            subword_binary_file='')
    if encoder_type == 'hash_emb':
        return HashEmbeddingEncoder(
            num_embeddings=vocab_size, embedding_dim=d_model,
            K=2, B_buckets=10000)
    raise ValueError(encoder_type)


def build_hyena_causal_lm(vocab_size, encoder_type, num_bits=64, interdim='256',
                          dropout=0.1, max_seq_len=1024, itos=None):
    """Return a HyenaDNAForCausalLM with our chosen encoder swapped in."""
    config = AutoConfig.from_pretrained(HYENA_MODEL_ID, trust_remote_code=True)
    config.vocab_size = vocab_size
    config.max_seq_len = max(max_seq_len + 16, config.max_seq_len)
    # randomize the body — we want from-scratch Hyena pretrain on WikiText
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    # Replace the word embedding
    new_emb = make_encoder(encoder_type, vocab_size, config.d_model,
                            num_bits=num_bits, interdim=interdim, dropout=dropout,
                            itos=itos)
    model.set_input_embeddings(new_emb)
    return model, config


def build_hyena_classifier(vocab_size, encoder_type, num_classes, num_bits=64,
                            interdim='256', dropout=0.1, max_seq_len=1024, itos=None):
    """HyenaDNAForSequenceClassification with our encoder."""
    config = AutoConfig.from_pretrained(HYENA_MODEL_ID, trust_remote_code=True)
    config.vocab_size = vocab_size
    config.max_seq_len = max(max_seq_len + 16, config.max_seq_len)
    config.num_labels = num_classes
    config.problem_type = 'single_label_classification'
    config.pad_token_id = 0
    model = AutoModelForSequenceClassification.from_config(
        config, trust_remote_code=True)
    new_emb = make_encoder(encoder_type, vocab_size, config.d_model,
                            num_bits=num_bits, interdim=interdim, dropout=dropout,
                            itos=itos)
    model.set_input_embeddings(new_emb)
    return model, config


def transfer_pretrain_to_classifier(classifier_model, pretrain_state, verbose=True):
    """Copy weights from pretrain (HyenaDNAForCausalLM-style) into classifier
    (HyenaDNAForSequenceClassification-style). Skip vocab-dependent buffers
    (subword2binary, subword2bytes) and the LM head (output projection)."""
    md = classifier_model.state_dict()
    transferred, skipped = [], []
    for k, v in pretrain_state.items():
        # Vocab-dependent buffers regenerated for new vocab
        if 'subword2binary' in k or 'subword2bytes' in k or 'subword' in k:
            skipped.append((k, 'vocab-dependent buffer'))
            continue
        if k.startswith('lm_head.'):
            skipped.append((k, 'LM head replaced by classifier'))
            continue
        if k in md and md[k].shape == v.shape:
            md[k] = v.clone()
            transferred.append((k, tuple(v.shape)))
        else:
            skipped.append((k, 'shape mismatch / absent'))
    classifier_model.load_state_dict(md, strict=False)
    if verbose:
        n_t = sum(int(np.prod(t[1])) for t in transferred)
        print(f'  transferred {len(transferred)} tensors ({n_t:,} params)',
              flush=True)
        print(f'  skipped {len(skipped)} tensors', flush=True)
        for k, r in skipped[:6]:
            print(f'    SKIP: {k}  ({r})', flush=True)


# ============================================================================
# Pretrain (WikiText causal LM)
# ============================================================================

class WikiTextDataset(Dataset):
    """Pre-tokenized WikiText-103 with sliding-window samples."""
    def __init__(self, tokenizer, split='train', seq_length=256, stride=128,
                 max_samples=None, chunk_size=1024 * 1024,
                 shuffle_tokens=False, shuffle_seed=42):
        print(f'Loading WikiText-103 {split}...', flush=True)
        ds = datasets.load_dataset('wikitext', 'wikitext-103-v1', split=split)
        token_chunks = []
        buf = []; buf_chars = 0

        def _flush():
            nonlocal buf, buf_chars
            if buf:
                token_chunks.append(tokenizer.encode(' '.join(buf)))
                buf = []; buf_chars = 0

        for item in ds:
            text = item['text'].strip()
            if text and not text.startswith('='):
                buf.append(text); buf_chars += len(text)
                if buf_chars >= chunk_size:
                    _flush()
        _flush()
        total = sum(len(c) for c in token_chunks)
        print(f'  total tokens: {total:,}', flush=True)
        self._tokens = np.empty(total, dtype=np.int64)
        off = 0
        for c in token_chunks:
            self._tokens[off:off + len(c)] = c
            off += len(c)
        del token_chunks
        if shuffle_tokens:
            # Mechanism control: permute the whole token stream to DESTROY
            # word order / syntax while keeping the exact unigram distribution.
            np.random.RandomState(shuffle_seed).shuffle(self._tokens)
            print(f'  [shuffle_tokens] permuted {total:,} tokens (seed={shuffle_seed}) '
                  '-- syntax destroyed, unigram stats preserved', flush=True)
        self.seq_length = seq_length
        self.stride = stride
        n_p = (total - seq_length) // stride
        self._n = min(n_p, max_samples) if max_samples else max(n_p, 0)
        print(f'  samples: {self._n:,}', flush=True)

    def __len__(self): return self._n
    def __getitem__(self, i):
        s = i * self.stride
        x = torch.as_tensor(self._tokens[s:s + self.seq_length].copy(), dtype=torch.long)
        y = torch.as_tensor(self._tokens[s + 1:s + self.seq_length + 1].copy(), dtype=torch.long)
        return {'input_ids': x, 'labels': y}


def pretrain(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  encoder={args.encoder}  full WikiText-103',
          flush=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    vocab_size = tokenizer.vocab_size

    train_ds = WikiTextDataset(tokenizer, split='train',
                                 seq_length=args.seq_length, stride=args.stride,
                                 max_samples=args.max_samples,
                                 shuffle_tokens=getattr(args, 'shuffle_tokens', False),
                                 shuffle_seed=args.seed)
    val_ds = WikiTextDataset(tokenizer, split='validation',
                              seq_length=args.seq_length, stride=args.seq_length)

    # For simhash_seb we need a per-token byte view (itos) to compute SimHash.
    pretrain_itos = None
    if args.encoder == 'simhash_seb':
        pretrain_itos = [tokenizer.convert_ids_to_tokens(i) for i in range(vocab_size)]
        # GPT-2 BPE uses 'Ġ' to mark word-leading space; treat as space for hashing.
        pretrain_itos = [(t or '').replace('Ġ', ' ') for t in pretrain_itos]
    model, config = build_hyena_causal_lm(vocab_size, args.encoder,
                                           num_bits=args.num_bits,
                                           interdim=args.interdim,
                                           dropout=args.dropout,
                                           max_seq_len=args.seq_length,
                                           itos=pretrain_itos)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  trainable params: {n_params:,}', flush=True)
    print(f'  d_model={config.d_model}  n_layer={config.n_layer}  '
          f'vocab={config.vocab_size}', flush=True)

    # Optional: load init weights from a previous checkpoint (e.g. hg38
    # pretrain → continue on WikiText). Skip vocab-dependent buffers and the
    # LM head (different vocab size).
    if getattr(args, 'init_ckpt', None) and args.init_ckpt.lower() != 'none':
        if not os.path.exists(args.init_ckpt):
            raise FileNotFoundError(args.init_ckpt)
        print(f'\n[init] continuing from {args.init_ckpt}', flush=True)
        ck = torch.load(args.init_ckpt, map_location='cpu', weights_only=False)
        sd_init = ck.get('model_state_dict', ck)
        md = model.state_dict()
        n_t, n_s = 0, 0
        for k, v in sd_init.items():
            if 'subword2binary' in k or 'subword2bytes' in k or 'subword' in k:
                n_s += 1; continue
            if k.startswith('lm_head.'):
                n_s += 1; continue
            if k in md and md[k].shape == v.shape:
                md[k] = v.clone(); n_t += 1
            else:
                n_s += 1
        model.load_state_dict(md, strict=False)
        print(f'  transferred {n_t} tensors, skipped {n_s}', flush=True)

    model = model.to(device)

    tr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                     num_workers=2, pin_memory=True)
    va = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                     num_workers=2, pin_memory=True)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                      betas=(0.9, 0.98))
    crit = nn.CrossEntropyLoss()
    n_steps_per_epoch = len(tr)
    pretrain_eps = args.pretrain_epochs if args.pretrain_epochs is not None else args.n_epochs
    total_steps = pretrain_eps * n_steps_per_epoch
    print(f'  steps per epoch={n_steps_per_epoch}  total={total_steps}  '
          f'(pretrain_epochs={pretrain_eps})', flush=True)

    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                            total_steps=max(total_steps, 1),
                                            pct_start=0.05)

    best_ppl = float('inf')
    t0 = time.time()
    for ep in range(pretrain_eps):
        model.train()
        running = 0.0; n_batch = 0
        ep_t0 = time.time()
        for step, batch in enumerate(tr):
            ids = batch['input_ids'].to(device, non_blocking=True)
            lab = batch['labels'].to(device, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(input_ids=ids).logits
                loss = crit(logits.reshape(-1, logits.size(-1)),
                            lab.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            running += float(loss.item()); n_batch += 1
            if step % max(1, n_steps_per_epoch // 5) == 0:
                print(f'  ep {ep+1} step {step}/{n_steps_per_epoch} '
                      f'loss={loss.item():.3f}', flush=True)
            ckpt_every = getattr(args, 'ckpt_every', 0)
            if ckpt_every > 0 and step > 0 and step % ckpt_every == 0:
                torch.save({'model_state_dict':
                            {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()},
                            'config': vars(args), 'vocab_size': vocab_size,
                            'epoch': ep + 1, 'step': step, 'partial': True},
                           os.path.join(args.out_dir, 'pretrain_partial.pt'))
                print(f'  [ckpt] saved partial at ep {ep+1} step {step}',
                      flush=True)
        # validate
        model.eval()
        v_loss = 0.0; v_n = 0
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for batch in va:
                ids = batch['input_ids'].to(device); lab = batch['labels'].to(device)
                logits = model(input_ids=ids).logits
                v_loss += float(crit(logits.reshape(-1, logits.size(-1)),
                                     lab.reshape(-1)).item()) * lab.size(0)
                v_n += lab.size(0)
        v_ppl = float(np.exp(v_loss / max(v_n, 1)))
        ep_time = time.time() - ep_t0
        print(f'  ep {ep+1}/{pretrain_eps}  train_loss={running/max(n_batch,1):.3f} '
              f'val_ppl={v_ppl:.2f}  ({ep_time:.1f}s)', flush=True)
        if v_ppl < best_ppl:
            best_ppl = v_ppl
            torch.save({'model_state_dict': {k: v.detach().cpu().clone()
                                              for k, v in model.state_dict().items()},
                        'config': vars(args), 'vocab_size': vocab_size,
                        'val_ppl': v_ppl, 'epoch': ep + 1},
                       os.path.join(args.out_dir, 'pretrain.pt'))
    print(f'\npretrain done in {time.time() - t0:.1f}s, best val PPL={best_ppl:.2f}',
          flush=True)
    return os.path.join(args.out_dir, 'pretrain.pt')


# ============================================================================
# Fine-tune (DNA classification)
# ============================================================================

def kmerize(seq, k=6, stride=1):
    seq = seq.upper()
    if len(seq) < k:
        return [seq]
    return [seq[i:i+k] for i in range(0, len(seq) - k + 1, stride)]


_RC_TBL = str.maketrans('ACGTN', 'TGCAN')
def revcomp(s):
    return s.translate(_RC_TBL)[::-1]


def build_kmer_vocab(train_kmer_lists, min_freq=2):
    counter = Counter()
    for toks in train_kmer_lists:
        counter.update(toks)
    stoi = {'<pad>': 0, '<unk>': 1}
    for tok, c in counter.most_common():
        if c >= min_freq:
            stoi[tok] = len(stoi)
    return stoi


def load_gb(name='gb_human_promoters', k=6, max_len=512, seed=42, keep_raw=False):
    entry = GB_HF_IDS[name]
    if isinstance(entry, tuple):
        repo, task_filter = entry
        ds = datasets.load_dataset(repo)
        ds = {sp: ds[sp].filter(lambda r: r['task'] == task_filter) for sp in ds}
    else:
        ds = datasets.load_dataset(entry)
    full_train = list(ds['train'])
    test = list(ds['test'])
    rng = random.Random(seed); rng.shuffle(full_train)
    n_val = max(1, int(len(full_train) * 0.1))
    val = full_train[:n_val]; train = full_train[n_val:]
    n_classes = max(int(r['label']) for r in full_train) + 1

    # NT subsets use 'sequence' column; GB datasets use 'seq'. Normalise.
    seq_key = 'seq' if 'seq' in full_train[0] else 'sequence'

    def proc(rows):
        out = []
        for r in rows:
            tokens = kmerize(r[seq_key], k=k)
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            rec = {'tokens': tokens, 'label': int(r['label'])}
            if keep_raw:
                rec['seq'] = r[seq_key].upper()  # for rc-aug
                rec['k'] = k
                rec['max_len'] = max_len
            out.append(rec)
        return out
    return proc(train), proc(val), proc(test), n_classes


class TokenDataset(Dataset):
    """DNA k-mer dataset. If rc_aug=True and training=True, with p=0.5 each
    epoch's sample is replaced by its reverse-complement (re-kmerized on the
    fly). Requires examples to carry the raw `seq` (set keep_raw=True in load_gb)."""
    def __init__(self, examples, stoi, max_len=512, rc_aug=False, training=False):
        self.exs = examples; self.stoi = stoi; self.max_len = max_len
        self.rc_aug = rc_aug; self.training = training
    def __len__(self): return len(self.exs)
    def __getitem__(self, i):
        ex = self.exs[i]
        if self.rc_aug and self.training and 'seq' in ex and random.random() < 0.5:
            seq = revcomp(ex['seq'])
            tokens = kmerize(seq, k=ex.get('k', 6))[: ex.get('max_len', self.max_len)]
        else:
            tokens = ex['tokens'][:self.max_len]
        ids = [self.stoi.get(t, self.stoi['<unk>']) for t in tokens]
        if not ids:
            ids = [self.stoi['<unk>']]
        # pad to max_len
        while len(ids) < self.max_len:
            ids.append(self.stoi['<pad>'])
        return torch.tensor(ids, dtype=torch.long), ex['label']


def collate(batch):
    ids = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return {'input_ids': ids, 'label': labels}


def finetune(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  encoder={args.encoder}  dataset={args.dataset}',
          flush=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

    train, val, test, n_classes = load_gb(args.dataset, k=args.k, keep_raw=getattr(args, 'rc_aug', False),
                                            max_len=args.max_len, seed=args.seed)
    frac = getattr(args, 'train_frac', 1.0)
    if frac < 1.0:
        n_keep = max(1, int(len(train) * frac))
        train = train[:n_keep]  # train already shuffled by seed in load_gb
        print(f'  [train_frac={frac}] subsampled train -> {n_keep}', flush=True)
    print(f'  train={len(train)} val={len(val)} test={len(test)} '
          f'n_classes={n_classes}', flush=True)
    stoi = build_kmer_vocab([s['tokens'] for s in train])
    print(f'  vocab={len(stoi)}', flush=True)
    rc = getattr(args, 'rc_aug', False)
    tr_ds = TokenDataset(train, stoi, max_len=args.max_len, rc_aug=rc, training=True)
    va_ds = TokenDataset(val, stoi, max_len=args.max_len)
    te_ds = TokenDataset(test, stoi, max_len=args.max_len)

    summary = {'config': vars(args), 'vocab_size': len(stoi),
               'n_classes': n_classes, 'splits': {'train': len(train),
                                                    'val': len(val),
                                                    'test': len(test)},
               'conditions': {}}

    itos_list = [None] * len(stoi)
    for tok, idx in stoi.items():
        itos_list[idx] = tok

    def make_classifier():
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
        m, _ = build_hyena_classifier(
            vocab_size=len(stoi), encoder_type=args.encoder,
            num_classes=n_classes, num_bits=args.num_bits,
            interdim=args.interdim, dropout=args.dropout,
            max_seq_len=args.max_len, itos=itos_list)
        return m

    def eval_loader(model, loader):
        model.eval()
        c = t = 0
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for batch in loader:
                ids = batch['input_ids'].to(device); lab = batch['label'].to(device)
                pred = model(input_ids=ids).logits.argmax(-1)
                c += (pred == lab).sum().item(); t += lab.size(0)
        return c / max(t, 1)

    def train_one(model, name):
        coll = functools.partial(collate)
        tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=coll,
                         num_workers=4, pin_memory=True, persistent_workers=True)
        va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll,
                         num_workers=2, pin_memory=True, persistent_workers=True)
        te = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll,
                         num_workers=2, pin_memory=True, persistent_workers=True)
        opt = optim.AdamW(model.parameters(), lr=args.lr_finetune, weight_decay=0.01)
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        best_val, best_state = 0.0, None
        for ep in range(args.n_epochs):
            model.train()
            for batch in tr:
                ids = batch['input_ids'].to(device); lab = batch['label'].to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    out = model(input_ids=ids, labels=lab)
                opt.zero_grad(); out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            v = eval_loader(model, va)
            print(f'    [{name}] epoch {ep+1}/{args.n_epochs} val={v:.4f}',
                  flush=True)
            if v > best_val:
                best_val = v
                best_state = {k: vv.detach().cpu().clone()
                              for k, vv in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        train_time = time.time() - t0
        test_acc = float(eval_loader(model, te))
        ckpt_path = os.path.join(args.out_dir, f'model_{name}.pt')
        torch.save({
            'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'optimizer_state_dict': opt.state_dict(),
            'epochs_completed': args.n_epochs,
            'best_val_acc': float(best_val),
            'test_acc': test_acc,
            'config': vars(args),
            'condition': name,
        }, ckpt_path)
        print(f'  [{name}] saved checkpoint to {ckpt_path}', flush=True)
        return {
            'val_acc': float(best_val),
            'test_acc': test_acc,
            'train_time_s': train_time,
            'n_params': sum(p.numel() for p in model.parameters()),
            'peak_mem_gb': round(torch.cuda.max_memory_allocated()/1e9, 2) if torch.cuda.is_available() else None,
            'epochs_completed': args.n_epochs,
        }

    # W: load pretrained
    if args.pretrain_ckpt and args.pretrain_ckpt.lower() != 'none':
        if not os.path.exists(args.pretrain_ckpt):
            raise FileNotFoundError(args.pretrain_ckpt)
        print(f'\n[W] load pretrain {args.pretrain_ckpt}', flush=True)
        ck = torch.load(args.pretrain_ckpt, map_location='cpu', weights_only=False)
        pretrain_state = ck['model_state_dict']
        m = make_classifier()
        transfer_pretrain_to_classifier(m, pretrain_state)
        m = m.to(device)
        rW = train_one(m, 'W')
        summary['conditions']['W_wikitext_init'] = rW
        print(f'  W: test={rW["test_acc"]:.4f}  val={rW["val_acc"]:.4f}',
              flush=True)
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    # R: random init
    print('\n[R] random init', flush=True)
    m = make_classifier().to(device)
    rR = train_one(m, 'R')
    summary['conditions']['R_random_init'] = rR
    print(f'  R: test={rR["test_acc"]:.4f}  val={rR["val_acc"]:.4f}', flush=True)

    if 'W_wikitext_init' in summary['conditions']:
        delta = (summary['conditions']['W_wikitext_init']['test_acc']
                 - summary['conditions']['R_random_init']['test_acc'])
        summary['delta_W_minus_R'] = delta
        print(f'\n=== {args.encoder} on {args.dataset} ===', flush=True)
        print(f'  W: {summary["conditions"]["W_wikitext_init"]["test_acc"]:.4f}',
              flush=True)
        print(f'  R: {rR["test_acc"]:.4f}', flush=True)
        print(f'  delta (W-R): {delta:+.4f}', flush=True)

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {args.out_dir}/summary.json', flush=True)


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['pretrain', 'finetune', 'both'],
                   default='both')
    p.add_argument('--encoder',
                   choices=['standard', 'binary_seb', 'byte_seb',
                            'binary_struct', 'hash_emb', 'simhash_seb'],
                   default='binary_seb')
    p.add_argument('--rc_aug', action='store_true',
                   help='Reverse-complement augmentation during finetune (Hyena baseline matching)')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--pretrain_ckpt', default=None,
                   help='For finetune phase: path to pretrain checkpoint, or "none"')
    p.add_argument('--init_ckpt', default=None,
                   help='For pretrain phase: optional checkpoint to initialize '
                        'from (e.g. hg38-pretrain ckpt for B1 sequence). '
                        'Set to "none" or leave blank for random init.')
    p.add_argument('--ckpt_every', type=int, default=50000,
                   help='Save partial checkpoint every N steps (0=disable)')
    p.add_argument('--dataset', choices=list(GB_HF_IDS.keys()),
                   default='gb_human_promoters')

    p.add_argument('--n_epochs', type=int, default=10,
                   help='Pretrain epochs (and fine-tune epochs unless --pretrain_epochs is set)')
    p.add_argument('--pretrain_epochs', type=int, default=None,
                   help='Override pretrain epoch count (leaves --n_epochs as finetune count)')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=6e-4)
    p.add_argument('--lr_finetune', type=float, default=1e-4)
    p.add_argument('--seq_length', type=int, default=256)
    p.add_argument('--stride', type=int, default=128)
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--max_len', type=int, default=512)
    p.add_argument('--k', type=int, default=6)
    p.add_argument('--num_bits', type=int, default=64)
    p.add_argument('--interdim', type=str, default='256')
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--shuffle_tokens', action='store_true',
                   help='Mechanism control (#2): permute the WikiText token stream '
                        'during pretrain to destroy syntax, keep unigram stats.')
    p.add_argument('--train_frac', type=float, default=1.0,
                   help='Data-efficiency curve (#3): fraction of DNA train set to keep.')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.phase in ('pretrain', 'both'):
        ckpt_path = pretrain(args)
        if args.pretrain_ckpt is None:
            args.pretrain_ckpt = ckpt_path

    if args.phase in ('finetune', 'both'):
        if args.pretrain_ckpt is None:
            args.pretrain_ckpt = os.path.join(args.out_dir, 'pretrain.pt')
        finetune(args)


if __name__ == '__main__':
    main()
