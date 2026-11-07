#!/usr/bin/env python3
"""Bits multimodal smoke — endangered-language style.

Hypothesis: Bits enables cross-modality transfer for low-resource targets,
analogous to byte-level transfer for endangered languages.

All modalities use their natural unit at k=1:
  - DNA      : single nucleotide (V=5 = ACGTN), IDs 2..6 (0=pad, 1=unk)
  - Protein  : single amino acid (V=20), IDs 7..26
  - Text     : GPT-2 BPE (V=50257), IDs 27..50283

The Bits encoder is V-independent (BinarySubwordConcateOnehot, untouched from
the original paper module) and processes IDs from any vocab range identically.

Cells:
  A1  pretrain on DNA only           -> fine-tune on nt_promoter_tata (low-res)
  B1  pretrain on DNA + protein      -> fine-tune on nt_promoter_tata
  C1  pretrain on DNA + protein + text -> fine-tune on nt_promoter_tata

Win condition: B1 > A1 and C1 >= B1 (cross-modal lift). All cells use the same
fine-tune setup (DNA k=1, same epoch count, same LR), so the only difference is
the pretrain corpus.
"""
import argparse, functools, json, os, random, sys, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import datasets
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, GPT2Tokenizer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
for cand in [os.path.join(ROOT, 'software', 'Senti_analysis'),
             os.path.join(ROOT, 'software'),
             os.path.join(THIS_DIR, '..')]:
    if os.path.isfile(os.path.join(cand, 'binary_subword_sentiment.py')):
        sys.path.insert(0, cand)
        break

from binary_subword_sentiment import BinarySubwordConcateOnehot  # paper module, untouched

HYENA_TINY = 'LongSafari/hyenadna-tiny-1k-seqlen-hf'
HYENA_LARGE = 'LongSafari/hyenadna-large-1m-seqlen-hf'
CADUCEUS_PS = 'kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16'
HYENA_MODEL_ID = HYENA_TINY  # default; overridden by --body flag in main()

# === Vocab layout (offsets so each modality lives in a disjoint ID range) ===
PAD_ID = 0
UNK_ID = 1
DNA_OFFSET = 2                          # ACGTN -> 2..6
DNA_VSIZE_LOCAL = 5
PROTEIN_OFFSET = DNA_OFFSET + DNA_VSIZE_LOCAL   # 7
AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'      # 20 standard
PROTEIN_VSIZE_LOCAL = len(AA_LETTERS)    # 20
TEXT_OFFSET = PROTEIN_OFFSET + PROTEIN_VSIZE_LOCAL   # 27
GPT2_VSIZE = 50257
VSIZE_TOTAL = TEXT_OFFSET + GPT2_VSIZE   # 50284

DNA_CHAR_TO_ID = {c: DNA_OFFSET + i for i, c in enumerate('ACGTN')}
# DNA IDs (with offset 2): A=2, C=3, G=4, T=5, N=6
DNA_A, DNA_C, DNA_G, DNA_T, DNA_N = DNA_OFFSET + 0, DNA_OFFSET + 1, DNA_OFFSET + 2, DNA_OFFSET + 3, DNA_OFFSET + 4
AA_TO_ID = {a: PROTEIN_OFFSET + i for i, a in enumerate(AA_LETTERS)}


def apply_rcps_bits_mapping(encoder):
    """Path B-half: Override DNA token bit mappings to satisfy reverse-complement
    symmetry at the BIT level only (T_bits = ~A_bits, G_bits = ~C_bits).
    The downstream FFN is NOT constrained, so output embeddings are NOT
    channel-mirror symmetric. This is a partial fix — Caduceus's RCPS layer
    sees flipped bits at input but the FFN can still scramble the symmetry.
    Protein/text tokens unaffected (no RC equivalent). N is its own RC.
    """
    sb = encoder.subword2binary
    sb[DNA_T] = 1 - sb[DNA_A]
    sb[DNA_G] = 1 - sb[DNA_C]
    print(f'  Path B-half (RC bits, unconstrained FFN): T=~A, G=~C', flush=True)


class RCEquivariantBits(nn.Module):
    """Path B-full: RC-equivariant Bits encoder.
    Output dim = 2 * half_dim where half_dim = embedding_dim // 2.
       emb(token_id) = concat( h(bits[id]),  h(NOT bits[id]) )
    Combined with apply_rcps_bits_mapping (T_bits = ~A_bits, G_bits = ~C_bits),
    this gives:
       emb(A) = [h(A_b), h(T_b)],   emb(T) = [h(T_b), h(A_b)] = mirror(emb(A))
    so Caduceus's channel-mirror symmetric layer is fed the structured input
    its theorem expects.
    """
    def __init__(self, num_embeddings, embedding_dim, num_bits=64,
                 interdim='256', relu_dropout=0.1):
        super().__init__()
        assert embedding_dim % 2 == 0, f'embedding_dim must be even, got {embedding_dim}'
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_bits = num_bits
        self.half_dim = embedding_dim // 2
        self.relu_dropout = relu_dropout

        # Stochastic mapping (same scheme as BinarySubwordConcateOnehot)
        subword2binary = torch.randint(0, 2, (num_embeddings, num_bits), dtype=torch.long)
        seen = set()
        for i in range(num_embeddings):
            seq = tuple(subword2binary[i].tolist())
            attempts = 0
            while seq in seen and attempts < 1000:
                subword2binary[i] = torch.randint(0, 2, (num_bits,), dtype=torch.long)
                seq = tuple(subword2binary[i].tolist())
                attempts += 1
            seen.add(seq)
        self.register_buffer('subword2binary', subword2binary)

        # Frozen one-hot bit embedding (0→[1,0], 1→[0,1])
        onehot = F.one_hot(torch.arange(2)).float()
        self.bit_embed = nn.Embedding(2, 2, _weight=onehot)
        self.bit_embed.requires_grad_(False)
        self.weight = self.bit_embed.weight

        # SHARED FFN h: input dim = 2*num_bits, output dim = half_dim
        interdim_list = [int(i) for i in str(interdim).split('_')] if interdim else [self.half_dim]
        dims = [2 * num_bits] + interdim_list + [self.half_dim]
        self.linears = nn.ModuleList([nn.Linear(dims[i-1], dims[i]) for i in range(1, len(dims))])
        for i, lin in enumerate(self.linears):
            if i + 1 != len(self.linears):
                nn.init.kaiming_normal_(lin.weight, nonlinearity='relu', mode='fan_out')
                nn.init.zeros_(lin.bias)
            else:
                nn.init.xavier_normal_(lin.weight)
                nn.init.zeros_(lin.bias)

        subword = torch.arange(num_embeddings)
        self.register_buffer('subword', subword)

    def _ffn(self, bits):
        """bits: (B, T, num_bits) long-{0,1} → half_dim embedding"""
        emb = self.bit_embed(bits)
        emb = emb.reshape(*emb.shape[:-2], -1)
        for i, lin in enumerate(self.linears):
            emb = lin(emb)
            if i + 1 != len(self.linears):
                emb = F.relu(emb)
                emb = F.dropout(emb, p=self.relu_dropout, training=self.training)
        return emb

    def forward(self, input, dropout=None):
        bits = self.subword2binary[input]            # (B, T, num_bits)
        h_x   = self._ffn(bits)                       # (B, T, half_dim)
        h_not = self._ffn(1 - bits)                   # (B, T, half_dim)
        return torch.cat([h_x, h_not], dim=-1)        # (B, T, embedding_dim)

    def construct_matrix_for_output_layer(self):
        return self.forward(self.subword)

CELL_CONFIGS = {
    'A1': {'modalities': ('dna',),                                       'vocab_size': PROTEIN_OFFSET},
    'B1': {'modalities': ('dna', 'protein'),                              'vocab_size': TEXT_OFFSET},
    'C1': {'modalities': ('dna', 'protein', 'text'),                      'vocab_size': VSIZE_TOTAL},
    # D1 = NT-v2-style multi-species DNA pretrain (hg38 + 850 species) + protein.
    # Same DNA alphabet as A1/B1, so vocab_size = TEXT_OFFSET (no new IDs needed).
    'D1': {'modalities': ('dna', 'multispecies_dna', 'protein'),          'vocab_size': TEXT_OFFSET},
    # E1 = D1 + text on top, full central-dogma + text union.
    'E1': {'modalities': ('dna', 'multispecies_dna', 'protein', 'text'),  'vocab_size': VSIZE_TOTAL},
}


# ============================================================================
# Tokenizers — return token-ID lists for a single sequence
# ============================================================================

def tokenize_dna_k1(seq, max_len):
    seq = seq.upper()
    return [DNA_CHAR_TO_ID.get(c, UNK_ID) for c in seq[:max_len]]


def tokenize_protein_aa(seq, max_len):
    seq = seq.upper()
    return [AA_TO_ID.get(a, UNK_ID) for a in seq[:max_len]]


def tokenize_text_bpe(text, gpt2_tok, max_len):
    ids = gpt2_tok.encode(text)[:max_len]
    return [TEXT_OFFSET + t for t in ids]


# ============================================================================
# Model construction (Bits encoder + HyenaDNA-tiny body)
# ============================================================================

def make_bits_encoder(vocab_size, embedding_dim, num_bits, interdim, dropout, variant='random'):
    """variant in {'random', 'rcps_half', 'rc_equivariant'}.
       - random        : standard BinarySubwordConcateOnehot (Path A)
       - rcps_half     : same module + RC-symmetric bit mapping for DNA, FFN unconstrained
       - rc_equivariant: RCEquivariantBits (RC-symmetric bits + RC-equivariant FFN)
    """
    if variant == 'rc_equivariant':
        enc = RCEquivariantBits(num_embeddings=vocab_size, embedding_dim=embedding_dim,
                                  num_bits=num_bits, interdim=interdim, relu_dropout=dropout)
        apply_rcps_bits_mapping(enc)
    else:
        enc = BinarySubwordConcateOnehot(
            num_embeddings=vocab_size, embedding_dim=embedding_dim,
            num_bits=num_bits, interdim=interdim, relu_dropout=dropout,
            subword_binary_file='')
        if variant == 'rcps_half':
            apply_rcps_bits_mapping(enc)
    return enc


def _find_caduceus_embedding_attr(backbone):
    """Caduceus's MixerModel wraps embeddings deep in the model. Probe common paths."""
    for path in [
        ['backbone', 'embeddings', 'word_embeddings'],
        ['embeddings', 'word_embeddings'],
        ['embed_tokens'],
        ['caduceus', 'embeddings', 'word_embeddings'],
    ]:
        cur = backbone
        ok = True
        for p in path:
            if hasattr(cur, p):
                cur = getattr(cur, p)
            else:
                ok = False; break
        if ok:
            return path
    return None


def _set_caduceus_embedding(backbone, path, new_emb):
    cur = backbone
    for p in path[:-1]:
        cur = getattr(cur, p)
    setattr(cur, path[-1], new_emb)


class CaduceusClassifierWithBits(nn.Module):
    """Caduceus backbone + Bits encoder (monkey-patched) + mean-pool + linear head.
    Matches the run_caduceus_baseline.py pattern but with our V-indep encoder.
    """
    def __init__(self, model_id, vocab_size, n_classes, hf_body, bits_variant,
                 num_bits=64, interdim='256', dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        if hf_body:
            self.backbone = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        else:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            self.backbone = AutoModel.from_config(config, trust_remote_code=True)
        d_model = self.backbone.config.d_model if hasattr(self.backbone.config, 'd_model') else self.backbone.config.hidden_size

        # Build and inject our Bits encoder
        enc = make_bits_encoder(vocab_size, d_model, num_bits, interdim, dropout, variant=bits_variant)
        emb_path = _find_caduceus_embedding_attr(self.backbone)
        if emb_path is None:
            raise RuntimeError('Could not locate Caduceus embedding attribute')
        print(f'  Caduceus embedding monkey-patched at: {".".join(emb_path)}', flush=True)
        _set_caduceus_embedding(self.backbone, emb_path, enc)

        # Probe hidden dim with a small forward pass
        with torch.no_grad():
            probe = torch.zeros(1, 32, dtype=torch.long)
            try:
                out = self.backbone(input_ids=probe)
                h = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
                hidden_dim = h.size(-1)
            except Exception as e:
                print(f'  WARNING: probe forward failed ({e}); falling back to 2*d_model', flush=True)
                hidden_dim = 2 * d_model
        print(f'  Caduceus actual output hidden dim: {hidden_dim}', flush=True)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.encoder = enc  # for transfer_pretrain_to_classifier compatibility

    def forward(self, input_ids, labels=None):
        out = self.backbone(input_ids=input_ids)
        h = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
        h = h.mean(1)
        logits = self.classifier(h)
        from types import SimpleNamespace
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return SimpleNamespace(loss=loss, logits=logits)
        return SimpleNamespace(logits=logits)


class CaduceusCausalLMWithBits(nn.Module):
    """Caduceus backbone + Bits encoder + tied LM head (predict next token from union vocab)."""
    def __init__(self, model_id, vocab_size, hf_body, bits_variant,
                 num_bits=64, interdim='256', dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        if hf_body:
            self.backbone = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        else:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            self.backbone = AutoModel.from_config(config, trust_remote_code=True)
        d_model = self.backbone.config.d_model if hasattr(self.backbone.config, 'd_model') else self.backbone.config.hidden_size

        enc = make_bits_encoder(vocab_size, d_model, num_bits, interdim, dropout, variant=bits_variant)
        emb_path = _find_caduceus_embedding_attr(self.backbone)
        if emb_path is None:
            raise RuntimeError('Could not locate Caduceus embedding attribute')
        _set_caduceus_embedding(self.backbone, emb_path, enc)

        with torch.no_grad():
            probe = torch.zeros(1, 32, dtype=torch.long)
            out = self.backbone(input_ids=probe)
            h = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
            hidden_dim = h.size(-1)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.encoder = enc

    def forward(self, input_ids, labels=None):
        out = self.backbone(input_ids=input_ids)
        h = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
        logits = self.lm_head(h)
        from types import SimpleNamespace
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), ignore_index=PAD_ID)
            return SimpleNamespace(loss=loss, logits=logits)
        return SimpleNamespace(logits=logits)


def build_causal_lm(vocab_size, num_bits=64, interdim='256', dropout=0.1, hf_body=False,
                    body='tiny', bits_variant='random'):
    if body == 'caduceus':
        return CaduceusCausalLMWithBits(CADUCEUS_PS, vocab_size, hf_body, bits_variant,
                                          num_bits=num_bits, interdim=interdim, dropout=dropout)
    config = AutoConfig.from_pretrained(HYENA_MODEL_ID, trust_remote_code=True)
    config.vocab_size = vocab_size
    if hf_body:
        model = AutoModelForCausalLM.from_pretrained(
            HYENA_MODEL_ID, config=config, trust_remote_code=True,
            ignore_mismatched_sizes=True)
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    enc = make_bits_encoder(vocab_size, config.d_model, num_bits, interdim, dropout,
                              variant=bits_variant)
    model.set_input_embeddings(enc)
    return model


def build_classifier(vocab_size, n_classes, num_bits=64, interdim='256', dropout=0.1,
                     hf_body=False, body='tiny', bits_variant='random'):
    if body == 'caduceus':
        return CaduceusClassifierWithBits(CADUCEUS_PS, vocab_size, n_classes, hf_body,
                                            bits_variant, num_bits=num_bits,
                                            interdim=interdim, dropout=dropout)
    config = AutoConfig.from_pretrained(HYENA_MODEL_ID, trust_remote_code=True)
    config.vocab_size = vocab_size
    config.num_labels = n_classes
    config.problem_type = 'single_label_classification'
    config.pad_token_id = PAD_ID
    if hf_body:
        model = AutoModelForSequenceClassification.from_pretrained(
            HYENA_MODEL_ID, config=config, trust_remote_code=True,
            ignore_mismatched_sizes=True)
    else:
        model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True)
    enc = make_bits_encoder(vocab_size, config.d_model, num_bits, interdim, dropout,
                              variant=bits_variant)
    model.set_input_embeddings(enc)
    return model


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
# Datasets
# ============================================================================

class TokensDataset(Dataset):
    def __init__(self, tokens, max_len):
        self.x = tokens
        self.max_len = max_len
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        ids = self.x[i][:self.max_len]
        ids = ids + [PAD_ID] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class LabeledDataset(Dataset):
    def __init__(self, pairs, max_len):
        self.pairs = pairs
        self.max_len = max_len
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        ids, lab = self.pairs[i]
        ids = ids[:self.max_len]
        ids = ids + [PAD_ID] * (self.max_len - len(ids))
        return {'input_ids': torch.tensor(ids, dtype=torch.long),
                'label': torch.tensor(lab, dtype=torch.long)}


# ============================================================================
# Pretrain corpus assembly
# ============================================================================

def load_pretrain_corpus(modalities, per_modality_size, max_len):
    """Returns interleaved token-id lists from the chosen modalities."""
    corpus = []

    if 'dna' in modalities:
        print(f'loading DNA pretrain corpus ({per_modality_size} seqs)...', flush=True)
        nt = datasets.load_dataset('InstaDeepAI/nucleotide_transformer_downstream_tasks',
                                    split='train')
        # Use a mix of NT tasks excluding promoter_tata (target) to keep pretrain
        # vs target separation.
        nt = nt.filter(lambda r: r['task'] != 'promoter_tata')
        count = 0
        for row in nt:
            if count >= per_modality_size:
                break
            ids = tokenize_dna_k1(row['sequence'], max_len)
            if len(ids) >= 16:
                corpus.append(ids); count += 1
        print(f'  DNA (hg38 via NT): {count} seqs', flush=True)

    if 'multispecies_dna' in modalities:
        # InstaDeepAI/multi_species_genomes — NT-v2's actual pretrain corpus.
        # 850 species, 174B nucleotides. Stream to avoid disk-cache blowup.
        print(f'loading multi-species DNA (streaming, {per_modality_size} seqs)...',
              flush=True)
        try:
            ms = datasets.load_dataset('InstaDeepAI/multi_species_genomes',
                                        split='train', streaming=True)
            count = 0
            for row in ms:
                if count >= per_modality_size:
                    break
                # Chunk long contigs into max_len-windows so each contig yields
                # one (or a few) training sequences (matches MT-style "concat
                # all data, byte-level").
                seq = row.get('sequence') or row.get('text') or ''
                if not seq:
                    continue
                # Take first max_len*4 chars and yield one chunk per max_len worth.
                for start in range(0, min(len(seq), max_len * 4), max_len):
                    if count >= per_modality_size:
                        break
                    ids = tokenize_dna_k1(seq[start:start + max_len], max_len)
                    if len(ids) >= 16:
                        corpus.append(ids); count += 1
            print(f'  multi-species DNA: {count} seqs', flush=True)
        except Exception as e:
            print(f'  WARNING: multi_species_genomes failed ({e}); skipping', flush=True)

    if 'protein' in modalities:
        print(f'loading protein corpus (UniRef50 streaming, {per_modality_size} seqs)...',
              flush=True)
        try:
            ur = datasets.load_dataset('agemagician/uniref50',
                                        split='train', streaming=True)
            count = 0
            for row in ur:
                if count >= per_modality_size:
                    break
                seq = row.get('text') or row.get('sequence') or ''
                ids = tokenize_protein_aa(seq, max_len)
                if len(ids) >= 16:
                    corpus.append(ids); count += 1
            print(f'  Protein: {count} seqs', flush=True)
        except Exception as e:
            print(f'  WARNING: UniRef50 failed ({e}); skipping protein', flush=True)

    if 'text' in modalities:
        print(f'loading text corpus (WikiText-103, {per_modality_size} seqs)...', flush=True)
        gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
        wt = datasets.load_dataset('wikitext', 'wikitext-103-v1', split='train')
        count = 0
        for row in wt:
            if count >= per_modality_size:
                break
            text = row['text'].strip()
            if not text:
                continue
            ids = tokenize_text_bpe(text, gpt2, max_len)
            if len(ids) >= 16:
                corpus.append(ids); count += 1
        print(f'  Text: {count} seqs', flush=True)

    random.shuffle(corpus)
    return corpus


GB_HF_IDS = {
    'gb_human_promoters':          'katarinagresova/Genomic_Benchmarks_human_nontata_promoters',
    'gb_human_enhancers':          'katarinagresova/Genomic_Benchmarks_human_enhancers_cohn',
    'gb_human_enhancers_ensembl':  'katarinagresova/Genomic_Benchmarks_human_enhancers_ensembl',
    'gb_human_ocr_ensembl':        'katarinagresova/Genomic_Benchmarks_human_ocr_ensembl',
    'gb_human_regulatory':         'katarinagresova/Genomic_Benchmarks_human_ensembl_regulatory',
    'gb_demo_human_or_worm':       'katarinagresova/Genomic_Benchmarks_demo_human_or_worm',
    'gb_demo_coding_vs_intergenomic': 'katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs',
    'gb_dummy_mouse_enhancers_ensembl': 'katarinagresova/Genomic_Benchmarks_dummy_mouse_enhancers_ensembl',
}


def load_finetune_task(name, max_len, finetune_size=None):
    """Load fine-tune task. Supports NT subtask names (e.g. 'promoter_tata',
    'regulatory_elements') AND GB dataset keys ('gb_human_regulatory' etc.)."""
    if name.startswith('gb_'):
        repo = GB_HF_IDS[name]
        print(f'loading fine-tune task: {name} ({repo})', flush=True)
        ds = datasets.load_dataset(repo)
        train_rows = list(ds['train']); test_rows = list(ds['test'])
        seq_key = 'seq' if 'seq' in train_rows[0] else 'sequence'
    else:
        print(f'loading fine-tune task: nt_{name}', flush=True)
        ds = datasets.load_dataset('InstaDeepAI/nucleotide_transformer_downstream_tasks')
        ds_train = ds['train'].filter(lambda r: r['task'] == name)
        ds_test = ds['test'].filter(lambda r: r['task'] == name)
        train_rows = list(ds_train); test_rows = list(ds_test)
        seq_key = 'sequence'
    random.shuffle(train_rows)
    nv = max(1, int(len(train_rows) * 0.1))
    val_rows, train_rows = train_rows[:nv], train_rows[nv:]
    if finetune_size:
        train_rows = train_rows[:finetune_size]
    def to_pair(r):
        return (tokenize_dna_k1(r[seq_key], max_len), int(r['label']))
    train = [to_pair(r) for r in train_rows]
    val   = [to_pair(r) for r in val_rows]
    test  = [to_pair(r) for r in test_rows]
    n_classes = max([p[1] for p in train + val + test]) + 1
    print(f'  train={len(train)}  val={len(val)}  test={len(test)}  n_classes={n_classes}',
          flush=True)
    return train, val, test, n_classes


# ============================================================================
# Training loops (MT-style: Adam β=0.9/0.98 eps=1e-9)
# ============================================================================

def make_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)


def causal_pretrain(model, tokens, args, device):
    ds = TokensDataset(tokens, max_len=args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    opt = make_optimizer(model, args.lr)
    model.train()
    t0 = time.time()
    losses = []
    step = 0
    for ep in range(args.n_epochs):
        for batch in loader:
            ids = batch.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(input_ids=ids, labels=ids)
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(out.loss.item()); step += 1
            if step % 100 == 0:
                ppl = math.exp(min(20, float(np.mean(losses[-100:]))))
                print(f'  step {step:5d}  loss={float(np.mean(losses[-100:])):.4f}  '
                      f'ppl={ppl:.2f}', flush=True)
    final = float(np.mean(losses[-50:]) if losses else float('nan'))
    print(f'  pretrain done in {time.time()-t0:.1f}s  final_loss={final:.4f}', flush=True)
    return {'final_loss': final, 'steps': step, 'time_s': time.time() - t0}


def evaluate(model, loader, device):
    model.eval(); c = t = 0
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for batch in loader:
            ids = batch['input_ids'].to(device); lab = batch['label'].to(device)
            out = model(input_ids=ids)
            c += (out.logits.argmax(-1) == lab).sum().item(); t += lab.size(0)
    return c / max(t, 1)


def finetune(model, train, val, test, args, device):
    tr = DataLoader(LabeledDataset(train, args.max_len), batch_size=args.batch_size, shuffle=True)
    va = DataLoader(LabeledDataset(val,   args.max_len), batch_size=args.batch_size, shuffle=False)
    te = DataLoader(LabeledDataset(test,  args.max_len), batch_size=args.batch_size, shuffle=False)
    opt = make_optimizer(model, args.ft_lr)
    t0 = time.time(); best_val = -1; best_state = None
    for ep in range(args.ft_epochs):
        model.train()
        for batch in tr:
            ids = batch['input_ids'].to(device); lab = batch['label'].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(input_ids=ids, labels=lab)
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        v = evaluate(model, va, device)
        print(f'  epoch {ep+1}/{args.ft_epochs}  val_acc={v:.4f}', flush=True)
        if v > best_val:
            best_val = v
            best_state = {k: vv.detach().cpu().clone() for k, vv in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc = evaluate(model, te, device)
    print(f'  test_acc={test_acc:.4f}  time={time.time()-t0:.1f}s', flush=True)
    return {'best_val': float(best_val), 'test_acc': float(test_acc),
            'time_s': time.time() - t0}


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cell', choices=list(CELL_CONFIGS.keys()), required=True)
    p.add_argument('--target_task', default='promoter_tata',
                   help='NT task name for fine-tune (default: promoter_tata, low-res)')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--per_modality_size', type=int, default=20000)
    p.add_argument('--finetune_size', type=int, default=None)
    p.add_argument('--max_len', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=2)
    p.add_argument('--ft_epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--ft_lr', type=float, default=1e-4)
    p.add_argument('--num_bits', type=int, default=64)
    p.add_argument('--interdim', default='256')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--body', choices=['tiny', 'large', 'caduceus'], default='tiny',
                   help='Backbone: HyenaDNA-tiny, -large, or Caduceus-PS')
    p.add_argument('--hf_body', action='store_true',
                   help='Use HF-pretrained body (vs random init)')
    p.add_argument('--bits_variant', choices=['random', 'rcps_half', 'rc_equivariant'],
                   default='random',
                   help='Bits encoder variant: random (default), rcps_half (RC-symmetric '
                        'bits, unconstrained FFN), rc_equivariant (RC-symmetric bits + '
                        'mirror-symmetric FFN, Path B-full)')
    args = p.parse_args()
    global HYENA_MODEL_ID
    if args.body == 'large':
        HYENA_MODEL_ID = HYENA_LARGE
    elif args.body == 'caduceus':
        HYENA_MODEL_ID = CADUCEUS_PS
    else:
        HYENA_MODEL_ID = HYENA_TINY
    print(f'using body: {args.body} ({HYENA_MODEL_ID})', flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = CELL_CONFIGS[args.cell]
    V = cfg['vocab_size']
    print(f'cell={args.cell}  modalities={cfg["modalities"]}  V={V}  device={device}',
          flush=True)

    # --- Pretrain ---
    pre_tokens = load_pretrain_corpus(cfg['modalities'], args.per_modality_size, args.max_len)
    print(f'pretrain corpus: {len(pre_tokens)} sequences', flush=True)
    if len(pre_tokens) < 100:
        print('not enough pretrain data; skipping pretrain', flush=True)
        pre_state = None
        pre_stats = None
    else:
        print('=== Pretrain (causal LM) ===', flush=True)
        pre_model = build_causal_lm(V, num_bits=args.num_bits, interdim=args.interdim,
                                      hf_body=args.hf_body, body=args.body,
                                      bits_variant=args.bits_variant).to(device)
        pre_stats = causal_pretrain(pre_model, pre_tokens, args, device)
        pre_state = {k: v.detach().cpu().clone() for k, v in pre_model.state_dict().items()}
        del pre_model; torch.cuda.empty_cache()

    # --- Fine-tune ---
    train, val, test, n_classes = load_finetune_task(
        args.target_task, args.max_len, args.finetune_size)
    print('=== Fine-tune ===', flush=True)
    cls = build_classifier(V, n_classes, num_bits=args.num_bits, interdim=args.interdim,
                            hf_body=args.hf_body, body=args.body,
                            bits_variant=args.bits_variant)
    if pre_state is not None:
        transfer_pretrain_to_classifier(cls, pre_state)
    cls = cls.to(device)
    ft_stats = finetune(cls, train, val, test, args, device)

    summary = {
        'cell': args.cell, 'modalities': list(cfg['modalities']),
        'vocab_size': V, 'n_classes': n_classes,
        'target_task': args.target_task,
        'config': vars(args),
        'pretrain': pre_stats, 'finetune': ft_stats,
        'splits': {'train': len(train), 'val': len(val), 'test': len(test)},
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved {args.out_dir}/summary.json  test_acc={ft_stats["test_acc"]:.4f}',
          flush=True)


if __name__ == '__main__':
    main()
