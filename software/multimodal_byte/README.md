# Multi-modal byte/binary tokenization experiment

## Goal
Test whether the EmByte / 001 byte/binary embedding gain we observed on
multilingual MT (quy/gn → es) extends to multi-modal classification, where
each modality is treated like a separate "language" sharing a universal
byte/bit alphabet.

## Architecture

```
sample = [TEXT_tokens] [SEP] [GENE_tokens] [SEP] [VARIANT_tokens] [SEP] ... [DISEASE_tokens]
                            ↓
       byte/bit embedding (EmByte / 001 / Subword)
                            ↓
                  [optional BiLSTM body]
                            ↓
                       mean-pool
                            ↓
                       linear head
                            ↓
                        logits
```

* **Concatenation order is fixed across all samples** so position 1 = modality 1, etc.
* All modalities share a single token vocabulary (with disjoint id ranges or
  separator tokens distinguishing modalities — both variants are run).
* EmByte uses 8 bytes per token (alphabet = 256, same as previous work).
* 001 uses 64 bits per token (alphabet = 2, same as previous work).
* Per-modality token budgets are derived from the dataset's empirical length
  distribution (see budget rule below).

## Code reuse vs new code

### Reused unchanged
* `software/Senti_analysis/binary_subword_sentiment.py` — `BinarySubwordConcateOnehot` (001 embedding module)
* `software/Senti_analysis/byte_subword_concate_onehot.py` — `ByteSubwordConcateOnehot` (EmByte embedding module)
* `software/healthcare/film_lamp_classification.py:88-110` — `TinyClassifier` base pattern (mean-pool head)
* `software/healthcare/extra_attacks.py` — `tag_attack`, `mia_attack` helpers (called for the privacy table)
* `software/healthcare/film_lamp_classification.py:get_token_grad_norms` — FILM attack helper

### New (this folder)
* `civic_loader.py` — pulls CIViC TSV, parses (text, gene, variant, disease, ...)
  into per-modality token sequences plus 5-way label.
* `multimodal_collator.py` — pads each modality to its budget, concatenates
  in fixed order, applies separator-token or disjoint-id-range method.
* `multimodal_classifier.py` — wraps the byte/bit embedding + optional BiLSTM
  body + mean-pool + linear head. Same TinyClassifier API.
* `run_civic_multimodal.py` — full grid: 3 methods × 2 boundary methods ×
  3 ablations × 2 bodies = 36 cells; per cell reports test acc + FILM + TAG + MIA.

## Token budget rule (the actual rule used)

```
text    = min(512, max-in-dataset)
gene    = min(2048, max-in-dataset)
variant = max-in-dataset  if max ≤ 1.5 × p95
        = p95             otherwise
other   = same rule as variant
```

For CIViC the empirical numbers (max / p95 / p99 / mean):
* text: 345 / 155 / 194 / 77 → budget 345
* gene: 2 / 1 / 1 / 1 → budget 2
* variant: 19 / 5 / 6 / 2 → budget 5 (max/p95 = 3.8 > 1.5)
* disease: 13 / 5 / 7 / 3 → budget 5 (max/p95 = 2.6 > 1.5)
* others (evidence_type, evidence_direction, evidence_level, variant_origin):
  budget 4 each (short categorical strings)

Total per sample ≈ 365 tokens + separators (when separator method is used).

## Modality boundary (Q2 — both variants run)

* **separator**: insert a learnable `[SEP_<modality>]` id between modalities.
* **disjoint id range**: text ids in [0, 30k); gene ids in [30k, 35k);
  variant ids in [35k, 40k); etc. No separator needed; the model learns
  which modality a token came from from its id range.

## Ablations (Q-C)

* `gene-only` — pad text/variant/etc. with PAD; only the gene modality has content.
* `gene+text` — gene + text; rest padded.
* `all` — every available modality, full concatenation.

## Bodies (Q4)

* **A** = `embedding → mean-pool → linear` (no body); identical to TinyClassifier.
* **B** = `embedding → BiLSTM → final hidden state → linear`; matches
  HealthcareClassifier.

## How to run

```bash
cd bitsseal
PYTHONUNBUFFERED=1 venv/bin/python -u software/multimodal_byte/run_civic_multimodal.py \
    --out_dir results/multimodal_byte/civic
```

Results land in `results/multimodal_byte/civic/{cell}.json` and a summary
`summary.json`.
