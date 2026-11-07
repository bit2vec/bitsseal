# Bits: Can Bits Seal Language?

Code for "Can Bits Seal Language?" (NeurIPS 2026). A token is mapped to a fixed random binary code and embedded by a small shared feed-forward network, removing the per-token embedding row.

## Layout

- `software/` classification, privacy attacks, defenses, MT, LM, and federated experiments as run for the paper (see `software/readme.md`)
- `llm_scaling/` from-scratch Qwen2.5-1.5B and 7B training with the Bits encoder (`bits_llm.py`, `bits_lm_traj.py`), held-out evaluation (`eval_ppl.py`, `eval_seed.sh`), the three-seed 1B-token driver (`run_3seeds_1B.sh`), bf16 conversion, upload script, and the figure script `plot_3seed.py`

## Checkpoints

Qwen2.5-7B checkpoints, three seeds, every 100M tokens up to 1B, plus logs and the held-out set, are on Hugging Face: https://huggingface.co/bit2vec/bitsseal (`qwen2.5-7b_c4_1B/`).

## Datasets

No data is shipped. Every dataset is public and is downloaded by the loader in the script that uses it. The Hugging Face `datasets` identifiers called in the code are `glue/sst2`, `stanfordnlp/imdb`, `fancyzhx/yelp_polarity`, `pubmed_qa/pqa_labeled`, `openlifescienceai/medmcqa`, `ade_corpus_v2`, `TimSchopf/medical_abstracts`, `mteb/banking77`, `clinc_oos/plus`, `mteb/amazon_massive_intent`, `lex_glue` (`ledgar`, `unfair_tos`, `ecthr_a`), `zeroshot/twitter-financial-news-sentiment`, `dair-ai/emotion`, `go_emotions`, `code_x_glue_cc_defect_detection` (Devign), `InstaDeepAI/nucleotide_transformer_downstream_tasks`, `wikitext/wikitext-103-v1`, `iwslt2017` (`de-en`, `ar-en`), and `allenai/c4/en` for the 1.5B and 7B runs. Genomic Benchmarks tasks are the `katarinagresova/Genomic_Benchmarks_*` Hugging Face datasets, Sleep-EDF from PhysioNet via `pyedflib`, UCI Diabetes 130 via `ucimlrepo`, and the CIViC, ClinVar, PharmGKB, and TCGA tables from their public portals through the loaders in `software/multimodal_byte/`.

## Package

`pip install bitsseal`
