## Datasets

### IWSLT14 (Pretraining)
- **Source**: TED Talks translations
- **Languages**: German, Arabic, Farsi, Turkish, Chinese → English
- **Size**: ~100K-160K sentence pairs per language
- **Preprocessing**: Remove sentences > 800 bytes

### IndCorpus (Finetuning)
- **Source**: Indigenous languages of the Americas
- **Languages**: 10 endangered languages → Spanish
  - quy (Quechua) - highest resource
  - gn (Guarani)
  - nah (Nahuatl)
  - shp (Shipibo-Konibo)
  - aym (Aymara)
  - cni (Asháninka)
  - bzd (Bribri)
  - oto (Otomí)
  - tar (Rarámuri)
  - hch (Wixarika)

## Running the Experiment

### Quick Test
```bash
python generalizability_coling.py --quick_test
```

### Full Experiment
```bash
python generalizability_coling.py --run_all
```

### Step by Step
```bash
# Pretrain only
python generalizability_coling.py --pretrain_only --pretrain_epochs 200

# Finetune only (requires pretrained checkpoints)
python generalizability_coling.py --finetune_only --finetune_epochs 50
```

