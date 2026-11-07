

## 1. Environment Setup

```bash
# Create conda environment
conda create -n binary_seb python=3.10 -y
conda activate binary_seb

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install dependencies
pip install datasets==2.16.0
pip install transformers==4.36.0
pip install torchtext==0.16.0
pip install numpy==1.24.0
pip install tqdm
pip install psutil

# Verify
python -c "import torch; import datasets; import transformers; print('OK!')"
```

## Running Experiments

### Run Everything at Once (Recommended)

```bash
# Run all experiments and generate Table 5
python generate_table5.py --run_all

# Quick test (fewer epochs, smaller data)
python generate_table5.py --quick_test

# Generate table from existing results
python generate_table5.py --generate_only
```

### Run Individual Experiments

#### Language Modeling (WikiText-103)

```bash
# Run all three methods
python lm_wikitext_comparison.py --run_all

# Run just Binary SEB
python lm_wikitext_comparison.py --embedding_type binary_seb --num_bits 64

# Use LSTM instead of Transformer
python lm_wikitext_comparison.py --run_all --model_type lstm
```

