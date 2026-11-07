
## 1. Create Conda Environment

```bash
# Create new conda environment with Python 3.10
conda create -n binary_seb python=3.10 -y

# Activate the environment
conda activate binary_seb

# Install PyTorch (CUDA 11.8 version - adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Alternative: CPU only version
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install other dependencies
pip install transformers==4.36.0
pip install datasets==2.16.0
pip install rouge-score==0.1.2
pip install numpy==1.24.0
pip install tqdm
pip install pandas
pip install matplotlib
pip install seaborn
pip install fairseq  # Optional: only if using Fairseq transformer

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets: {datasets.__version__}')"
```


## Download WikiText-103

The script automatically downloads WikiText-103 from Hugging Face, but you can pre-download:

```python
from datasets import load_dataset

# This will download and cache the dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")
print(f"Train: {len(dataset['train'])} samples")
print(f"Validation: {len(dataset['validation'])} samples")
print(f"Test: {len(dataset['test'])} samples")
```

## 4. Running the Experiment

### Quick Test Run
```bash
# Test with small number of samples
python film_attack_wikitext.py \
    --num_samples 10 \
    --batch_sizes 1 4 \
    --embedding_types subword byte_seb binary_seb \
    --output_dir ./results/test_run
```

### Full Experiment (Reproducing Paper Results)
```bash
# Run full experiment as in the paper
python film_attack_wikitext.py \
    --num_samples 100 \
    --batch_sizes 1 4 16 32 \
    --embedding_types subword byte_seb binary_seb \
    --num_bytes 8 \
    --num_bits 64 \
    --gpt2_model gpt2 \
    --output_dir ./results/full_experiment \
    --seed 42
```

### Run with Different Configurations
```bash
# Test different bit lengths for Binary SEB
for bits in 32 64 128; do
    python film_attack_wikitext.py \
        --num_samples 50 \
        --batch_sizes 1 4 16 32 \
        --embedding_types binary_seb \
        --num_bits $bits \
        --output_dir ./results/bits_${bits}
done
```
