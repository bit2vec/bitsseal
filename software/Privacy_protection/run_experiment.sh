#!/bin/bash
# =============================================================================
# FILM Attack Experiment Runner
# =============================================================================
# This script runs the FILM attack experiment comparing:
# 1. Standard Subword Embedding
# 2. Byte SEB (256 vocabulary)
# 3. Binary SEB (2 vocabulary)
# =============================================================================

set -e  # Exit on error

# Configuration
OUTPUT_DIR="./results"
NUM_SAMPLES=100
BATCH_SIZES="1 4 16 32"
NUM_BYTES=8
NUM_BITS=64
GPT2_MODEL="gpt2"
SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --batch_sizes)
            BATCH_SIZES="$2"
            shift 2
            ;;
        --num_bits)
            NUM_BITS="$2"
            shift 2
            ;;
        --quick)
            # Quick test mode
            NUM_SAMPLES=10
            BATCH_SIZES="1 4"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --output_dir DIR    Output directory (default: ./results)"
            echo "  --num_samples N     Number of samples to attack (default: 100)"
            echo "  --batch_sizes SIZES Batch sizes to test (default: '1 4 16 32')"
            echo "  --num_bits N        Bits per subword for Binary SEB (default: 64)"
            echo "  --quick             Quick test mode (10 samples, batch sizes 1,4)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "=============================================="
echo "FILM Attack Experiment Configuration"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo "Number of samples: $NUM_SAMPLES"
echo "Batch sizes: $BATCH_SIZES"
echo "Bytes per subword (Byte SEB): $NUM_BYTES"
echo "Bits per subword (Binary SEB): $NUM_BITS"
echo "GPT-2 model: $GPT2_MODEL"
echo "Random seed: $SEED"
echo "=============================================="

# Check if conda environment is activated
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "Warning: No conda environment detected."
    echo "Consider running: conda activate binary_seb"
fi

# Check dependencies
echo ""
echo "Checking dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || { echo "PyTorch not found!"; exit 1; }
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" || { echo "Transformers not found!"; exit 1; }
python -c "import datasets; print(f'Datasets: {datasets.__version__}')" || { echo "Datasets not found!"; exit 1; }
python -c "import rouge_score; print('Rouge-score: OK')" || { echo "Rouge-score not found!"; exit 1; }
echo "All dependencies OK!"

# Run experiment
echo ""
echo "=============================================="
echo "Starting FILM Attack Experiment"
echo "=============================================="

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/experiment_${TIMESTAMP}.log"

python film_attack_wikitext.py \
    --num_samples $NUM_SAMPLES \
    --batch_sizes $BATCH_SIZES \
    --num_bytes $NUM_BYTES \
    --num_bits $NUM_BITS \
    --gpt2_model $GPT2_MODEL \
    --output_dir "$OUTPUT_DIR" \
    --seed $SEED \
    --save_examples \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================="
echo "Experiment Complete!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR/results.json"
echo "Log file: $LOG_FILE"
echo "=============================================="
