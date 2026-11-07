#!/bin/bash
# Ensemble Binary SEB Comparison on IMDB Dataset
# Compares: Baseline vs Independent Ensemble vs Multi-Head Ensemble


echo "=========================================="
echo "Ensemble Binary SEB - IMDB Dataset"
echo "=========================================="
echo "Comparing three approaches:"
echo "  1. Baseline: Single Binary SEB model"
echo "  2. Independent: 10 independent Binary SEB models (seeds 42-51)"
echo "  3. Multi-Head: Shared embedding with 10 projection heads"
echo "=========================================="
echo ""

# Run all three experiments on dedicated GPU 3
# IMDB has longer reviews, so use max_len=512
CUDA_VISIBLE_DEVICES=3 python ensemble_comparison_experiment.py \
    --dataset imdb \
    --models baseline independent multihead \
    --epochs 20 \
    --batch_size 64 \
    --max_len 512 \
    --lr 5e-4 \
    --num_bits 64 \
    --interdim 128 \
    --num_models 10 \
    --output_dir ./ensemble_results_imdb \
    2>&1 | tee log_ensemble_imdb.txt

echo ""
echo "=========================================="
echo "IMDB Ensemble experiment complete!"
echo "Results saved to: ./ensemble_results_imdb/"
echo "=========================================="
