#!/bin/bash
# Run Ensemble Binary SEB Comparison Experiment on SST-2
# Compares: Baseline (1 model) vs Independent (10 models) vs Multi-Head (10 heads)



# Run comparison experiment
# - Compare all three approaches: baseline, independent, multihead
# - 20 epochs for proper convergence
# - 128 embedding dim per model (total 1280 for ensemble)
# - 10 models/heads for ensemble approaches
# - num_bits=64 (8 bytes equivalent)

python ensemble_comparison_experiment.py \
    --dataset sst2 \
    --models baseline independent multihead \
    --epochs 20 \
    --batch_size 64 \
    --max_len 64 \
    --min_freq 5 \
    --embedding_dim 128 \
    --num_models 10 \
    --num_bits 64 \
    --interdim 128 \
    --hidden_dim 300 \
    --num_layers 2 \
    --dropout 0.4 \
    --relu_dropout 0.1 \
    --lr 5e-4 \
    --seed 42 \
    --output_dir ./ensemble_results \
    2>&1 | tee log_ensemble_comparison.txt

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "Results saved to: ./ensemble_results/comparison_results.json"
echo "Log saved to: log_ensemble_comparison.txt"
echo "=========================================="
