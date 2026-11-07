#!/bin/bash


# Set GPU
export CUDA_VISIBLE_DEVICES=3

# Run quick test first to verify everything works
echo "Starting Generalizability Experiment (COLING Replication)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"

python generalizability_coling.py \
    --run_all \
    --pretrain_epochs 200 \
    --finetune_epochs 50 \
    --output_dir ./results_generalizability \
    2>&1 | tee log_generalizability.txt
