#!/bin/bash


python yelp_to_imdb_zeroshot.py \
    --embedding_types standard byte_seb binary_seb \
    --yelp_train_samples 100000 \
    --pretrain_epochs 5 \
    --batch_size 64 \
    --output_dir ./results_yelp_to_imdb \
    2>&1 | tee log_yelp_to_imdb_zeroshot.txt
