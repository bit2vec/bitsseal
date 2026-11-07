#!/bin/bash
# 3 seed pairs at 1B tokens, both arms of a seed at once on 4 GPUs each, seeds back to back.
cd /root
export FSDP_MODE="full_shard auto_wrap"
OUT=/root/traj/s3_1B; mkdir -p $OUT
for seed in 43 44 42; do
  echo "[$(date -u +%FT%TZ)] START seed $seed" >> $OUT/driver.log
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 29501 bits_lm_traj.py --arm bits     --bs 16 --grad_accum 1 --tokens 1000000000 --seed $seed --fsdp --out $OUT/s${seed}_bits > $OUT/s${seed}_bits.log 2>&1 &
  P1=$!
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port 29502 bits_lm_traj.py --arm standard --bs 16 --grad_accum 1 --tokens 1000000000 --seed $seed --fsdp --out $OUT/s${seed}_std  > $OUT/s${seed}_std.log 2>&1 &
  P2=$!
  wait $P1; R1=$?; wait $P2; R2=$?
  echo "[$(date -u +%FT%TZ)] END seed $seed bits_rc=$R1 std_rc=$R2" >> $OUT/driver.log
done
echo "[$(date -u +%FT%TZ)] ALL_SEEDS_DONE" >> $OUT/driver.log
