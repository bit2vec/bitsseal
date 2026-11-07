#!/bin/bash
# usage: eval_seed.sh SEED ARM GPU  -> appends RESULT lines to /root/traj/s3_1B/eval_s${SEED}_${ARM}.log
seed=$1; arm=$2; gpu=$3
source /venv/main/bin/activate; cd /root
dir=/root/traj/s3_1B/s${seed}_$([ $arm = bits ] && echo bits || echo std)
out=/root/traj/s3_1B/eval_s${seed}_${arm}.log
for c in $(ls -d $dir/checkpoint-* | sort -t- -k2 -n) $dir/final; do
  grep -q "ckpt=$c " $out 2>/dev/null && continue
  CUDA_VISIBLE_DEVICES=$gpu python eval_ppl.py --arm $arm --ckpt $c --seed $seed 2>&1 | grep -aE "RESULT|FATAL|Traceback|Error" >> $out
done
echo "EVAL_DONE seed=$seed arm=$arm" >> $out
