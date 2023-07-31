#!/bin/bash -xe
set -o pipefail
GPU_ID=$1
SEED=$2
if [ -z "$GPU_ID" ]; then
  GPU_ID=0
fi
if [ -z "$SEED" ]; then
  SEED=10
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU_ID

CORPUS='conll'
MSL=150
E1=5
E2=7
OUT_DIR="out/$RUN_NAME/$CORPUS/$SEED"
TEMP_DIR="$OUT_DIR/tmp"
mkdir -p $TEMP_DIR

python -u src/thunder.py --data_dir data/$CORPUS --output_dir $OUT_DIR --temp_dir $TEMP_DIR \
  --seed $SEED --max_seq_length $MSL --dual_train_epochs $E1 --self_train_epochs $E2 \
  --do_train --do_eval --eval_on "test" | tee $OUT_DIR/train_log.txt
