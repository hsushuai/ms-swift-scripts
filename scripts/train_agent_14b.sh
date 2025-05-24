#!/bin/bash
set -e  # if any command fails, the script will exit immediately
set -u  # if any variable is not set, the script will exit immediately

# arguments
PER_DEVICE_TRAIN_BATCH_SIZE=15
GRADIENT_ACCUMULATION_STEPS=2
DATA_VERSION=6
BASE_OUTPUT_DIR="/data01/xushuai/code/output/new_agent/new_agent_14b_v${DATA_VERSION}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}"

# auto increment output directory
TRAINING_ARGS_VERSION=1
INCREMENT_TRAINING=0
while [ -d "$OUTPUT_DIR" ]; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}_${INCREMENT_TRAINING}_${TRAINING_ARGS_VERSION}"
    ((TRAINING_ARGS_VERSION++))
done

echo "[INFO] Using output directory: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OMP_NUM_THREADS=10 \
NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
swift sft \
    --model /data01/LLM_model/Qwen3-14B \
    --dataset /data01/xushuai/code/data/new_agent-${DATA_VERSION}/train.jsonl \
    --num_train_epochs 4 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_length 3400 \
    --warmup_ratio 0.1 \
    --learning_rate 7e-6 \
    --eval_strategy no \
    --deepspeed zero3 \
    --save_only_model true \
    --gradient_checkpointing \
    --ddp_backend nccl \
    --save_strategy epoch \
    --save_total_limit 1 \
    --train_type full \
    --torch_dtype bfloat16 \
    --add_version false \
    --output_dir $OUTPUT_DIR \
    --dataloader_num_workers 10 \
    --dataset_num_proc 10 \
    --logging_steps 1 \
    --report_to swanlab \
    --attn_impl flash_attn \
    --use_liger_kernel true

# evaluation
echo "[INFO] Starting evaluation..."
CUDA_VISIBLE_DEVICES=3,4 \
python src/eval.py --model "$OUTPUT_DIR"  --qwen3

echo "[INFO] Done."

# bash scripts/train_all.sh > run.log 2>&1 &
