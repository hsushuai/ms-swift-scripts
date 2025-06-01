#!/bin/bash
set -e  # if any command fails, the script will exit immediately
set -u  # if any variable is not set, the script will exit immediately

# arguments
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
DATA_VERSION=3

BASE_OUTPUT_DIR="/data01/xushuai/code/output/test/condense_v${DATA_VERSION}"
OUTPUT_DIR="$BASE_OUTPUT_DIR"

# auto increment output directory
i=2
while [ -d "$OUTPUT_DIR" ]; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}_$i"
    ((i++))
done

echo "[INFO] Using output directory: $OUTPUT_DIR"

# training
echo "[INFO] Starting training..."
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OMP_NUM_THREADS=10 \
NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
swift sft \
    --model /data01/LLM_model/Qwen3-1.7B \
    --dataset "/data01/xushuai/code/data/agent-9/brain_0526.jsonl" \
    --num_train_epochs 4 \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --max_length 3400 \
    --warmup_ratio 0.05 \
    --learning_rate 1e-5 \
    --eval_strategy no \
    --deepspeed zero2 \
    --save_only_model true \
    --gradient_checkpointing \
    --ddp_backend nccl \
    --save_strategy epoch \
    --save_total_limit 1 \
    --train_type full \
    --torch_dtype bfloat16 \
    --add_version false \
    --output_dir "$OUTPUT_DIR" \
    --dataloader_num_workers 10 \
    --dataset_num_proc 10 \
    --attn_impl flash_attn \
    --use_liger_kernel true \
    --logging_steps 1 \
    --loss_type think_empty
