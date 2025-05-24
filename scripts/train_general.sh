#!/bin/bash
set -e  # if any command fails, the script will exit immediately
set -u  # if any variable is not set, the script will exit immediately

# arguments
PER_DEVICE_TRAIN_BATCH_SIZE=6
GRADIENT_ACCUMULATION_STEPS=16
DATA_VERSION=2

BASE_OUTPUT_DIR="/home/xushuai/code/output/general/general_1.7b_v${DATA_VERSION}"
OUTPUT_DIR="$BASE_OUTPUT_DIR"

# auto increment output directory
i=2
while [ -d "$OUTPUT_DIR" ]; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}-$i"
    ((i++))
done

echo "[INFO] Using output directory: $OUTPUT_DIR"

# training
echo "[INFO] Starting training..."
CUDA_VISIBLE_DEVICES=3,4 \
OMP_NUM_THREADS=32 \
NPROC_PER_NODE=2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
swift sft \
    --model /home/LLM_model/Qwen3-1.7B \
    --dataset "/home/xushuai/code/data/general-${DATA_VERSION}/train.jsonl" \
    --num_train_epochs 4 \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --max_length 1500 \
    --warmup_ratio 0.05 \
    --learning_rate 1e-4 \
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
    --output_dir "$OUTPUT_DIR" \
    --dataloader_num_workers 10 \
    --dataset_num_proc 10 \
    --logging_steps 1 \
    --report_to swanlab \
    --attn_impl flash_attn \
    --use_liger_kernel true \

# evaluation
echo "[INFO] Starting evaluation..."
CUDA_VISIBLE_DEVICES=3,4 \
python src/eval.py --model "$OUTPUT_DIR"

echo "[INFO] Done."
