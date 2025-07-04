#!/bin/bash
set -euo pipefail

# Trap interrupts (e.g. Ctrl+C) to kill all subprocesses
trap "echo 'Interrupted. Killing subprocesses...'; pkill -P $$; exit 1" SIGINT SIGTERM

# -------------------------------------
# Qwen3-32B Training Script (DeepSpeed Zero3 + Swift)
# -------------------------------------
# Usage:
#   bash scripts/train_agent_32b.sh > logs/train_agent_32b.log 2>&1 &

# =====================
#     CONFIGURATION
# =====================
MODEL_PATH="/data01/LLM_model/DeepSeek-R1-Distill-Qwen-7B"
DATA_VERSION=20  # data_size = 13788
DATASET_PATH="/data01/xushuai/code/data/agent-${DATA_VERSION}/train.jsonl"
BASE_OUTPUT_DIR="/data01/xushuai/code/output/agent/agent_ds_7b_v${DATA_VERSION}"
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=32
MAX_STEPS=216

# ====================
#      OUTPUT DIR
# ====================
TRAINING_ARGS_VERSION=1
OUTPUT_DIR="$BASE_OUTPUT_DIR"
while [ -d "$OUTPUT_DIR" ]; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}_${TRAINING_ARGS_VERSION}"
    ((TRAINING_ARGS_VERSION++))
done

echo "[INFO] Using output directory: $OUTPUT_DIR"

# ====================
#       TRAINING
# ====================
echo "[INFO] Starting training..."

CUDA_VISIBLE_DEVICES=2,3 \
NPROC_PER_NODE=2 \
OMP_NUM_THREADS=16 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
swift sft \
    --model "$MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_length 4500 \
    --warmup_ratio 0.1 \
    --learning_rate 1e-5 \
    --eval_strategy no \
    --deepspeed zero3 \
    --save_only_model true \
    --gradient_checkpointing \
    --ddp_backend nccl \
    --save_strategy steps \
    --max_steps $MAX_STEPS \
    --save_steps $MAX_STEPS \
    --save_total_limit 1 \
    --train_type full \
    --torch_dtype bfloat16 \
    --add_version false \
    --output_dir "$OUTPUT_DIR" \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    --logging_steps 1 \
    --report_to swanlab \
    --swanlab_project dipeak-agent \
    --attn_impl flash_attn \
    --use_liger_kernel true \
    --padding_free true

# =====================
#      EVALUATION
# =====================
echo "[INFO] Starting evaluation..."

CUDA_VISIBLE_DEVICES=3,4 \
python src/eval.py --model "$OUTPUT_DIR" --qwen3

echo "[INFO] Done. Training and evaluation complete."

# End of script
