#!/bin/bash
set -euo pipefail

# Trap interrupts (e.g. Ctrl+C) to kill all subprocesses
trap "echo 'Interrupted. Killing subprocesses...'; pkill -P $$; exit 1" SIGINT SIGTERM

# Description:
# This script trains the Qwen3-30B-A3B model with specified parameters,
# exports the model to Hugging Face format, and runs evaluation.

# Usage:
#   bash scripts/train_agent_30b.sh > logs/train_agent_30b.log 2>&1 &

#######################
# CONFIGURATION
#######################
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=232  # MUST BE MULTIPLE OF 8
TP=2
PP=2
EP=2
# TP * EP * PP = WORLD_SIZE
# DP = WORLD_SIZE / (TP * PP)
DATA_VERSION=16  # data_size = 3074  # packing
BASE_OUTPUT_DIR="/data01/xushuai/code/output/agent/agent_30b_v${DATA_VERSION}"
DATASET_PATH="/data01/xushuai/code/data/agent-${DATA_VERSION}/train.jsonl"
MODEL_LOAD_PATH="/data01/LLM_model/Qwen3-30B-A3B-mcore"

#######################
# GENERATE OUTPUT DIR
#######################
TRAINING_ARGS_VERSION=1
OUTPUT_DIR="$BASE_OUTPUT_DIR"
while [ -d "$OUTPUT_DIR" ]; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}_${TRAINING_ARGS_VERSION}"
    ((TRAINING_ARGS_VERSION++))
done

MEGATRON_OUTPUT_DIR="$OUTPUT_DIR/megatron_output"
HF_OUTPUT_DIR="$OUTPUT_DIR/hf_output"

echo "[INFO] Output directory: $OUTPUT_DIR"

#######################
# TRAINING
#######################
echo "[INFO] Starting training..."

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OMP_NUM_THREADS=16 \
NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
megatron sft \
    --load "$MODEL_LOAD_PATH" \
    --dataset "$DATASET_PATH" \
    --tensor_model_parallel_size $TP \
    --expert_model_parallel_size $EP \
    --pipeline_model_parallel_size $PP \
    --moe_grouped_gemm true \
    --moe_aux_loss_coeff 0.01 \
    --use_distributed_optimizer \
    --moe_token_dispatcher_type alltoall \
    --moe_expert_capacity_factor 1.0 \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 24 \
    --max_epochs 4 \
    --packing true \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_iters 0 \
    --min_lr 1e-6 \
    --eval_iters 0 \
    --save "$MEGATRON_OUTPUT_DIR" \
    --max_length 4000 \
    --num_workers 16 \
    --bf16 true \
    --ddp_backend nccl \
    --dataset_num_proc 16 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --use_flash_attn true \
    --log_interval 1 \
    --add_version false \
    --split_dataset_ratio 0 \
    --no_load_optim \
    --no_load_rng
    # --wandb_project dipeak-agent \
    # --wandb_exp_name $OUTPUT_DIR \

echo "[INFO] Training complete."

#######################
# EXPORT TO HF FORMAT
#######################
echo "[INFO] Exporting model to HuggingFace format..."

CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift export \
    --mcore_model "$MEGATRON_OUTPUT_DIR" \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir "$HF_OUTPUT_DIR"

echo "[INFO] Export complete."

#######################
# EVALUATION
#######################
echo "[INFO] Starting evaluation..."

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python src/eval.py --model "$HF_OUTPUT_DIR" --qwen3

echo "[INFO] Evaluation complete."
echo "[INFO] All tasks finished successfully."
