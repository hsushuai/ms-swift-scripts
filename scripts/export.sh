#!/bin/bash
set -e  # if any command fails, the script will exit immediately
set -u  # if any variable is not set, the script will exit immediately


# This script is used to convert the HF model to Megatron format.


CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift export \
    --model /data01/LLM_model/Qwen1.5-MoE-A2.7B \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir /data01/LLM_model/Qwen1.5-MoE-A2.7B-mcore