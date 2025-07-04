#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
swift deploy \
    --model /code/ft-models/v0-20250507-072010/checkpoint-2636 \
    --infer_backend vllm \
    --max_new_tokens 2048 \
    --served_model_name qwen3-1.7b