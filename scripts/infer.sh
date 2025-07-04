#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model /code/ft-models/v0-20250507-072010/checkpoint-2636 \
    --stream true \
    --infer_backend pt \
    --max_new_tokens 2048