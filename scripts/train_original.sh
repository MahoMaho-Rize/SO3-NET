#!/bin/bash
# Train original UprightNet baseline (with rotation augmentation)
# Expected: ~111000 training samples, ~47 hours on RTX 5880

cd "$(dirname "$0")/.."

python3 train.py \
    --network uprightnet \
    --epoch 50 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --num_points 2048 \
    --gpu_idx "0"
