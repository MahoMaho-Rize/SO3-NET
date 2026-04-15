#!/bin/bash
# Train L=2 equivariant model — cuet accelerated
# Online random rotation: 1110 train objects, infinite augmentation per epoch
# Usage:
#   bash scripts/train_l2.sh          # default k=32
#   bash scripts/train_l2.sh 64       # k=64

K=${1:-32}

cd "$(dirname "$0")/.."

echo "=== L=2, online rotation, k=$K ==="

python3 train.py \
    --network equivariant \
    --conv_type depthwise \
    --irreps_hidden "128x0e+128x1o+128x2e" \
    --lmax 2 \
    --loss_type vmf \
    --epoch 200 \
    --batch_size 4 \
    --learning_rate 0.001 \
    --weight_decay 1e-5 \
    --num_points 2048 \
    --max_radius 0.1 \
    --num_neighbors $K \
    --equi_layers 6 \
    --radial_neurons 128 \
    --num_radial_basis 16 \
    --vmf_kappa_init 1.0 \
    --beta 0.5 \
    --gpu_idx "0"
