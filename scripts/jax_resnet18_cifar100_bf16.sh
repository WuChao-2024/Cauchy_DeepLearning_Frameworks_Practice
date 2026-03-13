#!/bin/bash
# JAX ResNet-18 训练（CIFAR100，bfloat16 混合精度）
# 用法: bash scripts/jax_resnet18_cifar100_bf16.sh

set -e
cd "$(dirname "$0")/.."

python jax/cnn/resnet18_train.py \
    --dataset      cifar100 \
    --dataset_dir  ./dataset \
    --epochs       50 \
    --batch_size   128 \
    --lr           0.1 \
    --save_freq    10 \
    --dtype        bfloat16
