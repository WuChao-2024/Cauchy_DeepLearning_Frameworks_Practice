#!/bin/bash
# JAX ResNet-18 多卡 pmap 训练（CIFAR100）
# 用法: bash scripts/jax_resnet18_cifar100_pmap.sh

set -e
cd "$(dirname "$0")/.."

python jax/cnn/resnet18_train.py \
    --dataset      cifar100 \
    --dataset_dir  ./dataset \
    --epochs       50 \
    --batch_size   128 \
    --lr           0.1 \
    --save_freq    10 \
    --dtype        float32 \
    --pmap
