#!/bin/bash
# JAX ResNet-18 训练（CIFAR100，单卡，float32）
# 用法: bash scripts/jax_resnet18_cifar100.sh

set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0

python jax/cnn/resnet18_train.py \
    --dataset      cifar100 \
    --dataset_dir  ./dataset \
    --epochs       50 \
    --batch_size   128 \
    --lr           0.1 \
    --save_freq    10 \
    --dtype        float32
