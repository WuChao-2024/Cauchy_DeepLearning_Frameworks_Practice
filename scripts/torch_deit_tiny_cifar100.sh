#!/bin/bash
# PyTorch DeiT-Tiny 训练（CIFAR100，单卡，float32）
# 用法: bash scripts/torch_deit_tiny_cifar100.sh

set -e
cd "$(dirname "$0")/.."

python pytorch/transformer/deit_tiny_train.py \
    --dataset      cifar100 \
    --dataset_dir  ./dataset \
    --epochs       100 \
    --batch_size   128 \
    --lr           1e-3 \
    --warmup       5 \
    --save_freq    10 \
    --dtype        float32
