#!/bin/bash
# PyTorch MobileNetV2 训练（CIFAR100，单卡，float32）
# 用法: bash scripts/torch_mobilenetv2_cifar100.sh

set -e
cd "$(dirname "$0")/.."

python pytorch/cnn/mobilenetv2_train.py \
    --dataset      cifar100 \
    --dataset_dir  ./dataset \
    --epochs       100 \
    --batch_size   128 \
    --lr           0.01 \
    --save_freq    10 \
    --dtype        float32
