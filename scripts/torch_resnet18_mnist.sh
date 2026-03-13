#!/bin/bash
# PyTorch ResNet-18 训练（MNIST，单卡）
# 用法: bash scripts/torch_resnet18_mnist.sh

set -e
cd "$(dirname "$0")/.."

python pytorch/cnn/resnet18_train.py \
    --dataset      mnist \
    --dataset_dir  ./dataset \
    --epochs       30 \
    --batch_size   128 \
    --lr           0.1 \
    --save_freq    10 \
    --dtype        float32
