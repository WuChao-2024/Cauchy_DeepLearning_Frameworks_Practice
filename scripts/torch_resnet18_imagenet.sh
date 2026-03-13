#!/bin/bash
# PyTorch ResNet-18 训练（ImageNet，单卡）
# 用法: bash scripts/torch_resnet18_imagenet.sh

set -e
cd "$(dirname "$0")/.."

python pytorch/cnn/resnet18_train.py \
    --dataset      imagenet \
    --dataset_dir  ./dataset \
    --epochs       90 \
    --batch_size   256 \
    --lr           0.1 \
    --save_freq    10 \
    --dtype        float32
