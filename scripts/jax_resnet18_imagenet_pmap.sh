#!/bin/bash
# JAX ResNet-18 训练（ImageNet，多卡 pmap）
# 用法: bash scripts/jax_resnet18_imagenet_pmap.sh

set -e
cd "$(dirname "$0")/.."

python jax/cnn/resnet18_train.py \
    --dataset      imagenet \
    --dataset_dir  ./dataset \
    --epochs       90 \
    --batch_size   256 \
    --lr           0.1 \
    --save_freq    10 \
    --dtype        bfloat16 \
    --pmap
