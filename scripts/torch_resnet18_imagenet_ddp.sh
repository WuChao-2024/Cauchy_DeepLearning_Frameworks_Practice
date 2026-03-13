#!/bin/bash
# PyTorch ResNet-18 训练（ImageNet，多卡 DDP）
# 用法: bash scripts/torch_resnet18_imagenet_ddp.sh

set -e
cd "$(dirname "$0")/.."

NPROC=4  # 修改为实际 GPU 数量

torchrun --nproc_per_node=$NPROC \
    pytorch/cnn/resnet18_train.py \
    --dataset      imagenet \
    --dataset_dir  ./dataset \
    --epochs       90 \
    --batch_size   256 \
    --lr           0.1 \
    --save_freq    10 \
    --dtype        bfloat16 \
    --ddp
