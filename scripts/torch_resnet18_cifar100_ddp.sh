#!/bin/bash
# PyTorch ResNet-18 多卡 DDP 训练（CIFAR100，2卡）
# 用法: bash scripts/torch_resnet18_cifar100_ddp.sh

set -e
cd "$(dirname "$0")/.."

DATASET="cifar100"
DATASET_DIR="./dataset"
EPOCHS=100
BATCH_SIZE=128
LR=0.1
SAVE_FREQ=10
DTYPE="float32"
NPROC=2  # 修改为实际 GPU 数量

echo "=== PyTorch ResNet-18 DDP 多卡训练（${NPROC} 卡）==="
echo "数据集: $DATASET  精度: $DTYPE  epochs: $EPOCHS"

torchrun --nproc_per_node=$NPROC \
    pytorch/cnn/resnet18_train.py \
    --dataset      $DATASET \
    --dataset_dir  $DATASET_DIR \
    --epochs       $EPOCHS \
    --batch_size   $BATCH_SIZE \
    --lr           $LR \
    --save_freq    $SAVE_FREQ \
    --dtype        $DTYPE \
    --ddp
