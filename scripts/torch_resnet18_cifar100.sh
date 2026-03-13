#!/bin/bash
# PyTorch ResNet-18 训练脚本（CIFAR100，单卡，float32）
# 用法: bash scripts/torch_resnet18_cifar100.sh

set -e
cd "$(dirname "$0")/.."

DATASET="cifar100"
DATASET_DIR="./dataset"
EPOCHS=100
BATCH_SIZE=128
LR=0.1
SAVE_FREQ=10
DTYPE="float32"

echo "=== PyTorch ResNet-18 训练 ==="
echo "数据集: $DATASET  精度: $DTYPE  epochs: $EPOCHS"

python pytorch/cnn/resnet18_train.py \
    --dataset      $DATASET \
    --dataset_dir  $DATASET_DIR \
    --epochs       $EPOCHS \
    --batch_size   $BATCH_SIZE \
    --lr           $LR \
    --save_freq    $SAVE_FREQ \
    --dtype        $DTYPE
