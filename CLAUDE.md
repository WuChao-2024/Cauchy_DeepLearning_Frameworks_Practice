# CLAUDE.md — 项目开发摘要

## 项目概述

PyTorch vs JAX 对比教学项目，面向熟悉 PyTorch 的开发者，通过对比式学习帮助快速理解 JAX 框架。

## 项目状态：已完成

所有代码文件均已实现，项目可直接使用。

## 文件结构

```
.
├── dataset/
│   ├── download.py             # HuggingFace 数据集下载（支持断点续传）
│   ├── mnist/                  # MNIST parquet 文件
│   ├── cifar100/               # CIFAR-100 parquet 文件
│   └── imagenet/               # ImageNet parquet 文件
│
├── shared/
│   ├── data_utils.py           # 数据集加载（PyTorch DataLoader + numpy 迭代器）
│   └── model_utils.py          # 模型统计（参数量、FLOPs，PyTorch + JAX 两套）
│
├── pytorch/
│   ├── cnn/
│   │   ├── resnet18.py         # ResNet-18（从零实现）
│   │   ├── resnet18_train.py   # 训练（单卡/DDP 多卡/bfloat16）
│   │   ├── resnet18_eval.py    # TOP-1/TOP-5 评测
│   │   ├── resnet18_export.py  # 导出 ONNX
│   │   ├── resnet18_onnx_eval.py
│   │   ├── mobilenetv2.py      # MobileNetV2（从零实现）
│   │   ├── mobilenetv2_train.py
│   │   ├── mobilenetv2_eval.py
│   │   ├── mobilenetv2_export.py
│   │   └── mobilenetv2_onnx_eval.py
│   └── transformer/
│       ├── vit_small.py        # ViT-Small（从零实现，embed_dim=384, depth=12）
│       ├── vit_small_train.py
│       ├── vit_small_eval.py
│       ├── vit_small_export.py
│       ├── vit_small_onnx_eval.py
│       ├── deit_tiny.py        # DeiT-Tiny（从零实现，embed_dim=192, depth=12）
│       ├── deit_tiny_train.py
│       ├── deit_tiny_eval.py
│       ├── deit_tiny_export.py
│       └── deit_tiny_onnx_eval.py
│
├── jax/
│   ├── cnn/                    # 与 pytorch/cnn/ 结构对齐
│   │   ├── resnet18.py         # JAX/Flax ResNet-18
│   │   ├── resnet18_train.py   # 训练（单卡/pmap 多卡/bfloat16）
│   │   ├── resnet18_eval.py
│   │   ├── resnet18_export.py  # 优先 jax2onnx，fallback 保存 npz
│   │   ├── resnet18_onnx_eval.py
│   │   ├── mobilenetv2.py
│   │   ├── mobilenetv2_train.py
│   │   ├── mobilenetv2_eval.py
│   │   ├── mobilenetv2_export.py
│   │   └── mobilenetv2_onnx_eval.py
│   └── transformer/
│       ├── vit_small.py
│       ├── vit_small_train.py
│       ├── vit_small_eval.py
│       ├── vit_small_export.py
│       ├── vit_small_onnx_eval.py
│       ├── deit_tiny.py
│       ├── deit_tiny_train.py
│       ├── deit_tiny_eval.py
│       ├── deit_tiny_export.py
│       └── deit_tiny_onnx_eval.py
│
├── scripts/                    # 一键启动脚本（17 个）
│   ├── torch_resnet18_cifar100.sh
│   ├── torch_resnet18_cifar100_bf16.sh
│   ├── torch_resnet18_cifar100_ddp.sh
│   ├── torch_resnet18_imagenet.sh
│   ├── torch_resnet18_imagenet_ddp.sh
│   ├── torch_resnet18_mnist.sh
│   ├── torch_mobilenetv2_cifar100.sh
│   ├── torch_vit_small_cifar100.sh
│   ├── torch_deit_tiny_cifar100.sh
│   ├── jax_resnet18_cifar100.sh
│   ├── jax_resnet18_cifar100_bf16.sh
│   ├── jax_resnet18_cifar100_pmap.sh
│   ├── jax_resnet18_imagenet_pmap.sh
│   ├── jax_mobilenetv2_cifar100.sh
│   ├── jax_vit_small_cifar100.sh
│   └── jax_deit_tiny_cifar100.sh
│
├── requirements.txt
├── README_cn.md
└── CLAUDE.md
```

## 关键设计决策

### 数据集加载
- 数据来源：HuggingFace parquet 格式（ylecun/mnist, tanganke/cifar100, ILSVRC/imagenet-1k）
- shared/data_utils.py 提供两套接口：
  - `get_dataloader()` → PyTorch DataLoader（channels-first: B,C,H,W）
  - `get_numpy_iterator()` → numpy 迭代器（channels-last: B,H,W,C，供 JAX 使用）
- JAX 版本复用 PyTorch DataLoader，在迭代时做 transpose 转换

### 网络设计
- 所有网络从零实现，不引用 torchvision/timm 等库的预训练模型
- 小输入适配（MNIST/CIFAR）：ResNet-18 第一层改为 3x3 conv，去掉 maxpool；MobileNetV2 压缩 stride；ViT/DeiT 用 patch_size=4
- DeiT 训练时返回 (cls_out, dist_out) 两个 logits，eval 时返回平均值

### JAX 与 PyTorch 的核心差异（代码层面）
- JAX 模型无状态：参数在 params 字典中，BatchNorm 需要额外的 batch_stats
- JAX 训练步骤是纯函数，用 @jax.jit 编译
- JAX 多卡用 jax.pmap（单进程内并行），PyTorch 用 DDP（多进程）
- JAX 权重保存用 orbax.checkpoint，PyTorch 用 torch.save(state_dict())
- JAX bfloat16：直接 images.astype(jnp.bfloat16)；PyTorch：torch.autocast

### 训练产物命名规则
格式：`{模型名}_{框架}_{数据集}_{精度}_{YYYYMMDD_HHMMSS}`
例如：`resnet18_torch_cifar100_float32_20260312_215012`

每个训练产物目录包含：
- `config.txt`：训练配置
- `tensorboard/`：TensorBoard 日志
- PyTorch：`{model}_epoch{N:03d}.pth`、`{model}_best.pth`
- JAX：`checkpoints/epoch_{N:03d}/`、`checkpoints/best/`（orbax 格式）

### ONNX 导出策略
- PyTorch：直接 torch.onnx.export，稳定可靠
- JAX：优先尝试 jax2onnx（pip install jax2onnx），失败则保存 .npz 参数文件

### 模型统计工具（shared/model_utils.py）
- PyTorch：`print_model_summary(model, input_size, title)` / `print_submodule_stats(model, module_name, input_size)`
- JAX：`print_jax_model_summary(params, title)` / `print_jax_submodule_stats(params, module_path)`
- FLOPs 统计依赖 thop 库（PyTorch 专用）

## 环境要求
- Python 3.12
- PyTorch 2.10.0 + CUDA 12.8
- JAX >= 0.4.35（cuda12 版本）
- Flax >= 0.10.0, Optax >= 0.2.4, Orbax-checkpoint >= 0.10.0

## 开发机器上的conda环境名称
```bash
conda activate le_jax_torch
```

## 待完善（可选）
- README_cn.md 中的性能参考表格（需实际训练后填入）
- .gitignore 文件（建议添加 outputs/, dataset/, __pycache__/）
- 英文 README.md（由用户决定翻译）
