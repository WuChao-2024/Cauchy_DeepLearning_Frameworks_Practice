![](imgs/CauchyDelta.png)

# PyTorch vs JAX 对比学习项目

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/JAX-0.9-9cf?logo=google&logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/ONNX-1.20-61DAFB?logo=onnx&logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/TensorBoard-2.20-4CAF50?logo=tensorflow&logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/NVIDIA-CUDA-76B900?logo=nvidia&logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/NVIDIA-TensorRT-76B900?logo=nvidia&logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/RDK-S100-FF6F00?logo=rockstargames&logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/Cauchy-Kesai-FF0000?logo=github&logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/License-AGPL--3.0-blue?logo=gnu&logoColor=white" height="25" />
  
</p>


这个项目面向熟悉 PyTorch 的开发者，通过对比式学习帮助你快速理解 JAX 框架。每个网络都有 PyTorch 和 JAX 两个版本，代码结构对齐，方便直接对比。

## 0. 项目结构

```
.
├── dataset/                    # 数据集目录
│   ├── download.py             # 数据集下载脚本
│   ├── mnist/                  # MNIST 数据集
│   ├── cifar100/               # CIFAR-100 数据集
│   └── imagenet/               # ImageNet 数据集
│
├── shared/                     # 共享工具
│   ├── data_utils.py           # 数据集加载（PyTorch DataLoader + numpy 迭代器）
│   └── model_utils.py          # 模型统计（参数量、FLOPs）
│
├── pytorch/
│   ├── cnn/
│   │   ├── resnet18.py         # ResNet-18 网络定义
│   │   ├── resnet18_train.py   # 训练（单卡/DDP 多卡）
│   │   ├── resnet18_eval.py    # 精度评测（TOP-1/TOP-5）
│   │   ├── resnet18_export.py  # 导出 ONNX
│   │   ├── resnet18_onnx_eval.py
│   │   ├── mobilenetv2.py
│   │   ├── mobilenetv2_train.py
│   │   ├── mobilenetv2_eval.py
│   │   ├── mobilenetv2_export.py
│   │   └── mobilenetv2_onnx_eval.py
│   └── transformer/
│       ├── vit_small.py        # ViT-Small 网络定义
│       ├── vit_small_train.py
│       ├── vit_small_eval.py
│       ├── vit_small_export.py
│       ├── vit_small_onnx_eval.py
│       ├── deit_tiny.py        # DeiT-Tiny 网络定义
│       ├── deit_tiny_train.py
│       ├── deit_tiny_eval.py
│       ├── deit_tiny_export.py
│       └── deit_tiny_onnx_eval.py
│
├── jax/
│   ├── cnn/
│   │   ├── resnet18.py         # JAX/Flax ResNet-18
│   │   ├── resnet18_train.py   # 训练（单卡/pmap 多卡）
│   │   ├── resnet18_eval.py
│   │   ├── resnet18_export.py
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
├── scripts/                    # 一键启动脚本
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
└── README_cn.md
```

## 1. Pytorch和JAX对比

![](imgs/nano_banana_pytorch_jax.png)

## 2. 环境准备

### 2.1 创建 conda 环境

```bash
conda create -n le_jax_torch python=3.12
conda activate le_jax_torch
```

### 2.2 安装依赖

GPU 环境（CUDA 12.8）：

```bash
pip install -r requirements.txt
```

CPU 环境（无 GPU，适合 ARM 设备）：

```bash
# 替换 requirements.txt 中的 GPU 包
pip install torch torchvision torchaudio
pip install "jax[cpu]"
pip install flax optax orbax-checkpoint
pip install huggingface_hub datasets pyarrow Pillow
pip install tensorboard tensorboardX
pip install onnx onnxruntime thop numpy tqdm
```

JAX 的 ONNX 导出（可选）：

```bash
pip install jax2onnx
```

## 3. 数据集下载

### 3.1 MNIST 和 CIFAR-100

不需要 HuggingFace Token，直接下载：

```bash
python dataset/download.py mnist cifar100
```

### 3.2 ImageNet

ImageNet 需要先在 HuggingFace 申请访问权限：

1. 在 https://huggingface.co/settings/tokens 创建一个 Access Token
2. 在 https://huggingface.co/datasets/ILSVRC/imagenet-1k 页面点击 "Access repository" 申请访问
3. 等待审核通过后下载：

```bash
HF_TOKEN=your_token python dataset/download.py imagenet
```

或者：

```bash
python dataset/download.py --token your_token imagenet
```

下载支持断点续传，中断后重新运行会从上次进度继续。数据集默认保存在 `dataset/` 目录下。

### 3.3 下载全部数据集

```bash
HF_TOKEN=your_token python dataset/download.py
```

## 4. PyTorch 训练体验

### 4.1 ResNet-18 训练（CIFAR-100）

单卡训练：

```bash
cd pytorch/cnn
python resnet18_train.py --dataset cifar100 --dataset_dir ../../dataset --epochs 100
```

或者使用预制脚本：

```bash
bash scripts/torch_resnet18_cifar100.sh
```

bfloat16 混合精度训练（对比 float32 速度和精度差异）：

```bash
bash scripts/torch_resnet18_cifar100_bf16.sh
```

多卡 DDP 训练（2 卡）：

```bash
bash scripts/torch_resnet18_cifar100_ddp.sh
```

训练产物会保存在 `pytorch/cnn/outputs/resnet18_torch_cifar100_float32_YYYYMMDD_HHMMSS/` 目录下，包含：
- `config.txt`：训练配置
- `tensorboard/`：TensorBoard 日志
- `resnet18_epoch010.pth`、`resnet18_epoch020.pth`...：定期保存的权重
- `resnet18_best.pth`：最优权重

### 4.2 查看 TensorBoard

```bash
tensorboard --logdir pytorch/cnn/outputs/
```

然后在浏览器打开 http://localhost:6006

### 4.3 精度评测（TOP-1 / TOP-5）

```bash
cd pytorch/cnn
python resnet18_eval.py \
    --dataset cifar100 \
    --dataset_dir ../../dataset \
    --weights ./outputs/resnet18_torch_cifar100_float32_xxx/resnet18_best.pth
```

输出示例：

```
==================================================
  ResNet-18 框架评测（PyTorch）
==================================================
  数据集:       cifar100 (test)
  权重文件:     ./outputs/.../resnet18_best.pth
  总样本数:     10000
  批次大小:     128
  设备:         cuda
  --------------------------------------------------
  TOP-1 准确率: 76.23%
  TOP-5 准确率: 93.45%
  平均推理时间: 8.3 ms/batch
==================================================
```

### 4.4 导出 ONNX 并评测

```bash
cd pytorch/cnn

# 导出
python resnet18_export.py \
    --dataset cifar100 \
    --weights ./outputs/.../resnet18_best.pth \
    --output  ./outputs/.../resnet18_cifar100.onnx

# ONNX 评测
python resnet18_onnx_eval.py \
    --dataset cifar100 \
    --dataset_dir ../../dataset \
    --onnx ./outputs/.../resnet18_cifar100.onnx
```

### 4.5 模型统计（参数量和 FLOPs）

在 Python 中：

```python
import sys
sys.path.insert(0, "shared")
from model_utils import print_model_summary, print_submodule_stats

from pytorch.cnn.resnet18 import ResNet18
model = ResNet18(num_classes=100, input_size=(3, 32, 32))

# 打印整体统计
print_model_summary(model, input_size=(1, 3, 32, 32), title="ResNet-18 on CIFAR-100")

# 打印子模块统计
print_submodule_stats(model, "layer1", input_size=(1, 64, 32, 32))
print_submodule_stats(model, "fc",     input_size=(1, 512))
```

### 4.6 MobileNetV2 训练

```bash
bash scripts/torch_mobilenetv2_cifar100.sh
```

### 4.7 ViT-Small 训练

```bash
bash scripts/torch_vit_small_cifar100.sh
```

### 4.8 DeiT-Tiny 训练

```bash
bash scripts/torch_deit_tiny_cifar100.sh
```

### 4.9 ImageNet 训练

```bash
# 单卡
bash scripts/torch_resnet18_imagenet.sh

# 多卡 DDP（4 卡，bfloat16）
bash scripts/torch_resnet18_imagenet_ddp.sh
```

### 4.10 MNIST 训练

```bash
bash scripts/torch_resnet18_mnist.sh
```

## 5. JAX 训练体验

JAX 与 PyTorch 的核心区别体现在代码里：

| 概念 | PyTorch | JAX/Flax |
|------|---------|----------|
| 模型状态 | 存在 `model` 对象里 | 存在 `params` 字典里（无状态） |
| 前向传播 | `model(x)` | `model.apply({'params': params}, x)` |
| 训练步骤 | 命令式，逐行执行 | 纯函数，用 `jax.jit` 编译 |
| 多卡训练 | `DistributedDataParallel` | `jax.pmap` |
| 权重保存 | `torch.save(state_dict)` | `orbax.checkpoint` |
| 混合精度 | `torch.autocast` | 直接 `.astype(jnp.bfloat16)` |

### 5.1 ResNet-18 训练（CIFAR-100）

单卡训练：

```bash
bash scripts/jax_resnet18_cifar100.sh
```

bfloat16 混合精度：

```bash
bash scripts/jax_resnet18_cifar100_bf16.sh
```

多卡 pmap 训练：

```bash
bash scripts/jax_resnet18_cifar100_pmap.sh
```

训练产物保存在 `jax/cnn/outputs/resnet18_jax_cifar100_float32_YYYYMMDD_HHMMSS/` 目录下，包含：
- `config.txt`：训练配置
- `tensorboard/`：TensorBoard 日志
- `checkpoints/epoch_010/`、`checkpoints/epoch_020/`...：orbax checkpoint
- `checkpoints/best/`：最优权重

### 5.2 JAX 精度评测

```bash
cd jax/cnn
python resnet18_eval.py \
    --dataset cifar100 \
    --dataset_dir ../../dataset \
    --ckpt_dir ./outputs/resnet18_jax_cifar100_float32_xxx/checkpoints/best
```

### 5.3 JAX 导出 ONNX

需要安装 jax2onnx：

```bash
pip install jax2onnx
```

然后导出：

```bash
cd jax/cnn
python resnet18_export.py \
    --dataset cifar100 \
    --ckpt_dir ./outputs/.../checkpoints/best \
    --output  ./outputs/.../resnet18_cifar100_jax.onnx
```

如果没有安装 jax2onnx，脚本会将参数保存为 `.npz` 格式。

### 5.4 JAX 模型统计

```python
import sys
sys.path.insert(0, "shared")
from model_utils import print_jax_model_summary, print_jax_submodule_stats

# 初始化模型获取 params
import jax
import jax.numpy as jnp
from jax.cnn.resnet18 import create_resnet18

model = create_resnet18(num_classes=100, input_size=(3, 32, 32))
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 3)), training=False)['params']

print_jax_model_summary(params, title="ResNet-18 (JAX)")
print_jax_submodule_stats(params, "ResNetLayer_0")
```

### 5.5 其他 JAX 模型

```bash
bash scripts/jax_mobilenetv2_cifar100.sh
bash scripts/jax_vit_small_cifar100.sh
bash scripts/jax_deit_tiny_cifar100.sh
bash scripts/jax_resnet18_imagenet_pmap.sh
```

## 6. PyTorch vs JAX 关键对比

### 6.1 模型定义

PyTorch（命令式，面向对象）：

```python
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn1(self.conv1(x))
```

JAX/Flax（函数式，参数外置）：

```python
class BasicBlock(nn.Module):
    out_ch: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Conv(self.out_ch, (3, 3), strides=(self.stride, self.stride))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        return x
```

### 6.2 训练步骤

PyTorch（命令式）：

```python
optimizer.zero_grad()
logits = model(imgs)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

JAX（函数式 + jit 编译）：

```python
@jax.jit
def train_step(state, images, labels):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images, training=True)
        return cross_entropy(logits, labels), logits
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss
```

### 6.3 多卡训练

PyTorch DDP（进程级并行）：

```python
# 启动：torchrun --nproc_per_node=4 train.py --ddp
model = DistributedDataParallel(model, device_ids=[local_rank])
```

JAX pmap（设备级并行）：

```python
# 直接在单进程内并行
@partial(jax.pmap, axis_name="batch")
def train_step(state, images, labels):
    grads = jax.lax.pmean(grads, axis_name="batch")  # 跨设备梯度平均
    ...
```

### 6.4 权重保存与加载

PyTorch：

```python
# 保存（网络定义在 resnet18.py，权重单独保存）
torch.save(model.state_dict(), "resnet18_best.pth")

# 加载
model = ResNet18(num_classes=100, input_size=(3, 32, 32))
model.load_state_dict(torch.load("resnet18_best.pth", weights_only=True))
```

JAX（orbax checkpoint）：

```python
# 保存
checkpointer = ocp.StandardCheckpointer()
checkpointer.save("checkpoints/best", state)

# 加载（网络定义在 resnet18.py，权重在 checkpoints/best/）
state = checkpointer.restore("checkpoints/best", target=state)
```

### 6.5 混合精度训练

PyTorch（autocast）：

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
    logits = model(imgs)
    loss = criterion(logits, labels)
loss.backward()
```

JAX（直接类型转换）：

```python
images_bf16 = images.astype(jnp.bfloat16)
logits = model.apply(params, images_bf16)
logits = logits.astype(jnp.float32)  # 损失用 float32
```

## 7. 网络设计说明

### 7.1 CNN 网络

ResNet-18：经典残差网络，18 层，适合理解残差连接的设计思想。小输入（MNIST/CIFAR）时第一层改为 3x3 conv，去掉 maxpool。

MobileNetV2：轻量网络，使用深度可分离卷积和倒残差结构，参数量约为 ResNet-18 的 1/3，适合移动端部署。

### 7.2 Transformer 网络

ViT-Small：标准 Vision Transformer，embed_dim=384，12 层，6 头注意力。小输入用 patch_size=4，ImageNet 用 patch_size=16。

DeiT-Tiny：Data-efficient Image Transformer，embed_dim=192，12 层，3 头注意力。相比 ViT 增加了蒸馏 token，训练时返回两个 logits，推理时取平均。

### 7.3 输入尺寸适配

| 数据集 | 输入尺寸 | patch_size | 类别数 |
|--------|----------|------------|--------|
| MNIST | 1×28×28 | 4 | 10 |
| CIFAR-100 | 3×32×32 | 4 | 100 |
| ImageNet | 3×224×224 | 16 | 1000 |

## 8. 性能与精度参考

以下数据在训练完成后填入，供参考对比。

### 8.1 CIFAR-100 精度（100 epochs）

| 模型 | 框架 | 精度 | TOP-1 | TOP-5 | 推理时间(ms/batch) |
|------|------|------|-------|-------|-------------------|
| ResNet-18 | PyTorch float32 | - | - | - | - |
| ResNet-18 | PyTorch bfloat16 | - | - | - | - |
| ResNet-18 | JAX float32 | - | - | - | - |
| ResNet-18 | JAX bfloat16 | - | - | - | - |
| MobileNetV2 | PyTorch float32 | - | - | - | - |
| MobileNetV2 | JAX float32 | - | - | - | - |
| ViT-Small | PyTorch float32 | - | - | - | - |
| ViT-Small | JAX float32 | - | - | - | - |
| DeiT-Tiny | PyTorch float32 | - | - | - | - |
| DeiT-Tiny | JAX float32 | - | - | - | - |

### 8.2 CIFAR-100 ONNX 精度对比

| 模型 | 框架导出 | TOP-1 | TOP-5 | 推理时间(ms/batch) |
|------|---------|-------|-------|-------------------|
| ResNet-18 | PyTorch ONNX | - | - | - |
| ResNet-18 | JAX ONNX | - | - | - |
| MobileNetV2 | PyTorch ONNX | - | - | - |
| ViT-Small | PyTorch ONNX | - | - | - |
| DeiT-Tiny | PyTorch ONNX | - | - | - |

### 8.3 ImageNet 精度（90 epochs）

| 模型 | 框架 | 精度 | TOP-1 | TOP-5 |
|------|------|------|-------|-------|
| ResNet-18 | PyTorch float32 | - | - | - |
| ResNet-18 | PyTorch bfloat16 DDP | - | - | - |
| ResNet-18 | JAX bfloat16 pmap | - | - | - |

### 8.4 模型参数量和 FLOPs

| 模型 | 参数量 | FLOPs (CIFAR-100) | FLOPs (ImageNet) |
|------|--------|-------------------|------------------|
| ResNet-18 | ~11M | - | ~1.8G |
| MobileNetV2 | ~3.4M | - | ~300M |
| ViT-Small | ~22M | - | ~4.6G |
| DeiT-Tiny | ~5.7M | - | ~1.3G |

## 9. 常见问题

### JAX 第一次运行很慢

JAX 使用 XLA 编译，第一次调用 `jit` 编译的函数时会有编译开销（通常 10-60 秒），之后会很快。这是正常现象。

### ONNX 导出失败

JAX 的 ONNX 导出需要 `jax2onnx` 库，如果没有安装，脚本会将参数保存为 `.npz` 格式。推荐使用 PyTorch 版本的导出脚本，更稳定。

### DDP 训练时 batch_size 的含义

PyTorch DDP 中，`--batch_size` 是每张卡的 batch size，总 batch size = batch_size × GPU 数量。

### JAX pmap 要求 batch 能被设备数整除

训练脚本会自动将 batch_size 调整为设备数的整数倍。

### bfloat16 vs float16

本项目使用 bfloat16 而不是 float16，因为 bfloat16 数值范围与 float32 相同，不需要 GradScaler，训练更稳定。
