![](imgs/CauchyDelta.png)

# PyTorch vs JAX Comparative Learning Project

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/JAX-0.9-9cf?logo=google&logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/ONNX-1.20-61DAFB?logo=onnx&logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/TensorBoard-2.20-4CAF50?logo=tensorflow&logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/NVIDIA-CUDA-76B900?logo=nvidia&logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/RDK-S100-FF6F00?logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/License-AGPL--3.0-blue?logo=gnu&logoColor=white" height="25" />
</p>


This project is designed for developers familiar with PyTorch, helping you quickly understand the JAX framework through comparative learning. Each network has both PyTorch and JAX versions with aligned code structures for easy direct comparison.

## 0. Project Structure

```
.
├── dataset/                    # Dataset directory
│   ├── download.py             # Dataset download script
│   ├── mnist/                  # MNIST dataset
│   ├── cifar100/               # CIFAR-100 dataset
│   └── imagenet/               # ImageNet dataset
│
├── shared/                     # Shared utilities
│   ├── data_utils.py           # Dataset loading (PyTorch DataLoader + numpy iterator)
│   └── model_utils.py          # Model statistics (parameter count, FLOPs)
│
├── pytorch/
│   ├── cnn/
│   │   ├── resnet18.py         # ResNet-18 network definition
│   │   ├── resnet18_train.py   # Training (single GPU / DDP multi-GPU)
│   │   ├── resnet18_eval.py    # Accuracy evaluation (TOP-1/TOP-5)
│   │   ├── resnet18_export.py  # Export to ONNX
│   │   ├── resnet18_onnx_eval.py
│   │   ├── mobilenetv2.py
│   │   ├── mobilenetv2_train.py
│   │   ├── mobilenetv2_eval.py
│   │   ├── mobilenetv2_export.py
│   │   └── mobilenetv2_onnx_eval.py
│   └── transformer/
│       ├── vit_small.py        # ViT-Small network definition
│       ├── vit_small_train.py
│       ├── vit_small_eval.py
│       ├── vit_small_export.py
│       ├── vit_small_onnx_eval.py
│       ├── deit_tiny.py        # DeiT-Tiny network definition
│       ├── deit_tiny_train.py
│       ├── deit_tiny_eval.py
│       ├── deit_tiny_export.py
│       └── deit_tiny_onnx_eval.py
│
├── jax/
│   ├── cnn/
│   │   ├── resnet18.py         # JAX/Flax ResNet-18
│   │   ├── resnet18_train.py   # Training (single GPU / pmap multi-GPU)
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
├── scripts/                    # One-click launch scripts
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

## 1. Pytorch vs JAX

![](imgs/nano_banana_pytorch_jax.png)


## 2. Environment Setup

### 2.1 Create conda environment

```bash
conda create -n le_jax_torch python=3.12
conda activate le_jax_torch
```

### 2.2 Install dependencies

GPU environment (CUDA 12.8):

```bash
pip install -r requirements.txt
```

CPU environment (no GPU, suitable for ARM devices):

```bash
# Replace GPU packages in requirements.txt
pip install torch torchvision torchaudio
pip install "jax[cpu]"
pip install flax optax orbax-checkpoint
pip install huggingface_hub datasets pyarrow Pillow
pip install tensorboard tensorboardX
pip install onnx onnxruntime thop numpy tqdm
```

JAX ONNX export (optional):

```bash
pip install jax2onnx
```

## 3. Dataset Download

### 3.1 MNIST and CIFAR-100

No HuggingFace Token required, download directly:

```bash
python dataset/download.py mnist cifar100
```

### 3.2 ImageNet

ImageNet requires prior access permission from HuggingFace:

1. Create an Access Token at https://huggingface.co/settings/tokens
2. Click "Access repository" on https://huggingface.co/datasets/ILSVRC/imagenet-1k to request access
3. After approval, download:

```bash
HF_TOKEN=your_token python dataset/download.py imagenet
```

Or:

```bash
python dataset/download.py --token your_token imagenet
```

Downloads support resume functionality. If interrupted, re-running will continue from the last progress point. Datasets are saved by default in the `dataset/` directory.

### 3.3 Download all datasets

```bash
HF_TOKEN=your_token python dataset/download.py
```

## 4. PyTorch Training Experience

### 4.1 ResNet-18 Training (CIFAR-100)

Single GPU training:

```bash
cd pytorch/cnn
python resnet18_train.py --dataset cifar100 --dataset_dir ../../dataset --epochs 100
```

Or use pre-configured script:

```bash
bash scripts/torch_resnet18_cifar100.sh
```

bfloat16 mixed precision training (compare speed and accuracy differences with float32):

```bash
bash scripts/torch_resnet18_cifar100_bf16.sh
```

Multi-GPU DDP training (2 GPUs):

```bash
bash scripts/torch_resnet18_cifar100_ddp.sh
```

Training outputs are saved in `pytorch/cnn/outputs/resnet18_torch_cifar100_float32_YYYYMMDD_HHMMSS/` directory, containing:
- `config.txt`: Training configuration
- `tensorboard/`: TensorBoard logs
- `resnet18_epoch010.pth`, `resnet18_epoch020.pth`...: Periodically saved weights
- `resnet18_best.pth`: Best weights

### 4.2 View TensorBoard

```bash
tensorboard --logdir pytorch/cnn/outputs/
```

Then open http://localhost:6006 in your browser

### 4.3 Accuracy Evaluation (TOP-1 / TOP-5)

```bash
cd pytorch/cnn
python resnet18_eval.py \
    --dataset cifar100 \
    --dataset_dir ../../dataset \
    --weights ./outputs/resnet18_torch_cifar100_float32_xxx/resnet18_best.pth
```

Example output:

```
==================================================
  ResNet-18 Framework Evaluation (PyTorch)
==================================================
  Dataset:      cifar100 (test)
  Weight file:  ./outputs/.../resnet18_best.pth
  Total samples: 10000
  Batch size:   128
  Device:       cuda
  --------------------------------------------------
  TOP-1 Accuracy: 76.23%
  TOP-5 Accuracy: 93.45%
  Average inference time: 8.3 ms/batch
==================================================
```

### 4.4 Export ONNX and Evaluate

```bash
cd pytorch/cnn

# Export
python resnet18_export.py \
    --dataset cifar100 \
    --weights ./outputs/.../resnet18_best.pth \
    --output  ./outputs/.../resnet18_cifar100.onnx

# ONNX evaluation
python resnet18_onnx_eval.py \
    --dataset cifar100 \
    --dataset_dir ../../dataset \
    --onnx ./outputs/.../resnet18_cifar100.onnx
```

### 4.5 Model Statistics (Parameter Count and FLOPs)

In Python:

```python
import sys
sys.path.insert(0, "shared")
from model_utils import print_model_summary, print_submodule_stats

from pytorch.cnn.resnet18 import ResNet18
model = ResNet18(num_classes=100, input_size=(3, 32, 32))

# Print overall statistics
print_model_summary(model, input_size=(1, 3, 32, 32), title="ResNet-18 on CIFAR-100")

# Print submodule statistics
print_submodule_stats(model, "layer1", input_size=(1, 64, 32, 32))
print_submodule_stats(model, "fc",     input_size=(1, 512))
```

### 4.6 MobileNetV2 Training

```bash
bash scripts/torch_mobilenetv2_cifar100.sh
```

### 4.7 ViT-Small Training

```bash
bash scripts/torch_vit_small_cifar100.sh
```

### 4.8 DeiT-Tiny Training

```bash
bash scripts/torch_deit_tiny_cifar100.sh
```

### 4.9 ImageNet Training

```bash
# Single GPU
bash scripts/torch_resnet18_imagenet.sh

# Multi-GPU DDP (4 GPUs, bfloat16)
bash scripts/torch_resnet18_imagenet_ddp.sh
```

### 4.10 MNIST Training

```bash
bash scripts/torch_resnet18_mnist.sh
```

## 5. JAX Training Experience

Key differences between JAX and PyTorch are reflected in the code:

| Concept | PyTorch | JAX/Flax |
|------|---------|----------|
| Model state | Stored in `model` object | Stored in `params` dictionary (stateless) |
| Forward pass | `model(x)` | `model.apply({'params': params}, x)` |
| Training step | Imperative, executed line by line | Pure function, compiled with `jax.jit` |
| Multi-GPU training | `DistributedDataParallel` | `jax.pmap` |
| Weight saving | `torch.save(state_dict)` | `orbax.checkpoint` |
| Mixed precision | `torch.autocast` | Direct `.astype(jnp.bfloat16)` |

### 5.1 ResNet-18 Training (CIFAR-100)

Single GPU training:

```bash
bash scripts/jax_resnet18_cifar100.sh
```

bfloat16 mixed precision:

```bash
bash scripts/jax_resnet18_cifar100_bf16.sh
```

Multi-GPU pmap training:

```bash
bash scripts/jax_resnet18_cifar100_pmap.sh
```

Training outputs are saved in `jax/cnn/outputs/resnet18_jax_cifar100_float32_YYYYMMDD_HHMMSS/` directory, containing:
- `config.txt`: Training configuration
- `tensorboard/`: TensorBoard logs
- `checkpoints/epoch_010/`, `checkpoints/epoch_020/`...: orbax checkpoints
- `checkpoints/best/`: Best weights

### 5.2 JAX Accuracy Evaluation

```bash
cd jax/cnn
python resnet18_eval.py \
    --dataset cifar100 \
    --dataset_dir ../../dataset \
    --ckpt_dir ./outputs/resnet18_jax_cifar100_float32_xxx/checkpoints/best
```

### 5.3 JAX ONNX Export

Requires installing jax2onnx:

```bash
pip install jax2onnx
```

Then export:

```bash
cd jax/cnn
python resnet18_export.py \
    --dataset cifar100 \
    --ckpt_dir ./outputs/.../checkpoints/best \
    --output  ./outputs/.../resnet18_cifar100_jax.onnx
```

If jax2onnx is not installed, the script will save parameters in `.npz` format.

### 5.4 JAX Model Statistics

```python
import sys
sys.path.insert(0, "shared")
from model_utils import print_jax_model_summary, print_jax_submodule_stats

# Initialize model to get params
import jax
import jax.numpy as jnp
from jax.cnn.resnet18 import create_resnet18

model = create_resnet18(num_classes=100, input_size=(3, 32, 32))
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 3)), training=False)['params']

print_jax_model_summary(params, title="ResNet-18 (JAX)")
print_jax_submodule_stats(params, "ResNetLayer_0")
```

### 5.5 Other JAX Models

```bash
bash scripts/jax_mobilenetv2_cifar100.sh
bash scripts/jax_vit_small_cifar100.sh
bash scripts/jax_deit_tiny_cifar100.sh
bash scripts/jax_resnet18_imagenet_pmap.sh
```

## 6. PyTorch vs JAX Key Comparisons

### 6.1 Model Definition

PyTorch (imperative, object-oriented):

```python
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn1(self.conv1(x))
```

JAX/Flax (functional, external parameters):

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

### 6.2 Training Steps

PyTorch (imperative):

```python
optimizer.zero_grad()
logits = model(imgs)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

JAX (functional + jit compilation):

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

### 6.3 Multi-GPU Training

PyTorch DDP (process-level parallelism):

```python
# Launch: torchrun --nproc_per_node=4 train.py --ddp
model = DistributedDataParallel(model, device_ids=[local_rank])
```

JAX pmap (device-level parallelism):

```python
# Direct parallelism within a single process
@partial(jax.pmap, axis_name="batch")
def train_step(state, images, labels):
    grads = jax.lax.pmean(grads, axis_name="batch")  # Average gradients across devices
    ...
```

### 6.4 Weight Saving and Loading

PyTorch:

```python
# Save (network definition in resnet18.py, weights saved separately)
torch.save(model.state_dict(), "resnet18_best.pth")

# Load
model = ResNet18(num_classes=100, input_size=(3, 32, 32))
model.load_state_dict(torch.load("resnet18_best.pth", weights_only=True))
```

JAX (orbax checkpoint):

```python
# Save
checkpointer = ocp.StandardCheckpointer()
checkpointer.save("checkpoints/best", state)

# Load (network definition in resnet18.py, weights in checkpoints/best/)
state = checkpointer.restore("checkpoints/best", target=state)
```

### 6.5 Mixed Precision Training

PyTorch (autocast):

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
    logits = model(imgs)
    loss = criterion(logits, labels)
loss.backward()
```

JAX (direct type conversion):

```python
images_bf16 = images.astype(jnp.bfloat16)
logits = model.apply(params, images_bf16)
logits = logits.astype(jnp.float32)  # Use float32 for loss
```

## 7. Network Design Notes

### 7.1 CNN Networks

ResNet-18: Classic residual network with 18 layers, ideal for understanding residual connection design principles. For small inputs (MNIST/CIFAR), the first layer is changed to 3x3 conv and maxpool is removed.

MobileNetV2: Lightweight network using depthwise separable convolutions and inverted residual structures, with approximately 1/3 the parameters of ResNet-18, suitable for mobile deployment.

### 7.2 Transformer Networks

ViT-Small: Standard Vision Transformer with embed_dim=384, 12 layers, and 6-head attention. Uses patch_size=4 for small inputs and patch_size=16 for ImageNet.

DeiT-Tiny: Data-efficient Image Transformer with embed_dim=192, 12 layers, and 3-head attention. Compared to ViT, it adds a distillation token and returns two logits during training, averaging them during inference.

### 7.3 Input Size Adaptation

| Dataset | Input Size | patch_size | Num Classes |
|--------|----------|------------|--------|
| MNIST | 1×28×28 | 4 | 10 |
| CIFAR-100 | 3×32×32 | 4 | 100 |
| ImageNet | 3×224×224 | 16 | 1000 |

## 8. Performance and Accuracy Reference

The following data will be filled in after training completion for reference comparison.

### 8.1 CIFAR-100 Accuracy (100 epochs)

| Model | Framework | Precision | TOP-1 | TOP-5 | Inference Time(ms/batch) |
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

### 8.2 CIFAR-100 ONNX Accuracy Comparison

| Model | Export Framework | TOP-1 | TOP-5 | Inference Time(ms/batch) |
|------|---------|-------|-------|-------------------|
| ResNet-18 | PyTorch ONNX | - | - | - |
| ResNet-18 | JAX ONNX | - | - | - |
| MobileNetV2 | PyTorch ONNX | - | - | - |
| ViT-Small | PyTorch ONNX | - | - | - |
| DeiT-Tiny | PyTorch ONNX | - | - | - |

### 8.3 ImageNet Accuracy (90 epochs)

| Model | Framework | Precision | TOP-1 | TOP-5 |
|------|------|------|-------|-------|
| ResNet-18 | PyTorch float32 | - | - | - |
| ResNet-18 | PyTorch bfloat16 DDP | - | - | - |
| ResNet-18 | JAX bfloat16 pmap | - | - | - |

### 8.4 Model Parameter Count and FLOPs

| Model | Parameters | FLOPs (CIFAR-100) | FLOPs (ImageNet) |
|------|--------|-------------------|------------------|
| ResNet-18 | ~11M | - | ~1.8G |
| MobileNetV2 | ~3.4M | - | ~300M |
| ViT-Small | ~22M | - | ~4.6G |
| DeiT-Tiny | ~5.7M | - | ~1.3G |

## 9. Common Issues

### JAX First Run is Very Slow

JAX uses XLA compilation. The first call to a `jit`-compiled function incurs compilation overhead (typically 10-60 seconds), but subsequent calls are fast. This is normal behavior.

### ONNX Export Failure

JAX ONNX export requires the `jax2onnx` library. If not installed, the script will save parameters in `.npz` format. It's recommended to use the PyTorch version export script for better stability.

### Batch Size Meaning in DDP Training

In PyTorch DDP, `--batch_size` refers to the batch size per GPU. Total batch size = batch_size × number of GPUs.

### JAX pmap Requires Batch Divisible by Device Count

Training scripts automatically adjust batch_size to be a multiple of the device count.

### bfloat16 vs float16

This project uses bfloat16 instead of float16 because bfloat16 has the same numerical range as float32, doesn't require GradScaler, and provides more stable training.