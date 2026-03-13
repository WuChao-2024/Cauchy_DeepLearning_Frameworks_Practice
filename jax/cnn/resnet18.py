# JAX/Flax ResNet-18 网络定义（从零实现）
# JAX 与 PyTorch 的关键区别：
#   1. 模型是无状态的纯函数，参数通过 params 字典传入
#   2. BatchNorm 需要额外的 batch_stats 状态
#   3. 使用 nn.compact 装饰器简化子模块定义

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence


class BasicBlock(nn.Module):
    """ResNet 基本残差块"""
    out_ch: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, training: bool = True):
        residual = x
        out = nn.Conv(self.out_ch, (3, 3), strides=(self.stride, self.stride),
                      padding="SAME", use_bias=False)(x)
        out = nn.BatchNorm(use_running_average=not training)(out)
        out = nn.relu(out)
        out = nn.Conv(self.out_ch, (3, 3), strides=(1, 1),
                      padding="SAME", use_bias=False)(out)
        out = nn.BatchNorm(use_running_average=not training)(out)

        # shortcut：维度不匹配时用 1x1 conv 对齐
        if residual.shape != out.shape:
            residual = nn.Conv(self.out_ch, (1, 1), strides=(self.stride, self.stride),
                               use_bias=False)(residual)
            residual = nn.BatchNorm(use_running_average=not training)(residual)

        return nn.relu(out + residual)


class ResNetLayer(nn.Module):
    """ResNet 层：多个 BasicBlock 堆叠"""
    out_ch: int
    num_blocks: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, training: bool = True):
        x = BasicBlock(self.out_ch, stride=self.stride)(x, training)
        for _ in range(1, self.num_blocks):
            x = BasicBlock(self.out_ch, stride=1)(x, training)
        return x


class ResNet18(nn.Module):
    """
    ResNet-18（JAX/Flax 版本）
    输入格式：(B, H, W, C)，JAX 约定 channels-last
    small_input=True 时（mnist/cifar）：第一层用 3x3 conv，不加 maxpool
    """
    num_classes: int = 100
    small_input: bool = True  # h <= 64 时为 True

    @nn.compact
    def __call__(self, x, training: bool = True):
        if self.small_input:
            x = nn.Conv(64, (3, 3), strides=(1, 1), padding="SAME", use_bias=False)(x)
        else:
            x = nn.Conv(64, (7, 7), strides=(2, 2), padding="SAME", use_bias=False)(x)
            x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)

        x = ResNetLayer(64,  num_blocks=2, stride=1)(x, training)
        x = ResNetLayer(128, num_blocks=2, stride=2)(x, training)
        x = ResNetLayer(256, num_blocks=2, stride=2)(x, training)
        x = ResNetLayer(512, num_blocks=2, stride=2)(x, training)

        # 全局平均池化
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        return x


def create_resnet18(num_classes: int, input_size: tuple):
    """
    创建 ResNet-18 模型实例

    Returns:
        model: Flax 模型
        小输入（h<=64）自动使用 small_input=True
    """
    c, h, w = input_size
    small_input = (h <= 64)
    return ResNet18(num_classes=num_classes, small_input=small_input)
