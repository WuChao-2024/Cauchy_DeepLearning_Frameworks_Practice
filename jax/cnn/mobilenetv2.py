# JAX/Flax MobileNetV2 网络定义（从零实现）
# 深度可分离卷积 + 倒残差结构

import jax.numpy as jnp
import flax.linen as nn


def _make_divisible(v, divisor=8):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU6(nn.Module):
    """Conv + BN + ReLU6"""
    out_ch: int
    kernel: int = 3
    stride: int = 1
    groups: int = 1  # groups>1 时为深度卷积（feature_group_count）

    @nn.compact
    def __call__(self, x, training: bool = True):
        padding = (self.kernel - 1) // 2
        x = nn.Conv(
            self.out_ch,
            (self.kernel, self.kernel),
            strides=(self.stride, self.stride),
            padding=padding,
            feature_group_count=self.groups,
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        return jnp.clip(nn.relu(x), 0, 6)  # ReLU6


class InvertedResidual(nn.Module):
    """倒残差块：先升维，深度卷积，再降维"""
    in_ch: int
    out_ch: int
    stride: int
    expand_ratio: int

    @nn.compact
    def __call__(self, x, training: bool = True):
        hidden = int(round(self.in_ch * self.expand_ratio))
        use_res = (self.stride == 1 and self.in_ch == self.out_ch)

        out = x
        if self.expand_ratio != 1:
            out = ConvBNReLU6(hidden, kernel=1)(out, training)
        # 深度卷积（groups=hidden）
        out = ConvBNReLU6(hidden, kernel=3, stride=self.stride, groups=hidden)(out, training)
        # 逐点卷积（无激活）
        out = nn.Conv(self.out_ch, (1, 1), use_bias=False)(out)
        out = nn.BatchNorm(use_running_average=not training)(out)

        if use_res:
            return x + out
        return out


class MobileNetV2(nn.Module):
    """
    MobileNetV2（JAX/Flax 版本）
    输入格式：(B, H, W, C)，channels-last
    """
    num_classes: int = 100
    small_input: bool = True
    width_mult: float = 1.0

    # [expand_ratio, out_ch, num_blocks, stride]
    _SETTINGS = [
        (1,  16, 1, 1),
        (6,  24, 2, 2),
        (6,  32, 3, 2),
        (6,  64, 4, 2),
        (6,  96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    @nn.compact
    def __call__(self, x, training: bool = True):
        first_ch = _make_divisible(32 * self.width_mult)
        last_ch  = _make_divisible(1280 * max(1.0, self.width_mult))

        first_stride = 1 if self.small_input else 2
        x = ConvBNReLU6(first_ch, stride=first_stride)(x, training)

        in_ch = first_ch
        for t, c, n, s in self._SETTINGS:
            out_ch = _make_divisible(c * self.width_mult)
            # 小输入时压缩 stride
            if self.small_input and s == 2 and x.shape[1] <= 32:
                s = 1
            for i in range(n):
                stride = s if i == 0 else 1
                x = InvertedResidual(in_ch, out_ch, stride, t)(x, training)
                in_ch = out_ch

        x = ConvBNReLU6(last_ch, kernel=1)(x, training)
        # 全局平均池化
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dropout(0.2)(x, deterministic=not training)
        x = nn.Dense(self.num_classes)(x)
        return x


def create_mobilenetv2(num_classes: int, input_size: tuple):
    c, h, w = input_size
    return MobileNetV2(num_classes=num_classes, small_input=(h <= 64))
