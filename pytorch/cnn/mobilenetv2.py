# MobileNetV2 网络定义（从零实现，深度可分离卷积 + 倒残差结构）

import torch
import torch.nn as nn


def _make_divisible(v, divisor=8):
    """确保通道数是 divisor 的倍数"""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU6(nn.Sequential):
    """Conv + BN + ReLU6"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, groups=1):
        padding = (kernel - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """倒残差块：先升维，深度卷积，再降维"""

    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden = int(round(in_ch * expand_ratio))
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_ch, hidden, kernel=1))
        layers += [
            ConvBNReLU6(hidden, hidden, stride=stride, groups=hidden),  # 深度卷积
            nn.Conv2d(hidden, out_ch, 1, bias=False),                   # 逐点卷积
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2
    inverted_residual_setting: [t, c, n, s]
      t: expand_ratio, c: out_channels, n: num_blocks, s: stride
    """

    # [expand_ratio, out_ch, num_blocks, stride]
    _SETTINGS = [
        [1,  16, 1, 1],
        [6,  24, 2, 2],
        [6,  32, 3, 2],
        [6,  64, 4, 2],
        [6,  96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    def __init__(self, num_classes: int = 100, input_size: tuple = (3, 32, 32),
                 width_mult: float = 1.0):
        super().__init__()
        c, h, w = input_size
        small_input = (h <= 64)

        first_ch = _make_divisible(32 * width_mult)
        last_ch  = _make_divisible(1280 * max(1.0, width_mult))

        # 第一层：小输入用 stride=1，大输入用 stride=2
        first_stride = 1 if small_input else 2
        self.features = nn.ModuleList([
            ConvBNReLU6(c, first_ch, stride=first_stride)
        ])

        in_ch = first_ch
        for t, c, n, s in self._SETTINGS:
            out_ch = _make_divisible(c * width_mult)
            # 小输入时压缩 stride，避免特征图过小
            if small_input and s == 2 and h <= 32:
                s = 1
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(in_ch, out_ch, stride, t))
                in_ch = out_ch

        self.features.append(ConvBNReLU6(in_ch, last_ch, kernel=1))
        self.features = nn.Sequential(*self.features)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_ch, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
