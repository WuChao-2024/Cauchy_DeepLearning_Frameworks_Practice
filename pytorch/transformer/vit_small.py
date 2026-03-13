# ViT-Small 网络定义（从零实现，Vision Transformer）
# 参考：An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)
# patch_size 根据输入尺寸自动调整：mnist/cifar100 用 4，imagenet 用 16

import torch
import torch.nn as nn
import math


class PatchEmbed(nn.Module):
    """将图像切分为 patch 并线性投影"""

    def __init__(self, img_size, patch_size, in_ch, embed_dim):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # 用 stride=patch_size 的卷积实现 patch 切分 + 线性投影
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, N, embed_dim)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力（不使用 flash-attention，纯 PyTorch 实现）"""

    def __init__(self, embed_dim, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # 计算 Q K V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # 缩放点积注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class MLP(nn.Module):
    """Transformer FFN：两层全连接 + GELU"""

    def __init__(self, in_dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, hidden_dim)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_dim, in_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    """标准 Transformer 编码器块：LN -> MHSA -> LN -> MLP（Pre-Norm）"""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = MLP(embed_dim, int(embed_dim * mlp_ratio), drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTSmall(nn.Module):
    """
    ViT-Small
    embed_dim=384, depth=12, num_heads=6, mlp_ratio=4
    patch_size 根据输入自动选择：h<=64 用 4，否则用 16
    """

    def __init__(self, num_classes: int = 100, input_size: tuple = (3, 32, 32),
                 embed_dim: int = 384, depth: int = 12, num_heads: int = 6,
                 mlp_ratio: float = 4.0, drop: float = 0.1):
        super().__init__()
        c, h, w = input_size
        patch_size = 4 if h <= 64 else 16
        assert h % patch_size == 0, f"输入尺寸 {h} 不能被 patch_size {patch_size} 整除"

        self.patch_embed = PatchEmbed(h, patch_size, c, embed_dim)
        num_patches = self.patch_embed.num_patches

        # [CLS] token + 位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(drop)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)

        # 拼接 [CLS] token
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)

        # 取 [CLS] token 做分类
        return self.head(x[:, 0])
