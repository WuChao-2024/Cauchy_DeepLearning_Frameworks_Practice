# JAX/Flax ViT-Small 网络定义（从零实现）
# JAX 版本与 PyTorch 版本的关键区别：
#   1. 使用 nn.compact 装饰器，子模块在 __call__ 中定义
#   2. 参数通过 self.param() 或子模块自动管理
#   3. 输入格式：(B, H, W, C)，channels-last

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


class PatchEmbed(nn.Module):
    """将图像切分为 patch 并线性投影"""
    patch_size: int
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        # 用卷积实现 patch 切分 + 线性投影
        x = nn.Conv(self.embed_dim, (self.patch_size, self.patch_size),
                    strides=(self.patch_size, self.patch_size), use_bias=False)(x)
        # (B, H/P, W/P, embed_dim) -> (B, N, embed_dim)
        x = x.reshape(B, -1, self.embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力（纯 JAX 实现，不使用 flash-attention）"""
    embed_dim: int
    num_heads: int
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        B, N, C = x.shape
        head_dim = self.embed_dim // self.num_heads
        scale    = head_dim ** -0.5

        qkv = nn.Dense(self.embed_dim * 3)(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim)
        qkv = qkv.transpose(0, 2, 3, 1, 4)  # (B, 3, heads, N, head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(self.attn_drop)(attn, deterministic=deterministic)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = nn.Dense(self.embed_dim)(x)
        x = nn.Dropout(self.proj_drop)(x, deterministic=deterministic)
        return x


class MLP(nn.Module):
    """Transformer FFN"""
    hidden_dim: int
    drop: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        in_dim = x.shape[-1]
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.drop)(x, deterministic=deterministic)
        x = nn.Dense(in_dim)(x)
        x = nn.Dropout(self.drop)(x, deterministic=deterministic)
        return x


class TransformerBlock(nn.Module):
    """标准 Transformer 编码器块（Pre-Norm）"""
    embed_dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    drop: float = 0.0
    attn_drop: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        x = x + MultiHeadSelfAttention(
            self.embed_dim, self.num_heads, self.attn_drop, self.drop
        )(nn.LayerNorm()(x), deterministic)
        x = x + MLP(int(self.embed_dim * self.mlp_ratio), self.drop)(
            nn.LayerNorm()(x), deterministic
        )
        return x


class ViTSmall(nn.Module):
    """
    ViT-Small（JAX/Flax 版本）
    embed_dim=384, depth=12, num_heads=6
    输入格式：(B, H, W, C)，channels-last
    """
    num_classes: int = 100
    small_input: bool = True
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    drop: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = True):
        B, H, W, C = x.shape
        patch_size = 4 if self.small_input else 16
        deterministic = not training

        x = PatchEmbed(patch_size, self.embed_dim)(x)
        N = x.shape[1]

        # [CLS] token
        cls_token = self.param('cls_token', nn.initializers.zeros, (1, 1, self.embed_dim))
        cls_token = jnp.broadcast_to(cls_token, (B, 1, self.embed_dim))
        x = jnp.concatenate([cls_token, x], axis=1)

        # 位置编码
        pos_embed = self.param('pos_embed', nn.initializers.normal(0.02),
                               (1, N + 1, self.embed_dim))
        x = nn.Dropout(self.drop)(x + pos_embed, deterministic=deterministic)

        for _ in range(self.depth):
            x = TransformerBlock(self.embed_dim, self.num_heads,
                                 self.mlp_ratio, self.drop)(x, deterministic)

        x = nn.LayerNorm()(x)
        # 取 [CLS] token 做分类
        return nn.Dense(self.num_classes)(x[:, 0])


def create_vit_small(num_classes: int, input_size: tuple):
    c, h, w = input_size
    return ViTSmall(num_classes=num_classes, small_input=(h <= 64))
