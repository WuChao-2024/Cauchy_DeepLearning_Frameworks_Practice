# JAX/Flax DeiT-Tiny 网络定义（从零实现）
# 相比 ViT 增加了 dist_token，推理时取 cls+dist 平均

import jax
import jax.numpy as jnp
import flax.linen as nn

from vit_small import PatchEmbed, TransformerBlock


class DeiTTiny(nn.Module):
    """
    DeiT-Tiny（JAX/Flax 版本）
    embed_dim=192, depth=12, num_heads=3
    包含 cls_token 和 dist_token
    """
    num_classes: int = 100
    small_input: bool = True
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    mlp_ratio: float = 4.0
    drop: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = True):
        B, H, W, C = x.shape
        patch_size = 4 if self.small_input else 16
        deterministic = not training

        x = PatchEmbed(patch_size, self.embed_dim)(x)
        N = x.shape[1]

        cls_token  = self.param('cls_token',  nn.initializers.zeros, (1, 1, self.embed_dim))
        dist_token = self.param('dist_token', nn.initializers.zeros, (1, 1, self.embed_dim))
        cls_token  = jnp.broadcast_to(cls_token,  (B, 1, self.embed_dim))
        dist_token = jnp.broadcast_to(dist_token, (B, 1, self.embed_dim))

        # [CLS, DIST, patches]
        x = jnp.concatenate([cls_token, dist_token, x], axis=1)

        pos_embed = self.param('pos_embed', nn.initializers.normal(0.02),
                               (1, N + 2, self.embed_dim))
        x = nn.Dropout(self.drop)(x + pos_embed, deterministic=deterministic)

        for _ in range(self.depth):
            x = TransformerBlock(self.embed_dim, self.num_heads,
                                 self.mlp_ratio, self.drop)(x, deterministic)

        x = nn.LayerNorm()(x)

        cls_out  = nn.Dense(self.num_classes, name='head')(x[:, 0])
        dist_out = nn.Dense(self.num_classes, name='head_dist')(x[:, 1])

        if training:
            return cls_out, dist_out
        else:
            return (cls_out + dist_out) / 2


def create_deit_tiny(num_classes: int, input_size: tuple):
    c, h, w = input_size
    return DeiTTiny(num_classes=num_classes, small_input=(h <= 64))
