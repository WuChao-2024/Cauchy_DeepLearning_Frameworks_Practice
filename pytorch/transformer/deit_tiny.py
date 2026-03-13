# DeiT-Tiny 网络定义（从零实现，Data-efficient Image Transformer）
# 参考：Training data-efficient image transformers (Touvron et al., 2021)
# 相比 ViT 增加了蒸馏 token（distillation token），推理时取 cls+dist 平均

import torch
import torch.nn as nn

# 复用 ViT 的基础组件
from vit_small import PatchEmbed, MultiHeadSelfAttention, MLP, TransformerBlock


class DeiTTiny(nn.Module):
    """
    DeiT-Tiny
    embed_dim=192, depth=12, num_heads=3, mlp_ratio=4
    包含 cls_token 和 dist_token，推理时取两者 logits 的平均
    """

    def __init__(self, num_classes: int = 100, input_size: tuple = (3, 32, 32),
                 embed_dim: int = 192, depth: int = 12, num_heads: int = 3,
                 mlp_ratio: float = 4.0, drop: float = 0.1):
        super().__init__()
        c, h, w = input_size
        patch_size = 4 if h <= 64 else 16
        assert h % patch_size == 0

        self.patch_embed = PatchEmbed(h, patch_size, c, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 蒸馏 token
        # +2：cls_token + dist_token
        self.pos_embed  = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        self.pos_drop   = nn.Dropout(drop)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 两个分类头：cls head 和 distillation head
        self.head      = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)
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

        cls  = self.cls_token.expand(B, -1, -1)
        dist = self.dist_token.expand(B, -1, -1)
        x    = torch.cat([cls, dist, x], dim=1)
        x    = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)

        cls_out  = self.head(x[:, 0])
        dist_out = self.head_dist(x[:, 1])

        if self.training:
            # 训练时返回两个 logits（可用于蒸馏损失）
            return cls_out, dist_out
        else:
            # 推理时取平均
            return (cls_out + dist_out) / 2
