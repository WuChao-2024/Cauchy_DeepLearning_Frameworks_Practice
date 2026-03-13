# shared/model_utils.py
# 模型统计工具：参数量、FLOPs、子模块分析

import numpy as np
from typing import Optional


# ============================================================
# PyTorch 模型统计
# ============================================================

def count_params(model, module_name: Optional[str] = None) -> int:
    """
    统计 PyTorch 模型（或指定子模块）的参数量

    Args:
        model: PyTorch nn.Module
        module_name: 子模块名称，如 "layer1"、"fc"，None 表示整个模型
    Returns:
        参数总数
    """
    import torch.nn as nn
    target = model
    if module_name:
        for part in module_name.split("."):
            target = getattr(target, part)
    total = sum(p.numel() for p in target.parameters())
    trainable = sum(p.numel() for p in target.parameters() if p.requires_grad)
    return total, trainable


def count_flops(model, input_size: tuple, module_name: Optional[str] = None) -> tuple:
    """
    统计 PyTorch 模型（或指定子模块）的 FLOPs

    Args:
        model: PyTorch nn.Module
        input_size: 输入尺寸，如 (1, 3, 224, 224)
        module_name: 子模块名称，None 表示整个模型
    Returns:
        (flops, params) 元组
    """
    try:
        import copy
        from thop import profile
        from thop.utils import clever_format
        import torch
        target = model
        if module_name:
            for part in module_name.split("."):
                target = getattr(target, part)
        # 用深拷贝，避免 thop 在原模型上注册 total_ops/total_params buffer
        target_copy = copy.deepcopy(target)
        device = next(target_copy.parameters()).device
        dummy_input = torch.randn(*input_size, device=device)
        flops, params = profile(target_copy, inputs=(dummy_input,), verbose=False)
        del target_copy
        return int(flops), int(params)
    except (ImportError, StopIteration):
        return 0, 0


def _format_num(n: int) -> str:
    """格式化大数字，如 1234567 -> 1.23M"""
    if n >= 1e9:
        return f"{n/1e9:.2f}G"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def print_model_summary(model, input_size: tuple, title: str = "模型统计"):
    """
    打印 PyTorch 模型统计表格

    Args:
        model: PyTorch nn.Module
        input_size: 输入尺寸，如 (1, 3, 224, 224)
        title: 标题
    """
    import torch.nn as nn

    total_params, trainable_params = count_params(model)
    flops, _ = count_flops(model, input_size)

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'总参数量':<20} {_format_num(total_params):>15} ({total_params:,})")
    print(f"  {'可训练参数量':<20} {_format_num(trainable_params):>15} ({trainable_params:,})")
    if flops > 0:
        print(f"  {'FLOPs':<20} {_format_num(flops):>15} ({flops:,})")
    print(f"{'='*60}")

    # 打印各子模块参数量
    print(f"\n  子模块参数量：")
    print(f"  {'-'*55}")
    print(f"  {'模块名称':<30} {'参数量':>12} {'占比':>8}")
    print(f"  {'-'*55}")
    for name, module in model.named_children():
        p, _ = count_params(module)
        ratio = p / total_params * 100 if total_params > 0 else 0
        print(f"  {name:<30} {_format_num(p):>12} {ratio:>7.1f}%")
    print(f"  {'-'*55}\n")


def print_submodule_stats(model, module_name: str, input_size: tuple):
    """
    打印指定子模块的参数量和 FLOPs

    Args:
        model: PyTorch nn.Module
        module_name: 子模块名称，如 "layer1"
        input_size: 该子模块的输入尺寸
    """
    total, trainable = count_params(model, module_name)
    flops, _ = count_flops(model, input_size, module_name)

    print(f"\n  子模块 [{module_name}] 统计：")
    print(f"  {'参数量':<20} {_format_num(total):>12} ({total:,})")
    print(f"  {'可训练参数量':<20} {_format_num(trainable):>12} ({trainable:,})")
    if flops > 0:
        print(f"  {'FLOPs':<20} {_format_num(flops):>12} ({flops:,})")


# ============================================================
# JAX 模型统计
# ============================================================

def count_jax_params(params, module_path: Optional[str] = None) -> int:
    """
    统计 JAX/Flax 模型参数量

    Args:
        params: Flax 模型参数字典（pytree）
        module_path: 子模块路径，如 "ResBlock_0"，None 表示整个模型
    Returns:
        参数总数
    """
    import jax

    target = params
    if module_path:
        for key in module_path.split("."):
            target = target[key]

    leaves = jax.tree_util.tree_leaves(target)
    return sum(np.prod(leaf.shape) for leaf in leaves)


def print_jax_model_summary(params, title: str = "JAX 模型统计"):
    """
    打印 JAX/Flax 模型参数统计

    Args:
        params: Flax 模型参数字典
        title: 标题
    """
    import jax

    total = count_jax_params(params)

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'总参数量':<20} {_format_num(total):>15} ({total:,})")
    print(f"{'='*60}")

    # 打印顶层子模块
    if isinstance(params, dict):
        print(f"\n  子模块参数量：")
        print(f"  {'-'*55}")
        print(f"  {'模块名称':<30} {'参数量':>12} {'占比':>8}")
        print(f"  {'-'*55}")
        for key in params:
            p = count_jax_params(params[key])
            ratio = p / total * 100 if total > 0 else 0
            print(f"  {key:<30} {_format_num(p):>12} {ratio:>7.1f}%")
        print(f"  {'-'*55}\n")


def print_jax_submodule_stats(params, module_path: str):
    """
    打印 JAX 指定子模块的参数量

    Args:
        params: Flax 模型参数字典
        module_path: 子模块路径，如 "ResBlock_0"
    """
    p = count_jax_params(params, module_path)
    print(f"\n  JAX 子模块 [{module_path}] 参数量：{_format_num(p)} ({p:,})")
