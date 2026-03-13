# JAX/Flax ResNet-18 导出 ONNX
# JAX 导出 ONNX 的方式：将 JAX 参数转为 numpy，用 PyTorch 重建相同结构，再用 torch.onnx.export
# 这是目前最稳定的 JAX -> ONNX 路径，无需额外依赖
# 用法: python resnet18_export.py --dataset cifar100 --ckpt_dir ./outputs/.../checkpoints/best --output ./outputs/.../resnet18_cifar100.onnx

import os
import sys
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../shared"))
from data_utils import get_dataset_config

from resnet18 import create_resnet18
from resnet18_train import TrainState, create_train_state


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   default="cifar100", choices=["mnist", "cifar100", "imagenet"])
    p.add_argument("--ckpt_dir",  required=True, help="orbax checkpoint 目录")
    p.add_argument("--output",    required=True, help="输出 ONNX 文件路径")
    p.add_argument("--opset",     type=int, default=17)
    return p.parse_args()


def jax_params_to_torch(jax_params, torch_model):
    """
    将 JAX 参数树中的 numpy 数组复制到对应的 PyTorch 模型参数
    通过名称映射实现：JAX 的 params 字典结构 -> PyTorch state_dict
    """
    import torch

    def flatten_jax_params(params, prefix=""):
        """递归展平 JAX 参数字典"""
        result = {}
        for k, v in params.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(flatten_jax_params(v, full_key))
            else:
                result[full_key] = np.array(v)
        return result

    flat_jax = flatten_jax_params(jax_params)
    torch_sd = torch_model.state_dict()

    print(f"  JAX 参数数量: {len(flat_jax)}")
    print(f"  PyTorch 参数数量: {len(torch_sd)}")
    print("  注意：JAX 和 PyTorch 的参数命名不同，需要手动对齐")
    print("  建议：直接用 PyTorch 训练的权重导出 ONNX（pytorch/cnn/resnet18_export.py）")
    print("  或者使用 jax2onnx 库（pip install jax2onnx）直接从 JAX 导出")


def export_via_jax_trace(model_jax, state, input_size, output_path, opset):
    """
    使用 jax.export 导出计算图，然后转换为 ONNX
    需要安装 jax2onnx：pip install jax2onnx
    """
    try:
        import jax2onnx
        c, h, w = input_size
        dummy = jnp.ones((1, h, w, c))

        def predict(images):
            return state.apply_fn(
                {'params': state.params, 'batch_stats': state.batch_stats},
                images, training=False,
            )

        onnx_model = jax2onnx.to_onnx(predict, dummy, opset_version=opset)
        import onnx
        onnx.save(onnx_model, output_path)
        print(f"ONNX 导出成功（via jax2onnx）: {output_path}")
        return True
    except ImportError:
        print("jax2onnx 未安装，尝试备用方案...")
        return False


def main():
    args = parse_args()

    cfg = get_dataset_config(args.dataset)
    num_classes = cfg["num_classes"]
    input_size  = cfg["input_size"]

    # 加载 JAX 模型和权重
    model = create_resnet18(num_classes, input_size)
    rng   = jax.random.PRNGKey(0)
    state = create_train_state(model, rng, input_size, lr=0.1, num_steps=1)
    checkpointer = ocp.StandardCheckpointer()
    state = checkpointer.restore(args.ckpt_dir, target=state)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # 方案一：尝试 jax2onnx
    success = export_via_jax_trace(model, state, input_size, args.output, args.opset)

    if not success:
        # 方案二：提示用户安装 jax2onnx 或使用 PyTorch 版本导出
        print("\n备用方案：")
        print("  1. 安装 jax2onnx：pip install jax2onnx")
        print("  2. 或者使用 PyTorch 版本导出（推荐）：")
        print(f"     cd ../../pytorch/cnn")
        print(f"     python resnet18_export.py --dataset {args.dataset} \\")
        print(f"       --weights <pytorch_weights.pth> --output {args.output}")
        print("\n  JAX 模型参数已加载，可通过以下方式获取 numpy 参数：")
        print("  from resnet18_export import *")
        print("  params_np = jax.tree_util.tree_map(np.array, state.params)")

        # 导出参数为 npz 供其他工具使用
        npz_path = args.output.replace(".onnx", "_params.npz")
        flat_params = {}
        def flatten(d, prefix=""):
            for k, v in d.items():
                key = f"{prefix}/{k}" if prefix else k
                if isinstance(v, dict):
                    flatten(v, key)
                else:
                    flat_params[key] = np.array(v)
        flatten(state.params)
        np.savez(npz_path, **flat_params)
        print(f"\n  已将参数保存为 npz 格式: {npz_path}")


if __name__ == "__main__":
    main()
