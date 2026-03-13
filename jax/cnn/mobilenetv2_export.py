# JAX/Flax MobileNetV2 导出 ONNX
# 用法: python mobilenetv2_export.py --dataset cifar100 --ckpt_dir ./outputs/.../checkpoints/best --output ./outputs/.../mobilenetv2_cifar100.onnx

import os
import sys
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../shared"))
from data_utils import get_dataset_config

from mobilenetv2 import create_mobilenetv2
from mobilenetv2_train import TrainState, create_train_state


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   default="cifar100", choices=["mnist", "cifar100", "imagenet"])
    p.add_argument("--ckpt_dir",  required=True)
    p.add_argument("--output",    required=True)
    p.add_argument("--opset",     type=int, default=17)
    return p.parse_args()


def main():
    args = parse_args()

    cfg = get_dataset_config(args.dataset)
    num_classes = cfg["num_classes"]
    input_size  = cfg["input_size"]

    model = create_mobilenetv2(num_classes, input_size)
    rng   = jax.random.PRNGKey(0)
    state = create_train_state(model, rng, input_size, lr=0.01, num_steps=1)
    checkpointer = ocp.StandardCheckpointer()
    state = checkpointer.restore(args.ckpt_dir, target=state)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # 尝试 jax2onnx
    try:
        import jax2onnx
        import onnx
        c, h, w = input_size
        dummy = jnp.ones((1, h, w, c))

        def predict(images):
            return state.apply_fn(
                {'params': state.params, 'batch_stats': state.batch_stats},
                images, training=False,
            )

        onnx_model = jax2onnx.to_onnx(predict, dummy, opset_version=args.opset)
        onnx.save(onnx_model, args.output)
        print(f"ONNX 导出成功（via jax2onnx）: {args.output}")

    except ImportError:
        print("jax2onnx 未安装，导出参数为 npz 格式")
        print("安装方式：pip install jax2onnx")
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
        print(f"参数已保存: {npz_path}")


if __name__ == "__main__":
    main()
