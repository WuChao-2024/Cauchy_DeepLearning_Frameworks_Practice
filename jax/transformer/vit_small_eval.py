# JAX/Flax ViT-Small 精度评测，计算 TOP-1 和 TOP-5
# 用法: python vit_small_eval.py --dataset cifar100 --ckpt_dir ./outputs/.../checkpoints/best

import os
import sys
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../shared"))
from data_utils import get_numpy_iterator, get_dataset_config

from vit_small import create_vit_small
from vit_small_train import TrainState, create_train_state


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     default="cifar100", choices=["mnist", "cifar100", "imagenet"])
    p.add_argument("--dataset_dir", default="../../dataset")
    p.add_argument("--ckpt_dir",    required=True)
    p.add_argument("--batch_size",  type=int, default=128)
    return p.parse_args()


@jax.jit
def eval_step(state, images):
    logits = state.apply_fn({'params': state.params}, images, training=False)
    return logits


def main():
    args = parse_args()

    cfg = get_dataset_config(args.dataset)
    num_classes = cfg["num_classes"]
    input_size  = cfg["input_size"]

    model = create_vit_small(num_classes, input_size)
    rng   = jax.random.PRNGKey(0)
    state = create_train_state(model, rng, input_size, lr=1e-3, num_steps=1, warmup_steps=1)

    checkpointer = ocp.StandardCheckpointer()
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    state = checkpointer.restore(ckpt_dir, target=state)

    val_iter_fn, _ = get_numpy_iterator(args.dataset, "test", args.dataset_dir, args.batch_size)

    top1_correct, top5_correct, total = 0, 0, 0
    latencies = []

    for images, labels in val_iter_fn():
        images_jax = jnp.array(images)
        labels_np  = labels

        t0 = time.perf_counter()
        logits = eval_step(state, images_jax)
        logits.block_until_ready()
        latencies.append((time.perf_counter() - t0) * 1000)

        logits_np = np.array(logits)
        pred1 = logits_np.argmax(axis=1)
        top1_correct += (pred1 == labels_np).sum()

        top5_idx = np.argsort(logits_np, axis=1)[:, -5:]
        top5_correct += sum(labels_np[i] in top5_idx[i] for i in range(len(labels_np)))
        total += len(labels_np)

    top1 = top1_correct / total
    top5 = top5_correct / total
    avg_ms = sum(latencies) / len(latencies)

    print(f"\n{'='*50}")
    print(f"  ViT-Small 框架评测（JAX）")
    print(f"{'='*50}")
    print(f"  数据集:       {args.dataset} (test)")
    print(f"  权重目录:     {args.ckpt_dir}")
    print(f"  总样本数:     {total}")
    print(f"  批次大小:     {args.batch_size}")
    print(f"  JAX 设备:     {jax.devices()[0].device_kind}")
    print(f"  {'-'*46}")
    print(f"  TOP-1 准确率: {top1*100:.2f}%")
    print(f"  TOP-5 准确率: {top5*100:.2f}%")
    print(f"  平均推理时间: {avg_ms:.2f} ms/batch")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
