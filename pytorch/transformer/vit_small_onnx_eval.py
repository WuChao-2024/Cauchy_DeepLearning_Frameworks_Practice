# ViT-Small ONNX 模型评测，计算 TOP-1 和 TOP-5
# 用法: python vit_small_onnx_eval.py --dataset cifar100 --onnx ./outputs/.../vit_small_cifar100.onnx

import os
import sys
import argparse
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../shared"))
from data_utils import get_dataloader, get_dataset_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     default="cifar100", choices=["mnist", "cifar100", "imagenet"])
    p.add_argument("--dataset_dir", default="../../dataset")
    p.add_argument("--onnx",        required=True)
    p.add_argument("--batch_size",  type=int, default=128)
    return p.parse_args()


def evaluate_onnx(session, loader):
    top1_correct, top5_correct, total = 0, 0, 0
    latencies = []
    input_name = session.get_inputs()[0].name

    for imgs, labels in loader:
        imgs_np   = imgs.numpy()
        labels_np = labels.numpy()

        t0 = time.perf_counter()
        logits = session.run(None, {input_name: imgs_np})[0]
        latencies.append((time.perf_counter() - t0) * 1000)

        pred1 = logits.argmax(axis=1)
        top1_correct += (pred1 == labels_np).sum()

        top5_idx = np.argsort(logits, axis=1)[:, -5:]
        top5_correct += sum(labels_np[i] in top5_idx[i] for i in range(len(labels_np)))
        total += len(labels_np)

    return top1_correct / total, top5_correct / total, sum(latencies) / len(latencies)


def main():
    args = parse_args()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(args.onnx, sess_options=sess_opts, providers=providers)
    used_provider = session.get_providers()[0]

    val_loader = get_dataloader(args.dataset, "test", args.dataset_dir, args.batch_size)
    top1, top5, avg_ms = evaluate_onnx(session, val_loader)

    print(f"\n{'='*50}")
    print(f"  ViT-Small ONNX 评测")
    print(f"{'='*50}")
    print(f"  ONNX 模型:    {args.onnx}")
    print(f"  数据集:       {args.dataset} (test)")
    print(f"  总样本数:     {len(val_loader.dataset)}")
    print(f"  批次大小:     {args.batch_size}")
    print(f"  推理后端:     {used_provider}")
    print(f"  {'-'*46}")
    print(f"  TOP-1 准确率: {top1*100:.2f}%")
    print(f"  TOP-5 准确率: {top5*100:.2f}%")
    print(f"  平均推理时间: {avg_ms:.2f} ms/batch（ONNX Runtime）")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
