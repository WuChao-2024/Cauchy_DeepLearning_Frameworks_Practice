# MobileNetV2 精度评测，计算 TOP-1 和 TOP-5
# 用法: python mobilenetv2_eval.py --dataset cifar100 --weights ./outputs/mobilenetv2_torch_cifar100_float32_xxx/mobilenetv2_best.pth

import os
import sys
import argparse
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../shared"))
from data_utils import get_dataloader, get_dataset_config
from model_utils import print_model_summary

from mobilenetv2 import MobileNetV2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     default="cifar100", choices=["mnist", "cifar100", "imagenet"])
    p.add_argument("--dataset_dir", default="../../dataset")
    p.add_argument("--weights",     required=True, help="权重文件路径 (.pth)")
    p.add_argument("--batch_size",  type=int, default=128)
    return p.parse_args()


@torch.no_grad()
def evaluate_topk(model, loader, device):
    model.eval()
    top1_correct, top5_correct, total = 0, 0, 0
    latencies = []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        t0 = time.perf_counter()
        logits = model(imgs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

        pred1 = logits.argmax(dim=1)
        top1_correct += pred1.eq(labels).sum().item()

        _, pred5 = logits.topk(5, dim=1)
        top5_correct += pred5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

        total += imgs.size(0)

    return top1_correct / total, top5_correct / total, sum(latencies) / len(latencies)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = get_dataset_config(args.dataset)
    num_classes = cfg["num_classes"]
    input_size  = cfg["input_size"]

    model = MobileNetV2(num_classes=num_classes, input_size=input_size)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)

    val_loader = get_dataloader(args.dataset, "test", args.dataset_dir, args.batch_size)

    top1, top5, avg_ms = evaluate_topk(model, val_loader, device)

    print(f"\n{'='*50}")
    print(f"  MobileNetV2 框架评测（PyTorch）")
    print(f"{'='*50}")
    print(f"  数据集:       {args.dataset} (test)")
    print(f"  权重文件:     {args.weights}")
    print(f"  总样本数:     {len(val_loader.dataset)}")
    print(f"  批次大小:     {args.batch_size}")
    print(f"  设备:         {device}")
    print(f"  {'-'*46}")
    print(f"  TOP-1 准确率: {top1*100:.2f}%")
    print(f"  TOP-5 准确率: {top5*100:.2f}%")
    print(f"  平均推理时间: {avg_ms:.2f} ms/batch")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
