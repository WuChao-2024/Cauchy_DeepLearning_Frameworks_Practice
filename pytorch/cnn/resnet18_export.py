# ResNet-18 导出 ONNX
# 用法: python resnet18_export.py --dataset cifar100 --weights ./outputs/.../resnet18_best.pth --output ./outputs/.../resnet18_cifar100.onnx

import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../shared"))
from data_utils import get_dataset_config

from resnet18 import ResNet18


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  default="cifar100", choices=["mnist", "cifar100", "imagenet"])
    p.add_argument("--weights",  required=True, help="权重文件路径 (.pth)")
    p.add_argument("--output",   required=True, help="输出 ONNX 文件路径")
    p.add_argument("--opset",    type=int, default=17)
    return p.parse_args()


def main():
    args = parse_args()

    cfg = get_dataset_config(args.dataset)
    num_classes = cfg["num_classes"]
    input_size  = cfg["input_size"]

    # 加载网络（结构与权重分开）
    model = ResNet18(num_classes=num_classes, input_size=input_size)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    c, h, w = input_size
    dummy = torch.randn(1, c, h, w)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    torch.onnx.export(
        model, dummy, args.output,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"ONNX 导出成功: {args.output}")
    print(f"  数据集: {args.dataset}  输入尺寸: {(1, c, h, w)}  opset: {args.opset}")


if __name__ == "__main__":
    main()
