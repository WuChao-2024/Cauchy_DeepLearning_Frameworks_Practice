# MobileNetV2 导出 ONNX
# 用法: python mobilenetv2_export.py --dataset cifar100 --weights ./outputs/.../mobilenetv2_best.pth --output ./outputs/.../mobilenetv2_cifar100.onnx

import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../shared"))
from data_utils import get_dataset_config

from mobilenetv2 import MobileNetV2


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

    model = MobileNetV2(num_classes=num_classes, input_size=input_size)
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
