#!/usr/bin/env python3
"""
Download datasets from HuggingFace with resume support.
Datasets: MNIST, CIFAR100, ImageNet
"""

import os
import argparse
from huggingface_hub import snapshot_download


DATASETS = {
    "mnist": {
        "repo_id": "ylecun/mnist",
        "description": "MNIST handwritten digits",
    },
    "cifar100": {
        "repo_id": "tanganke/cifar100",
        "description": "CIFAR-100",
    },
    "imagenet": {
        "repo_id": "ILSVRC/imagenet-1k",
        "description": "ImageNet-1k (requires access approval)",
    },
}


def download_dataset(name: str, local_dir: str, token: str | None = None):
    info = DATASETS[name]
    dest = os.path.join(local_dir, name)
    print(f"Downloading {info['description']} -> {dest}")

    snapshot_download(
        repo_id=info["repo_id"],
        repo_type="dataset",
        local_dir=dest,
        token=token,
        resume_download=True,      # resume partial downloads
        max_workers=4,
    )
    print(f"Done: {name}")


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace datasets")
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which datasets to download (default: all)",
    )
    parser.add_argument(
        "--dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Root directory to save datasets (default: same dir as this script)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace access token (or set HF_TOKEN env var). Required for ImageNet.",
    )
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if "all" in args.datasets else args.datasets

    if "imagenet" in targets and not args.token:
        print(
            "Warning: ImageNet requires a HuggingFace token and dataset access approval.\n"
            "Set HF_TOKEN env var or pass --token <your_token>."
        )

    for name in targets:
        download_dataset(name, args.dir, args.token)


if __name__ == "__main__":
    main()
