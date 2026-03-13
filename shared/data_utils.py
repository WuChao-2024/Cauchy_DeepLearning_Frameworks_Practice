# shared/data_utils.py
# 数据集加载工具，支持 MNIST / CIFAR100 / ImageNet
# 数据来源：HuggingFace parquet 格式

import os
import glob
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# 各数据集配置
DATASET_CONFIG = {
    "mnist": {
        "num_classes": 10,
        "input_size": (1, 28, 28),
        "train_files": "mnist/mnist/train-*.parquet",
        "test_files": "mnist/mnist/test-*.parquet",
        "image_field": "image",
        "label_field": "label",
    },
    "cifar100": {
        "num_classes": 100,
        "input_size": (3, 32, 32),
        "train_files": "cifar100/data/train-*.parquet",
        "test_files": "cifar100/data/test-*.parquet",
        "image_field": "image",
        "label_field": "label",
    },
    "imagenet": {
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "train_files": "imagenet/data/train-*.parquet",
        "test_files": "imagenet/data/test-*.parquet",
        "image_field": "image",
        "label_field": "label",
    },
}


def get_dataset_config(dataset_name: str) -> dict:
    """获取数据集配置"""
    assert dataset_name in DATASET_CONFIG, f"不支持的数据集: {dataset_name}，可选: {list(DATASET_CONFIG.keys())}"
    return DATASET_CONFIG[dataset_name]


class ParquetDataset(Dataset):
    """从 HuggingFace parquet 文件加载数据集"""

    def __init__(self, parquet_files: list[str], image_field: str, label_field: str,
                 transform=None, dataset_name: str = "cifar100"):
        self.image_field = image_field
        self.label_field = label_field
        self.transform = transform
        self.dataset_name = dataset_name

        # 读取所有 parquet 文件
        tables = [pq.read_table(f) for f in sorted(parquet_files)]
        import pyarrow as pa
        self.table = pa.concat_tables(tables)
        self.length = len(self.table)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.table.slice(idx, 1)

        # 读取图像
        img_data = row[self.image_field][0].as_py()
        if isinstance(img_data, dict):
            # HuggingFace Image 格式：{"bytes": ..., "path": ...}
            img_bytes = img_data.get("bytes")
            if img_bytes:
                import io
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB" if self.dataset_name != "mnist" else "L")
            else:
                raise ValueError(f"无法读取图像数据: {img_data}")
        elif isinstance(img_data, bytes):
            import io
            img = Image.open(io.BytesIO(img_data)).convert("RGB" if self.dataset_name != "mnist" else "L")
        else:
            raise ValueError(f"未知图像格式: {type(img_data)}")

        label = row[self.label_field][0].as_py()

        if self.transform:
            img = self.transform(img)

        return img, label


def _get_transforms(dataset_name: str, is_train: bool):
    """获取数据预处理变换"""
    cfg = DATASET_CONFIG[dataset_name]
    c, h, w = cfg["input_size"]

    if dataset_name == "mnist":
        if is_train:
            return T.Compose([
                T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),
            ])
        else:
            return T.Compose([
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),
            ])

    elif dataset_name == "cifar100":
        if is_train:
            return T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        else:
            return T.Compose([
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

    elif dataset_name == "imagenet":
        if is_train:
            return T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            return T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])


def get_dataloader(dataset_name: str, split: str, dataset_dir: str,
                   batch_size: int = 128, num_workers: int = 4,
                   distributed: bool = False) -> DataLoader:
    """
    获取 PyTorch DataLoader

    Args:
        dataset_name: mnist / cifar100 / imagenet
        split: train / test
        dataset_dir: 数据集根目录（包含 mnist/, cifar100/, imagenet/ 子目录）
        batch_size: 批次大小
        num_workers: 数据加载线程数
        distributed: 是否使用分布式采样（DDP 训练时设为 True）
    """
    cfg = DATASET_CONFIG[dataset_name]
    is_train = (split == "train")

    file_pattern = cfg["train_files"] if is_train else cfg["test_files"]
    files = sorted(glob.glob(os.path.join(dataset_dir, file_pattern)))
    assert len(files) > 0, f"未找到数据文件: {os.path.join(dataset_dir, file_pattern)}"

    transform = _get_transforms(dataset_name, is_train)
    dataset = ParquetDataset(
        files, cfg["image_field"], cfg["label_field"],
        transform=transform, dataset_name=dataset_name
    )

    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=is_train)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(is_train and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )


def get_numpy_iterator(dataset_name: str, split: str, dataset_dir: str,
                       batch_size: int = 128):
    """
    获取 numpy 迭代器（用于 JAX 训练）
    每次迭代返回 (images: np.ndarray, labels: np.ndarray)
    images shape: (batch, H, W, C)，JAX 约定 channels-last
    """
    # 复用 PyTorch DataLoader，然后转为 numpy
    loader = get_dataloader(dataset_name, split, dataset_dir, batch_size, num_workers=4)

    def _iterator():
        for images, labels in loader:
            # PyTorch: (B, C, H, W) -> JAX: (B, H, W, C)
            imgs_np = images.numpy().transpose(0, 2, 3, 1)
            lbls_np = labels.numpy()
            yield imgs_np, lbls_np

    return _iterator, len(loader)
