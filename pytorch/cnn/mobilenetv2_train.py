# MobileNetV2 训练入口，支持单卡和 DDP 多卡训练，TensorBoard 日志
# 单卡: python mobilenetv2_train.py --dataset cifar100
# 多卡: torchrun --nproc_per_node=2 mobilenetv2_train.py --dataset cifar100 --ddp
# 精度对比: python mobilenetv2_train.py --dataset cifar100 --dtype bfloat16

import os
import sys
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../shared"))
from data_utils import get_dataloader, get_dataset_config
from model_utils import print_model_summary

from mobilenetv2 import MobileNetV2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     default="cifar100", choices=["mnist", "cifar100", "imagenet"])
    p.add_argument("--dataset_dir", default="../../dataset")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=0.01)
    p.add_argument("--save_freq",   type=int,   default=10, help="每隔多少 epoch 保存一次权重")
    p.add_argument("--dtype",       default="float32", choices=["float32", "bfloat16"],
                   help="训练精度，bfloat16 启用 autocast 混合精度")
    p.add_argument("--ddp",         action="store_true", help="启用 DDP 多卡训练")
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, writer, is_main, use_amp):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        # autocast 混合精度前向
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)

        # GradScaler 仅在 float16 时有意义，bfloat16 不需要 scale
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += logits.detach().argmax(1).eq(labels).sum().item()
        total += imgs.size(0)

        if is_main and step % 100 == 0:
            print(f"  step {step}/{len(loader)}  loss={loss.item():.4f}")

    avg_loss = total_loss / total
    acc = correct / total
    if is_main and writer:
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/acc",  acc,      epoch)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, writer, is_main, use_amp):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += imgs.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    if is_main and writer:
        writer.add_scalar("val/loss", avg_loss, epoch)
        writer.add_scalar("val/acc",  acc,      epoch)
    return avg_loss, acc


def main():
    args = parse_args()
    use_amp = (args.dtype == "bfloat16")

    # ── DDP 初始化 ──────────────────────────────────────────────
    # 单卡：直接运行；多卡：torchrun 自动设置 LOCAL_RANK
    if args.ddp:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        is_main = (local_rank == 0)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    # ── 训练产物目录（格式：mobilenetv2_torch_train_YYYYMMDD_HHMMSS）──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"mobilenetv2_torch_{args.dataset}_{args.dtype}_{timestamp}"
    run_dir   = os.path.join("./outputs", run_name)
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        # 保存训练配置
        with open(os.path.join(run_dir, "config.txt"), "w") as f:
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")

    # ── 数据集 ──────────────────────────────────────────────────
    cfg = get_dataset_config(args.dataset)
    num_classes = cfg["num_classes"]
    input_size  = cfg["input_size"]

    train_loader = get_dataloader(args.dataset, "train", args.dataset_dir,
                                  args.batch_size, distributed=args.ddp)
    val_loader   = get_dataloader(args.dataset, "test",  args.dataset_dir,
                                  args.batch_size)

    # ── 模型 ────────────────────────────────────────────────────
    model = MobileNetV2(num_classes=num_classes, input_size=input_size).to(device)

    if args.ddp:
        # DDP 包装：梯度自动在所有卡间同步
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    if is_main:
        print_model_summary(model.module if args.ddp else model,
                            (1,) + input_size, title="MobileNetV2")
        print(f"训练精度: {args.dtype}  {'(autocast 混合精度)' if use_amp else '(全精度 float32)'}")

    # ── 优化器 & 调度器 ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=4e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # bfloat16 不需要 GradScaler（数值范围足够），float16 才需要
    scaler = None

    # ── TensorBoard ─────────────────────────────────────────────
    writer = SummaryWriter(os.path.join(run_dir, "tensorboard")) if is_main else None

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, writer, is_main, use_amp)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, epoch, writer, is_main, use_amp)
        scheduler.step()

        if is_main:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}  {elapsed:.1f}s")

            # 按 save_freq 保存权重
            if epoch % args.save_freq == 0:
                ckpt_path = os.path.join(run_dir, f"mobilenetv2_epoch{epoch:03d}.pth")
                torch.save((model.module if args.ddp else model).state_dict(), ckpt_path)
                print(f"  -> 保存权重: {ckpt_path}")

            # 保存最优权重
            if val_acc > best_acc:
                best_acc = val_acc
                best_path = os.path.join(run_dir, "mobilenetv2_best.pth")
                torch.save((model.module if args.ddp else model).state_dict(), best_path)
                print(f"  -> 最优权重: {best_path}  acc={best_acc:.4f}")

    if is_main:
        writer.close()
        print(f"\n训练完成，最优 val acc={best_acc:.4f}")
        print(f"训练产物目录: {run_dir}")

    if args.ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
