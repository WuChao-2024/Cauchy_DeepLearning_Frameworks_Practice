# DeiT-Tiny 训练入口，支持单卡和 DDP 多卡训练，TensorBoard 日志
# DeiT 训练时返回 (cls_logits, dist_logits)，损失取两者之和
# 单卡:  python deit_tiny_train.py --dataset cifar100
# 多卡:  torchrun --nproc_per_node=2 deit_tiny_train.py --dataset cifar100 --ddp
# bf16:  python deit_tiny_train.py --dataset cifar100 --dtype bfloat16

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

from deit_tiny import DeiTTiny


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     default="cifar100", choices=["mnist", "cifar100", "imagenet"])
    p.add_argument("--dataset_dir", default="../../dataset")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--warmup",      type=int,   default=5)
    p.add_argument("--save_freq",   type=int,   default=10)
    p.add_argument("--dtype",       default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--ddp",         action="store_true")
    return p.parse_args()


def deit_loss(cls_out, dist_out, labels, criterion):
    """DeiT 损失：cls 和 dist 两个头的损失之和"""
    return criterion(cls_out, labels) + criterion(dist_out, labels)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer, is_main, use_amp):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            cls_out, dist_out = model(imgs)
            loss = deit_loss(cls_out, dist_out, labels, criterion)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        # 训练时用 cls_out 计算准确率
        correct += cls_out.detach().argmax(1).eq(labels).sum().item()
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
            # eval 模式下 DeiT 返回 (cls+dist)/2 的平均 logits
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"deit_tiny_torch_{args.dataset}_{args.dtype}_{timestamp}"
    run_dir   = os.path.join("./outputs", run_name)
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.txt"), "w") as f:
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")

    cfg = get_dataset_config(args.dataset)
    num_classes = cfg["num_classes"]
    input_size  = cfg["input_size"]

    train_loader = get_dataloader(args.dataset, "train", args.dataset_dir,
                                  args.batch_size, distributed=args.ddp)
    val_loader   = get_dataloader(args.dataset, "test",  args.dataset_dir, args.batch_size)

    model = DeiTTiny(num_classes=num_classes, input_size=input_size).to(device)

    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    if is_main:
        print_model_summary(model.module if args.ddp else model,
                            (1,) + input_size, title="DeiT-Tiny")
        print(f"训练精度: {args.dtype}  {'(autocast bfloat16)' if use_amp else '(全精度 float32)'}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    def lr_lambda(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / args.warmup
        progress = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        import math
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    writer   = SummaryWriter(os.path.join(run_dir, "tensorboard")) if is_main else None
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, is_main, use_amp)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, epoch, writer, is_main, use_amp)
        scheduler.step()

        if is_main:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}  {elapsed:.1f}s")

            if epoch % args.save_freq == 0:
                ckpt = os.path.join(run_dir, f"deit_tiny_epoch{epoch:03d}.pth")
                torch.save((model.module if args.ddp else model).state_dict(), ckpt)
                print(f"  -> 保存权重: {ckpt}")

            if val_acc > best_acc:
                best_acc = val_acc
                best_path = os.path.join(run_dir, "deit_tiny_best.pth")
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
