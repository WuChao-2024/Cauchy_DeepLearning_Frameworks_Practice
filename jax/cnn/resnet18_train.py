# JAX/Flax ResNet-18 训练入口
# JAX 训练与 PyTorch 的关键区别：
#   1. 模型参数存储在 params 字典中，不在模型对象里
#   2. 训练步骤是纯函数，用 jax.jit 编译加速
#   3. 多卡训练用 jax.pmap 替代 PyTorch DDP
#   4. BatchNorm 需要额外维护 batch_stats
#
# 单卡: python resnet18_train.py --dataset cifar100
# 多卡: python resnet18_train.py --dataset cifar100 --pmap
# bf16: python resnet18_train.py --dataset cifar100 --dtype bfloat16

import os
import sys
import argparse
import time
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from tensorboardX import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../shared"))
from data_utils import get_numpy_iterator, get_dataset_config
from model_utils import print_jax_model_summary

from resnet18 import create_resnet18


# ── 训练状态（扩展 train_state 以包含 batch_stats）──────────────
class TrainState(train_state.TrainState):
    batch_stats: any


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     default="cifar100", choices=["mnist", "cifar100", "imagenet"])
    p.add_argument("--dataset_dir", default="../../dataset")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=0.1)
    p.add_argument("--save_freq",   type=int,   default=10)
    p.add_argument("--dtype",       default="float32", choices=["float32", "bfloat16"],
                   help="bfloat16 启用混合精度训练")
    p.add_argument("--pmap",        action="store_true", help="启用 pmap 多卡数据并行")
    return p.parse_args()


def create_train_state(model, rng, input_size, lr, num_steps):
    """初始化模型参数和优化器状态"""
    c, h, w = input_size
    dummy = jnp.ones((1, h, w, c))  # JAX 约定 channels-last

    # init 返回 {'params': ..., 'batch_stats': ...}
    variables = model.init({'params': rng}, dummy, training=False)
    params     = variables['params']
    batch_stats = variables['batch_stats']

    # cosine 学习率调度
    schedule = optax.cosine_decay_schedule(lr, num_steps)
    tx = optax.chain(
        optax.add_decayed_weights(1e-4),  # weight decay
        optax.sgd(schedule, momentum=0.9),
    )
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )


@partial(jax.jit, static_argnames=["use_bf16"])
def train_step(state, images, labels, use_bf16=False):
    """单步训练（jit 编译为 XLA 计算图）"""

    def loss_fn(params):
        # bfloat16 混合精度：前向用 bf16，梯度累积用 float32
        if use_bf16:
            images_cast = images.astype(jnp.bfloat16)
        else:
            images_cast = images

        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            images_cast,
            training=True,
            mutable=['batch_stats'],
        )
        logits = logits.astype(jnp.float32)  # 损失始终用 float32
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, (logits, updates)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])

    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return state, loss, acc


@jax.jit
def eval_step(state, images, labels):
    """单步评测"""
    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        images,
        training=False,
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    acc  = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return loss, acc


# ── pmap 版本（多卡数据并行）──────────────────────────────────────
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3,))
def train_step_pmap(state, images, labels, use_bf16=False):
    """pmap 版训练步骤：每张卡处理一个 mini-batch，梯度 all-reduce"""

    def loss_fn(params):
        images_cast = images.astype(jnp.bfloat16) if use_bf16 else images
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            images_cast, training=True, mutable=['batch_stats'],
        )
        logits = logits.astype(jnp.float32)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, (logits, updates)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)

    # 跨设备梯度平均（等价于 DDP 的 all-reduce）
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss  = jax.lax.pmean(loss,  axis_name="batch")

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return state, loss, acc


@partial(jax.pmap, axis_name="batch")
def eval_step_pmap(state, images, labels):
    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        images, training=False,
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    acc  = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    loss = jax.lax.pmean(loss, axis_name="batch")
    acc  = jax.lax.pmean(acc,  axis_name="batch")
    return loss, acc


def main():
    args = parse_args()
    use_bf16 = (args.dtype == "bfloat16")
    num_devices = jax.device_count()

    print(f"JAX 设备数量: {num_devices}  设备类型: {jax.devices()[0].device_kind}")
    print(f"训练精度: {args.dtype}  {'(bfloat16 混合精度)' if use_bf16 else '(全精度 float32)'}")

    # ── 训练产物目录 ─────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"resnet18_jax_{args.dataset}_{args.dtype}_{timestamp}"
    run_dir   = os.path.abspath(os.path.join("./outputs", run_name))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # ── 数据集 ──────────────────────────────────────────────────
    cfg = get_dataset_config(args.dataset)
    num_classes = cfg["num_classes"]
    input_size  = cfg["input_size"]

    # pmap 时 batch 需要能被设备数整除
    batch_size = args.batch_size
    if args.pmap:
        batch_size = (batch_size // num_devices) * num_devices

    train_iter_fn, train_steps = get_numpy_iterator(
        args.dataset, "train", args.dataset_dir, batch_size)
    val_iter_fn, val_steps = get_numpy_iterator(
        args.dataset, "test", args.dataset_dir, batch_size)

    # ── 模型初始化 ───────────────────────────────────────────────
    model = create_resnet18(num_classes, input_size)
    rng   = jax.random.PRNGKey(42)
    state = create_train_state(model, rng, input_size, args.lr,
                               args.epochs * train_steps)

    print_jax_model_summary(state.params, title="ResNet-18 (JAX)")

    # pmap 时将 state 复制到所有设备
    if args.pmap:
        state = jax.device_put_replicated(state, jax.devices())

    writer   = SummaryWriter(os.path.join(run_dir, "tensorboard"))
    best_acc = 0.0

    # ── 权重保存（orbax）────────────────────────────────────────
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    checkpointer = ocp.StandardCheckpointer()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # 训练
        train_losses, train_accs = [], []
        for images, labels in train_iter_fn():
            images = jnp.array(images)
            labels = jnp.array(labels)

            if args.pmap:
                # 将 batch 切分到各设备：(B, H, W, C) -> (num_devices, B//D, H, W, C)
                images = images.reshape(num_devices, -1, *images.shape[1:])
                labels = labels.reshape(num_devices, -1)
                state, loss, acc = train_step_pmap(state, images, labels, use_bf16)
                loss = float(loss[0])
                acc  = float(acc[0])
            else:
                state, loss, acc = train_step(state, images, labels, use_bf16)
                loss = float(loss)
                acc  = float(acc)

            train_losses.append(loss)
            train_accs.append(acc)

        # 评测
        val_losses, val_accs = [], []
        for images, labels in val_iter_fn():
            images = jnp.array(images)
            labels = jnp.array(labels)

            if args.pmap:
                images = images[:num_devices * (len(images) // num_devices)]
                labels = labels[:num_devices * (len(labels) // num_devices)]
                images = images.reshape(num_devices, -1, *images.shape[1:])
                labels = labels.reshape(num_devices, -1)
                loss, acc = eval_step_pmap(state, images, labels)
                loss = float(loss[0])
                acc  = float(acc[0])
            else:
                loss, acc = eval_step(state, images, labels)
                loss = float(loss)
                acc  = float(acc)

            val_losses.append(loss)
            val_accs.append(acc)

        train_loss = np.mean(train_losses)
        train_acc  = np.mean(train_accs)
        val_loss   = np.mean(val_losses)
        val_acc    = np.mean(val_accs)
        elapsed    = time.time() - t0

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc",  train_acc,  epoch)
        writer.add_scalar("val/loss",   val_loss,   epoch)
        writer.add_scalar("val/acc",    val_acc,    epoch)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  {elapsed:.1f}s")

        # 保存权重（pmap 时取第 0 个设备的 state）
        save_state = jax.tree_util.tree_map(lambda x: x[0], state) if args.pmap else state

        if epoch % args.save_freq == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}")
            checkpointer.save(ckpt_path, save_state)
            print(f"  -> 保存权重: {ckpt_path}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(ckpt_dir, "best")
            checkpointer.save(best_path, save_state, force=True)
            print(f"  -> 最优权重: {best_path}  acc={best_acc:.4f}")

    writer.close()
    print(f"\n训练完成，最优 val acc={best_acc:.4f}")
    print(f"训练产物目录: {run_dir}")


if __name__ == "__main__":
    main()
