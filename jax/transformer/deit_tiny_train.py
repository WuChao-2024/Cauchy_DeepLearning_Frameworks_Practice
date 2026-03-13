# JAX/Flax DeiT-Tiny 训练入口
# 单卡: python deit_tiny_train.py --dataset cifar100
# 多卡: python deit_tiny_train.py --dataset cifar100 --pmap
# bf16: python deit_tiny_train.py --dataset cifar100 --dtype bfloat16

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

from deit_tiny import create_deit_tiny


class TrainState(train_state.TrainState):
    pass


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
    p.add_argument("--pmap",        action="store_true")
    return p.parse_args()


def create_train_state(model, rng, input_size, lr, num_steps, warmup_steps):
    c, h, w = input_size
    dummy = jnp.ones((1, h, w, c))
    # DeiT 训练模式返回 (cls_out, dist_out)，init 时用 training=False
    params = model.init({'params': rng}, dummy, training=False)['params']

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr,
        warmup_steps=warmup_steps, decay_steps=num_steps, end_value=lr * 0.01,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=0.05),
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@partial(jax.jit, static_argnames=["use_bf16"])
def train_step(state, images, labels, use_bf16=False):
    def loss_fn(params):
        images_cast = images.astype(jnp.bfloat16) if use_bf16 else images
        # 训练模式返回 (cls_out, dist_out)
        cls_out, dist_out = state.apply_fn({'params': params}, images_cast, training=True)
        cls_out  = cls_out.astype(jnp.float32)
        dist_out = dist_out.astype(jnp.float32)
        num_classes = cls_out.shape[-1]
        smooth = 0.1
        one_hot = jax.nn.one_hot(labels, num_classes)
        smooth_labels = one_hot * (1 - smooth) + smooth / num_classes
        loss_cls  = -jnp.sum(smooth_labels * jax.nn.log_softmax(cls_out),  axis=-1).mean()
        loss_dist = -jnp.sum(smooth_labels * jax.nn.log_softmax(dist_out), axis=-1).mean()
        return loss_cls + loss_dist, cls_out

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return state, loss, acc


@jax.jit
def eval_step(state, images, labels):
    # eval 模式返回 (cls+dist)/2
    logits = state.apply_fn({'params': state.params}, images, training=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    acc  = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return loss, acc


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3,))
def train_step_pmap(state, images, labels, use_bf16=False):
    def loss_fn(params):
        images_cast = images.astype(jnp.bfloat16) if use_bf16 else images
        cls_out, dist_out = state.apply_fn({'params': params}, images_cast, training=True)
        cls_out  = cls_out.astype(jnp.float32)
        dist_out = dist_out.astype(jnp.float32)
        num_classes = cls_out.shape[-1]
        smooth = 0.1
        one_hot = jax.nn.one_hot(labels, num_classes)
        smooth_labels = one_hot * (1 - smooth) + smooth / num_classes
        loss = (-jnp.sum(smooth_labels * jax.nn.log_softmax(cls_out),  axis=-1).mean()
              + -jnp.sum(smooth_labels * jax.nn.log_softmax(dist_out), axis=-1).mean())
        return loss, cls_out

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss  = jax.lax.pmean(loss,  axis_name="batch")
    state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return state, loss, acc


@partial(jax.pmap, axis_name="batch")
def eval_step_pmap(state, images, labels):
    logits = state.apply_fn({'params': state.params}, images, training=False)
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"deit_tiny_jax_{args.dataset}_{args.dtype}_{timestamp}"
    run_dir   = os.path.abspath(os.path.join("./outputs", run_name))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    cfg = get_dataset_config(args.dataset)
    num_classes = cfg["num_classes"]
    input_size  = cfg["input_size"]

    batch_size = args.batch_size
    if args.pmap:
        batch_size = (batch_size // num_devices) * num_devices

    train_iter_fn, train_steps = get_numpy_iterator(
        args.dataset, "train", args.dataset_dir, batch_size)
    val_iter_fn, val_steps = get_numpy_iterator(
        args.dataset, "test", args.dataset_dir, batch_size)

    model = create_deit_tiny(num_classes, input_size)
    rng   = jax.random.PRNGKey(42)
    warmup_steps = args.warmup * train_steps
    state = create_train_state(model, rng, input_size, args.lr,
                               args.epochs * train_steps, warmup_steps)

    print_jax_model_summary(state.params, title="DeiT-Tiny (JAX)")

    if args.pmap:
        state = jax.device_put_replicated(state, jax.devices())

    writer   = SummaryWriter(os.path.join(run_dir, "tensorboard"))
    best_acc = 0.0
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    checkpointer = ocp.StandardCheckpointer()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_losses, train_accs = [], []
        for images, labels in train_iter_fn():
            images = jnp.array(images)
            labels = jnp.array(labels)
            if args.pmap:
                images = images.reshape(num_devices, -1, *images.shape[1:])
                labels = labels.reshape(num_devices, -1)
                state, loss, acc = train_step_pmap(state, images, labels, use_bf16)
                loss, acc = float(loss[0]), float(acc[0])
            else:
                state, loss, acc = train_step(state, images, labels, use_bf16)
                loss, acc = float(loss), float(acc)
            train_losses.append(loss)
            train_accs.append(acc)

        val_losses, val_accs = [], []
        for images, labels in val_iter_fn():
            images = jnp.array(images)
            labels = jnp.array(labels)
            if args.pmap:
                n = num_devices * (len(images) // num_devices)
                images = images[:n].reshape(num_devices, -1, *images.shape[1:])
                labels = labels[:n].reshape(num_devices, -1)
                loss, acc = eval_step_pmap(state, images, labels)
                loss, acc = float(loss[0]), float(acc[0])
            else:
                loss, acc = eval_step(state, images, labels)
                loss, acc = float(loss), float(acc)
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
