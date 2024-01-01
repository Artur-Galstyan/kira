import functools as ft
from typing import Any, Optional, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Int, PRNGKeyArray, PyTree
from torch.utils.data import DataLoader
from tqdm import tqdm

from kira.model.model import Kira


def train(
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    learning_rate: float,
    kira: Kira,
    key: PRNGKeyArray,
    early_stop: int | None = None,
    wandb_client: Any | None = None,
    callback: Callable[[int, Kira, Any, int], None] | None = None,
    experiment: int = 7,
) -> Kira:
    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(eqx.filter(kira, eqx.is_inexact_array))
    total = len(train_dataloader) if early_stop is None else early_stop
    loss_tqdm = tqdm(
        total=total,
        desc="train loss",
        position=2,
        bar_format="{desc}",
        leave=False,
    )
    eval_loss_tqdm = tqdm(
        total=total, desc="eval loss", position=2, bar_format="{desc}", leave=False
    )
    for i, (x, y) in tqdm(
        enumerate(train_dataloader), desc="train", position=1, leave=False
    ):
        x = jnp.array(x)
        y = jnp.array(y)
        key, subkey = jax.random.split(key)
        kira, opt_state, loss_value = step(kira, opt_state, x, y, optimizer, key=subkey)
        if i % 100 == 0:
            loss_tqdm.set_description_str(f"train loss: {loss_value}")
            if wandb_client is not None:
                wandb_client.log({"train_loss": loss_value})
        if i % 1000 == 0:
            eval_loss = evaluate(test_dataloader, kira)
            eval_loss_tqdm.set_description_str(f"eval loss: {eval_loss}")
            if wandb_client is not None:
                wandb_client.log({"eval_loss": eval_loss})
        if early_stop is not None and i > early_stop:
            break
        if callback is not None:
            callback(i, kira, wandb_client, experiment)
    return kira


def evaluate(
    test_dataloader: DataLoader,
    kira: Kira,
):
    loss = 0
    for _, (x, y) in enumerate(test_dataloader):
        x = jnp.array(x)
        y = jnp.array(y)

        loss += loss_fn(kira, x, y, key=None)

    return loss / len(test_dataloader)


@eqx.filter_jit
def loss_fn(
    kira: Kira,
    x: Int[Array, "batch_size max_seq_len n_dims"],
    labels: Int[Array, "batch_size max_seq_len n_dims"],
    key: Optional[PRNGKeyArray],
) -> Array:
    partial_kira = ft.partial(kira, state=None, key=key)
    logits = eqx.filter_vmap(partial_kira)(x)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))


@eqx.filter_jit
def step(
    kira: PyTree,
    opt_state: PyTree,
    x: Array,
    y: Array,
    optimizer: optax.GradientTransformation,
    key: PRNGKeyArray,
) -> tuple[PyTree, PyTree, Any]:
    loss, grads = eqx.filter_value_and_grad(loss_fn)(kira, x, y, key)
    updates, opt_state = optimizer.update(grads, opt_state, kira)
    kira = eqx.apply_updates(kira, updates)

    return kira, opt_state, loss
