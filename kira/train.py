import functools as ft
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from icecream import ic
from jaxtyping import Array, Int, PRNGKeyArray, PyTree
from torch.utils.data import DataLoader

from kira.model.model import Kira


def train(
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    learning_rate: float,
    kira: Kira,
    key: PRNGKeyArray,
    early_stop: int | None = None,
) -> Kira:
    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(eqx.filter(kira, eqx.is_array_like))

    for i, (x, y) in enumerate(train_dataloader):
        x = jnp.array(x)
        y = jnp.array(y)
        key, subkey = jax.random.split(key)
        kira, opt_state, loss_value = step(kira, opt_state, x, y, optimizer, key=subkey)
        if i % 100 == 0:
            ic(i, loss_value)
        if i % 1000 == 0:
            eval_loss = evaluate(test_dataloader, kira)
            ic(i, loss_value, eval_loss)
        if early_stop is not None and i > early_stop:
            break
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
    partial_kira = ft.partial(kira, key=key)
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
