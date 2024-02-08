import functools as ft
from typing import Any, Optional, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Int, PRNGKeyArray, PyTree
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    learning_rate: float,
    model: PyTree,
    key: PRNGKeyArray,
    early_stop: int | None = None,
    wandb_client: Any | None = None,
    callback: Callable[[int, PyTree, Any, int], None] | None = None,
    experiment: int = 7,
    eval_every: Optional[int] = None,
    log_every: Optional[int] = 100,
) -> PyTree:
    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
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
        model, opt_state, loss_value = step(
            model, opt_state, x, y, optimizer, key=subkey
        )
        if log_every is not None and i % log_every == 0:
            loss_tqdm.set_description_str(f"train loss: {loss_value}")
            if wandb_client is not None:
                wandb_client.log({"train_loss": loss_value})
        if eval_every is not None and i % eval_every == 0:
            eval_loss = evaluate(test_dataloader, model)
            eval_loss_tqdm.set_description_str(f"eval loss: {eval_loss}")
            if wandb_client is not None:
                wandb_client.log({"eval_loss": eval_loss})
        if early_stop is not None and i > early_stop:
            break
        if callback is not None:
            callback(i, model, wandb_client, experiment)
    return model


def evaluate(
    test_dataloader: DataLoader,
    model: PyTree,
):
    loss = 0
    for _, (x, y) in enumerate(test_dataloader):
        x = jnp.array(x)
        y = jnp.array(y)

        loss += loss_fn(model, x, y, key=None)

    return loss / len(test_dataloader)


@eqx.filter_jit
def loss_fn(
    model: PyTree,
    x: Int[Array, "batch_size max_seq_len n_dims"],
    labels: Int[Array, "batch_size max_seq_len n_dims"],
    key: Optional[PRNGKeyArray],
) -> Array:
    partial_model = ft.partial(model, state=None, key=key)
    logits = eqx.filter_vmap(partial_model)(x)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))


@eqx.filter_jit
def step(
    model: PyTree,
    opt_state: PyTree,
    x: Array,
    y: Array,
    optimizer: optax.GradientTransformation,
    key: PRNGKeyArray,
) -> tuple[PyTree, PyTree, Any]:
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss
