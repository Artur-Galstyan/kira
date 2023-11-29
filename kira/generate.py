from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from kira.model.model import Kira


def generate_text(
    kira: Kira,
    max_seq_len: int,
    max_new_tokens: int,
    decode: Callable[[Array], str],
    vobab_size: int,
):
    jitted_kira = eqx.filter_jit(kira)
    x = jnp.zeros((max_seq_len,), dtype=jnp.int32)
    key = jax.random.PRNGKey(0)
    text = ""
    for _ in range(max_new_tokens):
        key, subkey, kira_key = jax.random.split(key, 3)
        logits = jitted_kira(x, key=kira_key)
        logits = logits[-1, :]
        probs = jax.nn.softmax(logits, axis=-1)

        next_token = jax.random.choice(
            subkey,
            jnp.arange(len(probs)),
            p=probs,
        )
        next_token = jnp.array(next_token, dtype=jnp.int32).reshape((1,))
        next_token = min(next_token.item(), vobab_size - 1)

        print(decode([next_token]), end="")  # type: ignore
        text += decode([next_token])  # type: ignore

        x = jnp.concatenate((x[1:], jnp.array([next_token])))
    return text
