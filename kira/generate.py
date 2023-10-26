import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Callable
from kira.model.model import Kira
from jaxtyping import Array
from icecream import ic


def generate_text(
    kira: Kira, max_new_tokens: int, decode: Callable[[Array], str], vobab_size: int
):
    jitted_kira = eqx.filter_jit(kira)
    max_seq_len = kira.max_seq_len
    x = jnp.zeros((max_seq_len,), dtype=jnp.int32)
    key = jax.random.PRNGKey(0)
    for i in range(max_new_tokens):
        key, subkey = jax.random.split(key)
        logits = jitted_kira(x)
        logits = logits[-1, :]
        probs = jax.nn.softmax(logits, axis=-1)

        next_token = jax.random.choice(
            subkey,
            jnp.arange(len(probs)),
            p=probs,
        )
        next_token = jnp.array(next_token, dtype=jnp.int32).reshape((1,))
        x = jnp.concatenate([x[1:], next_token])

        next_token = min(next_token.item(), vobab_size - 1)

        print(decode([next_token]), end="")
