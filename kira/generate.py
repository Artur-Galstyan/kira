from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from kira.model.model import Kira


def generate_text(
    kira: Kira,
    max_seq_len: int,
    max_new_tokens: int,
    decode: Callable,
    vobab_size: int,
    print_to_console: bool = True,
    random_key_seed: int = 0,
    state: eqx.nn.State = None,
) -> tuple[list[str], list[int]]:
    jitted_kira = eqx.filter_jit(kira)
    x = jnp.zeros((max_seq_len,), dtype=jnp.int32)
    key = jax.random.PRNGKey(random_key_seed)
    tokens = []
    decoded_tokens = []
    for _ in range(max_new_tokens):
        key, subkey, kira_key = jax.random.split(key, 3)
        if state is None:
            logits = jitted_kira(x, state=state, key=kira_key)
        else:
            logits, state = jitted_kira(x, state=state, key=kira_key)
        logits = logits[-1, :]
        probs = jax.nn.softmax(logits, axis=-1)

        next_token = jax.random.choice(
            subkey,
            jnp.arange(len(probs)),
            p=probs,
        )
        next_token = jnp.array(next_token, dtype=jnp.int32).reshape((1,))
        next_token = min(next_token.item(), vobab_size - 1)

        if print_to_console:
            print(decode([next_token]), end="")

        tokens.append(next_token)
        decoded_tokens.append(decode([next_token]))

        x = jnp.concatenate((x[1:], jnp.array([next_token])))
    return decoded_tokens, tokens
