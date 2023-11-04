import functools as ft

import equinox as eqx
import jax
from icecream import ic
from tinyshakespeareloader.hamlet import get_data

from kira.generate import generate_text
from kira.model.model import Kira
from kira.train import train


def main():
    max_seq_len = 32
    batch_size = 64
    tinyshakespeare = get_data(
        batch_size=batch_size, block_size=max_seq_len, shuffle=True
    )
    train_dataloader, test_dataloader = (
        tinyshakespeare.train_dataloader,
        tinyshakespeare.test_dataloader,
    )

    n_dims = tinyshakespeare.vocab_size if tinyshakespeare.vocab_size else 256
    n_embd = 384
    learning_rate = 3e-4
    num_heads = 6
    n_layers = 1
    max_new_tokens = 2000
    key = jax.random.PRNGKey(0)

    kira, state = eqx.nn.make_with_state(Kira)(
        n_dims,
        n_embd,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        key=key,
        n_layers=n_layers,
    )

    x, y = next(iter(train_dataloader))
    x = x.numpy()
    y = y.numpy()
    x = x[0][0].reshape(1, -1)

    key, subkey = jax.random.split(key)
    partial_kira = ft.partial(kira, key=subkey, state=state)
    output, state = eqx.filter_vmap(partial_kira)(x)
    ic(output.shape, state)

    key, subkey = jax.random.split(key)
    partial_kira = ft.partial(kira, key=subkey, state=state)
    output = eqx.filter_vmap(partial_kira)(x)
    ic(output.shape)


def test_train():
    max_seq_len = 64
    batch_size = 64
    tinyshakespeare = get_data(
        batch_size=batch_size, block_size=max_seq_len, shuffle=True
    )
    train_dataloader, test_dataloader = (
        tinyshakespeare.train_dataloader,
        tinyshakespeare.test_dataloader,
    )

    n_dims = tinyshakespeare.vocab_size if tinyshakespeare.vocab_size else 256
    n_embd = 384
    learning_rate = 3e-4
    num_heads = 6
    n_layers = 6
    max_new_tokens = 2000
    key = jax.random.PRNGKey(0)

    kira = Kira(
        n_dims,
        n_embd,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        key=key,
        n_layers=n_layers,
    )
    ic("Initial output")
    assert tinyshakespeare.decode is not None
    assert tinyshakespeare.vocab_size is not None
    generate_text(
        kira, max_new_tokens, tinyshakespeare.decode, tinyshakespeare.vocab_size
    )
    ic("Starting training...")
    key, subkey = jax.random.split(key)
    kira = train(
        train_dataloader,
        test_dataloader,
        learning_rate,
        kira,
        early_stop=5000,
        key=subkey,
    )
    ic("Final output")
    generate_text(
        kira, max_new_tokens, tinyshakespeare.decode, tinyshakespeare.vocab_size
    )


if __name__ == "__main__":
    main()
