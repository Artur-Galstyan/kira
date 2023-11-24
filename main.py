import equinox as eqx
import jax
import jax.numpy as jnp
from icecream import ic
from tinyshakespeareloader.hamlet import get_data

from kira.generate import generate_text
from kira.model.model import Kira
from kira.train import train


def main():
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

    kira, state = eqx.nn.make_with_state(Kira)(
        n_dims=n_dims,
        n_embd=n_embd,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        key=key,
        n_layers=n_layers,
    )
    states = eqx.filter_vmap(lambda: state, axis_size=batch_size)()

    x, y = next(iter(train_dataloader))
    x = x.numpy()
    y = y.numpy()

    key, subkey = jax.random.split(key)

    first_token = x[:, :1].reshape((batch_size, 1))

    output, states = eqx.filter_vmap(kira, in_axes=(0, 0, None))(
        first_token, states, subkey
    )
    output = jax.nn.softmax(output, axis=-1)
    first_token = jnp.argmax(output, axis=-1)
    key, subkey = jax.random.split(key)
    output, states = eqx.filter_vmap(kira, in_axes=(0, 0, None))(
        first_token, states, subkey
    )


def test_train():
    max_seq_len = 128  # 64
    batch_size = 64  # 64
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
    num_heads = 6  # 6
    n_layers = 6  # 6
    max_new_tokens = 200
    key = jax.random.PRNGKey(0)

    kira, state = eqx.nn.make_with_state(Kira)(
        n_dims,
        n_embd,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        key=key,
        n_layers=n_layers,
    )
    init_state = state
    ic("Initial output")
    assert tinyshakespeare.decode is not None
    assert tinyshakespeare.vocab_size is not None
    generate_text(
        kira,
        init_state,
        max_new_tokens,
        tinyshakespeare.decode,
        tinyshakespeare.vocab_size,
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
        kira,
        init_state,
        max_new_tokens,
        tinyshakespeare.decode,
        tinyshakespeare.vocab_size,
    )

    # save model
    eqx.tree_serialise_leaves("kira1.eqx", kira)


if __name__ == "__main__":
    with jax.checking_leaks():
        # main()
        test_train()
