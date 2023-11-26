import equinox as eqx
import jax
from tinyshakespeareloader.hamlet import get_data

from kira.model.model import Kira
from kira.train import train


def main():
    max_seq_len = 8
    batch_size = 4
    tinyshakespeare = get_data(
        batch_size=batch_size, block_size=max_seq_len, shuffle=True
    )
    train_dataloader, test_dataloader = (
        tinyshakespeare.train_dataloader,
        tinyshakespeare.test_dataloader,
    )

    n_dims = tinyshakespeare.vocab_size if tinyshakespeare.vocab_size else 256
    n_embd = 128  # 384
    learning_rate = 3e-4
    num_heads = 1  # 6
    n_layers = 1  # 6
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

    key, subkey = jax.random.split(key)
    kira = train(
        train_dataloader,
        test_dataloader,
        learning_rate,
        kira,
        subkey,
        early_stop=10000,
    )


if __name__ == "__main__":
    with jax.checking_leaks():
        main()
