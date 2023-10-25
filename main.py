import jax
from tinyshakespeareloader.hamlet import get_data

from kira.model.model import Kira
from kira.train import train


def main():
    tinyshakespeare = get_data(batch_size=256)
    train_dataloader, test_dataloader = (
        tinyshakespeare.train_dataloader,
        tinyshakespeare.test_dataloader,
    )

    n_dims = tinyshakespeare.vocab_size if tinyshakespeare.vocab_size else 256
    n_embd = tinyshakespeare.vocab_size if tinyshakespeare.vocab_size else 256
    learning_rate = 1e-3
    num_heads = 8
    key = jax.random.PRNGKey(0)

    kira = Kira(n_dims, n_embd, num_heads=num_heads, key=key)

    train(train_dataloader, test_dataloader, learning_rate, kira, early_stop=10000)


if __name__ == "__main__":
    main()
