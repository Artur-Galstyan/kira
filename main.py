import jax
from tinyshakespeareloader.hamlet import get_data
from kira.generate import generate_text

from kira.model.model import Kira
from kira.train import train
from icecream import ic


def main():
    max_seq_len = 8
    batch_size = 256
    tinyshakespeare = get_data(batch_size=batch_size, block_size=max_seq_len)
    train_dataloader, test_dataloader = (
        tinyshakespeare.train_dataloader,
        tinyshakespeare.test_dataloader,
    )

    n_dims = tinyshakespeare.vocab_size if tinyshakespeare.vocab_size else 256
    n_embd = 512
    learning_rate = 3e-4
    num_heads = 16
    max_new_tokens = 2000
    key = jax.random.PRNGKey(0)

    kira = Kira(n_dims, n_embd, num_heads=num_heads, max_seq_len=max_seq_len, key=key)
    ic("Initial output")
    assert tinyshakespeare.decode is not None
    assert tinyshakespeare.vocab_size is not None
    generate_text(
        kira, max_new_tokens, tinyshakespeare.decode, tinyshakespeare.vocab_size
    )
    ic("Starting training...")
    kira = train(
        train_dataloader, test_dataloader, learning_rate, kira, early_stop=5000
    )
    ic("Final output")
    generate_text(
        kira, max_new_tokens, tinyshakespeare.decode, tinyshakespeare.vocab_size
    )


if __name__ == "__main__":
    main()
