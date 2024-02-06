import json
import time

import equinox as eqx
import jax
from tinyshakespeareloader.hamlet import get_data

import wandb
from kira.generate import generate_text
from kira.model.mamba import Mamba, ModelArgs
from kira.model.model import Kira
from kira.train import train


def main():
    max_seq_len = 8
    batch_size = 64
    tinyshakespeare = get_data(
        batch_size=batch_size, block_size=max_seq_len, shuffle=True
    )
    train_dataloader, test_dataloader = (
        tinyshakespeare.train_dataloader,
        tinyshakespeare.test_dataloader,
    )

    n_dims = tinyshakespeare.vocab_size if tinyshakespeare.vocab_size else 256
    n_embd = 256  # 384
    learning_rate = 3e-4
    num_heads = 2  # 6
    query_multihead_dim = num_heads
    kv_multihead_dim = 2
    n_layers = 3  # 6
    max_new_tokens = 2000
    key = jax.random.PRNGKey(0)

    # kira = Kira(
    #     n_dims=n_dims,
    #     n_embd=n_embd,
    #     num_heads=num_heads,
    #     num_query_heads=query_multihead_dim,
    #     num_kv_heads=kv_multihead_dim,
    #     max_seq_len=max_seq_len,
    #     key=key,
    #     n_layers=n_layers,
    # )

    model_args = ModelArgs(
        d_model=n_embd,
        n_layer=n_layers,
        vocab_size=n_dims,
    )

    # wandb.init(
    #     project="mamba",
    #     name="mamba standard",
    #     config=model_args.__dict__,
    # )
    mamba = Mamba(model_args=model_args, key=key)

    key, subkey = jax.random.split(key)

    early_stop = 300
    # kira = train(
    #     train_dataloader,
    #     test_dataloader,
    #     learning_rate,
    #     kira,
    #     subkey,
    #     early_stop=early_stop,
    #     # wandb_client=wandb,
    # )
    #
    # generate_text(
    #     kira,
    #     max_seq_len,
    #     max_new_tokens,
    #     decode=tinyshakespeare.decode,
    #     vocab_size=tinyshakespeare.vocab_size,
    # )
    start_time = time.time()
    mamba = train(
        train_dataloader,
        test_dataloader,
        learning_rate,
        mamba,
        subkey,
        early_stop=early_stop,
        # wandb_client=wandb,
    )
    print("", flush=True)
    print("Training complete.")
    print(f"Training took {time.time() - start_time} seconds for {early_stop} steps.")

    generate_text(
        mamba,
        max_seq_len,
        max_new_tokens,
        decode=tinyshakespeare.decode,
        vocab_size=tinyshakespeare.vocab_size,
    )


if __name__ == "__main__":
    with jax.checking_leaks():
        main()
