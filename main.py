import time

import equinox as eqx
import jax
from tinyshakespeareloader.hamlet import get_data

from kira import Mamba
from kira.generate import generate_text
from kira.model_args import get_mamba_args, get_transformer_args
from kira.train import train
from tqdm import tqdm


def main():
    max_seq_len = 8
    early_stop = 300
    batch_size = 64
    tinyshakespeare = get_data(
        batch_size=batch_size, block_size=max_seq_len, shuffle=True
    )
    train_dataloader, test_dataloader = (
        tinyshakespeare.train_dataloader,
        tinyshakespeare.test_dataloader,
    )

    n_dims = tinyshakespeare.vocab_size if tinyshakespeare.vocab_size else 256
    n_embd = 384  # 384
    learning_rate = 3e-4
    num_heads = 2  # 6
    query_multihead_dim = num_heads
    kv_multihead_dim = 2
    n_layers = 3  # 6
    max_new_tokens = 200

    kira_model_args = get_transformer_args(
        n_dims=n_dims,
        n_embd=n_embd,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        num_query_heads=query_multihead_dim,
        num_kv_heads=kv_multihead_dim,
        width_size=256,
        depth=4,
        key_seed=0,
    )

    key = jax.random.PRNGKey(kira_model_args.key_seed)

    # kira = Kira(
    #     model_args=kira_model_args,
    #     key=key,
    # )
    # key, subkey = jax.random.split(key)
    # kira = train(
    #     train_dataloader,
    #     test_dataloader,
    #     learning_rate,
    #     kira,
    #     early_stop=early_stop,
    #     key=subkey,
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

    model_args = get_mamba_args(
        n_embd=n_embd,
        n_dims=n_dims,
        n_layers=n_layers,
    )
    # wandb.init(
    #     project="mamba",
    #     name="mamba standard",
    #     config=model_args.__dict__,
    # )
    mamba = Mamba(model_args=model_args, key=key)

    key, subkey = jax.random.split(key)

    # mamba = train(
    #     train_dataloader,
    #     test_dataloader,
    #     learning_rate,
    #     mamba,
    #     subkey,
    #     early_stop=early_stop,
    #     # wandb_client=wandb,
    # )

    # generate some dummy training data
    xs = []
    ys = []
    key, subkey, subkey2 = jax.random.split(key, 3)
    for i in range(2400):
        x = jax.random.randint(subkey, (batch_size, max_seq_len), 0, n_dims)
        y = jax.random.randint(subkey2, (batch_size, max_seq_len), 0, n_dims)
        xs.append(x)
        ys.append(y)

    start_time = time.time()
    mamba = eqx.filter_jit(mamba)
    for x, y in tqdm(zip(xs, ys)):
        eqx.filter_vmap(mamba)(x)

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
