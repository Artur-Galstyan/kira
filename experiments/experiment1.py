import equinox as eqx
import jax
from icecream import ic
from tinyshakespeareloader.hamlet import get_data
from tqdm import tqdm

import wandb
from kira.generate import generate_text
from kira.model.model import Kira
from kira.train import train

def checkpoint_callback(it: int, kira: Kira, wandb_client: wandb) -> None:
    if it % 1000 == 0:
        eqx.tree_serialise_leaves(f"kira-experiment-mkvh-checkpoint-{it}.eqx", kira)
        wandb_client.save(f"kira-experiment-mkvh-checkpoint-{it}.eqx")

max_seq_len = 128
batch_size = 128
tinyshakespeare = get_data(batch_size=batch_size, block_size=max_seq_len, shuffle=True)
train_dataloader, test_dataloader = (
    tinyshakespeare.train_dataloader,
    tinyshakespeare.test_dataloader,
)

iteration = 5
n_dims = tinyshakespeare.vocab_size if tinyshakespeare.vocab_size else 256
n_embd = 384
learning_rate = 3e-4
num_heads = 6
num_query_heads = 6
num_kv_heads = 1
n_layers = 6
max_new_tokens = 2000
early_stop = None
key_seed = 0
n_epochs = 1
key = jax.random.PRNGKey(key_seed)

wandb.init(
    project="kira",
    config={
        "iteration": iteration,
        "n_epochs": n_epochs,
        "n_dims": n_dims,
        "n_embd": n_embd,
        "num_heads": num_heads,
        "n_layers": n_layers,
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_new_tokens": max_new_tokens,
        "early_stop": early_stop,
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "key_seed": key_seed,
    },
)
kira = Kira(
    n_dims=n_dims,
    n_embd=n_embd,
    num_heads=num_heads,
    num_query_heads=num_query_heads,
    num_kv_heads=num_kv_heads,
    max_seq_len=max_seq_len,
    key=key,
    n_layers=n_layers,
)


init_text_with_state = generate_text(
    kira,
    max_seq_len,
    max_new_tokens,
    tinyshakespeare.decode,
    vobab_size=n_dims,
)


for epoch in tqdm(range(n_epochs)):
    key, subkey = jax.random.split(key)
    kira = train(
        train_dataloader,
        test_dataloader,
        learning_rate,
        kira,
        subkey,
        early_stop=early_stop,
        wandb_client=wandb,
        callback=checkpoint_callback,
    )

    print("Generating text with kv cache...")
    text_with_state = generate_text(
        kira,
        max_seq_len,
        max_new_tokens,
        tinyshakespeare.decode,
        vobab_size=n_dims,
    )
    ic(text_with_state)

eqx.tree_serialise_leaves(f"kira-experiment-mkvh-{iteration}.eqx", kira)
