import equinox as eqx
import icecream
import jax
from icecream import ic
from tinyshakespeareloader.hamlet import get_data
from tqdm import tqdm

import wandb
from kira.generate import generate_text
from kira.model.model import Kira
from kira.train import train

max_seq_len = 128
batch_size = 128
tinyshakespeare = get_data(batch_size=batch_size, block_size=max_seq_len, shuffle=True)
train_dataloader, test_dataloader = (
    tinyshakespeare.train_dataloader,
    tinyshakespeare.test_dataloader,
)

icecream.install()
n_dims = tinyshakespeare.vocab_size if tinyshakespeare.vocab_size else 256
n_embd = 384
learning_rate = 3e-4
num_heads = 6
n_layers = 6
max_new_tokens = 2000
early_stop = None
key_seed = 0
n_epochs = 1
key = jax.random.PRNGKey(key_seed)

# wandb.init(
#     project="kira",
#     config={
#         "n_epochs": n_epochs,
#         "n_dims": n_dims,
#         "n_embd": n_embd,
#         "num_heads": num_heads,
#         "n_layers": n_layers,
#         "max_seq_len": max_seq_len,
#         "batch_size": batch_size,
#         "learning_rate": learning_rate,
#         "max_new_tokens": max_new_tokens,
#         "early_stop": early_stop,
#         "key_seed": key_seed,
#     },
# )

kira = Kira(
    n_dims=n_dims,
    n_embd=n_embd,
    num_heads=num_heads,
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
    )

    print("Generating text with kv cache...")
    text_with_state = generate_text(
        kira,
        max_new_tokens,
        512,
        tinyshakespeare.decode,
        vobab_size=n_dims,
    )
    ic(text_with_state)

eqx.tree_serialise_leaves("kira-experiment2.eqx", kira)
