import equinox as eqx
import jax
from icecream import ic
from tinyshakespeareloader.hamlet import get_data
from tqdm import tqdm

import wandb
from kira.generate import generate_text_without_kv_cache, generate_text
from kira.model.model import Kira
from kira.train import train

max_seq_len = 8
batch_size = 4
tinyshakespeare = get_data(batch_size=batch_size, block_size=max_seq_len, shuffle=True)
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
early_stop = None
key_seed = 0
n_epochs = 50
key = jax.random.PRNGKey(key_seed)

wandb.init(
    project="kira",
    config={
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
        "key_seed": key_seed,
    },
)

kira, init_state = eqx.nn.make_with_state(Kira)(
    n_dims=n_dims,
    n_embd=n_embd,
    num_heads=num_heads,
    max_seq_len=max_seq_len,
    key=key,
    n_layers=n_layers,
)


init_text = generate_text_without_kv_cache(
    kira,
    max_seq_len,
    max_new_tokens,
    tinyshakespeare.decode,
    vobab_size=n_dims,
)
ic(init_text)
init_text_with_state = generate_text(
    kira,
    init_state,
    max_new_tokens,
    tinyshakespeare.decode,
    vobab_size=n_dims,
)
ic(init_text_with_state)


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

    text_without_kv = generate_text_without_kv_cache(
        kira,
        max_seq_len,
        max_new_tokens,
        tinyshakespeare.decode,
        vobab_size=n_dims,
    )

    text_with_state = generate_text(
        kira,
        init_state,
        max_new_tokens,
        tinyshakespeare.decode,
        vobab_size=n_dims,
    )
    ic(text_without_kv)

eqx.tree_serialise_leaves("kira-experiment1.eqx", kira)
