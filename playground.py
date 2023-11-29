import equinox as eqx
import icecream
import jax
from icecream import ic
from tinyshakespeareloader.hamlet import get_data

from kira.generate import generate_text, generate_text_without_kv_cache
from kira.model.model import Kira

icecream.install()
max_seq_len = 128
batch_size = 128
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
max_new_tokens = 512
early_stop = None
key_seed = 0
n_epochs = 1
key = jax.random.PRNGKey(key_seed)


kira, init_state = eqx.nn.make_with_state(Kira)(
    n_dims=n_dims,
    n_embd=n_embd,
    num_heads=num_heads,
    max_seq_len=max_seq_len,
    key=key,
    n_layers=n_layers,
)

kira = eqx.tree_deserialise_leaves("kira-experiment1.eqx", kira)

ic("WITH STATE")
text_with_state = generate_text(
    kira,
    init_state,
    max_new_tokens,
    tinyshakespeare.decode,
    tinyshakespeare.vocab_size,
    print_to_console=True,
)
ic(text_with_state)
ic("WITHOUT STATE")
text_without_state = generate_text_without_kv_cache(
    kira,
    max_seq_len,
    max_new_tokens,
    tinyshakespeare.decode,
    vobab_size=n_dims,
    print_to_console=True,
)
ic(text_without_state)
