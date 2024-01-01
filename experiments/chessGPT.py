import pathlib

import equinox as eqx
import jax
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb
from kira import Kira
from kira.train import train

max_seq_len = 8
batch_size = 4


path_to_chess_data = pathlib.Path(__file__).parent.absolute() / "chess_data/games.csv"
chess_data = pd.read_csv(path_to_chess_data)

game_start_token = "<START>"
game_end_token = "<END>"
all_moves = []

for index, row in chess_data.iterrows():
    moves = row["moves"].split(" ")
    moves.insert(0, game_start_token)
    moves.append(game_end_token)

    all_moves.extend(moves)

unique_moves = list(set(all_moves))
unique_moves.sort()

move_to_index = {}
index_to_move = {}

for i, move in enumerate(unique_moves):
    move_to_index[move] = i
    index_to_move[i] = move


logger.info(f"Number of unique moves: {len(unique_moves)}")
vocab_size = len(unique_moves)


class ChessDataset(Dataset):
    def __init__(self, moves, max_seq_len=128):
        self.moves = moves
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.moves) - self.max_seq_len

    def __getitem__(self, idx):
        return (
            torch.tensor(
                [
                    move_to_index[move]
                    for move in self.moves[idx : idx + self.max_seq_len]
                ]
            ),
            torch.tensor(
                [
                    move_to_index[move]
                    for move in self.moves[idx + 1 : idx + self.max_seq_len + 1]
                ]
            ),
        )


test_train_split = 0.8
train_dataset = ChessDataset(
    all_moves[: int(len(all_moves) * test_train_split)], max_seq_len=max_seq_len
)
test_dataset = ChessDataset(
    all_moves[int(len(all_moves) * test_train_split) :], max_seq_len=max_seq_len
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
logger.debug(chess_data["moves"][0])
x, y = next(iter(train_dataloader))
logger.info(f"x shape: {x.shape}")
logger.info(f"y shape: {y.shape}")

logger.debug(x[0])
logger.debug(y[0])

decoded_x = [index_to_move[int(i)] for i in x[0]]
decoded_y = [index_to_move[int(i)] for i in y[0]]

logger.debug(decoded_x)
logger.debug(decoded_y)

n_dims = vocab_size
n_embd = 384
learning_rate = 3e-4
num_heads = 6
num_query_heads = 6
num_kv_heads = 6
n_layers = 6
max_new_tokens = 200
early_stop = None
key_seed = 0
n_epochs = 1
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

weights_path = pathlib.Path(__file__).parent.parent.absolute() / "weights"
kira = eqx.tree_deserialise_leaves(weights_path / "kira-chess-weights.eqx", kira)
