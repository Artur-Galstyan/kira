import pathlib

import equinox as eqx
import fire
import jax
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb
from kira import Kira
from kira.train import train


def training(
    batch_size: int = 128,
    n_epochs: int = 10,
    n_embd: int = 512,
    num_heads: int = 8,
    n_layers: int = 8,
    max_seq_len: int = 128,
    learning_rate: float = 3e-4,
    max_new_tokens: int = 200,
    early_stop: int = None,
    num_query_heads: int = 8,
    num_kv_heads: int = 8,
    key_seed: int = 0,
    experiment: str = "experiment1",
    test_train_split: float = 0.8,
):
    max_seq_len = 128
    batch_size = 128

    path_to_chess_data = (
        pathlib.Path(__file__).parent.absolute() / "chess_data/games.csv"
    )
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

    train_dataset = ChessDataset(
        all_moves[: int(len(all_moves) * test_train_split)], max_seq_len=max_seq_len
    )
    test_dataset = ChessDataset(
        all_moves[int(len(all_moves) * test_train_split) :], max_seq_len=max_seq_len
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    n_dims = vocab_size
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
        eqx.tree_serialise_leaves(
            weights_path / f"kira-chess-weights-{epoch}.eqx", kira
        )
        wandb.save(weights_path / f"kira-chess-weights-{epoch}.eqx")


if __name__ == "__main__":
    fire.Fire(training)
