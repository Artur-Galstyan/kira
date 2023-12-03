import math
import sys
from typing import Callable, Optional

import pathlib
import jaxtyping as jt
import mido
import numpy as np
import torch.utils.data.dataset as dataset
from jaxtyping import Array
from torch.utils.data.dataloader import DataLoader
import pickle
from tqdm import tqdm
import pandas as pd
import equinox as eqx
import jax
from icecream import ic
from tinyshakespeareloader.hamlet import get_data
from tqdm import tqdm

import wandb
from kira.generate import generate_text
from kira.model.model import Kira
from kira.train import train


class MidiDataset(dataset.Dataset):
    def __init__(self, max_seq_len: int, data: np.ndarray):
        self.max_seq_len = max_seq_len
        self.data = data

    def __getitem__(self, index):
        if index == -1:
            index = len(self.data) - 1
        x = self.data[index : index + self.max_seq_len]
        y = self.data[index + 1 : index + self.max_seq_len + 1]

        if index + self.max_seq_len + 1 > len(self.data):
            diff = index + self.max_seq_len + 1 - len(self.data)

            to_add_on_x = diff - 1
            to_add_on_y = diff

            x = np.concatenate((x, self.data[:to_add_on_x]))
            y = np.concatenate((y, self.data[:to_add_on_y]))

        return x, y

    def __len__(self):
        return len(self.data)


class GameMusicData:
    train_dataloader: DataLoader
    test_dataloader: DataLoader
    vocab_size: int | None
    encode: Callable[[str], Array] | None
    decode: Callable[[Array], str] | None

    def __init__(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        vocab_size: int | None,
        encode: Callable[[str], Array] | None,
        decode: Callable[[Array], str] | None,
    ):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.vocab_size = vocab_size
        self.encode = encode
        self.decode = decode


def get_data(
    batch_size: int,
    max_seq_len: int,
    shuffle: bool,
    train_ratio=0.9,
) -> GameMusicData:
    if pathlib.Path("data.pkl").exists():
        with open("data.pkl", "rb") as f:
            data = pickle.load(f)
    else:
        path_to_midis = pathlib.Path("music_data/vg_music")
        midis = list(path_to_midis.glob("*.mid"))
        data = []
        for midi in tqdm(midis):
            try:
                d = midi_to_text(midi)
                data.extend(d)
            except Exception as e:
                print(e, file=sys.stderr)
                continue

        with open("data.pkl", "wb") as f:
            pickle.dump(data, f)

    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size}")
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(string: str) -> Array:
        return char_to_idx[string]

    def decode(index: Array) -> str:
        return idx_to_char[index]

    encoder = encode
    decoder = decode

    data = np.array([encoder(c) for c in data])

    n = int(train_ratio * len(data))

    train_data = data[:n]
    test_data = data[n:]

    train_dataset = MidiDataset(max_seq_len, train_data)
    test_dataset = MidiDataset(max_seq_len, test_data)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return GameMusicData(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        vocab_size=vocab_size,
        encode=encoder,
        decode=decoder,
    )


def quantize_velocity(velocity: int) -> str:
    if velocity == 0:
        return "OFF"
    elif velocity < 40:
        return "PP"
    elif velocity < 60:
        return "P"
    elif velocity < 80:
        return "MP"
    elif velocity < 100:
        return "MF"
    else:
        return "F"


def quantize_duration(n_16th: int) -> str:
    if n_16th <= 1:
        return "16TH"
    elif n_16th <= 2:
        return "8TH"
    elif n_16th <= 4:
        return "QUARTER"
    elif n_16th <= 8:
        return "HALF"
    else:
        return "WHOLE"


def midi_to_text(midifile) -> list[str]:
    mid = mido.MidiFile(midifile)
    text_representation = []
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_16th = ticks_per_beat / 4
    max_time_duration = 480

    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on":
                note, velocity, time = msg.note, msg.velocity, msg.time
                note = min(note, 127)
                velocity = min(velocity, 127)
                time = min(time, max_time_duration)
                n_16th = math.ceil(time / ticks_per_16th)

                velocity_token = quantize_velocity(velocity)
                duration_token = quantize_duration(n_16th)

                if time == 0 and velocity > 0:
                    token = f"NOTE_{note}_{velocity_token}_IMMEDIATE"
                elif velocity > 0:
                    token = f"NOTE_{note}_{velocity_token}_{duration_token}"
                else:
                    token = f"REST_{duration_token}"
                text_representation.append(token)
    return text_representation


max_seq_len = 128
batch_size = 128
game_music_data = get_data(batch_size=batch_size, max_seq_len=max_seq_len, shuffle=True)


train_dataloader, test_dataloader = (
    game_music_data.train_dataloader,
    game_music_data.test_dataloader,
)

n_dims = game_music_data.vocab_size
n_embd = 384
learning_rate = 3e-4
num_heads = 6
num_query_heads = 6
num_kv_heads = 3
n_layers = 6
max_new_tokens = 2000
early_stop = None
key_seed = 0
n_epochs = 1
key = jax.random.PRNGKey(key_seed)

wandb.init(
    name="music-generation-experiment3",
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

eqx.tree_serialise_leaves("kira-music-generation1.eqx", kira)
