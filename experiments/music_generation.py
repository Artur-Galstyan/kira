import math
import pathlib
import pickle
import sys
from typing import Callable, Optional

import equinox as eqx
import jax
import jaxtyping as jt
import mido
import numpy as np
import pandas as pd
import torch.utils.data.dataset as dataset
from icecream import ic
from jaxtyping import Array
from tinyshakespeareloader.hamlet import get_data
from torch.utils.data.dataloader import DataLoader
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
    encode: Callable[[str], int] | None
    decode: Callable[[Array], str] | None

    def __init__(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        vocab_size: int | None,
        encode: Callable[[str], int] | None,
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
    midis_path: str = "music_data/vg_music",
    pickle_path: str = "data.pkl",
) -> GameMusicData:
    if pathlib.Path(pickle_path).exists():
        with open("data.pkl", "rb") as f:
            data = pickle.load(f)
    else:
        path_to_midis = pathlib.Path(midis_path)
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

    def encode(string: str) -> int:
        return char_to_idx[string]

    def decode(index: Array) -> str:
        return idx_to_char[index[0]]

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


def dequantize_velocity(velocity_token: str) -> int:
    if velocity_token == "OFF":
        return 0
    elif velocity_token == "PP":
        return np.random.randint(40)
    elif velocity_token == "P":
        return np.random.randint(40, 60)
    elif velocity_token == "MP":
        return np.random.randint(60, 80)
    elif velocity_token == "MF":
        return np.random.randint(80, 100)
    else:
        return 110


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


def dequantize_duration(duration_token: str) -> int:
    if duration_token == "16TH":
        return 1
    elif duration_token == "8TH":
        return 2
    elif duration_token == "QUARTER":
        return 4
    elif duration_token == "HALF":
        return 8
    else:
        return 16


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


def text_to_midi(notes: list[str]) -> list[tuple[int, int, int]]:
    ticks_per_beat = 480
    ticks_per_16th = ticks_per_beat / 4
    midi_representation = []

    for note in notes:
        if note.endswith("IMMEDIATE"):
            _, note, velocity_token, _ = note.split("_")
            velocity = dequantize_velocity(velocity_token)

            time = 0
        elif note.startswith("NOTE"):
            _, note, velocity_token, duration_token = note.split("_")
            velocity = dequantize_velocity(velocity_token)
            n_16th = dequantize_duration(duration_token)
            time = n_16th * ticks_per_16th
        else:
            _, duration_token = note.split("_")
            note = 0
            velocity = 0
            n_16th = dequantize_duration(duration_token)
            time = n_16th * ticks_per_16th

        midi_representation.append((int(note), int(velocity), int(time)))

    return midi_representation


def midi_representation_to_midi_file(
    midi_representation: list[tuple[int, int, int]], output_path: str
):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    for note, velocity, time in midi_representation:
        # print(note, velocity, time, type(note), type(velocity), type(time))
        track.append(mido.Message("note_on", note=note, velocity=velocity, time=time))

    mid.save(output_path)


max_seq_len = 128
batch_size = 128

print("Loading data...")
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

# wandb.init(
#     name="music-generation-experiment3",
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
#         "num_query_heads": num_query_heads,
#         "num_kv_heads": num_kv_heads,
#         "key_seed": key_seed,
#     },
# )
print("Done")
print("Building model...")
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

kira = eqx.tree_deserialise_leaves("../weights/kira-music-generation1.eqx", kira)

# for epoch in tqdm(range(n_epochs)):
#     key, subkey = jax.random.split(key)
#     kira = train(
#         train_dataloader,
#         test_dataloader,
#         learning_rate,
#         kira,
#         subkey,
#         early_stop=early_stop,
#         wandb_client=wandb,
#     )

# eqx.tree_serialise_leaves("kira-music-generation1.eqx", kira)
print("Done")

print("Generating text...")
decoded_tokens, tokens = generate_text(
    kira,
    max_seq_len,
    max_new_tokens,
    game_music_data.decode,
    game_music_data.vocab_size,
    print_to_console=False,
    random_key_seed=22,
)
print("Done")

print("Generating midi representation...")
midi_representation = text_to_midi(decoded_tokens)
print("Done")

print("Generating midi file...")
midi_representation_to_midi_file(midi_representation, "generated.mid")
print("Done")
