"""Data loading for character-level next-character prediction on Tiny Shakespeare.

Shared by all three models (bigram, LSTM, transformer) so they see
identical data, vocabulary and train/validation split.
"""
import math
import urllib.request
from pathlib import Path

import torch

DATA_PATH = Path(__file__).resolve().parent / "data" / "input.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TRAIN_FRACTION = 0.9
SEED = 1337

torch.manual_seed(SEED)

# CUDA on Colab, MPS on Apple silicon, CPU otherwise.
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# --- 1. Load text ---
if not DATA_PATH.exists():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
text = DATA_PATH.read_text(encoding="utf-8")

# --- 2. Vocabulary ---
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    """Convert a string to a list of integer token ids.

    Args:
        s (str): Input text; every character must be in the vocabulary.

    Returns:
        list[int]: Token ids.
    """
    return [stoi[c] for c in s]


def decode(ids):
    """Convert a list of integer token ids back to a string.

    Args:
        ids (list[int]): Token ids.

    Returns:
        str: Decoded text.
    """
    return "".join(itos[i] for i in ids)


# --- 3. Train/validation split (contiguous, not shuffled, to avoid leakage) ---
data = torch.tensor(encode(text), dtype=torch.long)
n = int(TRAIN_FRACTION * len(data))
train_data = data[:n]
val_data = data[n:]


def _split(split):
    if split == "train":
        return train_data
    if split == "val":
        return val_data
    raise ValueError(f"split must be 'train' or 'val', got {split!r}")


def get_batch(split, block_size, batch_size, generator=None):
    """Sample a random batch of input/target sequences.

    Targets are the inputs shifted one character to the right, so
    y[b, t] is the next character after x[b, t].

    Args:
        split      (str): "train" or "val".
        block_size (int): Context length (characters per sequence).
        batch_size (int): Number of sequences in the batch.
        generator  (torch.Generator, optional): CPU generator for the start
            indices. Passing one makes the batch sequence independent of any
            other random draws (e.g. dropout), so every model can be fed
            exactly the same batches.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: x, y, each (batch_size, block_size), on `device`.
    """
    d = _split(split)
    ix = torch.randint(len(d) - block_size, (batch_size,), generator=generator)
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def iter_windows(split, block_size, batch_size):
    """Yield every character of a split once, as non-overlapping windows.

    Used for the final reported losses, so they cover the whole split rather
    than a random sample. The last partial window is dropped.

    Yields:
        tuple[torch.Tensor, torch.Tensor]: x, y, each (<=batch_size, block_size), on `device`.
    """
    d = _split(split)
    n_windows = (len(d) - 1) // block_size
    x_all = d[:n_windows * block_size].view(n_windows, block_size)
    y_all = d[1:n_windows * block_size + 1].view(n_windows, block_size)
    for i in range(0, n_windows, batch_size):
        yield x_all[i:i + batch_size].to(device), y_all[i:i + batch_size].to(device)


def baselines():
    """Two reference losses (nats/char) that any useful model must beat.

    Returns:
        dict: "uniform" is ln(vocab_size), a guess that ignores the text.
        "unigram" is the cross-entropy on val of always predicting the
        training-set character frequencies, i.e. using no context at all.
    """
    counts = torch.bincount(train_data, minlength=vocab_size).float()
    log_p = torch.log(counts / counts.sum())
    return {
        "uniform": math.log(vocab_size),
        "unigram": -log_p[val_data].mean().item(),
    }


if __name__ == "__main__":
    # Sanity checks — run: python data.py
    print("device:", device)
    print("vocab_size:", vocab_size)
    print("train/val lengths:", len(train_data), len(val_data))
    print("val chars unseen in train:", len(set(val_data.tolist()) - set(train_data.tolist())))
    print("baselines:", {k: round(v, 4) for k, v in baselines().items()})
    assert decode(encode("hello")) == "hello"
    x, y = get_batch("train", block_size=8, batch_size=4)
    print("x[0]:", repr(decode(x[0].tolist())))
    print("y[0]:", repr(decode(y[0].tolist())))
