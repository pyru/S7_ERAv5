"""Tokenize the corpus once, cache it, and serve batches."""
from __future__ import annotations

import os

import numpy as np
import torch

from .vocab import DATA, corpus_files, load_or_train, token_bytes


def build_stream(vocab_size: int) -> np.ndarray:
    cache = os.path.join(DATA, f"stream_{vocab_size}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    tok = load_or_train(vocab_size)
    ids: list[int] = []
    for path in corpus_files():
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        # encode in slices; the tokenizer is happier with bounded inputs
        for s in range(0, len(text), 200_000):
            ids.extend(tok.encode(text[s : s + 200_000]).ids)
    arr = np.asarray(ids, dtype=np.int32)
    np.save(cache, arr)
    return arr


def vocab_byte_list(vocab_size: int) -> list[bytes]:
    """Row v = the UTF-8 bytes of token id v, in id order."""
    tok = load_or_train(vocab_size)
    v = tok.get_vocab()
    out = [b""] * (max(v.values()) + 1)
    for surface, idx in v.items():
        out[idx] = token_bytes(surface)
    return out


class Batcher:
    def __init__(self, stream: np.ndarray, seq: int, batch: int, seed: int = 0,
                 val_frac: float = 0.02):
        n_val = int(len(stream) * val_frac)
        self.train = torch.from_numpy(stream[:-n_val].astype(np.int64))
        self.val = torch.from_numpy(stream[-n_val:].astype(np.int64))
        self.seq, self.batch = seq, batch
        self.seed = seed
        self.g = torch.Generator().manual_seed(seed)

    def reset(self):
        """Rewind the stream so every arm sees the SAME batches in the same
        order.  Without this the generator carries over from the previous arm
        and each arm trains on a different slice of the corpus, which quietly
        turns a controlled comparison into an uncontrolled one."""
        self.g = torch.Generator().manual_seed(self.seed)

    def _draw(self, src, batch=None, gen=None):
        b = batch or self.batch
        hi = len(src) - self.seq - 1
        i = torch.randint(0, hi, (b,), generator=gen or self.g)
        x = torch.stack([src[j : j + self.seq] for j in i])
        y = torch.stack([src[j + 1 : j + 1 + self.seq] for j in i])
        return x, y

    def train_batch(self):
        return self._draw(self.train)

    def val_batches(self, n: int, batch: int = 16):
        g = torch.Generator().manual_seed(1234)  # same val batches for every arm
        return [self._draw(self.val, batch, g) for _ in range(n)]
