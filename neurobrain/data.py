"""Token data pipeline: memory-mapped uint16 token streams + mixed batching."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


class TokenSource:
    """A flat uint16 token stream on disk, sampled as random contiguous windows."""

    def __init__(self, path: str, ctx: int):
        self.path = path
        self.ctx = ctx
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        if len(self.tokens) < ctx + 2:
            raise ValueError(f"{path} too small: {len(self.tokens)} tokens")

    def __len__(self) -> int:
        return len(self.tokens)

    def sample(self, batch: int, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
        hi = len(self.tokens) - self.ctx - 2
        starts = [rng.randrange(0, hi) for _ in range(batch)]
        chunk = np.stack([self.tokens[s: s + self.ctx + 1] for s in starts]).astype(np.int64)
        x = torch.from_numpy(chunk[:, :-1])
        y = torch.from_numpy(chunk[:, 1:])
        return x, y


class Mixture:
    """Weighted mixture of token sources (e.g. 75% code, 25% desktop transcripts)."""

    def __init__(self, sources: dict[str, tuple[str, float]], ctx: int):
        self.ctx = ctx
        self.names = list(sources)
        self.weights = [sources[n][1] for n in self.names]
        self.sources = {n: TokenSource(sources[n][0], ctx) for n in self.names}

    def __len__(self) -> int:
        return sum(len(s) for s in self.sources.values())

    def sample(self, batch: int, rng: random.Random, source: str | None = None):
        name = source or rng.choices(self.names, weights=self.weights, k=1)[0]
        return name, *self.sources[name].sample(batch, rng)


def write_tokens(path: str, ids: list[int]) -> int:
    arr = np.array(ids, dtype=np.uint16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        arr.tofile(f)
    return len(arr)
