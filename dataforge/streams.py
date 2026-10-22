"""Streaming writers: token streams (uint16 LE) and per-sample image shards.

Nothing here ever holds a corpus in memory. ``TokenStreamWriter`` appends
token ids to an open file and hashes as it goes; ``ShardWriter`` appends images
one at a time into a preallocated on-disk memmap, so peak RSS is one sample.

Both hand back the exact ``{"name", "sha256", "bytes", ...}`` record that goes
into ``manifest.json``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import numpy as np

from .manifest import write_npz


class TokenStreamWriter:
    """Append-only uint16 little-endian token stream with an incremental sha256."""

    def __init__(self, path: str, flush_tokens: int = 1 << 20):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.name = os.path.basename(path)
        self._f = open(path, "wb")
        self._h = hashlib.sha256()
        self._buf: list[int] = []
        self._flush_at = flush_tokens
        self.tokens = 0
        self.docs = 0
        self.bytes = 0
        self._closed = False

    def write(self, ids, doc: bool = True) -> None:
        self._buf.extend(ids)
        if doc:
            self.docs += 1
        if len(self._buf) >= self._flush_at:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        blob = np.asarray(self._buf, dtype="<u2").tobytes()
        self._f.write(blob)
        self._h.update(blob)
        self.tokens += len(self._buf)
        self.bytes += len(blob)
        self._buf.clear()

    def close(self) -> dict:
        if self._closed:
            return self.record()
        self.flush()
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()
        self._closed = True
        return self.record()

    def record(self) -> dict:
        return {
            "name": self.name,
            "sha256": self._h.hexdigest(),
            "bytes": self.bytes,
            "tokens": self.tokens,
            "docs": self.docs,
        }


@dataclass
class ShardAccumulator:
    """Collects per-sample arrays into a fixed-size shard, then writes one npz.

    Images are appended to a memmap on disk, so a 5k-sample shard costs one
    image of RSS, not 245 MB. The other arrays are small (tasks/heat/offsets/
    actions are < 12 MB per 5k samples) and stay in RAM.
    """

    dirpath: str
    shard_size: int
    index: int
    tmpdir: str
    _n: int = 0
    _images_path: str = ""
    _images: np.memmap | None = None
    tasks: list = field(default_factory=list)
    heat: list = field(default_factory=list)
    offsets: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    roles: list = field(default_factory=list)
    themes: list = field(default_factory=list)

    def __post_init__(self) -> None:
        os.makedirs(self.tmpdir, exist_ok=True)
        self._images_path = os.path.join(self.tmpdir, f"images-{self.index:04d}.u8")
        self._images = np.memmap(self._images_path, dtype=np.uint8, mode="w+",
                                 shape=(self.shard_size, 128, 128, 3))

    @property
    def full(self) -> bool:
        return self._n >= self.shard_size

    def add(self, image: np.ndarray, task_ids, heat, offset, action, label, role, theme) -> None:
        if self._n >= self.shard_size:
            raise RuntimeError("shard already full")
        self._images[self._n] = image
        self.tasks.append(np.asarray(task_ids, dtype=np.int64))
        self.heat.append(np.asarray(heat, dtype=np.float16))
        self.offsets.append(np.asarray(offset, dtype=np.float32))
        self.actions.append(int(action))
        self.labels.append(int(label))
        self.roles.append(int(role))
        self.themes.append(int(theme))
        self._n += 1

    def flush(self) -> dict | None:
        if self._n == 0:
            return None
        n = self._n
        self._images.flush()
        # Reading a slice of a memmap gives a plain ndarray view that np.lib
        # .format can stream into the zip without materialising the whole shard.
        images = self._images[:n]
        arrays = {
            "images": images,
            "tasks": np.stack(self.tasks),
            "heat": np.stack(self.heat),
            "offsets": np.stack(self.offsets).reshape(n, 2),
            "actions": np.asarray(self.actions, dtype=np.int64),
            "labels": np.asarray(self.labels, dtype=np.int64),
            "roles": np.asarray(self.roles, dtype=np.int64),
            "themes": np.asarray(self.themes, dtype=np.int64),
        }
        name = f"shard-{self.index:04d}.npz"
        path = os.path.join(self.dirpath, name)
        write_npz(path, arrays)
        from .manifest import sha256_file
        rec = {
            "name": name,
            "sha256": sha256_file(path),
            "bytes": os.path.getsize(path),
            "n": n,
        }
        self._reset()
        return rec

    def _reset(self) -> None:
        self._images._mmap.close()  # type: ignore[union-attr]
        try:
            os.unlink(self._images_path)
        except OSError:
            pass
        self.index += 1
        self._n = 0
        self.tasks, self.heat, self.offsets = [], [], []
        self.actions, self.labels, self.roles, self.themes = [], [], [], []
        self._images_path = os.path.join(self.tmpdir, f"images-{self.index:04d}.u8")
        self._images = np.memmap(self._images_path, dtype=np.uint8, mode="w+",
                                 shape=(self.shard_size, 128, 128, 3))

    def close(self) -> None:
        if self._images is not None and self._n == 0:
            self._images._mmap.close()  # type: ignore[union-attr]
            try:
                os.unlink(self._images_path)
            except OSError:
                pass
            self._images = None
