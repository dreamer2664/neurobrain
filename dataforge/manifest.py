"""Manifest reading/writing and the deterministic npz container.

The manifest is the contract consumers code against (specs/data_formats.md):

    {"version": "v1", "n": int, "shards": [{"name", "sha256", "bytes", "n"}],
     "generated_by": "dataforge 0.1.0", ...}

It deliberately contains **no timestamp**: two builds with the same seed must
produce byte-identical manifests, and that is the determinism proof.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from typing import Any, Iterator

import numpy as np

from . import __version__

GENERATOR = f"dataforge {__version__}"
MANIFEST = "manifest.json"
VERSION = "v1"


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_manifest(dirpath: str, payload: dict[str, Any]) -> str:
    os.makedirs(dirpath, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("version", VERSION)
    payload.setdefault("generated_by", GENERATOR)
    path = os.path.join(dirpath, MANIFEST)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    return path


def read_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dataset_dirs(root: str) -> Iterator[tuple[str, dict]]:
    """Yield (dir, manifest) for every manifest under root (root itself last)."""
    if not os.path.isdir(root):
        return
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        mf = os.path.join(d, MANIFEST)
        if os.path.isdir(d) and os.path.isfile(mf):
            yield d, read_manifest(mf)


# ----------------------------------------------------------------- npz, stable
def _zipinfo(name: str, date_time=(1980, 1, 1, 0, 0, 0), mode=0o644) -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(name, date_time=date_time)
    zi.compress_type = zipfile.ZIP_DEFLATED
    zi.external_attr = (mode & 0xFFFF) << 16
    zi.create_system = 3
    return zi


def write_npz(path: str, arrays: dict[str, np.ndarray], compresslevel: int = 1) -> None:
    """A byte-deterministic ``.npz``.

    ``numpy.savez`` embeds the current time in the zip headers, which breaks the
    "same seed => same sha256" rule. Here every entry gets a fixed timestamp and
    the entries are written in sorted name order.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, mode="w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=compresslevel, allowZip64=True) as zf:
        for name in sorted(arrays):
            arr = np.ascontiguousarray(arrays[name])
            zi = _zipinfo(f"{name}.npy")
            with zf.open(zi, mode="w", force_zip64=True) as fp:
                np.lib.format.write_array(fp, arr, allow_pickle=False)
    os.replace(tmp, path)


def read_npz_keys(path: str) -> dict[str, tuple[tuple[int, ...], str]]:
    """(shape, dtype) per array without decompressing the payload."""
    out: dict[str, tuple[tuple[int, ...], str]] = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith(".npy"):
                continue
            with zf.open(name) as fp:
                version = np.lib.format.read_magic(fp)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(fp)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(fp)
                else:  # pragma: no cover - very new numpy
                    shape, fortran, dtype = np.lib.format._read_array_header(fp, version)
            out[name[:-4]] = (tuple(shape), np.dtype(dtype).str)
    return out
