"""Code token streams: the source the language brain reads.

v1 (this file): every readable stdlib ``.py`` file under ``sysconfig`` paths,
content-hash deduplicated, split train/val by file, streamed through the
2052-token BPE one file at a time.

v2 (M6) adds the AST filter, function-level dedupe, category tags and the
4096 code-only tokenizer. The file list here is deliberately simple so the
v1 stream stays comparable with what agents 04 already train on.
"""

from __future__ import annotations

import hashlib
import os
import random
import sysconfig

from .. import manifest as M
from ..profiles import sub_seed
from ..streams import TokenStreamWriter


def read_py(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            src = f.read()
    except (OSError, UnicodeDecodeError):
        return None
    if "\x00" in src or len(src) < 64:
        return None
    return src


def iter_candidates(roots: list[str], min_bytes: int = 400) -> list[tuple[str, str, int]]:
    """(root_name, relpath, size) for every plausible source file, path-sorted."""
    out = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dp, dn, fn in os.walk(root):
            dn[:] = sorted(d for d in dn if not d.startswith(".") and d != "__pycache__")
            for f in sorted(fn):
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dp, f)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    continue
                if sz < min_bytes:
                    continue
                out.append((os.path.dirname(root.rstrip("/")) and root, os.path.relpath(p, root), sz))
    out.sort(key=lambda r: (r[0], r[1]))
    return out


def build(dirpath: str, cfg: dict, seed: int, tok, log=print) -> dict:
    roots = [sysconfig.get_paths()[k] for k in ("stdlib", "purelib") if k in sysconfig.get_paths()]
    cands = iter_candidates(roots, cfg.get("min_bytes", 400))
    log(f"[code] {len(cands)} candidate files under {len(roots)} roots")

    rng = random.Random(sub_seed(seed, "code"))
    order = list(range(len(cands)))
    rng.shuffle(order)

    # phase 1: take files in the shuffled order until the byte budget is met
    # (sizes only -- hashing every candidate would read half a gigabyte)
    picked: list[tuple[str, str, int]] = []
    total = 0
    for i in order:
        root, rel, sz = cands[i]
        if sz > cfg["max_bytes"]:
            continue
        if total + sz > cfg["max_bytes"]:
            break
        picked.append((root, rel, sz))
        total += sz
    # phase 2: content-hash dedupe of the picked files
    seen: set[str] = set()
    files: list[tuple[str, str, int]] = []
    total = 0
    for root, rel, sz in picked:
        try:
            with open(os.path.join(root, rel), "rb") as f:
                h = hashlib.sha1(f.read()).hexdigest()
        except OSError:
            continue
        if h in seen:
            continue
        seen.add(h)
        files.append((root, rel, sz))
        total += sz

    n_val = max(4, int(cfg["val_frac"] * len(files)))
    val = files[:n_val]
    train = files[n_val:]
    log(f"[code] {len(files)} unique files, {total/1e6:.1f} MB, "
        f"{len(train)} train / {len(val)} val")

    eot = tok.encode("<|endoftext|>")[0]
    shards = []
    for split, part in (("train", train), ("val", val)):
        w = TokenStreamWriter(os.path.join(dirpath, f"code_{split}.bin"))
        for root, rel, _ in part:
            src = read_py(os.path.join(root, rel))
            if not src:
                continue
            w.write(tok.encode(src) + [eot])
        rec = w.close()
        log(f"[code] {rec['name']}: {rec['docs']} docs, {rec['tokens']/1e6:.2f}M tokens, "
            f"{rec['bytes']/1e6:.1f} MB")
        shards.append(rec)

    return {
        "kind": "token_streams",
        "dataset": "code",
        "n": sum(s["docs"] for s in shards),
        "units": "docs",
        "shards": shards,
        "content": {
            "source": "stdlib+purelib",
            "files": len(files),
            "bytes_read": total,
            "val_docs": len(val),
            "dedupe": "sha1(file bytes)",
        },
    }
