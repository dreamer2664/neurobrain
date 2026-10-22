"""dataforge -- agent 05's corpus factory.

One CLI, one deterministic contract:

    python -m dataforge build  --out DIR --profile {smoke,standard,max}
    python -m dataforge stats  --dir DIR [--json PATH]
    python -m dataforge verify --dir DIR
    python -m dataforge link   --dir DIR --to data

Design rules that the tests pin down:

* generation streams to disk, one sample at a time; no shard is held in RAM
* every dataset directory carries ``manifest.json`` with a sha256 per file
* same seed + profile => byte-identical shards, twice
* ``stats`` recomputes everything from the bytes on disk, never from the
  build-time counters, and exits non-zero when a number is out of range
* ``verify`` re-hashes every shard and re-checks the drive invariants
"""

__version__ = "0.1.0"
