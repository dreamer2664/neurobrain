"""Generator registry.

A generator is a module with ``build(dirpath, cfg, seed, log) -> dict`` that
streams its dataset to ``dirpath`` and returns the manifest body. Anything not
implemented yet is listed in ``TODO`` and skipped loudly by the CLI, so a
partial build is always an honest one.
"""

from __future__ import annotations

from . import (code_corpus, code_v2, drive_corpus, hard_ui, instructions,
               trajectories, ui_protocol, ui_seed)

IMPLEMENTED = {
    "code": code_corpus.build,
    "ui_protocol": ui_protocol.build,
    "drive": drive_corpus.build,
    "ui_seed": ui_seed.build,
    "hard_ui": hard_ui.build,
    "instructions": instructions.build,
    "trajectories": trajectories.build,
    "code_v2": code_v2.build,
}

TODO = {
}


def get(name: str):
    if name not in IMPLEMENTED:
        raise KeyError(f"{name}: not implemented yet ({TODO.get(name, 'unknown')})")
    return IMPLEMENTED[name]
