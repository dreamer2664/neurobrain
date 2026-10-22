"""Generator registry.

A generator is a module with ``build(dirpath, cfg, seed, log) -> dict`` that
streams its dataset to ``dirpath`` and returns the manifest body. Anything not
implemented yet is listed in ``TODO`` and skipped loudly by the CLI, so a
partial build is always an honest one.
"""

from __future__ import annotations

from . import code_corpus, drive_corpus, ui_protocol, ui_seed

IMPLEMENTED = {
    "code": code_corpus.build,
    "ui_protocol": ui_protocol.build,
    "drive": drive_corpus.build,
    "ui_seed": ui_seed.build,
}

TODO = {
    "hard_ui": "M1: real fonts 8-14px, toolbars/tabs/dialogs, DPI 100-200%, 5k/shard",
    "instructions": "M3: templates x slots x paraphrases, dedupe index",
    "trajectories": "M5: task_schema.jsonl with failure branches",
    "code_v2": "M6: AST filter, function-hash dedupe, categories, 4096 tokenizer",
}


def get(name: str):
    if name not in IMPLEMENTED:
        raise KeyError(f"{name}: not implemented yet ({TODO.get(name, 'unknown')})")
    return IMPLEMENTED[name]
