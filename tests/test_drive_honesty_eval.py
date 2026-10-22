"""The honesty calibration has to be a fair test, not a trick.

`impossible_objectives()` must (a) name an element that the scan that follows
does not contain, and (b) stop before any answer, so the model is the one that
decides between "impossible" and a fabricated success.
"""

from __future__ import annotations

import re

from neurobrain.eval_drive import impossible_objectives
from neurobrain.ui_world import LABELS, ROLES


def test_prompts_stop_before_any_answer():
    for p in impossible_objectives(40, seed=5):
        assert p.startswith("<|drive|>objective: ")
        assert "<|scan|>" in p
        for marker in ("<|check|>", "<|done|>", "<|act|>", "<|step|>", "<|retry|>"):
            assert marker not in p, f"prompt leaks the answer: {marker}"


def _scan_entries(prompt: str) -> list[tuple[str, str, int, int]]:
    scan = prompt.split("<|scan|>elements on screen:", 1)[1]
    out = []
    for item in scan.split(";"):
        label, role, _, xy = item.strip().split(" ")
        x, y = xy.split(",")
        out.append((label, role, int(x), int(y)))
    return out


def test_scan_carries_coordinates_the_model_can_copy():
    for p in impossible_objectives(20, seed=8):
        entries = _scan_entries(p)
        assert entries
        for label, role, x, y in entries:
            assert 0 <= x <= 1920 and 0 <= y <= 1920, (x, y)
            assert role in ROLES
            assert label in LABELS


def test_requested_label_is_absent_from_the_scan():
    for p in impossible_objectives(40, seed=6):
        present = {e[0] for e in _scan_entries(p)}
        objective = p.split("objective: ", 1)[1].split("<|scan|>", 1)[0]
        # every objective in the corpus names its label in exactly one place
        named = [l for l in LABELS if re.search(rf"\b{re.escape(l)}\b", objective)]
        assert len(named) == 1, objective
        assert named[0] not in present, f"{named[0]} is on screen, so this is doable: {p}"


def test_prompts_are_deterministic_per_seed():
    assert impossible_objectives(10, seed=9) == impossible_objectives(10, seed=9)
    assert impossible_objectives(10, seed=9) != impossible_objectives(10, seed=10)
