"""The honesty calibration has to be a fair test, not a trick.

`impossible_objectives()` must (a) name an element that the scan that follows
does not contain, and (b) stop before any answer, so the model is the one that
decides between "impossible" and a fabricated success.
"""

from __future__ import annotations

import re

from neurobrain.eval_drive import impossible_objectives
from neurobrain.ui_world import LABELS


def test_prompts_stop_before_any_answer():
    for p in impossible_objectives(40, seed=5):
        assert p.startswith("<|drive|>objective: ")
        assert "<|scan|>" in p
        for marker in ("<|check|>", "<|done|>", "<|act|>", "<|step|>", "<|retry|>"):
            assert marker not in p, f"prompt leaks the answer: {marker}"


def test_requested_label_is_absent_from_the_scan():
    for p in impossible_objectives(40, seed=6):
        scan = p.split("<|scan|>elements on screen:", 1)[1]
        present = {s.strip() for s in scan.split(",")}
        objective = p.split("objective: ", 1)[1].split("<|scan|>", 1)[0]
        # every objective in the corpus names its label in exactly one place
        named = [l for l in LABELS if re.search(rf"\b{re.escape(l)}\b", objective)]
        assert len(named) == 1, objective
        assert named[0] not in present, f"{named[0]} is on screen, so this is doable: {p}"


def test_prompts_are_deterministic_per_seed():
    assert impossible_objectives(10, seed=9) == impossible_objectives(10, seed=9)
    assert impossible_objectives(10, seed=9) != impossible_objectives(10, seed=10)
