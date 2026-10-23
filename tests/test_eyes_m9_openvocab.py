"""Tests for M9 Open-Vocabulary Desktop + SERP + Cross-Role UI generator & co-training."""

from __future__ import annotations

import random

from neurobrain.desktop_openvocab import (
    HELD_OUT_OOD_PHRASES,
    TRAIN_OPEN_WORDS,
    _sample_serp_quad,
    render_openvocab_example,
    sample_openvocab_desktop_example,
)


def test_hard_serp_lexical_and_digit_siblings():
    rng = random.Random(2026)
    shared_prefix_count = 0
    for _ in range(40):
        quad = _sample_serp_quad(rng, split="train")
        assert len(quad) == 4
        assert len(set(quad)) == 4
        first_words = [p.split()[0] for p in quad]
        if len(set(first_words)) == 1:
            shared_prefix_count += 1
    assert shared_prefix_count >= 5, "SERP sampler must produce same-prefix numeric sibling quads"


def test_strict_disjointness_between_train_and_heldout_phrases():
    train_set = set(w.lower() for w in TRAIN_OPEN_WORDS)
    for phrase in HELD_OUT_OOD_PHRASES:
        # Held-out multi-word phrases must not appear verbatim in train words
        assert phrase.lower() not in train_set


def test_openvocab_desktop_and_serp_rendering_invariants():
    rng = random.Random(1234)
    modes_seen = set()
    has_digits = False
    for _ in range(30):
        ex = sample_openvocab_desktop_example(rng, split="train")
        modes_seen.add(ex.layout.archetype)
        rgb = render_openvocab_example(ex, size=256)
        assert rgb.shape == (256, 256, 3)
        assert rgb.dtype.name == "uint8"
        assert 0 <= ex.target.cx <= 256 and 0 <= ex.target.cy <= 256
        assert ex.target.state != "disabled"
        if any(ch.isdigit() for ch in ex.target.label):
            has_digits = True

    assert "desktop_serp" in modes_seen
    assert "crowded_crossrole" in modes_seen
    assert has_digits, "Open-vocab generator must include numeric tokens (ports, codes, versions)"


def test_heldout_ood_phrase_split_renders_cleanly():
    rng = random.Random(99)
    ex = sample_openvocab_desktop_example(rng, split="ood_phrases")
    rgb = render_openvocab_example(ex, size=256)
    assert rgb.shape == (256, 256, 3)
    assert ex.target.label in HELD_OUT_OOD_PHRASES
