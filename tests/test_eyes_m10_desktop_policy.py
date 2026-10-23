"""Tests for M10 Autonomous From-Scratch `DesktopBrainPolicy` + Unscripted X11 Goal Suite."""

from __future__ import annotations

import numpy as np
import torch

from neurobrain.desktop_policy import (
    MAX_GOAL_CHARS,
    PRIMITIVE_OPS,
    DesktopBrainPolicy,
    build_unscripted_goal_tasks,
    encode_goal_chars,
)


def test_desktop_brain_policy_forward_and_param_budget():
    model = DesktopBrainPolicy()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 1_000_000, f"Policy exceeded 1.0M param budget: {n_params}"

    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    goal = "google postgres 5432 and save the fact in notes"
    decision = model.act(rgb, goal, history_ops=[2, 4], last_changed=1)

    assert decision["op_name"] in PRIMITIVE_OPS
    assert 0 <= decision["op_id"] < len(PRIMITIVE_OPS)
    assert isinstance(decision["query_text"], str)
    assert isinstance(decision["target_text"], str)


def test_goal_char_encoding_and_mask_alignment():
    goal = "search minigoogle for sqlite wal and bookmark it"
    ids = encode_goal_chars(goal)
    assert len(ids) == MAX_GOAL_CHARS
    assert all(isinstance(x, int) for x in ids)


def test_unscripted_goal_suite_has_30_tasks_with_seen_and_ood_splits():
    tasks = build_unscripted_goal_tasks()
    assert len(tasks) == 30
    splits = {t.split for t in tasks}
    assert splits == {"seen_phrasing", "ood_phrasing"}
    assert sum(1 for t in tasks if t.split == "ood_phrasing") == 10
