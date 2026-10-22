"""Tests for M7 60-task deterministic headless Xvfb + Tkinter UI eval suite (`specs/eval_suite.md`)."""

from __future__ import annotations

import json
import os

from neurobrain.eval_suite import build_eval_tasks, run_eval_suite


def test_suite_manifest_has_60_spec_compliant_tasks():
    tasks = build_eval_tasks()
    assert len(tasks) == 60
    ids = [t["id"] for t in tasks]
    assert len(set(ids)) == 60

    cats = {}
    for t in tasks:
        for req in ("id", "category", "instruction", "app", "setup", "success", "budget"):
            assert req in t, f"Task {t.get('id')} missing {req}"
        assert "check" in t["success"]
        assert "max_steps" in t["budget"] and "max_seconds" in t["budget"]
        cats[t["category"]] = cats.get(t["category"], 0) + 1

    assert set(cats.keys()) == {"file_manager", "editor", "settings", "browser", "recovery"}
    for c, count in cats.items():
        assert count == 12, f"Expected 12 tasks in {c}, got {count}"


def test_failure_strip_saved_when_task_fails(tmp_path):
    class AlwaysCornerGrounder:
        step = 0

        def ground(self, rgb_or_png, instruction: str, exclude_box=None):
            return {"x": 2, "y": 2, "fx": 2.0, "fy": 2.0, "action": "click", "score": 0.0}

    sb_path = str(tmp_path / "scoreboard.json")
    strips_dir = str(tmp_path / "strips")
    sb = run_eval_suite(
        checkpoint_dir=None,
        grounder_override=AlwaysCornerGrounder(),
        task_limit=2,
        out_path=sb_path,
        strips_dir=strips_dir,
    )
    assert os.path.exists(sb_path)
    assert sb["n"] == 2
    assert sb["success"] == 0.0
    assert len(sb["failures"]) == 2
    for f in sb["failures"]:
        assert "id" in f and "why" in f and "strip" in f
        assert os.path.exists(f["strip"])


def test_real_grounder_subset_runs_headless_xvfb(tmp_path):
    sb_path = str(tmp_path / "scoreboard_sub.json")
    strips_dir = str(tmp_path / "strips_sub")
    sb = run_eval_suite(
        checkpoint_dir="checkpoints/grip_v2",
        task_ids=["fm-001", "ed-001", "st-001", "br-001", "rc-001"],
        out_path=sb_path,
        strips_dir=strips_dir,
    )
    assert sb["n"] == 5
    assert sb["success"] >= 0.60
    with open(sb_path) as f:
        loaded = json.load(f)
    for k in ("commit", "model", "n", "success", "by_category", "median_steps", "median_seconds", "ram_peak_mb", "failures"):
        assert k in loaded
