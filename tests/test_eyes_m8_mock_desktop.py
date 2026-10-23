"""Tests for M8 Mock Desktop OS, Offline Search Engine ("MiniGoogle"), and 40-task Desktop & Googling Suite."""

from __future__ import annotations

import json
import os

from neurobrain.mock_desktop import (
    OfflineSearchEngine,
    build_desktop_eval_tasks,
    run_desktop_suite,
)


def test_offline_search_engine_ranking_and_ad_trap():
    eng = OfflineSearchEngine()
    assert len(eng.documents) >= 48

    res = eng.search("postgres default port")
    assert len(res) == 4
    assert res[0].slug == "postgres 5432"
    assert "5432" in res[0].fact
    assert not res[0].is_ad

    res_ad = eng.search("postgres default port", include_ad_trap=True)
    assert len(res_ad) == 4
    assert res_ad[0].is_ad is True
    assert any(r.slug == "postgres 5432" and not r.is_ad for r in res_ad[1:])


def test_desktop_eval_tasks_manifest_has_40_tasks():
    tasks = build_desktop_eval_tasks()
    assert len(tasks) == 40
    ids = [t["id"] for t in tasks]
    assert len(set(ids)) == 40

    by_cat = {}
    for t in tasks:
        for k in ("id", "category", "instruction", "setup", "plan", "success", "budget"):
            assert k in t
        by_cat[t["category"]] = by_cat.get(t["category"], 0) + 1

    assert set(by_cat.keys()) == {
        "desktop_nav",
        "google_search",
        "research_workflow",
        "adversarial_desktop",
    }
    for cat, n in by_cat.items():
        assert n == 10, f"{cat} expected 10 tasks, got {n}"


def test_mock_desktop_headless_x11_smoke(tmp_path):
    out_json = str(tmp_path / "desktop_sb.json")
    strips = str(tmp_path / "strips")
    sb = run_desktop_suite(
        checkpoint_dir="checkpoints/grip_v2",
        task_ids=["dn-001", "gs-001", "rw-001", "ad-001"],
        out_path=out_json,
        strips_dir=strips,
    )
    assert os.path.exists(out_json)
    assert sb["n"] == 4
    loaded = json.load(open(out_json))
    assert "by_category" in loaded and "success" in loaded
