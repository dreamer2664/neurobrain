"""Tests for codebench. Test first: the bench must be correct before any
model number coming out of it means anything."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurobrain import codebench as cb  # noqa: E402
from neurobrain.codebench_tasks import TASKS  # noqa: E402


def test_task_bank_shape():
    assert 40 <= len(TASKS) <= 80, "deliverable: 40-80 tasks"
    ids = [t["id"] for t in TASKS]
    fns = [t["fn"] for t in TASKS]
    assert len(set(ids)) == len(ids)
    assert len(set(fns)) == len(fns)
    for t in TASKS:
        assert t["prompt"].startswith("def ") and t["prompt"].endswith("\n")
        assert f"def {t['fn']}(" in t["prompt"]
        assert t["fn"] in t["tests"]
        assert t["reference"].strip()
        assert t["category"] and t["difficulty"]
        assert t["timeout_s"] >= 1


def test_build_pins_hashes(tmp_path):
    man = cb.build(task_dir=str(tmp_path / "tasks"), manifest_path=str(tmp_path / "manifest.json"))
    assert man["n"] == len(TASKS)
    files = sorted(os.listdir(tmp_path / "tasks"))
    assert len(files) == len(TASKS)
    # rebuild: identical content -> identical hashes (manifest is stable)
    man2 = cb.build(task_dir=str(tmp_path / "tasks"), manifest_path=str(tmp_path / "manifest.json"))
    assert [e["sha256"] for e in man["tasks"]] == [e["sha256"] for e in man2["tasks"]]
    loaded = cb.load_tasks(task_dir=str(tmp_path / "tasks"), manifest_path=str(tmp_path / "manifest.json"))
    assert len(loaded) == len(TASKS)
    # tamper -> hash pin must fail
    victim = tmp_path / "tasks" / files[0]
    victim.write_text(victim.read_text() + "\n# tampered\n")
    try:
        cb.load_tasks(task_dir=str(tmp_path / "tasks"), manifest_path=str(tmp_path / "manifest.json"))
        raise AssertionError("tampered task file must fail its sha256 pin")
    except ValueError:
        pass


def test_every_reference_passes():
    for t in TASKS:
        r = cb.run_candidate(t["prompt"], t["reference"], t["tests"], t["fn"], timeout_s=10)
        assert r["passed"], f"{t['id']}: reference failed: {r['why']}"
        assert r["defines"], f"{t['id']}: reference does not define fn"


def test_wrong_body_fails():
    t = next(t for t in TASKS if t["fn"] == "add")
    r = cb.run_candidate(t["prompt"], "    return a - b\n", t["tests"], "add")
    assert not r["passed"] and r["parses"] and r["defines"]


def test_syntax_error_detected():
    t = next(t for t in TASKS if t["fn"] == "add")
    r = cb.run_candidate(t["prompt"], "    return a +\n", t["tests"], "add")
    assert not r["passed"] and not r["parses"]
    assert r["why"].startswith("syntax:")


def test_timeout_enforced():
    t = next(t for t in TASKS if t["fn"] == "add")
    r = cb.run_candidate(t["prompt"], "    while True:\n        pass\n", t["tests"], "add",
                         timeout_s=1)
    assert not r["passed"] and r["why"] == "timeout"


def test_extract_completion_stops():
    p = "def add(a, b):\n"
    assert cb.extract_completion(p + "    return a + b\n\n\ndef sub(a, b):\n    pass\n", p) == "    return a + b\n"
    assert cb.extract_completion(p + "    return a + b\n<|fim_suffix|>rest", p) == "    return a + b\n"
    assert cb.extract_completion(p + "    return a + b\n<|drive|>objective: x", p) == "    return a + b\n"
    assert cb.extract_completion(p + "nonsense", p) == "nonsense\n"
    assert cb.extract_completion(p + "", p) == ""


def test_prompts_fit_context():
    tok_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "tokenizer.json")
    if not os.path.exists(tok_path):
        return
    from neurobrain.tokenizer import BPETokenizer
    tok = BPETokenizer.load(tok_path)
    for t in TASKS:
        n = len(tok.encode(t["prompt"]))
        assert n + 128 <= 256, f"{t['id']}: prompt {n} tokens leaves no room to generate"
