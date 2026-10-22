"""Tests for the dataforge CLI (agent 05).

Written before the implementation: every requirement M0 has to meet is pinned
here first. Run with `python -m pytest tests/test_dataforge.py -q`.

Contracts under test (specs/data_formats.md):
  * `build` writes a manifest.json with version/n/generated_by and a sha256 per
    shard; token streams are uint16 LE; UI shards are npz with the exact keys
    images/tasks/heat/offsets/actions.
  * `verify` recomputes every sha256 and re-checks the drive invariants from the
    bytes on disk, not from the build-time counters.
  * `stats` prints samples / tokens / duplicates / label balance / coordinate
    distribution / image entropy / retry counts / terminal rate, and fails the
    build when a number is out of range.
  * determinism: same seed + profile => identical hashes, twice.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataforge import drive_rules  # noqa: E402


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "dataforge", *args],
        cwd=str(ROOT), capture_output=True, text=True, timeout=1800,
    )


def build(out: Path, profile: str = "smoke", seed: int = 7, datasets: str | None = None):
    args = ["build", "--out", str(out), "--profile", profile, "--seed", str(seed)]
    if datasets:
        args += ["--datasets", datasets]
    return run(*args)


@pytest.fixture(scope="session")
def built(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("forge")
    proc = build(out, "smoke", 7, "drive,ui_protocol,ui_seed,code")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out


# ------------------------------------------------------------------ manifest
def test_build_creates_manifest_per_dataset(built: Path):
    for name in ["drive", "ui_protocol", "ui_seed", "code"]:
        mf = built / name / "manifest.json"
        assert mf.exists(), f"missing {mf}"
        m = json.loads(mf.read_text())
        assert m["version"] == "v1"
        assert m["generated_by"].startswith("dataforge")
        assert m["profile"] == "smoke"
        assert m["seed"] == 7
        assert isinstance(m["n"], int) and m["n"] > 0
        assert m["shards"], "manifest lists no shards"
        for sh in m["shards"]:
            assert len(sh["sha256"]) == 64
            assert (built / name / sh["name"]).exists()


def test_top_level_manifest_indexes_datasets(built: Path):
    idx = json.loads((built / "manifest.json").read_text())
    assert idx["version"] == "v1"
    assert set(idx["datasets"]) >= {"drive", "ui_protocol", "ui_seed", "code"}


def test_token_stream_is_uint16_le(built: Path):
    p = built / "drive" / "drive_train.bin"
    raw = p.read_bytes()
    assert len(raw) % 2 == 0
    toks = np.frombuffer(raw, dtype="<u2")
    assert len(toks) > 1000
    assert toks.max() < 1 << 16


def test_ui_shard_matches_spec(built: Path):
    shard = sorted((built / "ui_seed").glob("shard-*.npz"))[0]
    z = np.load(shard)
    for k in ["images", "tasks", "heat", "offsets", "actions"]:
        assert k in z.files, f"missing key {k}"
    assert z["images"].dtype == np.uint8
    assert z["images"].shape[1:] == (128, 128, 3)
    assert z["tasks"].dtype == np.int64 and z["tasks"].shape[1] == 48
    assert z["heat"].dtype == np.float16 and z["heat"].shape[1:] == (32, 32)
    assert z["offsets"].dtype == np.float32 and z["offsets"].shape[1] == 2
    assert z["actions"].dtype == np.int64
    n = z["images"].shape[0]
    assert n == 5000  # 5k per shard, per the brief
    for k in ["tasks", "heat", "offsets", "actions"]:
        assert z[k].shape[0] == n


# -------------------------------------------------------------------- verify
def test_verify_passes_on_fresh_build(built: Path):
    proc = run("verify", "--dir", str(built))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FAIL" not in proc.stdout


def test_verify_detects_bitrot(tmp_path: Path):
    out = tmp_path / "forge"
    assert build(out, "smoke", 7, "drive").returncode == 0
    p = out / "drive" / "drive_train.bin"
    raw = bytearray(p.read_bytes())
    raw[1234] ^= 0xFF
    p.write_bytes(bytes(raw))
    proc = run("verify", "--dir", str(out))
    assert proc.returncode != 0
    assert "sha256" in proc.stdout.lower()


def test_verify_detects_missing_terminal_token(tmp_path: Path):
    """A shard whose episodes stop early must be rejected by verify."""
    out = tmp_path / "forge"
    assert build(out, "smoke", 7, "drive").returncode == 0
    from neurobrain.tokenizer import BPETokenizer
    tok = BPETokenizer.load(str(ROOT / "data" / "tokenizer.json"))
    eot = tok.encode("<|endoftext|>")[0]
    bad = "<|drive|>objective: click save<|step|>1<|act|>click(1,2)<|obs|>no change"  # no <|done|>
    ids = tok.encode(bad) + [eot]
    p = out / "drive" / "drive_train.bin"
    p.write_bytes(np.asarray(ids, dtype="<u2").tobytes())
    proc = run("verify", "--dir", str(out))
    assert proc.returncode != 0
    assert "terminal" in proc.stdout.lower()


# --------------------------------------------------------------- determinism
def test_determinism_same_seed(tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        assert build(d, "smoke", 11, "ui_seed,drive").returncode == 0
    ha = {d.name: d.read_text() for d in [a / "drive" / "manifest.json", a / "ui_seed" / "manifest.json"]}
    hb = {d.name: d.read_text() for d in [b / "drive" / "manifest.json", b / "ui_seed" / "manifest.json"]}
    assert ha == hb, "same seed produced different manifests"


def test_determinism_different_seed_changes_hashes(tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert build(a, "smoke", 11, "drive").returncode == 0
    assert build(b, "smoke", 12, "drive").returncode == 0
    ma = json.loads((a / "drive" / "manifest.json").read_text())
    mb = json.loads((b / "drive" / "manifest.json").read_text())
    assert [s["sha256"] for s in ma["shards"]] != [s["sha256"] for s in mb["shards"]]


# --------------------------------------------------------------------- stats
def test_stats_reports_required_numbers(built: Path):
    proc = run("stats", "--dir", str(built))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    for key in ["samples", "tokens", "duplicates", "label balance",
                "coordinate", "entropy", "retry", "terminal_rate", "OK"]:
        assert key in out, f"stats output missing {key!r}"


def test_stats_json_is_machine_readable(built: Path, tmp_path: Path):
    p = tmp_path / "stats.json"
    proc = run("stats", "--dir", str(built), "--json", str(p))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    d = json.loads(p.read_text())
    assert d["drive"]["invariants"]["terminal_rate"] == 1.0
    assert d["drive"]["invariants"]["retry_after_failure_rate"] == 1.0
    assert d["drive"]["invariants"]["failure_fraction"] >= 0.5
    assert d["ui_seed"]["label_balance"]["n_labels"] >= 12
    assert 0.0 <= d["ui_seed"]["duplicates"]["exact_rate"] < 0.05


def test_stats_fails_on_broken_drive_corpus(tmp_path: Path):
    """An episode that ends between a failure and the next attempt = build fail."""
    out = tmp_path / "forge"
    assert build(out, "smoke", 7, "drive").returncode == 0
    from neurobrain.tokenizer import BPETokenizer
    tok = BPETokenizer.load(str(ROOT / "data" / "tokenizer.json"))
    eot = tok.encode("<|endoftext|>")[0]
    bad = ("<|drive|>objective: x<|step|>1<|act|>click(1,2)<|obs|>no change"
           "<|check|>objective not satisfied<|retry|>aim better<|done|>success")
    ids = tok.encode(bad) + [eot]
    (out / "drive" / "drive_train.bin").write_bytes(np.asarray(ids, dtype="<u2").tobytes())
    proc = run("stats", "--dir", str(out))
    assert proc.returncode != 0


# ---------------------------------------------------------------- drive rules
def test_drive_rules_classify_episode():
    ep = ("<|drive|>objective: click save<|step|>1<|act|>click(1,2)<|obs|>nothing"
          "<|check|>not satisfied<|retry|>try again<|step|>2<|act|>click(3,4)"
          "<|obs|>saved<|check|>satisfied<|done|>success")
    r = drive_rules.classify(ep)
    assert r["n_done"] == 1 and r["n_retry"] == 1 and r["has_failure"]
    assert r["retry_after_failure_ok"]
    assert r["kind"] == "ui"
    code = "<|task|>make the tests pass for add()<|try|>1<|run|>pytest<|err|>AssertionError<|retry|>patch<|try|>2<|ok|>1 passed<|done|>success"
    assert drive_rules.classify(code)["kind"] == "code"
    plan = "<|drive|>objective: tidy<|plan|>1.a 2.b <|step|>1<|act|>a()<|obs|>ok<|done|>success"
    assert drive_rules.classify(plan)["kind"] == "plan"


def test_drive_rules_flags_early_stop():
    ep = ("<|drive|>objective: x<|step|>1<|act|>click(1,2)<|obs|>no<|check|>no"
          "<|retry|>again<|done|>success")
    r = drive_rules.classify(ep)
    assert not r["retry_after_failure_ok"]


def test_build_rejects_out_of_range_profile(tmp_path: Path):
    proc = build(tmp_path / "x", "nonsense")
    assert proc.returncode != 0
    assert "profile" in (proc.stdout + proc.stderr).lower()


# ------------------------------------------- two ways of checking must agree
def test_token_scan_agrees_with_text_rules(tmp_path: Path):
    """The numpy token scan and the text classifier must return the same
    verdict for every episode -- otherwise `stats` and `build` disagree."""
    from dataforge import tokenscan
    from neurobrain.drive import drive_corpus
    from neurobrain.tokenizer import BPETokenizer

    tok = BPETokenizer.load(str(ROOT / "data" / "tokenizer.json"))
    eot = tok.encode("<|endoftext|>")[0]
    eps = list(drive_corpus(120, seed=5))
    ids = []
    for e in eps:
        ids.extend(tok.encode(e))
        ids.append(eot)
    arr = np.asarray(ids, dtype="<u2")
    scan = tokenscan.scan(arr, tok)
    rows = [drive_rules.classify(e) for e in eps]
    text = drive_rules.summarise(rows)
    assert scan["episodes"] == text["episodes"] == 120
    assert abs(scan["terminal_rate"] - text["terminal_rate"]) < 1e-9
    assert abs(scan["failure_fraction"] - text["failure_fraction"]) < 1e-9
    assert abs(scan["retry_after_failure_rate"] - text["retry_after_failure_rate"]) < 1e-9
    for k in ("ui", "code", "plan"):
        assert abs(scan["mix"][k] - text["mix"][k]) < 1e-9, (k, scan["mix"], text["mix"])


def test_verify_detects_corrupt_shard(tmp_path: Path):
    out = tmp_path / "forge"
    assert build(out, "smoke", 7, "ui_seed").returncode == 0
    shard = sorted((out / "ui_seed").glob("shard-*.npz"))[0]
    raw = bytearray(shard.read_bytes())
    raw[len(raw) // 2] ^= 0x5A
    shard.write_bytes(bytes(raw))
    proc = run("verify", "--dir", str(out))
    assert proc.returncode != 0
    assert "sha256" in proc.stdout.lower()


def test_diff_proves_determinism(tmp_path: Path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d in (a, b, c):
        assert build(d, "smoke", 31, "drive,ui_seed").returncode == 0
    assert run("diff", "--a", str(a), "--b", str(b)).returncode == 0
    # a third build with a different seed must differ
    assert build(c, "smoke", 32, "drive,ui_seed").returncode == 0
    assert run("diff", "--a", str(a), "--b", str(c)).returncode == 1
