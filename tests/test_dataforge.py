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


# --------------------------------------------------------------- M1: hard UI
def test_hard_ui_shards_and_tiers(tmp_path: Path):
    out = tmp_path / "forge"
    proc = run("build", "--out", str(out), "--profile", "smoke", "--seed", "5",
               "--datasets", "hard_ui", "--set", "hard_ui.samples=600")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    mf = json.loads((out / "hard_ui" / "manifest.json").read_text())
    assert mf["kind"] == "ui_shards" and mf["n"] == 600
    shard = out / "hard_ui" / mf["shards"][0]["name"]
    z = np.load(shard)
    for k in ["images", "tasks", "heat", "offsets", "actions", "labels", "roles", "themes"]:
        assert k in z.files, k
    off = z["offsets"]
    assert np.all(np.abs(off) <= 1.0)
    # the hard world must not saturate its regression target
    assert np.mean(np.abs(off) < 0.999) > 0.9, "offsets clip like the seed world"
    tiers = out / "hard_ui" / "tiers" / (shard.stem + "-tier.npy")
    assert tiers.exists() and np.load(tiers).shape == (600,)


def test_hard_ui_touches_many_heat_cells(tmp_path: Path):
    out = tmp_path / "forge"
    proc = run("build", "--out", str(out), "--profile", "smoke", "--seed", "5",
               "--datasets", "hard_ui", "--set", "hard_ui.samples=600")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    z = np.load(out / "hard_ui" / "shard-0000.npz")
    cells = z["heat"].astype(np.float32).reshape(600, -1).argmax(axis=1)
    assert len(set(cells.tolist())) > 150, "hard world still parks targets on a grid"


# ---------------------------------------------------------- M3: instructions
def test_instructions_records_and_dedupe(tmp_path: Path):
    import gzip
    out = tmp_path / "forge"
    proc = run("build", "--out", str(out), "--profile", "smoke", "--seed", "9",
               "--datasets", "instructions", "--set", "instructions.n=2000")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    mf = json.loads((out / "instructions" / "manifest.json").read_text())
    c = mf["content"]
    assert c["unique"] >= 2000
    assert c["reject_rate"] < 0.10
    with gzip.open(out / "instructions" / mf["records"][0]["name"], "rt") as f:
        rec = json.loads(f.readline())
    for key in ["text", "target", "family", "variant"]:
        assert key in rec
    assert rec["target"]["label"] and rec["target"]["role"]

    # every emitted instruction is unique (normalised)
    from neurobrain.instructions import normalize
    lines = []
    for sh in mf["records"]:
        with gzip.open(out / "instructions" / sh["name"], "rt") as f:
            lines += [json.loads(l)["text"] for l in f]
    norm = [normalize(t) for t in lines]
    assert len(set(norm)) == len(norm)


def test_instructions_include_the_hard_cases(tmp_path: Path):
    out = tmp_path / "forge"
    proc = run("build", "--out", str(out), "--profile", "smoke", "--seed", "9",
               "--datasets", "instructions", "--set", "instructions.n=4000")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    mf = json.loads((out / "instructions" / "manifest.json").read_text())
    fam = mf["content"]["family"]
    for family in ["multi", "relative", "type", "slider"]:
        assert fam.get(family, 0) > 0.01, f"{family} under-represented: {fam}"
    var = mf["content"]["variant"]
    assert var.get("typo", 0) > 0.05 and var.get("lower", 0) > 0.2


def test_build_set_override(tmp_path: Path):
    out = tmp_path / "forge"
    proc = run("build", "--out", str(out), "--profile", "smoke", "--seed", "3",
               "--datasets", "drive", "--set", "drive.episodes=120")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    mf = json.loads((out / "drive" / "manifest.json").read_text())
    assert mf["n"] == 120


# ------------------------------------------------------- M5: trajectories
def test_trajectories_schema_and_invariants(tmp_path: Path):
    import gzip
    out = tmp_path / "forge"
    proc = run("build", "--out", str(out), "--profile", "smoke", "--seed", "4",
               "--datasets", "trajectories", "--set", "trajectories.n=800")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    mf = json.loads((out / "trajectories" / "manifest.json").read_text())
    assert mf["kind"] == "jsonl" and mf["n"] == 800
    from neurobrain.trajectories import validate_trajectory, has_failure
    n = fails = 0
    ids = set()
    for sh in mf["shards"]:
        with gzip.open(out / "trajectories" / sh["name"], "rt") as f:
            for line in f:
                rec = json.loads(line)
                assert validate_trajectory(rec) == []
                assert rec["id"] not in ids
                ids.add(rec["id"])
                n += 1
                fails += has_failure(rec)
    assert n == 800
    assert fails / n >= 0.5


def test_verify_rejects_a_bad_trajectory(tmp_path: Path):
    import gzip
    out = tmp_path / "forge"
    assert run("build", "--out", str(out), "--profile", "smoke", "--seed", "4",
               "--datasets", "trajectories", "--set", "trajectories.n=50").returncode == 0
    mf = json.loads((out / "trajectories" / "manifest.json").read_text())
    p = out / "trajectories" / mf["shards"][0]["name"]
    with gzip.open(p, "rt") as f:
        lines = f.readlines()
    rec = json.loads(lines[0])
    rec["history"][0]["verdict"] = "brilliant"          # not in the enum
    lines[0] = json.dumps(rec) + "\n"
    with gzip.GzipFile(p, "wb", mtime=0) as f:
        f.write("".join(lines).encode())
    # hash no longer matches the manifest -> verify must fail either way
    proc = run("verify", "--dir", str(out))
    assert proc.returncode != 0


# ---------------------------------------------------------- M6: code corpus v2
def test_code_v2_ast_dedupe_and_tokenizers(tmp_path: Path):
    out = tmp_path / "forge"
    proc = run("build", "--out", str(out), "--profile", "smoke", "--seed", "8",
               "--datasets", "code_v2")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    mf = json.loads((out / "code_v2" / "manifest.json").read_text())
    c = mf["content"]
    assert c["functions"] > 200
    assert c["functions_val"] > 0 and c["functions_train"] > c["functions_val"]
    # the 4096 tokenizer must actually pack code tighter than the 2052 one
    assert c["bytes_per_token_4096"] > c["bytes_per_token_2052"] * 1.05
    assert (out / "code_v2" / c["tokenizer_4096"]["name"]).exists()
    names = {s["name"] for s in mf["shards"]}
    assert {"codev2_2052_train.bin", "codev2_4096_train.bin",
            "codev2_instr_train.bin"} <= names


def test_code_v2_no_function_crosses_the_split(tmp_path: Path):
    """A function hash may appear on exactly one side of the wall."""
    import ast as _ast
    import hashlib as _hashlib
    out = tmp_path / "forge"
    proc = run("build", "--out", str(out), "--profile", "smoke", "--seed", "8",
               "--datasets", "code_v2")
    assert proc.returncode == 0, proc.stdout + proc.stderr

    from neurobrain.tokenizer import BPETokenizer
    tok = BPETokenizer.load(str(ROOT / "data" / "tokenizer.json"))
    eot = tok.encode("<|endoftext|>")[0]

    def hashes_of(path):
        arr = np.fromfile(path, dtype="<u2")
        out_, cur = set(), []
        for t in arr:
            if t == eot:
                text = tok.decode(cur)
                try:
                    node = _ast.parse(text).body[0]
                    out_.add(_hashlib.sha1(_ast.dump(node, include_attributes=False)
                                           .encode()).hexdigest())
                except Exception:  # noqa: BLE001
                    pass
                cur = []
            else:
                cur.append(t)
        return out_

    tr = hashes_of(out / "code_v2" / "codev2_2052_train.bin")
    va = hashes_of(out / "code_v2" / "codev2_2052_val.bin")
    assert tr and va
    assert not (tr & va), f"{len(tr & va)} functions leaked across the split"


def test_code_tokenizer_roundtrips(tmp_path: Path):
    out = tmp_path / "forge"
    assert run("build", "--out", str(out), "--profile", "smoke", "--seed", "8",
               "--datasets", "code_v2").returncode == 0
    from neurobrain.tokenizer import BPETokenizer
    tok = BPETokenizer.load(str(out / "code_v2" / "code_tokenizer_4096.json"))
    assert tok.vocab_size == 4096 and tok.n_bpe == 4096 - 4
    s = "def f(x):\n    return x * 2\n"
    assert tok.decode(tok.encode(s)) == s


def test_terminal_mix_counts_impossible_objectives(tmp_path: Path):
    """'failed' must be counted as a terminal outcome, not as the word that
    appears in code retry text."""
    from dataforge import tokenscan
    from neurobrain.drive import drive_corpus
    from neurobrain.tokenizer import BPETokenizer

    tok = BPETokenizer.load(str(ROOT / "data" / "tokenizer.json"))
    eot = tok.encode("<|endoftext|>")[0]
    eps = list(drive_corpus(1500, seed=12))
    ids = []
    for e in eps:
        ids.extend(tok.encode(e))
        ids.append(eot)
    arr = np.asarray(ids, dtype="<u2")
    mix = tokenscan.scan(arr, tok)["terminal_mix"]
    raw = {}
    for e in eps:
        tail = e.rsplit("<|done|>", 1)[-1]
        k = tail.split(":")[0].strip().split()[0]
        raw[k] = raw.get(k, 0) + 1
    n = len(eps)
    for k, v in raw.items():
        assert abs(mix[k] - v / n) < 0.002, (k, mix[k], v / n)
    assert mix["failed"] > 0.01


# ------------------------------------------------- protocol: runtime dialect
def test_scan_line_matches_the_runtime(tmp_path: Path):
    """The corpus scan must be byte-identical in shape to what the runtime
    hands the model (brainbox/lm_drive.py: scan_from_obs). If these drift, the
    model learns a language nothing at runtime speaks."""
    import importlib.util
    import random as _random

    src = ROOT / "brainbox" / "lm_drive.py"
    if not src.exists():
        # brainbox lives on main; fetch it if this branch does not carry it
        got = subprocess.run(["git", "show", "origin/main:brainbox/lm_drive.py"],
                             cwd=str(ROOT), capture_output=True, text=True)
        if got.returncode != 0:
            pytest.skip("brainbox/lm_drive.py not available")
        (ROOT / "brainbox").mkdir(exist_ok=True)
        (ROOT / "brainbox" / "lm_drive.py").write_text(got.stdout)
        (ROOT / "brainbox" / "__init__.py").write_text("")
    from brainbox.lm_drive import scan_from_obs
    from neurobrain.drive import scan_line
    from neurobrain.ui_world import SCALE, make_layout

    rng = _random.Random(3)
    for _ in range(25):
        layout = make_layout(rng)
        obs = {"elements": [{"label": e.label, "role": e.role, "cx": e.cx, "cy": e.cy}
                            for e in layout.elements]}
        assert scan_line(layout) == scan_from_obs(obs), "scan formats diverged"


def test_every_terminal_is_one_the_runtime_parses():
    from brainbox.lm_drive import TERMINALS as RUNTIME_TERMINALS
    from neurobrain.drive import drive_corpus, SCAN_TERMINALS
    assert tuple(SCAN_TERMINALS) == tuple(RUNTIME_TERMINALS)
    for ep in drive_corpus(600, seed=21):
        tail = ep.rsplit("<|done|>", 1)[-1]
        word = tail.split(":")[0].strip().split()[0]
        assert word in RUNTIME_TERMINALS, (word, ep[-120:])


def test_desktop_episodes_carry_a_scan_before_acting():
    from neurobrain.drive import drive_corpus
    for ep in drive_corpus(600, seed=22):
        if ep.startswith("<|drive|>"):
            head = ep.split("<|step|>", 1)[0]
            assert "<|scan|>elements on screen: " in head, ep[:200]


def test_scan_coordinates_are_clickable():
    """Every coordinate the scan advertises must be a legal action coordinate."""
    import re
    import random as _random
    from neurobrain.drive import drive_corpus
    pat = re.compile(r"at (\d+),(\d+)")
    for ep in drive_corpus(300, seed=23):
        for x, y in pat.findall(ep.split("<|step|>", 1)[0]):
            assert 0 <= int(x) <= 1920 and 0 <= int(y) <= 1920
