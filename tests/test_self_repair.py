"""Tests for the self-repair loop. Uses a stub model (no torch checkpoint) so
the mechanics are verified deterministically; the real model runs in the CLI."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurobrain import self_repair as sr  # noqa: E402
from neurobrain.codebench import load_tasks, build as build_tasks, TASK_DIR, MANIFEST_PATH  # noqa: E402
from neurobrain.tokenizer import BPETokenizer  # noqa: E402

TOK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "tokenizer.json")


class Cfg:
    ctx = 256


class StubModel:
    """generate() returns prefix + the next scripted completion, token-exact."""

    def __init__(self, tok, scripted):
        self.tok = tok
        self.cfg = Cfg()
        self.scripted = list(scripted)
        self.calls = 0

    def generate(self, idx, max_new_tokens=64, temperature=1.0, top_k=1, eos_id=None):
        text = self.scripted[min(self.calls, len(self.scripted) - 1)]
        self.calls += 1
        new = self.tok.encode(text)[:max_new_tokens]
        return torch.tensor([idx[0].tolist() + new], dtype=torch.long)


def _tasks():
    if not os.path.exists(TASK_DIR):
        build_tasks()
    return load_tasks()


def test_build_repair_ctx_structure_and_budget():
    tok = BPETokenizer.load(TOK_PATH)
    ids = sr.build_repair_ctx(tok, "def add(a, b):" + chr(10), "    return a - b" + chr(10),
                              "AssertionError", "add")
    assert len(ids) <= sr.CTX_BUDGET
    text = tok.decode(ids)
    assert text.startswith(sr.TM + "make the tests pass for add()")
    assert sr.TRY + "1" in text and sr.TRY + "2" in text
    assert sr.ERR + "AssertionError" in text
    assert text.endswith(sr.code_marker(tok))       # generation continues right after it
    # a huge attempt-1 gets trimmed, but the error feedback survives
    big = "    x = 1" + chr(10) * 0 + "  # pad" * 400
    ids2 = sr.build_repair_ctx(tok, "def add(a, b):" + chr(10), big + chr(10), "boom", "add")
    assert len(ids2) <= sr.CTX_BUDGET
    assert "boom" in tok.decode(ids2)


def test_assemble_whole_def_vs_body():
    prompt = "def add(a, b):" + chr(10)
    whole = "def add(a, b):" + chr(10) + "    return a + b" + chr(10)
    assert sr.assemble(prompt, whole) == whole
    body_only = "    return a + b" + chr(10)
    assert sr.assemble(prompt, body_only) == prompt + body_only
    # markers in the generation are cut before assembly
    with_marker = "def add(a, b):" + chr(10) + "    return a + b" + chr(10) + sr.TRY + "3"
    assert sr.assemble(prompt, with_marker) == whole


def test_repair_task_fixes_wrong_body():
    tok = BPETokenizer.load(TOK_PATH)
    tasks = _tasks()
    t = next(x for x in tasks if x["fn"] == "add")
    nl = chr(10)
    wrong = "    return a - b" + nl
    right = "def add(a, b):" + nl + '    """return the sum of a and b"""' + nl + "    return a + b" + nl
    model = StubModel(tok, [wrong, right])
    tr = sr.repair_task(model, tok, t, temperature=0.8, top_k=40)
    assert tr["passed1"] is False
    assert tr["passed2"] is True, tr
    assert model.calls == 2
    assert "AssertionError" in tr["error"] or tr["error"]


def test_repair_task_first_pass_short_circuits():
    tok = BPETokenizer.load(TOK_PATH)
    tasks = _tasks()
    t = next(x for x in tasks if x["fn"] == "add")
    nl = chr(10)
    right_body = '    """return the sum of a and b"""' + nl + "    return a + b" + nl
    model = StubModel(tok, [right_body])
    tr = sr.repair_task(model, tok, t)
    assert tr["passed1"] is True
    assert model.calls == 1, "no second generation when the first passes"


def test_repair_task_reports_second_failure():
    tok = BPETokenizer.load(TOK_PATH)
    tasks = _tasks()
    t = next(x for x in tasks if x["fn"] == "add")
    model = StubModel(tok, ["    return a - b" + chr(10), "    return a * b" + chr(10)])
    tr = sr.repair_task(model, tok, t)
    assert tr["passed1"] is False and tr["passed2"] is False
    assert tr["why2"]
