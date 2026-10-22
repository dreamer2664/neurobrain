"""The three drive invariants, expressed once, checked two ways.

    I1  every episode ends in exactly one terminal token
    I2  no episode stops between a failure and the next attempt
    I3  >= 50% of episodes contain at least one failure

``classify`` works on decoded text (used while generating, where the text is
already in hand). ``tokenscan`` works on the uint16 array (used by stats and
verify, which must judge the bytes on disk, not the generator's bookkeeping).
``test_token_scan_agrees_with_text_rules`` pins the two to each other.
"""

from __future__ import annotations

TERMINAL = "<|done|>"
FAILURE = "<|retry|>"
ERR = "<|err|>"
# a marker that counts as "the next attempt" after a failure
ATTEMPT = ("<|act|>", "<|try|>", "<|step|>")

MARKERS = {
    "drive": "<|drive|>", "task": "<|task|>", "plan": "<|plan|>",
    "step": "<|step|>", "act": "<|act|>", "obs": "<|obs|>", "check": "<|check|>",
    "retry": "<|retry|>", "done": "<|done|>", "try": "<|try|>", "err": "<|err|>",
    "ok": "<|ok|>", "code": "<|code|>", "run": "<|run|>", "think": "<|think|>",
    "scan": "<|scan|>", "budget": "<|budget|>",
    "endoftext": "<|endoftext|>",
}

# The runtime's vocabulary (brainbox/lm_drive.py: TERMINALS). Anything else after
# <|done|> is a protocol violation, not a style choice.
OUTCOMES = ("success", "failed", "blocked", "impossible")


def classify(text: str) -> dict:
    """Per-episode verdict from its text. Keys are stable: trainers log them."""
    n_done = text.count(TERMINAL)
    retry_pos = []
    i = text.find(FAILURE)
    while i >= 0:
        retry_pos.append(i)
        i = text.find(FAILURE, i + len(FAILURE))

    retry_ok = True
    for p in retry_pos:
        tail = text[p + len(FAILURE):]
        end = tail.find("<|done|>")
        seg = tail if end < 0 else tail[:end]
        if not any(a in seg for a in ATTEMPT):
            retry_ok = False
            break

    if text.startswith("<|task|>"):
        kind = "code"
    elif "<|plan|>" in text:
        kind = "plan"
    else:
        kind = "ui"

    return {
        "kind": kind,
        "n_done": n_done,
        "n_retry": len(retry_pos),
        "n_act": sum(text.count(a) for a in ATTEMPT),
        "has_failure": bool(retry_pos) or ERR in text,
        "retry_after_failure_ok": retry_ok,
        "terminal_ok": n_done == 1,
        "chars": len(text),
    }


def summarise(rows: list[dict]) -> dict:
    n = len(rows) or 1
    kinds = {"ui": 0, "code": 0, "plan": 0}
    for r in rows:
        kinds[r["kind"]] += 1
    return {
        "episodes": len(rows),
        "terminal_rate": round(sum(r["n_done"] == 1 for r in rows) / n, 6),
        "retry_after_failure_rate": round(sum(r["retry_after_failure_ok"] for r in rows) / n, 6),
        "failure_fraction": round(sum(r["has_failure"] for r in rows) / n, 6),
        "mix": {k: round(v / n, 4) for k, v in kinds.items()},
        "mean_retries": round(sum(r["n_retry"] for r in rows) / n, 3),
        "mean_chars": round(sum(r["chars"] for r in rows) / n, 1),
        "failures": {
            "terminal": [i for i, r in enumerate(rows) if r["n_done"] != 1][:5],
            "retry_gap": [i for i, r in enumerate(rows) if not r["retry_after_failure_ok"]][:5],
        },
    }


def check(stats: dict) -> list[str]:
    """Return a list of violated invariants (empty == healthy)."""
    bad = []
    if stats["terminal_rate"] < 1.0:
        bad.append(f"I1 terminal_rate={stats['terminal_rate']} != 1.0 "
                   f"(episodes {stats['failures']['terminal']})")
    if stats["retry_after_failure_rate"] < 1.0:
        bad.append(f"I2 retry_after_failure_rate={stats['retry_after_failure_rate']} != 1.0 "
                   f"(episodes {stats['failures']['retry_gap']})")
    if stats["failure_fraction"] < 0.5:
        bad.append(f"I3 failure_fraction={stats['failure_fraction']} < 0.5")
    return bad
