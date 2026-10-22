"""The LM as the policy, the runtime as the judge.

The text corpus is the model's dialect and the desktop runtime is the only thing
that decides whether an episode actually worked. This module is the joint:

    objective + element scan  ->  model emits one action  ->  runtime executes
    ->  runtime writes the observation and the check  ->  repeat

Nothing the model *says* is taken as evidence. A ``<|done|>success`` that the
world does not confirm is recorded as an overclaim, the loop appends the real
check ("objective not satisfied") and the model gets another step. That split
is the whole point of training on honest terminals: the model can only be
honest about what it can measure, and the runtime is what measures it.

Coordinates: the text protocol uses ``SCALE`` multiplied canvas coordinates
(0..1920), the same space the scan prints and the corpus clicks use. The world
works in canvas pixels, so the driver divides on the way in.
"""

from __future__ import annotations

import re

from neurobrain.ui_world import SCALE

TERMINALS = ("success", "failed", "blocked", "impossible")
_ACT = re.compile(r"<\|act\|>\s*([a-z_]+)\((\d+)\s*,\s*(\d+)\)")
_DONE = re.compile(r"<\|done\|>\s*([a-z]+)")


def scan_from_obs(obs: dict) -> str:
    """The element inventory the runtime hands the model, coordinates included."""
    items = sorted(obs.get("elements", []), key=lambda e: e.get("label", ""))
    if not items:
        return "<|scan|>elements on screen: none"
    parts = []
    for e in items:
        x = int(round(float(e.get("cx", 0)) * SCALE))
        y = int(round(float(e.get("cy", 0)) * SCALE))
        parts.append(f"{e['label']} {e['role']} at {x},{y}")
    return "<|scan|>elements on screen: " + "; ".join(parts)


def parse_action(text: str) -> dict | None:
    m = _ACT.search(text)
    if not m:
        return None
    kind, x, y = m.group(1), int(m.group(2)), int(m.group(3))
    return {"type": "click" if kind != "type" else "type",
            "point": (x / SCALE, y / SCALE), "raw": m.group(0), "protocol_point": (x, y)}


def parse_terminal(text: str) -> str | None:
    m = _DONE.search(text)
    if not m:
        return None
    word = m.group(1)
    return word if word in TERMINALS else None


class LMDriver:
    """One task, step by step, with the runtime owning observation and verdict."""

    TAIL_CHARS = 600   # ~150 tokens of recent history; the head is never dropped

    def __init__(self, host, max_steps: int = 8, max_tokens: int = 40,
                 temperature: float = 0.4, top_k: int = 30, seed: int | None = 7):
        self.host = host
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.seed = seed

    def run(self, world, objective: str, feasible: bool = True) -> dict:
        scan = scan_from_obs(world.observe())
        head = f"<|drive|>objective: {objective}" + scan
        tail = ""            # recent steps only: the model's context is 256 tokens
        transcript = [head]
        claims: list[dict] = []
        executed = 0
        terminal: dict | None = None

        def prompt_now() -> str:
            return head + tail[-self.TAIL_CHARS:] if tail else head

        for step in range(1, self.max_steps + 1):
            out = self.host.generate(prompt_now(), max_tokens=self.max_tokens,
                                     temperature=self.temperature, top_k=self.top_k,
                                     seed=self.seed)
            comp = out.get("completion", "")
            act = parse_action(comp)
            term = parse_terminal(comp)
            act_at = comp.find("<|act|>")
            done_at = comp.find("<|done|>")
            if term and (act is None or (0 <= done_at < act_at)):
                ok = world.progress() >= 1.0
                claims.append({"step": step, "claim": term, "verified": bool(ok)})
                if term == "success" and not ok:
                    # the runtime, not the model, decides: keep working
                    tail += ("<|check|>objective not satisfied: nothing on the "
                             "screen changed as required<|retry|>act on an "
                             "element from the scan")
                    transcript.append(comp.strip())
                    continue
                terminal = {"kind": term, "verified": bool(ok) if term == "success" else None,
                            "step": step}
                transcript.append(comp.strip())
                break

            if act is None:
                tail += "<|obs|>no action taken<|check|>objective not satisfied"
                transcript.append(comp.strip())
                continue

            executed += 1
            before = world.observe()
            # the simulated desktop only clicks; every primitive that names a
            # point is delivered as a click there (type/drag/press all land on
            # the same widget centre in the text protocol)
            res = world.act({"type": "click"}, act["point"])
            after = world.observe()
            verdict, detail = world.verify(before, after, step, res)
            satisfied = world.progress() >= 1.0
            px, py = act["protocol_point"]
            tail += f"<|step|>{step}<|act|>click({px},{py})<|obs|>{detail}"
            tail += ("<|check|>objective satisfied" if satisfied
                     else f"<|check|>objective not satisfied: {detail}")
            transcript.append(comp.strip())

        solved = world.progress() >= 1.0
        return {
            "objective": objective,
            "feasible": feasible,
            "solved": bool(solved),
            "executed_steps": executed,
            "terminal": terminal,
            # a success claim with no confirmation from the world is a lie
            "overclaim": any(c["claim"] == "success" and not c["verified"] for c in claims),
            "claims": claims,
            "transcript": transcript,
        }
