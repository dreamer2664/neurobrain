#!/usr/bin/env python3
"""Assemble the morning report from the night's artifacts.

    python3 scripts/morning_report.py --out REPORT.md

Nothing here computes a score of its own: every number is read from the json the
eval scripts wrote, so the report cannot drift from the measurements.
"""

from __future__ import annotations

import argparse
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def jload(path: str, default=None):
    try:
        with open(os.path.join(ROOT, path)) as f:
            return json.load(f)
    except Exception:
        return default


def pct(x) -> str:
    try:
        return f"{100 * float(x):.1f}%"
    except Exception:
        return "n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "REPORT.md"))
    args = ap.parse_args()

    qa = jload("reports/qa_eval.json", {})
    web = jload("reports/web_eval.json", {})
    drive = jload("reports/e2e_lm_drive.json", {})
    webc = jload("reports/web_corpus.json", {})
    qac = jload("reports/qa_corpus.json", {})

    lines = []
    lines.append("# Morning report\n")
    lines.append(f"_generated {time.strftime('%Y-%m-%d %H:%M')} local, "
                 f"checkpoint step {qa.get('step', web.get('step', '?'))}_\n")
    lines.append("## The two bars\n")
    lines.append("| bar | target | measured |")
    lines.append("|---|---|---|")
    lines.append(f"| 50 basic questions | 90% | **{pct(qa.get('accuracy'))}** "
                 f"({qa.get('n', 0)} questions) |")
    lines.append(f"| open chrome, research, report in google docs | 95% | "
                 f"**{pct(web.get('accuracy'))}** ({web.get('tasks', 0)} tasks) |")
    lines.append("")

    if qa:
        lines.append("### Question answering\n")
        lines.append(f"* seen facts, unseen question templates: "
                     f"**{pct(qa.get('seen_accuracy'))}** (n={qa.get('seen_n')})")
        lines.append(f"* facts never trained on: **{pct(qa.get('heldout_accuracy'))}** "
                     f"(n={qa.get('heldout_n')}) - this is generalisation, not recall\n")
        lines.append("| question | expected | model | |")
        lines.append("|---|---|---|---|")
        for s in qa.get("samples", [])[:10]:
            mark = "ok" if s.get("ok") else "**miss**"
            lines.append(f"| {s.get('q','')} | {s.get('expected','')} | "
                         f"`{s.get('got','')[:40]}` | {mark} |")
        lines.append("")
        if qac:
            lines.append(f"_(qa corpus: {qac.get('facts_total')} facts, "
                         f"{qac.get('heldout_facts') and len(qac['heldout_facts'])} held out, "
                         f"{qac.get('train_tokens', 0)/1e6:.2f}M tokens)_\n")

    if web:
        lines.append("### Research and report on the mock desktop\n")
        ok = web.get("solved", 0)
        lines.append(f"{ok}/{web.get('tasks')} tasks finished with the right answer in a "
                     f"saved document, mean {web.get('mean_steps')} actions, "
                     f"overclaim {pct(web.get('overclaim'))}.\n")
        rows = web.get("rows", [])[:12]
        if rows:
            lines.append("| topic | answer | document | saved | |")
            lines.append("|---|---|---|---|---|")
            for r in rows:
                lines.append(f"| {r.get('topic','')} | {r.get('answer','')} | "
                             f"`{r.get('doc','') or '(empty)'}` | {r.get('saved')} | "
                             f"{'ok' if r.get('solved') else '**miss**'} |")
            lines.append("")
        if webc:
            toks = sum(s.get("tokens", 0) for s in webc.get("streams", []))
            lines.append(f"_(web corpus: {toks/1e6:.2f}M tokens of click-by-click transcripts)_\n")

    if drive:
        lines.append("### Honesty, unchanged from the drive loop\n")
        lines.append(f"* terminal_rate {pct(drive.get('terminal_rate'))}, "
                     f"overclaim {pct(drive.get('overclaim_rate'))}, "
                     f"impossible answered honestly {pct(drive.get('impossible_honesty'))}\n")

    lines.append("## What is still not true\n")
    lines.append("* The research is done against a mock desktop, not a real Chrome. "
                 "The X11 primitives exist and are tested, but no browser has been driven.")
    lines.append("* \"Research\" means reading an article page whose text the runtime put there; "
                 "the model copies the answer, it does not search the open web.")
    lines.append("* The knowledge base is 411 hand-checked facts, not the internet.\n")

    md = "\n".join(lines) + "\n"
    with open(args.out, "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
