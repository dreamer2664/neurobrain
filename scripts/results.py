#!/usr/bin/env python3
"""Collate live metrics from reports/ into RESULTS.md."""

from __future__ import annotations

import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(name, default=None):
    path = os.path.join(ROOT, "reports", name)
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return default


def read_jsonl(name):
    path = os.path.join(ROOT, "reports", name)
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def main() -> None:
    corpus = load_json("corpus_report.json", {})
    lm = [r for r in read_jsonl("lm_log.jsonl") if "eval" in r]
    grip = [r for r in read_jsonl("grip_log.jsonl") if "eval" in r]
    code_eval = load_json("code_eval.json", {})
    ui_eval = load_json("ui_eval.json", {})

    lines = ["# live results", "",
             f"_updated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}_", ""]

    if corpus:
        t = corpus.get("tokens", {})
        lines += ["## corpus", "",
                  "| stream | tokens |", "|---|---|"]
        for k, v in t.items():
            lines.append(f"| {k} | {v/1e6:.2f}M |")
        lines += [f"| vocab | {corpus.get('vocab_size')} |",
                  f"| code bytes/token | {corpus.get('code_bytes_per_token')} |", ""]

    if lm:
        last = lm[-1]
        lines += ["## language brain", "",
                  f"- step: **{last['step']}**",
                  f"- held-out loss: code **{last['eval']['code']:.3f}** "
                  f"(ppl {2.718281828 ** last['eval']['code']:.1f}), "
                  f"desktop **{last['eval']['ui']:.3f}** "
                  f"(ppl {2.718281828 ** last['eval']['ui']:.1f})"]
        if code_eval:
            lines.append(f"- syntax-valid generations: **{code_eval.get('syntax_valid')}/"
                         f"{code_eval.get('syntax_total')}** "
                         f"({100*code_eval.get('syntax_rate',0):.0f}%) at step {code_eval.get('step')}")
        if ui_eval:
            lines.append(f"- desktop protocol: format {100*ui_eval.get('format_rate',0):.0f}%, "
                         f"action type {100*ui_eval.get('action_type_rate',0):.0f}%, "
                         f"click inside target widget {100*ui_eval.get('in_widget_rate',0):.0f}%")
        lines += ["", "![curves](samples/training_curves.png)", ""]

    if grip:
        last = grip[-1]["eval"]
        lines += ["## gripnet (screen pointing)", "",
                  f"- step: **{grip[-1]['step']}**",
                  f"- click lands on the instructed widget: **{100*last['hit_widget']:.1f}%**",
                  f"- click within 6 px of widget centre: {100*last['hit_point']:.1f}%",
                  f"- action-type accuracy: {100*last['action_acc']:.1f}%", "",
                  "![grip](samples/grip_predictions.png)", ""]

    out = os.path.join(ROOT, "RESULTS.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[results] wrote {out}")


if __name__ == "__main__":
    main()
