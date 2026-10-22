#!/usr/bin/env python3
"""Pack a checkpoint into the fleet's shippable directory layout (specs/model_io.md).

    checkpoints/<name>/
      config.json      arch + tokenizer_sha256 + weights_sha256 (agent-01 spec) + step
      weights.pt       bare state_dict, fp32 (source of truth)
      tokenizer.json   exact tokenizer used for training
      card.md          numbers, lineage, sample generations (failures included)

Usage:
    python3 scripts/pack_checkpoint.py --ckpt checkpoints/run12k/lm_best.pt \
        --name lm-run12k --codebench reports/codebench.json \
        --drive-eval reports/drive_eval.json --code-eval reports/code_eval.json \
        --samples 2 --notes "trained with: python3 -m neurobrain.train_lm ..."
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from neurobrain.sample import load  # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_card(name: str, config: dict, metrics: dict, samples: list[dict], notes: str) -> str:
    lines = [
        f"# checkpoint card: {name}",
        "",
        f"- packed: {config['packed_utc']}",
        f"- arch: {config['n_layer']}L d{config['d_model']} h{config['n_head']} "
        f"ctx{config['ctx']} vocab{config['vocab_size']} ({config['params']:,} params)",
        f"- step: {config['step']}",
        f"- weights_sha256: `{config['weights_sha256']}`",
        f"- tokenizer_sha256: `{config['tokenizer_sha256']}`",
        "",
        "## numbers",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## sample generations (verbatim, failures included)", ""]
    for s in samples:
        lines += [f"### {s.get('kind', 'sample')}", "", "prompt:", "```", s["prompt"], "```",
                  "generation:", "```", s["generation"], "```", ""]
    if notes:
        lines += ["## lineage / notes", "", notes, ""]
    return chr(10).join(lines) + chr(10)


def pack(ckpt: str, out_dir: str, tok_path: str, name: str = "lm",
         metrics: dict | None = None, samples: list[dict] | None = None,
         notes: str = "") -> dict:
    model, tok, step = load(ckpt, tok_path)
    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, "weights.pt")
    torch.save(model.state_dict(), weights_path)
    shutil.copyfile(tok_path, os.path.join(out_dir, "tokenizer.json"))
    cfgd = model.cfg.to_dict()
    config = {
        "kind": "lm", "name": name,
        "vocab_size": cfgd["vocab_size"], "n_layer": cfgd["n_layer"],
        "n_head": cfgd["n_head"], "d_model": cfgd["d_model"], "ctx": cfgd["ctx"],
        "tie_embeddings": cfgd["tie_embeddings"],
        "step": step, "params": model.num_params(),
        "tokenizer_sha256": sha256_file(tok_path),
        "weights_sha256": sha256_file(weights_path),
        "source_ckpt_sha256": sha256_file(ckpt),
        "packed_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    card = make_card(name, config, metrics or {}, samples or [], notes)
    with open(os.path.join(out_dir, "card.md"), "w") as f:
        f.write(card)
    return config


def _collect_metrics(paths: dict[str, str | None]) -> dict:
    m: dict = {}
    ce = paths.get("codebench")
    if ce and os.path.exists(ce):
        d = json.load(open(ce))
        g = d["modes"]["greedy"]
        s = d["modes"].get("sampled", {})
        m["codebench pass@1 (greedy)"] = g["pass_at_1"]
        m["codebench pass@1 (sampled)"] = s.get("pass_at_1")
        m["codebench syntax_valid (greedy)"] = g["syntax_valid_rate"]
        m["codebench body_rate (greedy)"] = g.get("body_rate")
        m["codebench step"] = d.get("step")
    de = paths.get("drive")
    if de and os.path.exists(de):
        d = json.load(open(de))
        m["terminal_rate"] = d.get("terminal_rate")
        m["retry_after_failure"] = d.get("retry_after_failure")
        m["honest_completion"] = d.get("honest_completion")
    cv = paths.get("code")
    if cv and os.path.exists(cv):
        d = json.load(open(cv))
        m["code val_loss"] = d.get("val_loss")
        m["code val_ppl"] = d.get("val_ppl")
    return m


def _samples(model, tok, n: int) -> list[dict]:
    from neurobrain.eval_drive import objectives
    out = []
    if n <= 0:
        return out
    code_prompt = "def average(values):" + chr(10)
    ids = torch.tensor([tok.encode(code_prompt)], dtype=torch.long)
    gen = model.generate(ids, max_new_tokens=96, temperature=0.7, top_k=30,
                         eos_id=tok.special_id(tok.specials[0]))
    out.append({"kind": "code", "prompt": code_prompt,
                "generation": tok.decode(gen[0].tolist())[len(code_prompt):]})
    if n > 1:
        drive_prompt = objectives(1, seed=5)[0]
        ids = torch.tensor([tok.encode(drive_prompt)], dtype=torch.long)
        gen = model.generate(ids, max_new_tokens=160, temperature=0.5, top_k=30,
                             eos_id=tok.special_id(tok.specials[0]))
        out.append({"kind": "drive", "prompt": drive_prompt,
                    "generation": tok.decode(gen[0].tolist())[len(drive_prompt):]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None, help="output dir (default checkpoints/<name>)")
    ap.add_argument("--name", default="lm")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer.json"))
    ap.add_argument("--codebench", default=None)
    ap.add_argument("--drive-eval", default=None)
    ap.add_argument("--code-eval", default=None)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    torch.set_num_threads(2)
    out_dir = args.out or os.path.join(ROOT, "checkpoints", args.name)
    metrics = _collect_metrics({"codebench": args.codebench, "drive": args.drive_eval,
                                "code": args.code_eval})
    model, tok, _ = load(args.ckpt, args.tokenizer)
    samples = _samples(model, tok, args.samples)
    config = pack(args.ckpt, out_dir, args.tokenizer, name=args.name,
                  metrics=metrics, samples=samples, notes=args.notes)
    print(json.dumps(config, indent=1))
    print("[pack] wrote", out_dir)


if __name__ == "__main__":
    main()
