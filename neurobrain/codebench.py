"""codebench: the code deliverable benchmark (owner: agent 04).

48 tasks (signature + docstring -> body) from `codebench_tasks.py`. Each task
is written to its own JSON file under `data/codebench/tasks/` and pinned by
sha256 in `data/codebench/manifest.json` -- held out by file hash, so the
bench is reproducible and the corpus builder can exclude any file that ever
collides with it.

Scoring: greedy and sampled decoding, then every candidate is executed in a
sandboxed subprocess (`python -I`, minimal env, hard timeout). Headline
numbers:

  pass@1             completion passes every hidden assert
  syntax_valid_rate  prompt+completion parses AND defines the requested fn

Usage:
    python3 -m neurobrain.codebench build
    python3 -m neurobrain.codebench score --ckpt checkpoints/lm_best.pt
    python3 -m neurobrain.codebench holdout
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

from .codebench_tasks import TASKS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(ROOT, "data", "codebench")
TASK_DIR = os.path.join(BENCH_DIR, "tasks")
MANIFEST_PATH = os.path.join(BENCH_DIR, "manifest.json")

BENCH_VERSION = "codebench-v1.1"
FIELDS = ("id", "category", "difficulty", "prompt", "fn", "tests", "reference", "timeout_s")
STOP_MARKERS = ("<|" + "end" + "oftext" + "|>", chr(10) + "def ", chr(10) + "class ", "<|" + "ta" + "sk|>", "<|" + "dri" + "ve|>", "<|" + "u" + "i|>")
# corpus-v2 models are trained to emit the code-separator special right before
# a function body (instruction format task...sep...code). The extractor must
# treat that marker as "code starts here", not as part of the candidate.
CODE_SEP = "<|" + "co" + "de|>"


# ------------------------------------------------------------------ build/pin
def _canonical(t: dict) -> str:
    return json.dumps({k: t[k] for k in FIELDS}, indent=1, sort_keys=True) + "\n"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(task_dir: str = TASK_DIR, manifest_path: str = MANIFEST_PATH) -> dict:
    """Write one JSON file per task plus a sha256-pinned manifest."""
    os.makedirs(task_dir, exist_ok=True)
    entries = []
    for t in TASKS:
        name = f"{t['id']}.json"
        path = os.path.join(task_dir, name)
        text = _canonical(t)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        entries.append({
            "file": name, "id": t["id"], "category": t["category"],
            "difficulty": t["difficulty"], "fn": t["fn"],
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
    manifest = {
        "bench": BENCH_VERSION, "n": len(entries),
        "built_utc": time.strftime("%FT%TZ", time.gmtime()),
        "generated_by": "agent-04 neurobrain.codebench",
        "tasks": entries,
    }
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def load_tasks(task_dir: str = TASK_DIR, manifest_path: str = MANIFEST_PATH,
               verify: bool = True) -> list[dict]:
    """Load tasks from disk; verify file hashes against the manifest."""
    tasks = []
    for name in sorted(os.listdir(task_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(task_dir, name)
        if verify and os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as f:
                man = json.load(f)
            pin = {e["file"]: e["sha256"] for e in man["tasks"]}
            if name in pin and sha256_file(path) != pin[name]:
                raise ValueError(f"codebench task {name} fails its sha256 pin")
        with open(path, encoding="utf-8") as f:
            tasks.append(json.load(f))
    return tasks


# --------------------------------------------------------------- extraction
def extract_completion(text: str, prompt: str) -> str:
    """Cut a raw generation down to the candidate body."""
    comp = text[len(prompt):] if text.startswith(prompt) else text
    # if the model announced code with the separator marker, the candidate is
    # whatever follows the LAST such marker (final announced code block)
    j = comp.rfind(CODE_SEP)
    if j != -1:
        comp = comp[j + len(CODE_SEP):]
    cut = len(comp)
    for m in STOP_MARKERS:
        i = comp.find(m)
        if i != -1:
            cut = min(cut, i)
    comp = comp[:cut].rstrip()
    return comp + chr(10) if comp else ""


# ------------------------------------------------------------------ execution
def run_candidate(prompt: str, completion: str, tests: str, fn: str,
                  timeout_s: float = 5.0) -> dict:
    """Execute prompt+completion+hidden tests in a sandboxed subprocess."""
    body = prompt + completion
    try:
        tree = ast.parse(body)
    except SyntaxError as e:
        return {"passed": False, "parses": False, "defines": False, "why": f"syntax: {e.msg}"}
    defines = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn
                  for n in tree.body)
    source = body + "\n\n" + tests + "\nprint('CB_OK')\n"
    with tempfile.TemporaryDirectory(prefix="cb-") as td:
        path = os.path.join(td, "candidate.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(source)
        env = {"PATH": "/usr/bin:/bin", "TMPDIR": td}
        try:
            proc = subprocess.run([sys.executable, "-I", path], capture_output=True,
                                  text=True, timeout=timeout_s, cwd=td, env=env)
        except subprocess.TimeoutExpired:
            return {"passed": False, "parses": True, "defines": defines, "why": "timeout"}
    passed = proc.returncode == 0 and "CB_OK" in proc.stdout
    why = ""
    if not passed:
        err = [l for l in proc.stderr.strip().splitlines() if l.strip()]
        why = (err[-1] if err else f"rc={proc.returncode}")[:200]
    return {"passed": passed, "parses": True, "defines": defines, "why": why}


# --------------------------------------------------------------------- scoring
def peak_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return -1.0


def score(ckpt: str, data_dir: str = os.path.join(ROOT, "data"),
          out: str = os.path.join(ROOT, "reports", "codebench.json"),
          max_tokens: int = 128, seed: int = 0, modes: tuple[str, ...] = ("greedy", "sampled"),
          task_dir: str = TASK_DIR, limit: int | None = None) -> dict:
    import torch
    from .sample import load
    from .tokenizer import BPETokenizer  # noqa: F401  (loaded via sample.load)

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    model, tok, step = load(ckpt, tok_path=os.path.join(data_dir, "tokenizer.json"))
    eot = tok.special_id("<|endoftext|>")
    tasks = load_tasks(task_dir)
    if limit:
        tasks = tasks[:limit]

    per_mode = {}
    for mode in modes:
        temperature, top_k = (1.0, 1) if mode == "greedy" else (0.8, 40)
        rows = []
        t0 = time.time()
        gen_tokens = 0
        for t in tasks:
            ids = tok.encode(t["prompt"])
            x = torch.tensor([ids], dtype=torch.long)
            out_ids = model.generate(x, max_new_tokens=max_tokens, temperature=temperature,
                                     top_k=top_k, eos_id=eot)
            full = tok.decode(out_ids[0].tolist())
            gen_tokens += out_ids.shape[1] - len(ids)
            comp = extract_completion(full, t["prompt"])
            raw_head = full[len(t["prompt"]):][:160] if not comp else ""
            r = run_candidate(t["prompt"], comp, t["tests"], t["fn"], t.get("timeout_s", 5))
            rows.append({"id": t["id"], "category": t["category"], "difficulty": t["difficulty"],
                         "completion": comp[:400], "raw_head": raw_head, **r})
        dt = time.time() - t0
        n = len(rows)
        passed = sum(r["passed"] for r in rows)
        syntax_ok = sum(r["parses"] and r["defines"] for r in rows)
        by_cat: dict[str, list[int]] = {}
        by_diff: dict[str, list[int]] = {}
        for r in rows:
            by_cat.setdefault(r["category"], [0, 0])
            by_cat[r["category"]][0] += r["passed"]
            by_cat[r["category"]][1] += 1
            by_diff.setdefault(r["difficulty"], [0, 0])
            by_diff[r["difficulty"]][0] += r["passed"]
            by_diff[r["difficulty"]][1] += 1
        per_mode[mode] = {
            "decode": {"temperature": temperature, "top_k": top_k, "max_tokens": max_tokens},
            "pass_at_1": round(passed / n, 4), "passed": passed, "n": n,
            "syntax_valid_rate": round(syntax_ok / n, 4),
            "body_rate": round(sum(bool(r["completion"]) for r in rows) / n, 4),
            "parses_rate": round(sum(r["parses"] for r in rows) / n, 4),
            "by_category": {k: round(v[0] / v[1], 3) for k, v in sorted(by_cat.items())},
            "by_difficulty": {k: round(v[0] / v[1], 3) for k, v in sorted(by_diff.items())},
            "gen_tok_s": round(gen_tokens / max(1e-9, dt), 1),
            "seconds": round(dt, 1),
            "tasks": rows,
        }
        print(f"[codebench] {mode}: pass@1 {passed}/{n}  syntax+defines {syntax_ok}/{n}  "
              f"{per_mode[mode]['gen_tok_s']} tok/s  {dt:.0f}s", flush=True)

    result = {
        "bench": BENCH_VERSION,
        "utc": time.strftime("%FT%TZ", time.gmtime()),
        "ckpt": os.path.relpath(ckpt, ROOT),
        "ckpt_sha256": sha256_file(ckpt),
        "step": step,
        "cfg": model.cfg.to_dict(),
        "params": model.num_params(),
        "seed": seed,
        "peak_rss_mb": peak_rss_mb(),
        "modes": per_mode,
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"[codebench] wrote {out}")
    return result


# --------------------------------------------------------------------- holdout
def _norm(s: str) -> str:
    return " ".join(s.split())


def holdout_scan(roots: list[str] | None = None,
                 task_dir: str = TASK_DIR) -> dict:
    """Prove the bench is held out: no corpus file contains a task verbatim.

    Two mechanisms:
      1. file-hash pinning: every task JSON is sha256-pinned in the manifest,
         so the task set cannot drift unnoticed; the corpus builder excludes
         any file whose sha256 is in the manifest.
      2. verbatim scan: the normalized prompt+reference solution of every task
         is searched across the corpus roots (stdlib + site-packages, the same
         roots build_corpus.py / retokenize.py consume).
    """
    if roots is None:
        import sysconfig
        roots = [sysconfig.get_paths()["stdlib"], sysconfig.get_paths()["purelib"]]
    tasks = load_tasks(task_dir, verify=False)
    needles = {t["id"]: _norm(t["prompt"] + t["reference"]) for t in tasks}
    collisions: list[dict] = []
    files = scanned = 0
    skip = {"__pycache__", ".git", "node_modules"}
    for root in roots:
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in skip and not d.startswith(".")]
            for name in fn:
                if not name.endswith(".py"):
                    continue
                p = os.path.join(dp, name)
                try:
                    with open(p, encoding="utf-8", errors="ignore") as f:
                        content = _norm(f.read())
                except OSError:
                    continue
                files += 1
                scanned += len(content)
                for tid, needle in needles.items():
                    if needle in content:
                        collisions.append({"task": tid, "file": p})
    return {
        "method": "sha256 manifest pin + verbatim prompt+reference scan of corpus roots",
        "roots": roots, "py_files_scanned": files,
        "chars_scanned": scanned, "collisions": collisions,
        "n_collisions": len(collisions),
        "task_files": len(tasks),
    }


# ------------------------------------------------------------------------ cli
def main() -> None:
    ap = argparse.ArgumentParser(description="codebench: signature+docstring -> body")
    ap.add_argument("cmd", choices=["build", "score", "holdout"])
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "lm_best.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "codebench.json"))
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.cmd == "build":
        man = build()
        print(f"[codebench] built {man['n']} tasks -> {TASK_DIR} (+ manifest)")
    elif args.cmd == "score":
        score(args.ckpt, data_dir=args.data, out=args.out,
              max_tokens=args.max_tokens, seed=args.seed, limit=args.limit)
    elif args.cmd == "holdout":
        rep = holdout_scan()
        os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
        p = os.path.join(ROOT, "reports", "codebench_holdout.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
        print(json.dumps({k: v for k, v in rep.items() if k != "collisions"}, indent=1))
        print(f"[codebench] wrote {p}")


if __name__ == "__main__":
    main()
