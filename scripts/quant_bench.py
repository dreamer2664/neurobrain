"""M7 acceptance: int8 accuracy / RSS delta table + governor evidence,
on the real checkpoints (ckpts/lm step 4600, ckpts/grip step 750).

    python3 scripts/quant_bench.py --out report.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def bench_subprocess(int8: bool) -> dict:
    cmd = [sys.executable, "-m", "brainbox", "bench", "--checkpoint",
           "checkpoints/lm", "--tokens", "20", "--json"]
    if int8:
        cmd.append("--int8")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=300)
    d = json.loads(r.stdout)
    return {"boot_s": d["boot_s"], "tok_s": d["tok_s"],
            "peak_rss_mb": d["peak_rss_mb"], "gen_s": d["gen_s"]}


def http(port, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                  method=method,
                                  headers={"Content-Type": "application/json"})
    try:
        with OPENER.open(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def serve_probe(port, int8: bool, cap_mb=None) -> dict:
    cmd = [sys.executable, "-m", "brainbox", "serve", "--port", str(port),
           "--checkpoint", "checkpoints/lm"]
    if int8:
        cmd.append("--int8")
    if cap_mb:
        cmd += ["--ram-cap-mb", str(cap_mb)]
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 60
        health = None
        while time.time() < deadline:
            if p.poll() is not None:
                raise RuntimeError(p.stdout.read())
            try:
                _, health = http(port, "GET", "/health")
                break
            except Exception:
                time.sleep(0.2)
        status, gen = http(port, "POST", "/generate",
                           {"prompt": "def f():\n", "max_tokens": 8})
        after = http(port, "GET", "/health")[1]["rss_mb"]
        with open(f"/proc/{p.pid}/status") as fh:
            hwm = next(int(l.split()[1]) / 1024 for l in fh
                       if l.startswith("VmHWM:"))
        return {"serve_rss_mb": after, "peak_hwm_mb": round(hwm, 1),
                "generate_status": status, "step": health["step"]}
    finally:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def lm_loss(host, tok, text: str, ctx=128, windows=24) -> float:
    import torch

    ids = tok.encode(text)
    losses = []
    step = max(1, (len(ids) - ctx - 1) // windows)
    with torch.no_grad():
        for i in range(windows):
            s = i * step
            chunk = ids[s:s + ctx + 1]
            if len(chunk) < ctx + 1:
                break
            x = torch.tensor([chunk[:-1]])
            y = torch.tensor([chunk[1:]])
            _, loss = host.model(x, y)
            losses.append(loss.item())
    return sum(losses) / len(losses), len(losses)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    from brainbox.grip import Grounder
    from brainbox.host import BrainHost
    from brainbox.quant import quantize_model
    from neurobrain.train_grip import evaluate as grip_evaluate

    torch.set_num_threads(2)
    rep = {"milestone": "M7",
           "note": "int8 = weight-only per-row symmetric, dequant-on-forward. "
                   "fp32 weights.pt remains the source of truth; the int8 "
                   "sidecar is derived (scripts/migrate_checkpoint.py --int8)."}

    def sizes(d):
        fp = os.path.getsize(os.path.join(d, "weights.pt"))
        q = os.path.getsize(os.path.join(d, "weights.int8.pt"))
        return {"fp32_bytes": fp, "int8_bytes": q, "ratio": round(q / fp, 3)}

    # ---- LM: loss + agreement in-process ----
    fp = BrainHost(checkpoint="checkpoints/lm", threads=2)
    fp.boot()
    q8 = BrainHost(checkpoint="checkpoints/lm", threads=2, quantize="int8")
    q8.boot()
    text = ""
    for f in ("neurobrain/model.py", "neurobrain/tokenizer.py",
              "neurobrain/ui_world.py"):
        with open(os.path.join(ROOT, f)) as fh:
            text += fh.read()
    loss_fp, nw = lm_loss(fp, fp.tok, text)
    loss_q, _ = lm_loss(q8, q8.tok, text)
    prompts = ["def ", "import ", "class ", "return ", "    self.",
               "# quicksort\n", "if __name__", "for i in range(",
               "click(", "<|ui|>"] * 4
    agree = sum(fp.score(p, top_k=1)["token_ids"] == q8.score(p, top_k=1)["token_ids"]
                for p in prompts)
    rep["lm"] = {**sizes("checkpoints/lm"),
                 "loss_holdout_src": {"fp32": round(loss_fp, 4),
                                      "int8": round(loss_q, 4),
                                      "delta": round(loss_q - loss_fp, 4),
                                      "windows": nw, "ctx": 128},
                 "top1_agreement": f"{agree}/{len(prompts)}",
                 "bench_fp32": bench_subprocess(False),
                 "bench_int8": bench_subprocess(True),
                 "serve_fp32": serve_probe(18331, False),
                 "serve_int8": serve_probe(18332, True)}
    del fp, q8

    # ---- Grip: hit rate fp32 vs int8, same sample stream ----
    g = Grounder.from_dir("checkpoints/grip")
    ev_fp = grip_evaluate(g.model, random.Random(0), 256)
    g.model.eval()
    qmodel = quantize_model(copy.deepcopy(g.model))
    ev_q = grip_evaluate(qmodel, random.Random(0), 256)
    rep["grip"] = {**sizes("checkpoints/grip"),
                   "hit_widget_fp32": ev_fp["hit_widget"],
                   "hit_widget_int8": ev_q["hit_widget"],
                   "action_acc_fp32": ev_fp["action_acc"],
                   "action_acc_int8": ev_q["action_acc"], "n": 256}

    # ---- Governor live: cap below live RSS refuses work, health alive ----
    gov = serve_probe(18333, False, cap_mb=200)
    rep["governor_live"] = {
        "cap_mb": 200, "note": "cap deliberately below the server's live RSS",
        "generate_status_over_cap": gov["generate_status"],
        "expect": 429}
    peaks = [rep["lm"]["bench_fp32"]["peak_rss_mb"],
             rep["lm"]["bench_int8"]["peak_rss_mb"],
             rep["lm"]["serve_fp32"]["peak_hwm_mb"],
             rep["lm"]["serve_int8"]["peak_hwm_mb"]]
    rep["max_measured_peak_mb"] = max(peaks)
    rep["budget_target_mb"] = 3000
    rep["pass"] = bool(gov["generate_status"] == 429 and max(peaks) <= 3000
                       and abs(loss_q - loss_fp) < 0.05
                       and ev_q["hit_widget"] >= ev_fp["hit_widget"] - 0.05)
    print(json.dumps(rep, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
    return 0 if rep["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
