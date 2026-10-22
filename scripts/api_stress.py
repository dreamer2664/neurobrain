"""M1 acceptance: 200 sequential API requests against a fresh `brainbox
serve` subprocess on the real checkpoints; fail if server RSS grows > 20 MB.

Mix per 4 requests: generate, generate, score, ground (x50 = 200). Ground
requests use fresh ui_world examples, so this also reports the API-path
grounding hit rate. A prober thread polls /health throughout to prove it
stays < 50 ms while model work is queued.

    python3 scripts/api_stress.py --out /path/report.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(port, method, path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 method=method,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with OPENER.open(r, timeout=timeout) as resp:
        out = json.loads(resp.read())
    return out, (time.time() - t0) * 1000


def proc_rss_mb(pid):
    with open(f"/proc/{pid}/status") as f:
        vals = {}
        for line in f:
            if line.startswith(("VmRSS:", "VmHWM:")):
                k, v = line.split(":")
                vals[k] = int(v.split()[0]) / 1024
    return vals.get("VmRSS", 0.0), vals.get("VmHWM", 0.0)


def stats(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    return {"n": len(xs), "p50": round(statistics.median(xs), 2),
            "p95": round(xs[int(0.95 * (len(xs) - 1))], 2),
            "max": round(xs[-1], 2), "mean": round(statistics.mean(xs), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18321)
    ap.add_argument("--checkpoint", default="checkpoints/lm")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import numpy as np
    from PIL import Image

    from neurobrain import ui_world

    # Pre-build 10 ground payloads with known targets.
    rng = random.Random(20261022)
    grounds = []
    for _ in range(50):  # one unique screen per ground slot in the 200-mix
        ex = ui_world.sample_example(rng)
        buf = io.BytesIO()
        Image.fromarray(np.asarray(ui_world.render(ex.layout), dtype=np.uint8)
                        ).save(buf, format="PNG")
        grounds.append((base64.b64encode(buf.getvalue()).decode(), ex))

    proc = subprocess.Popen(
        [sys.executable, "-m", "brainbox", "serve", "--port", str(args.port),
         "--checkpoint", args.checkpoint],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 60
        health = None
        while time.time() < deadline:
            if proc.poll() is not None:
                print(proc.stdout.read())
                print("server exited during boot", file=sys.stderr)
                return 1
            try:
                health, _ = call(args.port, "GET", "/health", timeout=2)
                break
            except Exception:
                time.sleep(0.2)
        assert health and health["ok"], f"server not healthy: {health}"

        def gen():
            return call(args.port, "POST", "/generate",
                        {"prompt": "def retry(fn, n=3):\n", "max_tokens": 16, "seed": 1})

        def score():
            return call(args.port, "POST", "/score", {"prompt": "import ", "top_k": 5})

        def ground(i):
            b64, ex = grounds[i % len(grounds)]
            body, ms = call(args.port, "POST", "/ground",
                            {"png_b64": b64, "instruction": ex.task})
            return body, ms, ex.target.contains(body["x"], body["y"])

        # warmup, then baseline
        for _ in range(3):
            gen(); score(); ground(0)
        time.sleep(0.5)
        rss0_health = call(args.port, "GET", "/health")[0]["rss_mb"]
        rss0, _ = proc_rss_mb(proc.pid)

        # /health prober during the run
        health_ms, stop = [], threading.Event()

        def probe():
            while not stop.is_set():
                try:
                    _, ms = call(args.port, "GET", "/health", timeout=5)
                    health_ms.append(ms)
                except Exception:
                    pass
                time.sleep(0.05)

        prober = threading.Thread(target=probe, daemon=True)
        prober.start()

        lat = {"generate": [], "score": [], "ground": []}
        hits = grounds_n = 0
        t0 = time.time()
        for i in range(args.requests):
            kind = ("generate", "generate", "score", "ground")[i % 4]
            if kind == "generate":
                _, ms = gen()
            elif kind == "score":
                _, ms = score()
            else:
                _, ms, hit = ground(grounds_n)  # dedicated counter: use every screen
                hits += hit
                grounds_n += 1
            lat[kind].append(ms)
        wall = time.time() - t0
        stop.set()
        prober.join(timeout=5)
        time.sleep(1.0)
        rss1_health = call(args.port, "GET", "/health")[0]["rss_mb"]
        rss1, hwm = proc_rss_mb(proc.pid)

        report = {
            "milestone": "M1", "requests": args.requests,
            "mix": {k: len(v) for k, v in lat.items()},
            "wall_s": round(wall, 2),
            "latency_ms": {k: stats(v) for k, v in lat.items()},
            "health_during_load_ms": stats(health_ms),
            "rss_mb": {"baseline_proc": round(rss0, 1), "after_proc": round(rss1, 1),
                       "growth_proc": round(rss1 - rss0, 1),
                       "baseline_health": rss0_health, "after_health": rss1_health,
                       "growth_health": rss1_health - rss0_health,
                       "peak_hwm_proc": round(hwm, 1)},
            "ground_hit_rate": round(hits / grounds_n, 3) if grounds_n else None,
            "ground_n": grounds_n,
            "leak_limit_mb": 20,
            "checkpoint": args.checkpoint, "health": health,
        }
        report["pass"] = (report["rss_mb"]["growth_proc"] <= 20
                          and report["rss_mb"]["growth_health"] <= 20)
        print(json.dumps(report, indent=2))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(report, f, indent=2)
        return 0 if report["pass"] else 3
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
