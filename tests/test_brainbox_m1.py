"""M1 tests: local API per specs/model_io.md.

Server runs in-process on an ephemeral loopback port with a tiny stub LM and
a random-init GripNet (test doubles injected into BrainHost). The wire
contract is what's under test here; real-checkpoint numbers come from
scripts/api_stress.py.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TOK = os.path.join(ROOT, "data", "tokenizer.json")
TINY = {"n_layer": 1, "n_head": 2, "d_model": 32, "ctx": 64}


def req(port, method, path, body=None, raw=None, timeout=30):
    data = raw
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=data, method=method, headers=headers)
    try:
        # The sandbox may define HTTP(S)_PROXY; loopback must bypass proxies.
        t0 = time.time()
        with NO_PROXY_OPENER.open(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read()), (time.time() - t0) * 1000
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), (time.time() - t0) * 1000


NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def sample_png_b64() -> str:
    import random

    import numpy as np
    from PIL import Image

    from neurobrain import ui_world

    ex = ui_world.sample_example(random.Random(3))
    img = ui_world.render(ex.layout)
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class ServerFixture:
    def __init__(self, governor=None):
        self.tmp = tempfile.mkdtemp(prefix="bb_m1_")

        import torch

        from brainbox.checkpoints import make_stub
        from brainbox.grip import Grounder
        from brainbox.host import BrainHost
        from brainbox.server import make_server
        from neurobrain.gripnet import GripNet

        ckdir = os.path.join(self.tmp, "lm-tiny")
        make_stub(ckdir, TOK, seed=5, config_override=TINY)
        torch.manual_seed(5)
        grip_model = GripNet()
        grip_model.eval()
        host = BrainHost(checkpoint=ckdir, threads=2, governor=governor,
                         grounder=Grounder(model=grip_model, step=-1,
                                            source="test-double"))
        host.boot()
        self.server = make_server(host, port=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = ServerFixture()

    @classmethod
    def tearDownClass(cls):
        cls.fx.close()

    def test_bound_to_loopback_only(self):
        self.assertEqual(self.fx.server.server_address[0], "127.0.0.1")

    def test_health_spec_exact_and_fast(self):
        status, body, ms = req(self.fx.port, "GET", "/health")  # warm
        status, body, ms = req(self.fx.port, "GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(set(body), {"ok", "rss_mb", "step"})
        self.assertTrue(body["ok"])
        self.assertIsInstance(body["rss_mb"], int)
        self.assertLess(ms, 50, f"health took {ms:.1f} ms")

    def test_generate_spec_exact(self):
        status, body, _ = req(self.fx.port, "POST", "/generate",
                              {"prompt": "def f():\n", "max_tokens": 5})
        self.assertEqual(status, 200)
        self.assertEqual(set(body), {"text", "tokens", "ms"})
        self.assertEqual(body["tokens"], 5)
        self.assertIsInstance(body["text"], str)

    def test_generate_defaults_64_tokens(self):
        status, body, _ = req(self.fx.port, "POST", "/generate", {"prompt": "x"})
        self.assertEqual(status, 200)
        # tiny test ctx=64: prompt(1) + 64 new may slide; host counts new tokens
        self.assertEqual(body["tokens"], 64)

    def test_score_spec_exact(self):
        status, body, _ = req(self.fx.port, "POST", "/score",
                              {"prompt": "import ", "top_k": 5})
        self.assertEqual(status, 200)
        self.assertEqual(set(body), {"tokens", "logprobs"})
        self.assertEqual(len(body["tokens"]), 5)
        self.assertEqual(len(body["logprobs"]), 5)

    def test_ground_spec_exact(self):
        status, body, _ = req(self.fx.port, "POST", "/ground",
                              {"png_b64": sample_png_b64(),
                               "instruction": "click the ok button"})
        self.assertEqual(status, 200)
        self.assertEqual(set(body), {"x", "y", "action"})
        self.assertTrue(0 <= body["x"] < 128 and 0 <= body["y"] < 128)
        self.assertIn(body["action"], ("click", "type", "toggle", "drag"))

    def test_errors(self):
        status, body, _ = req(self.fx.port, "POST", "/generate", raw=b"{not json")
        self.assertEqual(status, 400)
        status, body, _ = req(self.fx.port, "POST", "/generate", {"max_tokens": 3})
        self.assertEqual(status, 400)
        status, body, _ = req(self.fx.port, "GET", "/nope")
        self.assertEqual(status, 404)
        status, body, _ = req(self.fx.port, "GET", "/generate")
        self.assertEqual(status, 405)
        status, body, _ = req(self.fx.port, "POST", "/ground",
                              {"png_b64": "!!!not-base64!!!", "instruction": "x"})
        self.assertEqual(status, 400)

    def test_queue_serialises_model_execution(self):
        results = []

        def one(i):
            results.append(req(self.fx.port, "POST", "/generate",
                               {"prompt": "def f():\n", "max_tokens": 6})[0])

        threads = [threading.Thread(target=one, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual(sorted(results), [200] * 8)
        self.assertEqual(self.fx.server.stats["peak_concurrency"], 1)
        self.assertGreaterEqual(self.fx.server.stats["jobs"], 8)


class TestGovernorRefusal(unittest.TestCase):
    def test_429_over_cap_but_health_still_answers(self):
        from brainbox.governor import Governor

        fx = ServerFixture(governor=Governor(cap_mb=1, resume_mb=0.5))
        try:
            status, body, _ = req(fx.port, "POST", "/generate",
                                  {"prompt": "x", "max_tokens": 1})
            self.assertEqual(status, 429)
            self.assertIn("reason", body)
            status, body, _ = req(fx.port, "GET", "/health")
            self.assertEqual(status, 200)
        finally:
            fx.close()


class TestServeCLI(unittest.TestCase):
    def test_serve_help(self):
        import subprocess

        r = subprocess.run([sys.executable, "-m", "brainbox", "serve", "--help"],
                           capture_output=True, text=True, cwd=ROOT, timeout=60)
        self.assertEqual(r.returncode, 0)
        self.assertIn("--port", r.stdout)
        self.assertIn("8321", r.stdout)


if __name__ == "__main__":
    unittest.main()
