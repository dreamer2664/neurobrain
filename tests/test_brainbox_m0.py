"""M0 tests for brainbox: skeleton + doctor + bench + 20-token generate.

Run:  python3 -m unittest discover -s tests -v
Stdlib unittest only (no pytest dependency in the sandbox).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TOK = os.path.join(ROOT, "data", "tokenizer.json")
TINY = {"n_layer": 1, "n_head": 2, "d_model": 32, "ctx": 64}


def make_tiny_stub(root: str) -> str:
    from brainbox.checkpoints import make_stub

    out = os.path.join(root, "lm-tiny-stub")
    make_stub(out, TOK, seed=1234, config_override=TINY)
    return out


class TestGovernor(unittest.TestCase):
    def test_hysteresis_refuse_and_resume(self):
        from brainbox.governor import Governor

        rss = {"v": 1000.0}
        g = Governor(cap_mb=3800, resume_mb=3200, rss_fn=lambda: rss["v"])
        self.assertTrue(g.can_work())
        rss["v"] = 3900.0
        self.assertFalse(g.can_work())
        rss["v"] = 3500.0  # below cap but above resume: still tripped
        self.assertFalse(g.can_work())
        rss["v"] = 3100.0  # below resume: work again
        self.assertTrue(g.can_work())

    def test_rss_reader_returns_positive(self):
        from brainbox.governor import peak_rss_mb, rss_mb

        self.assertGreater(rss_mb(), 0)
        self.assertGreaterEqual(peak_rss_mb(), rss_mb() * 0.5)


class TestCheckpointRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bb_m0_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_stub_dir_layout_and_load(self):
        from brainbox.checkpoints import load_checkpoint_dir

        ckdir = make_tiny_stub(self.tmp)
        for name in ("config.json", "weights.pt", "tokenizer.json", "card.md"):
            self.assertTrue(os.path.exists(os.path.join(ckdir, name)), name)
        with open(os.path.join(ckdir, "config.json")) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["kind"], "lm")
        self.assertTrue(cfg["stub"])
        self.assertIn("tokenizer_sha256", cfg)
        self.assertIn("weights_sha256", cfg)

        model, tok, info = load_checkpoint_dir(ckdir)
        self.assertEqual(info.step, 0)
        self.assertTrue(info.stub)
        self.assertEqual(tok.vocab_size, 2052)
        self.assertEqual(model.cfg.vocab_size, 2052)

    def test_corrupt_weights_rejected_by_sha(self):
        from brainbox.checkpoints import CheckpointError, load_checkpoint_dir

        ckdir = make_tiny_stub(self.tmp)
        wpath = os.path.join(ckdir, "weights.pt")
        with open(wpath, "r+b") as f:
            f.seek(100)
            f.write(b"\xff")
        with self.assertRaises(CheckpointError):
            load_checkpoint_dir(ckdir)

    def test_tampered_tokenizer_rejected_by_sha(self):
        from brainbox.checkpoints import CheckpointError, load_checkpoint_dir

        ckdir = make_tiny_stub(self.tmp)
        tpath = os.path.join(ckdir, "tokenizer.json")
        with open(tpath) as f:
            payload = json.load(f)
        payload["vocab"] = payload["vocab"][:-1]
        with open(tpath, "w") as f:
            json.dump(payload, f)
        with self.assertRaises(CheckpointError):
            load_checkpoint_dir(ckdir)

    def test_legacy_pt_migration(self):
        import torch

        from brainbox.checkpoints import load_checkpoint_dir, migrate_legacy
        from neurobrain.model import GPT, GPTConfig

        cfg = GPTConfig(vocab_size=2052, **TINY)
        torch.manual_seed(7)
        model = GPT(cfg)
        legacy = os.path.join(self.tmp, "lm_best.pt")
        torch.save({"model": model.state_dict(), "step": 7,
                    "best_val": 1.5, "cfg": cfg.to_dict()}, legacy)

        out = os.path.join(self.tmp, "lm")
        migrate_legacy(legacy, TOK, out)
        _, _, info = load_checkpoint_dir(out)
        self.assertEqual(info.step, 7)
        self.assertFalse(info.stub)


class TestHostGenerate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bb_m0_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_generate_20_tokens(self):
        from brainbox.host import BrainHost

        ckdir = make_tiny_stub(self.tmp)
        host = BrainHost(checkpoint=ckdir, threads=2)
        info = host.boot()
        self.assertTrue(info["stub"])
        self.assertGreater(info["params"], 0)
        out = host.generate("def f(x):\n", max_tokens=20, seed=0)
        self.assertEqual(out["tokens"], 20)
        self.assertIsInstance(out["text"], str)
        self.assertGreater(out["ms"], 0)

    def test_score_topk(self):
        from brainbox.host import BrainHost

        ckdir = make_tiny_stub(self.tmp)
        host = BrainHost(checkpoint=ckdir, threads=2)
        host.boot()
        out = host.score("def f", top_k=5)
        self.assertEqual(len(out["tokens"]), 5)
        self.assertEqual(len(out["logprobs"]), 5)
        self.assertTrue(all(lp <= 0 for lp in out["logprobs"]))


class TestCLI(unittest.TestCase):
    def run_cli(self, argv):
        from brainbox.cli import main

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def test_doctor_json(self):
        code, out = self.run_cli(["doctor", "--json"])
        self.assertEqual(code, 0)
        d = json.loads(out)
        self.assertTrue(d["ok"])
        self.assertIn("torch", d["versions"])
        self.assertIn("python", d["versions"])
        self.assertIn("brainbox", d["versions"])
        self.assertIn("checkpoint", d)

    def test_bench_20_tokens_json(self):
        tmp = tempfile.mkdtemp(prefix="bb_m0_")
        try:
            ckdir = make_tiny_stub(tmp)
            code, out = self.run_cli(
                ["bench", "--checkpoint", ckdir, "--tokens", "20", "--json"])
            self.assertEqual(code, 0)
            d = json.loads(out)
            self.assertEqual(d["gen_tokens"], 20)
            self.assertEqual(d["threads"], 2)
            self.assertLess(d["peak_rss_mb"], 1600)   # sandbox ceiling
            self.assertLess(d["boot_s"], 20)          # done-when bound
            self.assertGreater(d["tok_s"], 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_help_lists_all_verbs_and_unimplemented_exit_2(self):
        r = subprocess.run([sys.executable, "-m", "brainbox", "--help"],
                           capture_output=True, text=True, cwd=ROOT, timeout=60)
        for verb in ("doctor", "bench", "serve", "run", "tasks", "logs", "kill", "undo"):
            self.assertIn(verb, r.stdout)
        r2 = subprocess.run([sys.executable, "-m", "brainbox", "serve"],
                            capture_output=True, text=True, cwd=ROOT, timeout=60)
        self.assertEqual(r2.returncode, 2)
        self.assertIn("M1", r2.stdout + r2.stderr)


if __name__ == "__main__":
    unittest.main()
