"""M2 tests: checkpoint directories + migration for both current .pt kinds."""

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
TINY_LM = {"n_layer": 1, "n_head": 2, "d_model": 32, "ctx": 64}


def save_legacy_grip(path, step=3):
    import torch

    from neurobrain.gripnet import GripNet

    torch.manual_seed(11)
    model = GripNet()
    torch.save({"model": model.state_dict(), "step": step,
                "eval": {"hit_widget": 0.5}}, path)
    return model


def sample_png() -> bytes:
    import random

    import numpy as np
    from PIL import Image

    from neurobrain import ui_world

    ex = ui_world.sample_example(random.Random(1))
    buf = io.BytesIO()
    Image.fromarray(np.asarray(ui_world.render(ex.layout), dtype=np.uint8)
                    ).save(buf, format="PNG")
    return buf.getvalue()


class TestGripMigration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bb_m2_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_grip_dir_roundtrip_and_config(self):
        from brainbox.checkpoints import migrate_grip_legacy
        from brainbox.grip import Grounder

        pt = os.path.join(self.tmp, "grip_best.pt")
        save_legacy_grip(pt)
        out = os.path.join(self.tmp, "grip")
        info = migrate_grip_legacy(pt, out)
        self.assertEqual(info.kind, "grip")
        self.assertEqual(info.step, 3)
        for name in ("config.json", "weights.pt", "card.md"):
            self.assertTrue(os.path.exists(os.path.join(out, name)), name)
        with open(os.path.join(out, "config.json")) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["kind"], "grip")
        self.assertEqual(cfg["char_vocab"], 40)
        self.assertEqual(cfg["max_len"], 48)
        self.assertEqual(cfg["d"], 64)
        self.assertEqual(cfg["grid"], 32)
        self.assertIn("weights_sha256", cfg)

        g = Grounder.from_dir(out)
        self.assertEqual(g.step, 3)
        r = g.ground(sample_png(), "click the save button")
        self.assertTrue(0 <= r["x"] < 128 and 0 <= r["y"] < 128)

    def test_grip_dir_tamper_rejected(self):
        from brainbox.checkpoints import CheckpointError, migrate_grip_legacy
        from brainbox.grip import Grounder

        pt = os.path.join(self.tmp, "grip_best.pt")
        save_legacy_grip(pt)
        out = os.path.join(self.tmp, "grip")
        migrate_grip_legacy(pt, out)
        with open(os.path.join(out, "weights.pt"), "r+b") as f:
            f.seek(50)
            f.write(b"\x00")
        with self.assertRaises(CheckpointError):
            Grounder.from_dir(out)

    def test_migrate_script_both_kinds(self):
        pt = os.path.join(self.tmp, "grip_best.pt")
        save_legacy_grip(pt)
        out = os.path.join(self.tmp, "grip2")
        r = subprocess.run([sys.executable, "scripts/migrate_checkpoint.py", pt,
                            "--kind", "grip", "--out", out],
                           capture_output=True, text=True, cwd=ROOT, timeout=180)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.exists(os.path.join(out, "config.json")))


class TestFindAndDoctor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bb_m2_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("BRAINBOX_CHECKPOINT", None)
        os.environ.pop("BRAINBOX_GRIP_CHECKPOINT", None)

    def _make_dirs(self):
        from brainbox.checkpoints import make_stub, migrate_grip_legacy

        lm = os.path.join(self.tmp, "lm")
        make_stub(lm, TOK, seed=2, config_override=TINY_LM)
        pt = os.path.join(self.tmp, "grip_best.pt")
        save_legacy_grip(pt, step=9999)  # higher step than LM: must not win LM search
        grip = os.path.join(self.tmp, "grip")
        migrate_grip_legacy(pt, grip)
        return lm, grip

    def test_find_checkpoint_ignores_grip_dirs(self):
        from brainbox.checkpoints import find_checkpoint

        lm, grip = self._make_dirs()
        os.environ["BRAINBOX_CHECKPOINT"] = lm  # explicit path wins
        self.assertEqual(find_checkpoint(), (lm, "dir"))
        del os.environ["BRAINBOX_CHECKPOINT"]
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            # cwd has no checkpoints/ wrapper here; point the search at tmp
            # roots via explicit subdirs instead:
            os.makedirs("checkpoints", exist_ok=True)
            shutil.copytree(lm, "checkpoints/lm")
            shutil.copytree(grip, "checkpoints/grip")
            found = find_checkpoint()
            self.assertIsNotNone(found)
            self.assertTrue(found[0].endswith(os.path.join("checkpoints", "lm")),
                            f"LM search picked {found}")
        finally:
            os.chdir(cwd)

    def test_doctor_reports_grip_and_verifies(self):
        from brainbox.cli import main

        lm, grip = self._make_dirs()
        os.environ["BRAINBOX_CHECKPOINT"] = lm
        os.environ["BRAINBOX_GRIP_CHECKPOINT"] = grip
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["doctor", "--json"])
        d = json.loads(buf.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(d["checkpoint"]["verified"])
        self.assertTrue(d["grip_checkpoint"]["found"])
        self.assertEqual(d["grip_checkpoint"]["step"], 9999)
        self.assertTrue(d["grip_checkpoint"]["verified"])

    def test_doctor_fails_on_corrupt_checkpoint(self):
        from brainbox.cli import main

        lm, grip = self._make_dirs()
        with open(os.path.join(lm, "weights.pt"), "r+b") as f:
            f.seek(10)
            f.write(b"\xff")
        os.environ["BRAINBOX_CHECKPOINT"] = lm
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["doctor", "--json"])
        d = json.loads(buf.getvalue())
        self.assertEqual(code, 1)
        self.assertFalse(d["ok"])
        self.assertFalse(d["checkpoint"]["verified"])


if __name__ == "__main__":
    unittest.main()
