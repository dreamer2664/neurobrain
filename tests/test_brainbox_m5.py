"""M5 tests: packaging — bundled tokenizer, zipapp, wheel."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class TestBundledTokenizer(unittest.TestCase):
    def test_bundled_copy_identical_to_data(self):
        from brainbox.checkpoints import bundled_tokenizer_path

        bundled = bundled_tokenizer_path()
        self.assertIsNotNone(bundled)
        self.assertEqual(sha(bundled), sha(os.path.join(ROOT, "data/tokenizer.json")))

    def test_bundled_loads(self):
        from brainbox.checkpoints import bundled_tokenizer_path
        from neurobrain.tokenizer import BPETokenizer

        tok = BPETokenizer.load(bundled_tokenizer_path())
        self.assertEqual(tok.vocab_size, 2052)


class TestZipapp(unittest.TestCase):
    def test_zipapp_builds_and_runs_doctor(self):
        tmp = tempfile.mkdtemp(prefix="bb_m5zip_")
        try:
            pyz = os.path.join(tmp, "brainbox.pyz")
            r = subprocess.run([sys.executable, "scripts/build_zipapp.py",
                                "--out", pyz],
                               capture_output=True, text=True, cwd=ROOT, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(os.path.exists(pyz))
            r = subprocess.run([sys.executable, pyz, "--version"],
                               capture_output=True, text=True, cwd=tmp, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("brainbox", r.stdout)
            env = {**os.environ, "BRAINBOX_HOME": os.path.join(tmp, "home")}
            r = subprocess.run([sys.executable, pyz, "doctor", "--json"],
                               capture_output=True, text=True, cwd=tmp,
                               timeout=180, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            d = json.loads(r.stdout)
            self.assertTrue(d["ok"], d)
            self.assertTrue(d["tokenizer"]["found"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestWheel(unittest.TestCase):
    def test_wheel_builds_with_entrypoint_and_data(self):
        tmp = tempfile.mkdtemp(prefix="bb_m5whl_")
        try:
            r = subprocess.run([sys.executable, "-m", "pip", "wheel",
                                "--no-deps", "--no-build-isolation",
                                "-w", tmp, "."],
                               capture_output=True, text=True, cwd=ROOT, timeout=600)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            wheels = [f for f in os.listdir(tmp) if f.endswith(".whl")]
            self.assertEqual(len(wheels), 1)
            with zipfile.ZipFile(os.path.join(tmp, wheels[0])) as z:
                names = z.namelist()
                self.assertIn("brainbox/cli.py", names)
                self.assertIn("brainbox/tokenizer.json", names)
                self.assertIn("neurobrain/model.py", names)
                ep = [n for n in names if n.endswith("entry_points.txt")][0]
                self.assertIn("brainbox = brainbox.cli:main",
                              z.read(ep).decode())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
