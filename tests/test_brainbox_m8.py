"""M8 tests: suite definition sanity, scoreboard math, clean-clone finders."""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TOK = os.path.join(ROOT, "data", "tokenizer.json")


class TestSuiteDefinition(unittest.TestCase):
    def test_twenty_tasks_unique_ids_valid_categories(self):
        from scripts.e2e_suite import CATEGORIES, SUITE

        self.assertEqual(len(SUITE), 20)
        ids = [i for i, _, _ in SUITE]
        self.assertEqual(len(set(ids)), 20)
        for _, cat, _ in SUITE:
            self.assertIn(cat, CATEGORIES)

    def test_scoreboard_math(self):
        from scripts.e2e_suite import scoreboard

        rows = [{"id": "a", "category": "editor", "success": True,
                 "attempts": 2, "wall_s": 0.5, "why": ""},
                {"id": "b", "category": "editor", "success": False,
                 "attempts": 9, "wall_s": 1.5, "why": "missed"},
                {"id": "c", "category": "code", "success": True,
                 "attempts": 1, "wall_s": 0.1, "why": ""}]
        sb = scoreboard(rows, "abc123", 4600, 750, 300.0)
        self.assertEqual(sb["n"], 3)
        self.assertAlmostEqual(sb["success"], 2 / 3, places=3)
        self.assertEqual(sb["by_category"]["editor"]["rate"], 0.5)
        self.assertEqual(sb["failures"], [{"id": "b", "why": "missed"}])
        self.assertEqual(sb["median_steps"], 2)


class TestCleanCloneFinders(unittest.TestCase):
    def test_weightless_checkpoint_dir_is_skipped(self):
        from brainbox.checkpoints import find_checkpoint, find_grip_checkpoint

        tmp = tempfile.mkdtemp(prefix="bb_m8clean_")
        cwd = os.getcwd()
        try:
            # committed-shape dirs: config + tokenizer + card, NO weights.pt
            for kind, name in (("lm", "lm"), ("grip", "grip")):
                d = os.path.join(tmp, "checkpoints", name)
                os.makedirs(d)
                with open(os.path.join(d, "config.json"), "w") as f:
                    json.dump({"kind": kind, "step": 4600, "stub": False,
                               "vocab_size": 2052, "weights_sha256": "0" * 64}, f)
                shutil.copyfile(TOK, os.path.join(d, "tokenizer.json"))
            os.chdir(tmp)
            for finder in (find_checkpoint, find_grip_checkpoint):
                found = finder()
                if found and found[1] == "dir":
                    # never a placeholder: a returned dir must be loadable
                    self.assertTrue(os.path.exists(
                        os.path.join(found[0], "weights.pt")), found)
                    self.assertNotIn(tmp, found[0])
        finally:
            os.chdir(cwd)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
