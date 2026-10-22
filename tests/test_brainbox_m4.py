"""M4 tests: undo/trash — destructive steps are reversible, byte-identical."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: str) -> str:
    with open(p, "rb") as f:
        return sha_bytes(f.read())


class TestTrash(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bb_m4_")
        self.home = os.path.join(self.tmp, "brainbox-home")
        self.work = os.path.join(self.tmp, "work")
        os.makedirs(self.work)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_file(self, rel, data: bytes, mode=0o644):
        p = os.path.join(self.work, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
        os.chmod(p, mode)
        return p

    def test_single_file_roundtrip_byte_identical(self):
        from brainbox.trash import restore, trash_path

        data = os.urandom(4097)
        p = self.make_file("a.bin", data, mode=0o640)
        mt = os.stat(p).st_mtime
        entry = trash_path("t-0001", p, home=self.home)
        self.assertFalse(os.path.exists(p))
        self.assertEqual(entry["sha256"], sha_bytes(data))
        rep = restore("t-0001", home=self.home)
        self.assertEqual(rep["failures"], [])
        self.assertEqual(sha_file(p), sha_bytes(data))
        self.assertEqual(oct(os.stat(p).st_mode & 0o777), "0o640")
        self.assertAlmostEqual(os.stat(p).st_mtime, mt, delta=1.0)

    def test_directory_tree_roundtrip(self):
        from brainbox.trash import restore, trash_path

        rng = random.Random(4)
        files = {}
        for rel in ("d/x.txt", "d/sub/y.bin", "d/sub/deep/z.dat"):
            data = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 5000)))
            files[rel] = sha_bytes(data)
            self.make_file(rel, data)
        os.makedirs(os.path.join(self.work, "d/empty"))
        trash_path("t-0002", os.path.join(self.work, "d"), home=self.home)
        self.assertFalse(os.path.exists(os.path.join(self.work, "d")))
        rep = restore("t-0002", home=self.home)
        self.assertEqual(rep["failures"], [])
        for rel, h in files.items():
            self.assertEqual(sha_file(os.path.join(self.work, rel)), h)
        self.assertTrue(os.path.isdir(os.path.join(self.work, "d/empty")))

    def test_same_basename_two_dirs(self):
        from brainbox.trash import restore, trash_path

        p1 = self.make_file("one/same.txt", b"first")
        p2 = self.make_file("two/same.txt", b"second")
        trash_path("t-0003", p1, home=self.home)
        trash_path("t-0003", p2, home=self.home)
        restore("t-0003", home=self.home)
        with open(p1, "rb") as f:
            self.assertEqual(f.read(), b"first")
        with open(p2, "rb") as f:
            self.assertEqual(f.read(), b"second")

    def test_restore_refuses_overwrite_then_allows(self):
        from brainbox.trash import restore, trash_path

        p = self.make_file("victim.txt", b"original")
        trash_path("t-0004", p, home=self.home)
        self.make_file("victim.txt", b"recreated")
        rep = restore("t-0004", home=self.home)
        self.assertEqual(len(rep["failures"]), 1)
        with open(p, "rb") as f:
            self.assertEqual(f.read(), b"recreated")  # not clobbered
        rep = restore("t-0004", home=self.home, overwrite=True)
        self.assertEqual(rep["failures"], [])
        with open(p, "rb") as f:
            self.assertEqual(f.read(), b"original")

    def test_errors(self):
        from brainbox.trash import TrashError, restore, trash_path

        with self.assertRaises((TrashError, FileNotFoundError)):
            trash_path("t-x", os.path.join(self.work, "nope"), home=self.home)
        with self.assertRaises(TrashError):
            restore("t-unknown", home=self.home)
        with self.assertRaises(ValueError):
            trash_path("../escape", self.make_file("e.txt", b"x"), home=self.home)
        with self.assertRaises(TrashError):  # never eat your own trash dir
            p = self.make_file("f.txt", b"x")
            trash_path("t-ok", p, home=self.home)
            trash_path("t-ok", self.home, home=self.home)

    def test_100_cycles_byte_identical(self):
        from brainbox.trash import list_trash, restore, trash_path

        rng = random.Random(100)
        hashes = {}
        restored_total = 0
        for i in range(100):
            data = bytes(rng.randrange(256) for _ in range(rng.choice(
                (0, 1, 17, 1024, 65535))))
            p = self.make_file(f"cycle_{i}.bin", data)
            hashes[p] = sha_bytes(data)
            trash_path("t-cyc", p, home=self.home)
            self.assertFalse(os.path.exists(p))
            if i % 3 == 0 or i == 99:  # restore in batches; verify every file
                rep = restore("t-cyc", home=self.home)
                self.assertEqual(rep["failures"], [], f"cycle {i}")
                restored_total += len(rep["restored"])
                for rp in rep["restored"]:
                    self.assertEqual(sha_file(rp), hashes[rp], f"{rp}")
                    os.remove(rp)
        self.assertEqual(restored_total, 100)
        self.assertEqual(list_trash(home=self.home), [])


class TestUndoCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bb_m4cli_")
        self.home = os.path.join(self.tmp, "home")
        os.environ["BRAINBOX_HOME"] = self.home
        self.addCleanup(lambda: os.environ.pop("BRAINBOX_HOME", None))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cli_undo_and_list(self):
        from brainbox.cli import main
        from brainbox.trash import trash_path

        p = os.path.join(self.tmp, "gone.txt")
        with open(p, "wb") as f:
            f.write(b"bring me back")
        trash_path("t-7777", p)  # default home from env
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["undo", "--list", "--json"])
        self.assertEqual(code, 0)
        listing = json.loads(buf.getvalue())
        self.assertEqual([t["task_id"] for t in listing], ["t-7777"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["undo", "t-7777"])
        self.assertEqual(code, 0)
        with open(p, "rb") as f:
            self.assertEqual(f.read(), b"bring me back")
        with redirect_stdout(io.StringIO()):
            code = main(["undo", "t-7777"])  # already restored
        self.assertNotEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
