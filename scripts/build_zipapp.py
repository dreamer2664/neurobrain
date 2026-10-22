"""Build a single-file brainbox zipapp (python >= 3.10 + torch required
on the target; the .pyz bundles brainbox + neurobrain code and the
tokenizer, not torch itself).

    python3 scripts/build_zipapp.py --out dist/brainbox.pyz
    python3 dist/brainbox.pyz doctor
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
import tempfile
import zipapp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist/brainbox.pyz")
    args = ap.parse_args()

    stage = tempfile.mkdtemp(prefix="bb_zipapp_")
    try:
        for pkg in ("brainbox", "neurobrain"):
            shutil.copytree(os.path.join(ROOT, pkg), os.path.join(stage, pkg),
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        with open(os.path.join(stage, "__main__.py"), "w") as f:
            f.write("import sys\nfrom brainbox.cli import main\n"
                    "if __name__ == '__main__':\n    sys.exit(main())\n")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        zipapp.create_archive(stage, target=args.out,
                              interpreter="/usr/bin/env python3")
        mode = os.stat(args.out).st_mode
        os.chmod(args.out, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"zipapp -> {args.out} ({os.path.getsize(args.out)} bytes)")
        return 0
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
