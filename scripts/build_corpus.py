#!/usr/bin/env python3
"""Build the training corpus: real Python source + synthetic desktop transcripts.

Code sources (all permissively licensed open source):
  1. The CPython standard library shipped with the interpreter.
  2. Installed site-packages (pure-python libraries).
  3. Source distributions downloaded into /tmp/sdist (flask, django, ...).

Desktop sources:
  4. Transcripts from the synthetic desktop world in neurobrain/ui_world.py,
     plus a hand-written seed corpus of canonical desktop-control examples.

Outputs uint16 token streams + a BPE tokenizer, ready for train_lm.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tarfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from neurobrain.tokenizer import train_bpe, BPETokenizer, SPECIAL_TOKENS  # noqa: E402
from neurobrain.ui_world import protocol_corpus  # noqa: E402
from neurobrain.data import write_tokens  # noqa: E402

SEED_UI_CORPUS = """\
<|ui|><|screen|>1920x1080<|window|>Files<el>button[1400,60,120,40]{{submit}}<el>field[300,60,700,40]{{search}}<el>tab[60,20,120,40]{{home}}<|task|>click the submit button<|action|>click(1462,79)<|endoftext|>
<|ui|><|screen|>1920x1080<|window|>Settings<el>checkbox[200,300,24,24]{{save}}<el>slider[400,500,300,20]{{filter}}<|task|>toggle the save checkbox<|action|>toggle(212,312)<|endoftext|>
<|ui|><|screen|>1920x1080<|window|>Editor<el>menu[1200,40,160,32]{{rename}}<el>button[1350,90,100,36]{{cancel}}<|task|>choose rename from the menu<|action|>click(1281,57)<|endoftext|>
<|ui|><|screen|>1920x1080<|window|>Browser<el>field[400,80,800,44]{{login}}<el>button[420,140,90,32]{{submit}}<|task|>type into the login field<|action|>type(803,102)<|endoftext|>
<|ui|><|screen|>1920x1080<|window|>Photos<el>icon[100,200,64,64]{{upload}}<el>icon[200,200,64,64]{{download}}<|task|>click the upload icon<|action|>click(131,231)<|endoftext|>
<|ui|><|screen|>1920x1080<|window|>Mail<el>button[900,300,110,38]{{archive}}<el>button[760,300,110,38]{{delete}}<|task|>press archive<|action|>click(953,318)<|endoftext|>
"""

SKIP_DIRS = {"test", "tests", "test_", "__pycache__", "site-packages", "dist-packages",
             ".git", "build", "dist", "node_modules"}


def iter_files(roots: list[str], min_bytes: int = 400, max_bytes: int = 400_000):
    for root in roots:
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in SKIP_DIRS and not d.startswith(".")]
            for f in fn:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dp, f)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    continue
                if min_bytes <= sz <= max_bytes:
                    yield p


def extract_sdists(dirpath: str, workdir: str) -> list[str]:
    out_roots = []
    if not os.path.isdir(dirpath):
        return out_roots
    for name in sorted(os.listdir(dirpath)):
        if not name.endswith((".tar.gz", ".tgz")):
            continue
        dest = os.path.join(workdir, name.replace(".tar.gz", "").replace(".tgz", ""))
        if not os.path.isdir(dest):
            os.makedirs(dest, exist_ok=True)
            try:
                with tarfile.open(os.path.join(dirpath, name)) as tf:
                    tf.extractall(dest, filter="data")
            except Exception as e:  # noqa: BLE001
                print(f"[corpus] skip {name}: {e}")
                continue
        out_roots.append(dest)
    return out_roots


def read_py(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except (UnicodeDecodeError, OSError):
        return None
    if src.count("\x00"):
        return None
    return src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    ap.add_argument("--sdist-dir", default="/tmp/sdist")
    ap.add_argument("--workdir", default="/tmp/sdist_extracted")
    ap.add_argument("--max-code-bytes", type=int, default=180_000_000)
    ap.add_argument("--ui-samples", type=int, default=200_000)
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import sysconfig
    stdlib = sysconfig.get_paths()["stdlib"]
    site = sysconfig.get_paths()["purelib"]
    sdist_roots = extract_sdists(args.sdist_dir, args.workdir)
    roots = [stdlib, site] + sdist_roots
    print(f"[corpus] scanning {len(roots)} roots")

    files = list(iter_files(roots))
    rng = random.Random(args.seed)
    rng.shuffle(files)

    seen: set[str] = set()
    docs: list[str] = []
    total = 0
    for p in files:
        if total >= args.max_code_bytes:
            break
        src = read_py(p)
        if not src:
            continue
        h = hashlib.sha1(src.encode("utf-8", "ignore")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        docs.append(src)
        total += len(src.encode("utf-8"))
    print(f"[corpus] collected {len(docs)} python files, {total/1e6:.1f} MB")

    rng.shuffle(docs)
    n_val = max(8, int(0.03 * len(docs)))
    val_docs, train_docs = docs[:n_val], docs[n_val:]

    t0 = time.time()
    ui_train = list(protocol_corpus(int(args.ui_samples * 0.97), seed=args.seed))
    ui_val = list(protocol_corpus(int(args.ui_samples * 0.03) + 8, seed=args.seed + 1))
    ui_train = [SEED_UI_CORPUS] * max(1, len(ui_train) // 200) + ui_train
    print(f"[corpus] ui transcripts: {len(ui_train)} train / {len(ui_val)} val "
          f"({sum(len(s) for s in ui_train)/1e6:.1f} MB) in {time.time()-t0:.1f}s")

    # Tokenizer is learned on a balanced sample: code dominates the mixture, so
    # learn merges mostly from code but reserve some of the budget for UI text.
    sample_rng = random.Random(args.seed + 99)
    code_sample = sample_rng.sample(train_docs, k=min(len(train_docs), 1200))
    code_text = "\n".join(code_sample)
    ui_text = "".join(ui_train[:4000])
    bpe_texts = [code_text[:14_000_000], ui_text]
    print(f"[corpus] learning BPE: {len(bpe_texts[0])/1e6:.1f}MB code + {len(bpe_texts[1])/1e6:.1f}MB ui")
    tok = train_bpe(bpe_texts, vocab_size=args.vocab_size, specials=SPECIAL_TOKENS)
    tok.save(os.path.join(args.out, "tokenizer.json"))
    print(f"[corpus] tokenizer saved: vocab {tok.vocab_size}")

    def encode_all(texts: list[str], tag: str) -> int:
        ids: list[int] = []
        t = time.time()
        for i, s in enumerate(texts):
            ids.extend(tok.encode(s))
            ids.append(tok.encode("<|endoftext|>")[0])
            if (i + 1) % 2000 == 0:
                print(f"[corpus] {tag}: {i+1}/{len(texts)} docs, {len(ids)/1e6:.2f}M tokens, "
                      f"{len(ids)/max(1e-9, time.time()-t)/1e3:.0f}k tok/s", flush=True)
        return ids

    train_ids = encode_all(train_docs, "code-train")
    val_ids = encode_all(val_docs, "code-val")
    ui_train_ids = encode_all(ui_train, "ui-train")
    ui_val_ids = encode_all(ui_val, "ui-val")

    stats = {}
    for name, ids in [("code_train", train_ids), ("code_val", val_ids),
                      ("ui_train", ui_train_ids), ("ui_val", ui_val_ids)]:
        write_tokens(os.path.join(args.out, f"{name}.bin"), ids)
        stats[name] = len(ids)
        print(f"[corpus] wrote {name}.bin: {len(ids)/1e6:.2f}M tokens")

    # report: compression ratio sanity check
    raw = sum(len(t.encode()) for t in train_docs)
    report = {
        "code_files": len(docs),
        "code_bytes": total,
        "tokens": stats,
        "vocab_size": tok.vocab_size,
        "code_bytes_per_token": round(raw / max(1, stats["code_train"]), 3),
        "ui_examples": len(ui_train) + len(ui_val),
        "sdist_roots": [os.path.basename(r) for r in sdist_roots],
    }
    with open(os.path.join(ROOT, "reports", "corpus_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("[corpus] report:", json.dumps(report)[:400])


if __name__ == "__main__":
    main()
