"""Code corpus v2 (M6): AST-filtered, function-deduped, categorised, split clean.

Pipeline per file:

    read -> ast.parse (syntax filter, no regex tricks) -> per function:
        docstring, source segment, semantic hash (ast.dump), keyword category

Splits are assigned on the **function hash**, not the file: a function that
appears in three files lands in exactly one split, so a validation function can
never be seen during training (the classic copy-paste leak in stdlib mirrors).

Outputs:

    codev2_2052_train.bin / _val.bin        functions packed for the 2052 BPE
    codev2_instr_train.bin / _val.bin       <|task|>docstring<|code|>body pairs
    codev2_4096_train.bin / _val.bin        same functions through the 4096 code BPE
    code_tokenizer_4096.json                the code-only tokenizer
    manifest.json                           counts, categories, bytes/token per tokenizer
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import sysconfig
import sys

from ..profiles import sub_seed
from ..streams import TokenStreamWriter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neurobrain.tokenizer import SPECIAL_TOKENS, train_bpe  # noqa: E402
from .code_corpus import iter_candidates, read_py  # noqa: E402

CATEGORIES = ["core", "web", "data", "test", "cli", "net", "fileio", "ui"]
CAT_SIGNALS = {
    "web": ["flask", "django", "fastapi", "http", "urllib", "requests", "wsgi", "asgi",
            "route", "server", "template", "cookie", "cgi", "html"],
    "data": ["numpy", "pandas", "csv", "json", "sqlite", "sql", "array", "dataframe",
             "statistics", "decimal", "fractions", "struct", "pickle"],
    "test": ["test", "unittest", "pytest", "assert", "mock", "fixture"],
    "cli": ["argparse", "click", "optparse", "optionparser", "sys.argv", "main(",
            "getopt", "console_scripts"],
    "net": ["socket", "asyncio", "ssl", "websocket", "tcp", "udp", "smtp", "ftp",
            "http.client", "select", "selectors"],
    "fileio": ["shutil", "pathlib", "os.path", "glob", "tempfile", "open(", "file",
               "path", "io.", "zipfile", "tarfile"],
    "ui": ["tkinter", "curses", "rich", "tty", "termios", "textual", "pygame", "qt"],
}
VERB_STARTS = {
    "return", "compute", "parse", "load", "save", "build", "get", "set", "check",
    "convert", "render", "fetch", "send", "read", "write", "open", "close", "create",
    "delete", "update", "find", "list", "add", "remove", "format", "encode", "decode",
    "validate", "normalise", "normalize", "extract", "merge", "split", "join", "run",
    "start", "stop", "handle", "process", "apply", "make", "generate", "scan", "walk",
    "resolve", "detect", "measure", "count", "compare", "sort", "map", "filter",
}


def _func_hash(node: ast.AST) -> str:
    return hashlib.sha1(ast.dump(node, include_attributes=False).encode()).hexdigest()


def _category(path: str, src: str, node: ast.AST) -> str:
    hay = (path + " " + src[:4000] + " " + getattr(node, "name", "")).lower()
    best, best_score = "core", 0
    for cat, words in CAT_SIGNALS.items():
        score = sum(hay.count(w) for w in words)
        if score > best_score:
            best, best_score = cat, score
    return best


def _instruction(doc: str) -> str | None:
    text = " ".join(doc.strip().split())
    if len(text) < 12:
        return None
    first = text.split(". ")[0].rstrip(".")
    words = first.split()
    if not words or len(words) < 3:
        return None
    if len(first) > 160:
        first = " ".join(words[:18])
    if words[0].lower().strip(":") in VERB_STARTS:
        return first[0].lower() + first[1:]
    return "write a function that " + first[0].lower() + first[1:]


def extract_functions(path: str, src: str, min_chars: int = 70, max_chars: int = 4000):
    """Yield (hash, source, docstring-or-None, category) for every real function."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return
    lines = src.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if not start or not end:
            continue
        body = "\n".join(lines[start - 1:end])
        if not (min_chars <= len(body) <= max_chars):
            continue
        if body.count("\n") < 1:
            continue
        yield (_func_hash(node), body, ast.get_docstring(node), _category(path, src, node))


def build(dirpath: str, cfg: dict, seed: int, tok, log=print) -> dict:
    roots = [sysconfig.get_paths()[k] for k in ("stdlib", "purelib") if k in sysconfig.get_paths()]
    cands = iter_candidates(roots, cfg.get("min_bytes", 400))
    rng = random.Random(sub_seed(seed, "code_v2"))
    order = list(range(len(cands)))
    rng.shuffle(order)

    # read files until the byte budget is met (sizes only: no hashing pass)
    budget = int(cfg.get("max_bytes", 40_000_000))
    picked, total = [], 0
    for i in order:
        root, rel, sz = cands[i]
        if total + sz > budget:
            break
        picked.append((root, rel, sz))
        total += sz
    log(f"[code_v2] {len(picked)} files, {total/1e6:.1f} MB on disk")

    val_frac = float(cfg.get("val_frac", 0.03))
    seen: dict[str, str] = {}          # function hash -> split
    docs: list[tuple[str, str, str | None, str]] = []   # split, body, doc, category
    parse_fail = 0
    files_used = 0
    for root, rel, _ in picked:
        p = os.path.join(root, rel)
        src = read_py(p)
        if not src:
            continue
        files_used += 1
        file_split = "val" if (int(hashlib.sha1(p.encode()).hexdigest()[:8], 16) % 1000
                               < val_frac * 1000) else "train"
        before = len(docs)
        for fh, body, doc, cat in extract_functions(p, src):
            # the same function copy-pasted across the stdlib is ONE sample:
            # emit the first occurrence only, in the split that file owns
            if fh in seen:
                continue
            seen[fh] = file_split
            docs.append((file_split, body, doc, cat))
        if len(docs) == before:
            parse_fail += 1
    train = [d for d in docs if d[0] == "train"]
    val = [d for d in docs if d[0] == "val"]
    log(f"[code_v2] {files_used} files parsed, {len(docs):,} unique functions "
        f"({parse_fail} files with none), split {len(train):,}/{len(val):,}")

    # ---- the 4096 code-only tokenizer: trained on the train split only -------
    tok_bytes = int(cfg.get("tok_bytes", 12_000_000))
    corpus, used = [], 0
    for _, body, _, _ in train:
        corpus.append(body)
        used += len(body.encode())
        if used >= tok_bytes:
            break
    log(f"[code_v2] training 4096 BPE on {used/1e6:.1f} MB of code")
    # 4096 total ids, of which 4 are reserved specials -- the same convention as
    # the checkpoint's 2052-token tokenizer (2048 BPE + 4 specials)
    tok4096 = train_bpe(corpus, 4096 - len(SPECIAL_TOKENS), specials=SPECIAL_TOKENS, verbose=False)
    tok_path = os.path.join(dirpath, "code_tokenizer_4096.json")
    tok4096.save(tok_path)
    log(f"[code_v2] {os.path.basename(tok_path)} vocab {tok4096.vocab_size}")

    # ---- streams ------------------------------------------------------------
    eot = tok.encode("<|endoftext|>")[0]
    eot4 = tok4096.encode("<|endoftext|>")[0]
    shards = []
    bytes_2052 = tokens_2052 = bytes_4096 = tokens_4096 = 0
    instr_docs = 0
    instr_chars = 0

    def write_stream(name: str, items, tokenizer, eot_id, count_bytes: str | None,
                     tok_file: str | None = None, role: str = "primary"):
        nonlocal bytes_2052, tokens_2052, bytes_4096, tokens_4096
        w = TokenStreamWriter(os.path.join(dirpath, name))
        b = t = 0
        for text in items:
            ids = tokenizer.encode(text)
            w.write(ids + [eot_id])
            b += len(text.encode())
            t += len(ids)
        rec = w.close()
        rec["role"] = role
        if tok_file:
            rec["tokenizer"] = tok_file   # stats/verify need the right vocabulary
        if count_bytes == "2052":
            bytes_2052 += b
            tokens_2052 += t
        elif count_bytes == "4096":
            bytes_4096 += b
            tokens_4096 += t
        log(f"[code_v2] {rec['name']}: {rec['docs']:,} docs, {rec['tokens']/1e6:.2f}M tokens, "
            f"{b/max(1,t):.2f} bytes/token")
        shards.append(rec)
        return rec

    bodies = {"train": [d[1] for d in train], "val": [d[1] for d in val]}
    for split in ("train", "val"):
        write_stream(f"codev2_2052_{split}.bin", bodies[split], tok, eot, "2052",
                     role="primary")
    for split in ("train", "val"):
        write_stream(f"codev2_4096_{split}.bin", bodies[split], tok4096, eot4, "4096",
                     tok_file=os.path.basename(tok_path), role="alt_tokenizer")

    pairs = {"train": [], "val": []}
    for split, body, doc, cat in docs:
        if not doc:
            continue
        ins = _instruction(doc)
        if not ins:
            continue
        pairs[split].append(f"<|task|>{ins}<|code|>{body}")
        instr_docs += 1
        instr_chars += len(ins)
    for split in ("train", "val"):
        write_stream(f"codev2_instr_{split}.bin", pairs[split], tok, eot, None,
                     role="instruction_pairs")

    cat_hist = {c: 0 for c in CATEGORIES}
    for split, body, doc, cat in docs:
        cat_hist[cat] = cat_hist.get(cat, 0) + 1
    content = {
        "source": "stdlib+purelib, ast.parse filtered",
        "files_parsed": files_used,
        "functions": len(docs),
        "functions_train": len(train),
        "functions_val": len(val),
        "dedupe": "sha1(ast.dump(node)) across all files, first-file-wins split",
        "categories": {k: round(v / max(1, len(docs)), 4) for k, v in cat_hist.items()},
        "instruction_pairs": instr_docs,
        "instruction_mean_chars": round(instr_chars / max(1, instr_docs), 1),
        "bytes_per_token_2052": round(bytes_2052 / max(1, tokens_2052), 4),
        "bytes_per_token_4096": round(bytes_4096 / max(1, tokens_4096), 4),
        "vocab_2052": tok.vocab_size,
        "vocab_4096": tok4096.vocab_size,
        "tokenizer_4096": {"name": os.path.basename(tok_path),
                           "sha256": _sha(tok_path), "bytes": os.path.getsize(tok_path)},
        "tok_train_bytes": used,
    }
    log(f"[code_v2] bytes/token 2052={content['bytes_per_token_2052']} "
        f"4096={content['bytes_per_token_4096']} "
        f"(+{100*(content['bytes_per_token_4096']/content['bytes_per_token_2052']-1):.1f}%), "
        f"{instr_docs:,} instruction pairs")
    return {
        "kind": "token_streams",
        "dataset": "code_v2",
        "n": len(docs),
        "units": "functions",
        "shards": shards,
        "content": content,
    }


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
