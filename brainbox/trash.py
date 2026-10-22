"""Undo/trash: destructive steps move targets aside, never straight to oblivion.

Layout: ``~/.brainbox/trash/<task-id>/`` (override root with ``BRAINBOX_HOME``)
holds the moved payloads plus a ``manifest.json`` recording, per entry, the
original absolute path, sha256 (per file; for a directory a hash over its
sorted file list), size, mode and mtime. ``restore`` moves everything back
and re-verifies hashes; a file only counts as restored when it is
byte-identical. Restores never clobber an existing path unless asked.

Spec: specs/task_schema.md rule 5, specs/drive.md (destructive actions).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time

_SAFE_TASK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class TrashError(RuntimeError):
    pass


def brainbox_home(home: str | None = None) -> str:
    return (home or os.environ.get("BRAINBOX_HOME")
            or os.path.expanduser("~/.brainbox"))


def trash_root(home: str | None = None) -> str:
    return os.path.join(brainbox_home(home), "trash")


def _check_task_id(task_id: str) -> str:
    if not _SAFE_TASK.match(task_id or ""):
        raise ValueError(f"unsafe task id: {task_id!r}")
    return task_id


def _task_dir(task_id: str, home: str | None) -> str:
    return os.path.join(trash_root(home), _check_task_id(task_id))


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _describe(path: str) -> dict:
    """Hash + size + stat for a file, symlink or directory tree."""
    st = os.lstat(path)
    base = {"mode": st.st_mode & 0o7777, "mtime": st.st_mtime}
    if os.path.islink(path):
        return {**base, "kind": "symlink", "target": os.readlink(path),
                "sha256": hashlib.sha256(os.readlink(path).encode()).hexdigest(),
                "size": 0}
    if os.path.isdir(path):
        h = hashlib.sha256()
        size = 0
        members = []
        for root, dirs, files in os.walk(path):
            dirs.sort()
            for name in sorted(files):
                fp = os.path.join(root, name)
                rel = os.path.relpath(fp, path)
                fh = _sha256_file(fp)
                members.append(rel)
                size += os.path.getsize(fp)
                h.update(f"{rel}\0{fh}\n".encode())
        return {**base, "kind": "dir", "sha256": h.hexdigest(), "size": size,
                "members": members}
    return {**base, "kind": "file", "sha256": _sha256_file(path),
            "size": os.path.getsize(path)}


def _load_manifest(task_dir: str) -> dict:
    path = os.path.join(task_dir, "manifest.json")
    if not os.path.exists(path):
        return {"task_id": os.path.basename(task_dir), "created_at": time.time(),
                "entries": []}
    with open(path) as f:
        return json.load(f)


def _save_manifest(task_dir: str, manifest: dict) -> None:
    path = os.path.join(task_dir, "manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)


def trash_path(task_id: str, path: str, home: str | None = None) -> dict:
    """Move `path` into the task's trash (the safe form of delete)."""
    src = os.path.abspath(path)
    root = trash_root(home)
    home_abs = os.path.abspath(brainbox_home(home))
    if src == home_abs or src == os.path.abspath(root) \
            or src.startswith(os.path.abspath(root) + os.sep):
        raise TrashError("refusing to trash the brainbox trash itself")
    if not os.path.lexists(src):
        raise TrashError(f"no such file: {src}")
    task_dir = _task_dir(task_id, home)
    os.makedirs(task_dir, exist_ok=True)
    manifest = _load_manifest(task_dir)
    if any(e["original"] == src for e in manifest["entries"]):
        raise TrashError(f"{src} is already in trash for {task_id}")
    entry = {"original": src,
             "stored": f"{len(manifest['entries']):04d}_{os.path.basename(src) or 'root'}",
             **_describe(src)}
    dest = os.path.join(task_dir, entry["stored"])
    if os.path.lexists(dest):
        raise TrashError(f"trash slot already occupied: {dest}")
    if entry["kind"] == "symlink":
        os.symlink(entry["target"], dest)
        os.remove(src)
    else:
        shutil.move(src, dest)
    if _describe(dest)["sha256"] != entry["sha256"]:
        raise TrashError(f"hash changed while trashing {src}; left at {dest}")
    manifest["entries"].append(entry)
    _save_manifest(task_dir, manifest)
    return entry


def restore(task_id: str, home: str | None = None, overwrite: bool = False) -> dict:
    """Move everything trashed under `task_id` back. Byte-identical or failed."""
    task_dir = _task_dir(task_id, home)
    if not os.path.isdir(task_dir):
        raise TrashError(f"no trash recorded for task {task_id}")
    manifest = _load_manifest(task_dir)
    restored, failures, remaining = [], [], []
    for entry in manifest["entries"]:
        src = os.path.join(task_dir, entry["stored"])
        dst = entry["original"]
        try:
            if not os.path.lexists(src):
                raise TrashError("stored payload missing")
            if os.path.lexists(dst):
                if not overwrite:
                    raise TrashError("original path exists (refusing to overwrite)")
                if os.path.isdir(dst) and not os.path.islink(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            if entry["kind"] == "symlink":
                os.symlink(entry["target"], dst)
                os.remove(src)
            else:
                shutil.move(src, dst)
            if _describe(dst)["sha256"] != entry["sha256"]:
                raise TrashError("restored bytes differ from recorded sha256")
            try:
                if entry["kind"] != "symlink":
                    os.chmod(dst, entry["mode"])
                    os.utime(dst, (entry["mtime"], entry["mtime"]))
            except OSError:
                pass  # content is the contract; metadata is best-effort
            restored.append(dst)
        except Exception as e:
            failures.append({"path": dst, "error": str(e)})
            remaining.append(entry)
    if remaining:
        manifest["entries"] = remaining
        _save_manifest(task_dir, manifest)
    else:
        shutil.rmtree(task_dir, ignore_errors=True)
    return {"task_id": task_id, "restored": restored, "failures": failures}


def list_trash(home: str | None = None) -> list[dict]:
    root = trash_root(home)
    out = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if os.path.isdir(d):
                m = _load_manifest(d)
                out.append({"task_id": name, "entries": len(m["entries"]),
                            "bytes": sum(e.get("size", 0) for e in m["entries"]),
                            "created_at": m.get("created_at")})
    return out


def purge(task_id: str, home: str | None = None) -> bool:
    """Permanently delete a task's trash. Explicit and irreversible."""
    task_dir = _task_dir(task_id, home)
    if not os.path.isdir(task_dir):
        return False
    shutil.rmtree(task_dir)
    return True
