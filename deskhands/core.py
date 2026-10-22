"""Portable facade for the twelve desktop actions.

M0 implements Linux/X11 directly with ctypes. Platform-specific backends are
selected here as they become available.
"""
from __future__ import annotations

import subprocess
import time
from typing import Any

from .x11 import X11Backend

ACTIONS = ("click", "double_click", "right_click", "move", "drag", "scroll", "type", "key", "wait", "focus_window", "launch", "screen")


class DesktopHands:
    def __init__(self) -> None:
        self.backend = X11Backend()

    def close(self) -> None:
        self.backend.close()

    def perform(self, action: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic_ns()
        kind = action.get("type", "")
        detail = ""
        ok = False
        for attempt in range(2):
            try:
                if kind not in ACTIONS:
                    raise ValueError(f"unknown action: {kind}")
                detail = self._dispatch(kind, action)
                ok = True
                break
            except Exception as exc:
                detail = str(exc)
                if attempt == 0:
                    time.sleep(0.002)
        elapsed = (time.monotonic_ns() - started) // 1_000_000
        return {"ok": ok, "detail": detail, "duration_ms": int(elapsed)}

    def _dispatch(self, kind: str, a: dict[str, Any]) -> str:
        if kind == "wait":
            ms = max(0, int(a.get("ms", 0)))
            time.sleep(ms / 1000)
            return f"waited {ms} ms"
        if kind == "launch":
            command = a["command"]
            p = subprocess.Popen(command, shell=isinstance(command, str), close_fds=True)
            return f"pid={p.pid}"
        if kind == "screen":
            return "observation requested"
        if kind == "focus_window":
            self.backend.focus_window(str(a["title_substr"]))
            return f"focused {a['title_substr']}"
        if kind == "type":
            self.backend.type_text(str(a.get("text", "")))
            return f"typed {len(str(a.get('text', '')))} chars"
        if kind == "key":
            self.backend.chord([str(k) for k in a["keys"]])
            return "chord sent"
        if kind == "drag":
            self.backend.drag(int(a["x"]), int(a["y"]), int(a["x2"]), int(a["y2"]), int(a.get("ms", 600)))
            return "dragged"
        if kind == "scroll":
            self.backend.scroll(int(a["x"]), int(a["y"]), int(a["dy"]))
            return "scrolled"
        x, y = int(a["x"]), int(a["y"])
        if kind == "move":
            self.backend.move(x, y)
        elif kind == "right_click":
            self.backend.click(x, y, 3, 1)
        elif kind == "double_click":
            self.backend.click(x, y, 1, 2)
        else:
            buttons = {"left": 1, "middle": 2, "right": 3}
            self.backend.click(x, y, buttons.get(str(a.get("button", "left")), 1), int(a.get("clicks", 1)))
        return kind.replace("_", " ") + " sent"
