"""M2 — demo app as the test rig: drive it with the real X11 primitives.

The scenario is the one in the brief: 20 clicks, 200 typed chars, toggles, and
a drag, executed through ``DesktopHands`` (XTest) against a live tkinter window
under Xvfb, asserting on the app's own dumped state.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from deskhands import DesktopHands
from deskhands.demo_app import (
    BTN_ABOUT, BTN_NEW, CHECK, ENTRY, SCALE,
)

REPO = Path(__file__).resolve().parent.parent


class Client:
    def __init__(self, port: int) -> None:
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.fh = self.sock.makefile("rwb")

    def cmd(self, **msg):
        self.fh.write((json.dumps(msg) + "\n").encode())
        self.fh.flush()
        return json.loads(self.fh.readline())

    def state(self) -> dict:
        return self.cmd(op="state")

    def rects(self) -> dict:
        return self.cmd(op="rects")

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def center(rect):
    return rect[0] + rect[2] // 2, rect[1] + rect[3] // 2


def frac(rect, fx, fy=0.5):
    return rect[0] + int(rect[2] * fx), rect[1] + int(rect[3] * fy)


def start_app():
    pf = tempfile.mktemp(prefix="demo.port.")
    proc = subprocess.Popen(
        [sys.executable, "-m", "deskhands.demo_app", "--port-file", pf],
        cwd=str(REPO), env=os.environ,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        if os.path.exists(pf):
            port = int(open(pf).read().strip())
            return proc, Client(port)
        if proc.poll() is not None:
            out = proc.stdout.read().decode() if proc.stdout else ""
            pytest.fail(f"demo app exited early (rc={proc.returncode}):\n{out}")
        time.sleep(0.1)
    proc.kill()
    pytest.fail("demo app did not publish a port in 30s")


@pytest.fixture
def app():
    proc, client = start_app()
    try:
        yield client
    finally:
        client.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def do(h, action):
    r = h.perform(action)
    assert r["ok"], r
    return r


def click_at(h, rect, fx=0.5, fy=0.5):
    x, y = frac(rect, fx, fy)
    do(h, {"type": "click", "x": x, "y": y})


def test_live_scenario_20_clicks_200_chars_toggles_drag(app):
    r = app.rects()
    h = DesktopHands()
    try:
        s0 = app.state()
        assert s0["counters"]["clicks"] == 0

        # ---- 20 real clicks ------------------------------------------------
        plan = (
            [BTN_NEW] * 6 + [f"item{i}" for i in (0, 1, 2)]
            + [CHECK] + [ENTRY] + [SCALE] * 2
            + [BTN_NEW] * 7
        )
        assert len(plan) == 20
        for target in plan:
            if target.startswith("item"):
                click_at(h, r["items"][target[4:]])
            else:
                click_at(h, r[target])
            time.sleep(0.03)
        time.sleep(0.2)
        s = app.state()

        assert s["counters"]["clicks"] >= 20, s["counters"]
        assert s["counters"]["new"] == 13, s["counters"]
        assert len(s["files"]) == len(s0["files"]) + 13
        assert s["counters"]["toggle"] == 1 and s["show_hidden"] is True
        assert s["selected"] == [s["files"][2]]  # last plain item click wins

        # ---- 200 typed characters ----------------------------------------
        # focus the entry at a *different* spot than the plan's click (clicking
        # the entry twice at the same pixel clears the listbox selection in Tk).
        click_at(h, r[ENTRY], fx=0.75)
        time.sleep(0.1)
        do(h, {"type": "type", "text": "x" * 200})
        time.sleep(0.2)
        s = app.state()
        assert len(s["entry_text"]) == 200, len(s["entry_text"])
        assert s["counters"]["type"] == 200

        # ---- toggles (1 more; total 2 -> back to False) -------------------
        click_at(h, r[CHECK])
        time.sleep(0.03)
        s = app.state()
        assert s["counters"]["toggle"] == 2
        assert s["show_hidden"] is False  # 2 toggles from initial False -> False

        # ---- drag the canvas item; assert it lands within 2 px ------------
        di = r["drag_item"]
        dx, dy = center(di)
        do(h, {"type": "drag", "x": dx, "y": dy, "x2": dx + 100, "y2": dy + 40, "ms": 60})
        time.sleep(0.15)
        s = app.state()
        assert s["counters"]["drag"] >= 1
        # item started at canvas (20,20); a +100/+40 drag puts its corner at
        # (120,60). The spec requires the drag to land within 2 px.
        assert abs(s["drag_pos"][0] - 120) <= 2, s["drag_pos"]
        assert abs(s["drag_pos"][1] - 60) <= 2, s["drag_pos"]

        # ---- dialog: open, then close (last, so it can't steal focus) -----
        click_at(h, r[BTN_ABOUT])
        time.sleep(0.15)
        assert app.state()["dialog_open"] is True
        r = app.rects()
        click_at(h, r["dialog_ok"])
        time.sleep(0.1)
        assert app.state()["dialog_open"] is False
    finally:
        h.close()


def test_script_mode_is_deterministic():
    ops = {
        "ops": [
            {"op": "click", "target": BTN_NEW},
            {"op": "click", "target": BTN_NEW},
            {"op": "click", "target": CHECK},
            {"op": "select", "indexes": [1, 2]},
            {"op": "type", "text": "hello world"},
            {"op": "set", "value": 48},
        ]
    }
    path = tempfile.mktemp(suffix=".json")
    with open(path, "w") as fh:
        json.dump(ops, fh)
    proc = subprocess.run(
        [sys.executable, "-m", "deskhands.demo_app", "--script", path],
        cwd=str(REPO), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    s = json.loads(proc.stdout)
    assert s["counters"]["new"] == 2
    assert s["show_hidden"] is True
    assert s["selected"] == ["app.py", "notes.txt"]
    assert s["entry_text"] == "hello world"
    assert s["icon_size"] == 48
