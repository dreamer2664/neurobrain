"""Dependency-free demo file-manager: the test rig for deskhands.

A tkinter lookalike (list, buttons, entry, checkbox, dialog, slider) that is
scriptable from JSON and can dump its own state for assertions. No third-party
GUI libraries — only the Python standard library.

Two driving modes
-----------------
* ``--script ops.json``  — run a list of high-level JSON ops in-process, print
  the final state as JSON, and exit. Deterministic; used by unit tests.
* ``--port-file P``      — start the mainloop and listen on a TCP socket for
  ``state`` / ``rects`` / ``script`` / ``quit`` JSON-line commands. This is the
  live rig: the X11 backend can move the real pointer over the window and the
  test asks the app what it sees.

``rects`` reports every widget's physical screen geometry (root coords, origin
top-left), so ``DesktopHands`` can click/type/drag on the live window.

Usage
-----
    xvfb-run -a python -m deskhands.demo_app --dump-state
    xvfb-run -a python -m deskhands.demo_app --script scenario.json
    xvfb-run -a python -m deskhands.demo_app --port-file /tmp/demo.port
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox

DEFAULT_FILES = [
    "README.md", "app.py", "notes.txt", "report.pdf",
    "data.csv", "image.png", "script.sh", "archive.zip",
]

# widget ids the rig and the script language use
BTN_NEW = "btn_new"
BTN_DELETE = "btn_delete"
BTN_ABOUT = "btn_about"
BTN_QUIT = "btn_quit"
ENTRY = "entry"
CHECK = "check_hidden"
SCALE = "scale"
LISTBOX = "listbox"
DIALOG = "dialog"


class DemoApp(tk.Tk):
    """A scriptable file-manager lookalike."""

    def __init__(self, files=None) -> None:
        super().__init__()
        self.title("Files — demo")
        self.geometry("960x520")
        self.files = list(files or DEFAULT_FILES)
        self.selected: list[str] = []
        self.entry_text = ""
        self.show_hidden = False
        self.icon_size = 32
        self.drag_pos = [20, 20]
        self.dialog: tk.Toplevel | None = None
        self.counters = {
            "clicks": 0, "new": 0, "delete": 0, "toggle": 0,
            "dialog": 0, "slider": 0, "type": 0, "drag": 0, "scroll": 0,
        }
        self._last_op = None
        self.last_action = "init"
        self._build()
        self._refresh()

    # ------------------------------------------------------------------ ui
    def _build(self) -> None:
        bar = tk.Frame(self)
        bar.pack(side="top", fill="x", padx=6, pady=6)

        tk.Label(bar, text="Search:").pack(side="left")
        self.entry = tk.Entry(bar, width=28)
        self.entry.pack(side="left", padx=(4, 12))
        self.entry.bind("<KeyRelease>", self._on_entry)

        self.check_hidden = tk.Checkbutton(bar, text="Show hidden",
                                           command=self.toggle_hidden)
        self.check_hidden.pack(side="left", padx=(0, 12))

        self.scale = tk.Scale(bar, from_=8, to=64, orient="horizontal")
        self.scale.set(self.icon_size)  # set before command so init doesn't count
        self.scale.configure(command=self._on_scale)
        self.scale.pack(side="left", padx=(0, 12))

        self.btn_new = tk.Button(bar, text="New", command=self.new_file)
        self.btn_new.pack(side="left", padx=2)
        self.btn_delete = tk.Button(bar, text="Delete", command=self.delete_files)
        self.btn_delete.pack(side="left", padx=2)
        self.btn_about = tk.Button(bar, text="About", command=self.open_dialog)
        self.btn_about.pack(side="left", padx=2)
        self.btn_quit = tk.Button(bar, text="Quit", command=self.destroy)
        self.btn_quit.pack(side="left", padx=2)

        self.content = tk.Frame(self)
        self.content.pack(side="top", fill="both", expand=True, padx=6, pady=(0, 6))
        self.listbox = tk.Listbox(self.content, selectmode="extended")
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        self.listbox.bind("<MouseWheel>", lambda e: self._touch("scroll"))
        self.listbox.bind("<Button-4>", lambda e: self._touch("scroll"))
        self.listbox.bind("<Button-5>", lambda e: self._touch("scroll"))

        # A draggable canvas item: the rig's drag target. The app records where
        # the item landed so a test can assert the drag primitive's accuracy.
        self.canvas = tk.Canvas(self.content, width=300, height=200,
                                bg="#eeeeee", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.drag_size = 60
        self.drag_pos = [20, 20]
        self.drag_item = self.canvas.create_rectangle(
            20, 20, 20 + self.drag_size, 20 + self.drag_size,
            fill="steelblue", outline="#333333", tags="drag",
        )
        self.canvas.tag_bind("drag", "<ButtonPress-1>", self._drag_start)
        self.canvas.tag_bind("drag", "<B1-Motion>", self._drag_move)
        self.canvas.tag_bind("drag", "<ButtonRelease-1>", self._drag_end)

        self.status = tk.Label(self, text="ready", anchor="w")
        self.status.pack(side="bottom", fill="x")

        # Count every real pointer press/drag at application level. An app-level
        # binding runs *after* the widget's own binding, so it cannot shadow the
        # Scale's built-in value-setting (a widget-level bind there would).
        self.bind_all("<ButtonPress>", lambda e: self._touch("click"))
        self.bind_all("<B1-Motion>", lambda e: self._touch("drag"))

    # --------------------------------------------------------------- state
    def _touch(self, what: str) -> None:
        self.counters["clicks"] += 1
        if what == "drag":
            self.counters["drag"] += 1
        elif what == "scroll":
            self.counters["scroll"] += 1
        self.last_action = what

    def state(self) -> dict:
        return {
            "files": list(self.files),
            "selected": list(self.selected),
            "curselection": list(self.listbox.curselection()),
            "entry_text": self.entry_text,
            "show_hidden": self.show_hidden,
            "icon_size": int(self.scale.get()),
            "drag_pos": list(self.drag_pos),
            "counters": dict(self.counters),
            "last_action": self.last_action,
            "dialog_open": bool(self.dialog and self.dialog.winfo_exists()),
        }

    # ------------------------------------------------------------ behaviour
    def _visible(self) -> list[str]:
        if self.show_hidden:
            return list(self.files)
        return [f for f in self.files if not f.startswith(".")]

    def _refresh(self) -> None:
        keep = set(self.selected)
        files = self._visible()
        self.listbox.delete(0, "end")
        for f in files:
            self.listbox.insert("end", f)
        sel = [i for i, f in enumerate(files) if f in keep]
        if sel:
            self.listbox.selection_set(*sel)
        else:
            self.listbox.selection_clear(0, "end")
        self.selected = [files[i] for i in sel]

    def _on_entry(self, _event) -> None:
        self.entry_text = self.entry.get()
        self.counters["type"] = len(self.entry_text)
        self.last_action = "type"

    def toggle_hidden(self) -> None:
        self.show_hidden = not self.show_hidden
        self.counters["toggle"] += 1
        self.last_action = "toggle"
        self._refresh()

    def new_file(self) -> None:
        self.files.append(f"new-file-{len(self.files)}.txt")
        self.counters["new"] += 1
        self.last_action = "new"
        self._refresh()

    def delete_files(self) -> None:
        if not self.selected:
            return
        for name in self.selected:
            if name in self.files:
                self.files.remove(name)
                self.counters["delete"] += 1
        self.selected = []
        self.last_action = "delete"
        self._refresh()

    def open_dialog(self) -> None:
        self.counters["dialog"] += 1
        self.last_action = "dialog"
        self.dialog = tk.Toplevel(self)
        self.dialog.title("About")
        self.dialog.geometry("320x140")
        tk.Label(self.dialog, text="demo file manager").pack(padx=16, pady=(16, 8))
        self.dialog_btn = tk.Button(self.dialog, text="OK",
                                    command=self._close_dialog)
        self.dialog_btn.pack(pady=16)

    def _close_dialog(self) -> None:
        if self.dialog is not None:
            self.dialog.destroy()
            self.dialog = None

    def _on_select(self, _event) -> None:
        self.selected = [self.listbox.get(i) for i in self.listbox.curselection()]
        self.last_action = "select"

    def _on_scale(self, _value) -> None:
        self.icon_size = int(self.scale.get())
        self.counters["slider"] += 1
        self.last_action = "slider"

    # ------------------------------------------------------- draggable item
    def _drag_start(self, event) -> None:
        self._drag_off = (event.x - self.drag_pos[0], event.y - self.drag_pos[1])

    def _drag_move(self, event) -> None:
        nx = event.x - self._drag_off[0]
        ny = event.y - self._drag_off[1]
        self.drag_pos = [nx, ny]
        self.canvas.coords(self.drag_item, nx, ny, nx + self.drag_size, ny + self.drag_size)
        self.counters["drag"] += 1
        self.last_action = "drag"

    def _drag_end(self, _event) -> None:
        self.last_action = "drag"

    # -------------------------------------------------------------- script
    def apply_op(self, op: dict) -> str:
        """Execute one high-level JSON op. Returns a short detail string."""
        name = op.get("op")
        if name == "click":
            target = op["target"]
            if target == BTN_NEW:
                self.new_file()
            elif target == BTN_DELETE:
                self.delete_files()
            elif target == BTN_ABOUT:
                self.open_dialog()
            elif target == BTN_QUIT:
                self.destroy()
            elif target == CHECK:
                self.toggle_hidden()
            elif target == ENTRY:
                self.entry.focus_set()
            elif target == SCALE:
                self.scale.focus_set()
            elif target == LISTBOX:
                self.listbox.focus_set()
            else:
                raise ValueError(f"unknown click target: {target}")
        elif name == "select":
            for i in op.get("indexes", []):
                self.listbox.selection_set(int(i))
            self._on_select(None)
        elif name == "type":
            self.entry.delete(0, "end")
            self.entry.insert(0, op.get("text", ""))
            self.entry_text = self.entry.get()
            self.counters["type"] = len(self.entry_text)
            self.last_action = "type"
        elif name == "toggle":
            self.toggle_hidden()
        elif name == "set":
            self.scale.set(int(op["value"]))
        elif name == "scroll":
            self.listbox.yview_scroll(int(op.get("dy", 1)), "units")
            self._touch("scroll")
        elif name == "dialog":
            if op.get("open", True):
                self.open_dialog()
            else:
                self._close_dialog()
        elif name == "quit":
            self.destroy()
        elif name == "noop":
            return "noop"
        else:
            raise ValueError(f"unknown op: {name}")
        return name

    def run_script(self, ops: list[dict]) -> dict:
        for op in ops:
            self.apply_op(op)
        self.update_idletasks()
        return self.state()

    # -------------------------------------------------------------- geometry
    def rects(self) -> dict:
        """Physical screen geometry for every scriptable widget."""
        self.update_idletasks()
        self.update()
        out: dict = {}
        for key, widget in ((ENTRY, self.entry), (CHECK, self.check_hidden),
                            (SCALE, self.scale), (BTN_NEW, self.btn_new),
                            (BTN_DELETE, self.btn_delete), (BTN_ABOUT, self.btn_about),
                            (BTN_QUIT, self.btn_quit), (LISTBOX, self.listbox)):
            out[key] = self._rect(widget)
        out["drag_item"] = [
            self.canvas.winfo_rootx() + self.drag_pos[0],
            self.canvas.winfo_rooty() + self.drag_pos[1],
            self.drag_size, self.drag_size,
        ]
        items = {}
        n = self.listbox.size()
        for i in range(n):
            bb = self.listbox.bbox(i)
            if bb and bb[2] > 0 and bb[3] > 0:
                x, y, w, h = bb
                items[str(i)] = [
                    self.listbox.winfo_rootx() + x,
                    self.listbox.winfo_rooty() + y,
                    w, h,
                ]
        out["items"] = items
        if self.dialog is not None and self.dialog.winfo_exists():
            for w in self.dialog.winfo_children():
                out[f"dialog/{w.winfo_name() or type(w).__name__}"] = self._rect(w)
            out["dialog_rect"] = self._rect(self.dialog)
            if getattr(self, "dialog_btn", None) is not None:
                out["dialog_ok"] = self._rect(self.dialog_btn)
        return out

    @staticmethod
    def _rect(widget) -> list[int]:
        return [widget.winfo_rootx(), widget.winfo_rooty(),
                widget.winfo_width(), widget.winfo_height()]


# -------------------------------------------------------------------- socket
class ControlServer:
    """JSON-line control socket for the live rig."""

    def __init__(self, app: DemoApp, port: int) -> None:
        self.app = app
        self._last_op = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self._stop = False

    def serve_forever(self) -> None:
        self.sock.settimeout(0.5)
        while not self._stop:
            try:
                conn, _ = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            buf = b""
            while True:
                try:
                    data = conn.recv(4096)
                except OSError:
                    return
                if not data:
                    return
                buf += data
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    self._one(conn, raw)
                    if self._last_op == "quit":
                        return

    def _one(self, conn: socket.socket, raw: bytes) -> None:
        raw = raw.strip()
        if not raw:
            return
        try:
            msg = json.loads(raw)
        except Exception as exc:
            self._send(conn, {"ok": False, "error": f"bad json: {exc}"})
            return
        op = msg.get("op") if isinstance(msg, dict) else None
        try:
            if op == "state":
                payload = self.app.state()
            elif op == "rects":
                payload = self.app.rects()
            elif op == "script":
                payload = self.app.run_script(msg.get("ops", []))
            elif op == "quit":
                self.app.destroy()
                payload = {"ok": True}
            else:
                payload = {"ok": False, "error": f"unknown op {op!r}"}
            self._send(conn, payload)
        except Exception as exc:  # never kill the loop on one bad message
            self._send(conn, {"ok": False, "error": str(exc)})
        self._last_op = op

    @staticmethod
    def _send(conn: socket.socket, payload: dict) -> None:
        try:
            conn.sendall((json.dumps(payload) + "\n").encode())
        except OSError:
            pass

    def stop(self) -> None:
        self._stop = True
        try:
            self.sock.close()
        except OSError:
            pass


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="deskhands.demo_app")
    ap.add_argument("--script", help="JSON file of ops to run in-process, then exit")
    ap.add_argument("--dump-state", action="store_true",
                    help="build the app, print its state as JSON, and exit")
    ap.add_argument("--port-file", help="write the control-socket port here and run")
    ap.add_argument("--seed", type=int, default=0, help="deterministic file seed")
    args = ap.parse_args(argv)

    app = DemoApp()
    app.update_idletasks()

    if args.dump_state:
        print(json.dumps(app.state()))
        return 0

    if args.script:
        with open(args.script) as fh:
            data = json.load(fh)
        ops = data["ops"] if isinstance(data, dict) else data
        print(json.dumps(app.run_script(ops)))
        return 0

    if args.port_file:
        port = _free_port()
        with open(args.port_file, "w") as fh:
            fh.write(str(port))
        server = ControlServer(app, port)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            app.mainloop()
        finally:
            server.stop()
    else:
        ap.error("one of --dump-state / --script / --port-file is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
