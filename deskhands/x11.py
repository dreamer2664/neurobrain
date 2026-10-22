"""X11/XTest backend using only the Python standard library."""
from __future__ import annotations
import ctypes as C
import ctypes.util
import time


class X11Backend:
    def __init__(self) -> None:
        self.x = C.CDLL(ctypes_path("X11"))
        self.xt = C.CDLL(ctypes_path("Xtst"))
        self.x.XOpenDisplay.restype = C.c_void_p
        self.x.XDefaultRootWindow.argtypes = [C.c_void_p]
        self.x.XDefaultRootWindow.restype = C.c_ulong
        self.x.XFlush.argtypes = [C.c_void_p]
        self.x.XCloseDisplay.argtypes = [C.c_void_p]
        self.x.XFree.argtypes = [C.c_void_p]
        self.x.XDisplayWidth.argtypes = [C.c_void_p, C.c_int]
        self.x.XDisplayHeight.argtypes = [C.c_void_p, C.c_int]
        self.x.XKeysymToKeycode.argtypes = [C.c_void_p, C.c_ulong]
        self.x.XKeysymToKeycode.restype = C.c_uint
        self.x.XStringToKeysym.argtypes = [C.c_char_p]
        self.x.XStringToKeysym.restype = C.c_ulong
        self.xt.XTestFakeMotionEvent.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_ulong]
        self.xt.XTestFakeButtonEvent.argtypes = [C.c_void_p, C.c_uint, C.c_int, C.c_ulong]
        self.xt.XTestFakeKeyEvent.argtypes = [C.c_void_p, C.c_uint, C.c_int, C.c_ulong]
        self.d = self.x.XOpenDisplay(None)
        if not self.d:
            raise RuntimeError("cannot open X display")
        self.root = self.x.XDefaultRootWindow(self.d)
        self.width = self.x.XDisplayWidth(self.d, 0)
        self.height = self.x.XDisplayHeight(self.d, 0)

    def close(self) -> None:
        if self.d:
            self.x.XCloseDisplay(self.d)
            self.d = None

    def _point(self, x: int, y: int) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"coordinate ({x},{y}) outside {self.width}x{self.height}")

    def _flush(self) -> None:
        self.x.XFlush(self.d)

    def move(self, x: int, y: int) -> None:
        self._point(x, y)
        if not self.xt.XTestFakeMotionEvent(self.d, 0, x, y, 0):
            raise RuntimeError("XTest motion failed")
        self._flush()

    def _button(self, button: int, down: bool) -> None:
        if not self.xt.XTestFakeButtonEvent(self.d, button, down, 0):
            raise RuntimeError("XTest button failed")

    def click(self, x: int, y: int, button: int = 1, clicks: int = 1) -> None:
        self.move(x, y)
        for i in range(max(1, clicks)):
            self._button(button, True); self._button(button, False)
            if i + 1 < clicks: time.sleep(0.04)
        self._flush()

    def drag(self, x: int, y: int, x2: int, y2: int, ms: int) -> None:
        self._point(x, y); self._point(x2, y2)
        self.move(x, y); self._button(1, True)
        steps = 20
        delay = max(0, ms) / steps / 1000
        for i in range(1, steps + 1):
            self.move(round(x + (x2-x)*i/steps), round(y + (y2-y)*i/steps))
            if delay: time.sleep(delay)
        self._button(1, False); self._flush()

    def scroll(self, x: int, y: int, dy: int) -> None:
        self.move(x, y)
        button = 5 if dy > 0 else 4
        for _ in range(abs(dy)):
            self._button(button, True); self._button(button, False)
        self._flush()

    def _keycode(self, name: str) -> int:
        aliases = {"ctrl":"Control_L", "control":"Control_L", "alt":"Alt_L", "shift":"Shift_L", "win":"Super_L", "enter":"Return", "esc":"Escape", "space":"space", "tab":"Tab", "backspace":"BackSpace", "delete":"Delete"}
        keysym = self.x.XStringToKeysym(aliases.get(name.lower(), name).encode())
        code = self.x.XKeysymToKeycode(self.d, keysym)
        if not code: raise ValueError(f"unsupported key: {name}")
        return code

    def chord(self, keys: list[str]) -> None:
        codes = [self._keycode(k) for k in keys]
        for code in codes: self.xt.XTestFakeKeyEvent(self.d, code, True, 0)
        for code in reversed(codes): self.xt.XTestFakeKeyEvent(self.d, code, False, 0)
        self._flush()

    def type_text(self, text: str) -> None:
        for char in text:
            if ord(char) > 127: raise ValueError("unicode clipboard fallback not yet available")
            shifted = char.isupper() or char in "!@#$%^&*()_+{}|:\"<>?"
            name = {"\n":"Return", "\t":"Tab", " ":"space"}.get(char, char.lower())
            codes = ([self._keycode("shift")] if shifted else []) + [self._keycode(name)]
            for code in codes: self.xt.XTestFakeKeyEvent(self.d, code, True, 0)
            for code in reversed(codes): self.xt.XTestFakeKeyEvent(self.d, code, False, 0)
        self._flush()

    def focus_window(self, title: str) -> None:
        self.x.XQueryTree.argtypes = [C.c_void_p, C.c_ulong, C.POINTER(C.c_ulong), C.POINTER(C.c_ulong), C.POINTER(C.POINTER(C.c_ulong)), C.POINTER(C.c_uint)]
        self.x.XFetchName.argtypes = [C.c_void_p, C.c_ulong, C.POINTER(C.c_char_p)]
        self.x.XSetInputFocus.argtypes = [C.c_void_p, C.c_ulong, C.c_int, C.c_ulong]
        needle = title.casefold()
        def walk(window: int) -> int | None:
            name = C.c_char_p()
            if self.x.XFetchName(self.d, window, C.byref(name)) and name.value:
                value = name.value.decode(errors="replace")
                self.x.XFree(name)
                if needle in value.casefold(): return window
            root, parent, children, count = C.c_ulong(), C.c_ulong(), C.POINTER(C.c_ulong)(), C.c_uint()
            if self.x.XQueryTree(self.d, window, C.byref(root), C.byref(parent), C.byref(children), C.byref(count)):
                try:
                    for i in range(count.value):
                        found = walk(children[i])
                        if found: return found
                finally:
                    if children: self.x.XFree(children)
            return None
        found = walk(self.root)
        if not found: raise LookupError(f"window not found: {title}")
        self.x.XMapRaised(self.d, found)
        self.x.XSetInputFocus(self.d, found, 1, 0)
        self._flush()


def ctypes_path(name: str) -> str:
    path = ctypes.util.find_library(name)
    if not path: raise RuntimeError(f"library {name} not found")
    return path
