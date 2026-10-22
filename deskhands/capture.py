"""Fast X11 screen capture with monotonic timestamps."""
from __future__ import annotations
import ctypes as C
import ctypes.util
import time
from dataclasses import dataclass
from PIL import Image


class XImage(C.Structure):
    _fields_ = [
        ("width", C.c_int), ("height", C.c_int), ("xoffset", C.c_int),
        ("format", C.c_int), ("data", C.c_void_p), ("byte_order", C.c_int),
        ("bitmap_unit", C.c_int), ("bitmap_bit_order", C.c_int),
        ("bitmap_pad", C.c_int), ("depth", C.c_int),
        ("bytes_per_line", C.c_int), ("bits_per_pixel", C.c_int),
    ]


@dataclass(frozen=True)
class Screenshot:
    image: Image.Image
    timestamp_ns: int
    rect: tuple[int, int, int, int]


class Capture:
    def __init__(self) -> None:
        path = ctypes.util.find_library("X11")
        if not path: raise RuntimeError("libX11 unavailable")
        self.x = C.CDLL(path)
        self.x.XOpenDisplay.restype = C.c_void_p
        self.x.XDefaultRootWindow.argtypes = [C.c_void_p]
        self.x.XDefaultRootWindow.restype = C.c_ulong
        self.x.XDisplayWidth.argtypes = [C.c_void_p, C.c_int]
        self.x.XDisplayHeight.argtypes = [C.c_void_p, C.c_int]
        self.x.XGetImage.argtypes = [C.c_void_p, C.c_ulong, C.c_int, C.c_int, C.c_uint, C.c_uint, C.c_ulong, C.c_int]
        self.x.XGetImage.restype = C.POINTER(XImage)
        self.x.XDestroyImage.argtypes = [C.POINTER(XImage)]
        self.x.XCloseDisplay.argtypes = [C.c_void_p]
        self.display = self.x.XOpenDisplay(None)
        if not self.display: raise RuntimeError("cannot open X display")
        self.root = self.x.XDefaultRootWindow(self.display)
        self.size = (self.x.XDisplayWidth(self.display, 0), self.x.XDisplayHeight(self.display, 0))

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    def close(self) -> None:
        if self.display:
            self.x.XCloseDisplay(self.display)
            self.display = None

    def full_screen(self) -> Screenshot:
        return self.region(0, 0, *self.size)

    def region(self, x: int, y: int, width: int, height: int) -> Screenshot:
        sw, sh = self.size
        if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > sw or y + height > sh:
            raise ValueError(f"region {(x,y,width,height)} outside {(sw,sh)}")
        timestamp = time.monotonic_ns()
        ptr = self.x.XGetImage(self.display, self.root, x, y, width, height, (1 << 64) - 1, 2)
        if not ptr: raise RuntimeError("XGetImage failed")
        try:
            info = ptr.contents
            if info.bits_per_pixel != 32:
                raise RuntimeError(f"unsupported XImage pixel depth {info.bits_per_pixel}")
            raw = C.string_at(info.data, info.bytes_per_line * height)
            image = Image.frombuffer("RGB", (width, height), raw, "raw", "BGRX", info.bytes_per_line, 1).copy()
        finally:
            self.x.XDestroyImage(ptr)
        return Screenshot(image, timestamp, (x, y, width, height))

    @staticmethod
    def downscale(shot: Screenshot | Image.Image, size: tuple[int, int]) -> Image.Image:
        image = shot.image if isinstance(shot, Screenshot) else shot
        return image.resize(size, Image.Resampling.BILINEAR)
