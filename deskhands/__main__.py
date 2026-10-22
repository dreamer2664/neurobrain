from __future__ import annotations
import sys
from . import ACTIONS, DesktopHands


def selftest() -> int:
    h = DesktopHands()
    cases = {
        "click": {"x":10,"y":10}, "double_click":{"x":10,"y":10},
        "right_click":{"x":10,"y":10}, "move":{"x":10,"y":10},
        "drag":{"x":10,"y":10,"x2":30,"y2":30,"ms":20},
        "scroll":{"x":10,"y":10,"dy":1}, "type":{"text":"abc"},
        "key":{"keys":["ctrl","a"]}, "wait":{"ms":1},
        "focus_window":{"title_substr":"__expected_missing__"},
        "launch":{"command":["/bin/true"]}, "screen":{},
    }
    print(f"{'action':16} {'result':8} latency_ms  detail")
    print("-" * 64)
    failed = 0
    try:
        for name in ACTIONS:
            r = h.perform({"type":name, **cases[name]})
            # A clean not-found response proves focus dispatch without requiring a WM.
            expected_absence = name == "focus_window" and "not found" in r["detail"]
            green = r["ok"] or expected_absence
            failed += not green
            print(f"{name:16} {'GREEN' if green else 'RED':8} {r['duration_ms']:10}  {r['detail']}")
    finally:
        h.close()
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(selftest() if len(sys.argv) > 1 and sys.argv[1] == "selftest" else "usage: python -m deskhands selftest")
