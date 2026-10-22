"""RAM governor: hard cap with hysteresis (specs/budget.md, specs/drive.md).

Above the cap (default 3800 MB RSS) new work is refused; work resumes only
once RSS falls below the resume line (default 3200 MB). Stdlib only: RSS is
read from /proc on Linux, peak RSS from getrusage.
"""

from __future__ import annotations

import gc
import os


def rss_mb() -> float:
    """Current resident set size of this process, in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0  # kB -> MB
    except (OSError, ValueError, IndexError):
        pass
    return peak_rss_mb()  # fallback: peak is an upper bound on current


def peak_rss_mb() -> float:
    """Peak resident set size of this process so far, in MB."""
    try:
        import resource

        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak_kb / 1024.0  # Linux: ru_maxrss is in kB
    except Exception:
        return 0.0


class GovernorRefused(RuntimeError):
    """Raised when the governor refuses work (maps to HTTP 429 in M1)."""


class Governor:
    def __init__(self, cap_mb: float | None = None, resume_mb: float = 3200.0,
                 rss_fn=rss_mb):
        if cap_mb is None:
            cap_mb = float(os.environ.get("BRAINBOX_RAM_CAP_MB", "3800"))
        self.cap_mb = float(cap_mb)
        self.resume_mb = float(resume_mb)
        self.rss_fn = rss_fn
        self.tripped = False

    def can_work(self) -> bool:
        rss = self.rss_fn()
        if self.tripped:
            if rss < self.resume_mb:
                self.tripped = False
        elif rss > self.cap_mb:
            self.tripped = True
            self.drop_caches()
        return not self.tripped

    def check_or_raise(self) -> None:
        if not self.can_work():
            raise GovernorRefused(
                f"ram budget: rss {self.rss_fn():.0f} MB over cap "
                f"{self.cap_mb:.0f} MB; resumes under {self.resume_mb:.0f} MB")

    @staticmethod
    def drop_caches() -> None:
        gc.collect()
