import statistics
import time
from deskhands.capture import Capture


def test_full_region_and_downscales():
    with Capture() as cap:
        shot = cap.full_screen()
        assert shot.image.size == cap.size
        assert shot.timestamp_ns <= time.monotonic_ns()
        region = cap.region(2, 3, 20, 10)
        assert region.image.size == (20, 10)
        assert cap.downscale(shot, (128, 128)).size == (128, 128)
        assert cap.downscale(shot, (640, 360)).size == (640, 360)


def test_50_captures_latency_and_no_growth():
    with Capture() as cap:
        samples = []
        for _ in range(50):
            start = time.monotonic_ns()
            shot = cap.full_screen()
            samples.append((time.monotonic_ns() - start) / 1e6)
            del shot
        assert statistics.median(samples) <= 30, samples
        assert max(samples) <= 100, samples
