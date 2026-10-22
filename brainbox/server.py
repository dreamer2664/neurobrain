"""Local HTTP API per specs/model_io.md (M1).

    POST /generate  {"prompt": str, "max_tokens": 64}     -> {"text", "tokens", "ms"}
    POST /score     {"prompt": str, "top_k": 5}           -> {"tokens", "logprobs"}
    POST /ground    {"png_b64": str, "instruction": str}  -> {"x", "y", "action"}
    GET  /health                                          -> {"ok", "rss_mb", "step"}

Bound to 127.0.0.1 (hard-coded; default port 8321). Model work runs one job
at a time on a single worker thread fed by a queue; /health never queues, so
it answers in <50 ms even while a generation is running. Over the RAM cap:
429 with `reason`. Stdlib only.
"""

from __future__ import annotations

import base64
import binascii
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .checkpoints import CheckpointError
from .governor import GovernorRefused

DEFAULT_PORT = 8321
MAX_BODY_BYTES = 12 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
JOB_TIMEOUT_S = 120

POST_ROUTES = ("/generate", "/score", "/ground")


class _Job:
    __slots__ = ("fn", "event", "result", "error")

    def __init__(self, fn):
        self.fn = fn
        self.event = threading.Event()
        self.result = None
        self.error = None


class BrainboxServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, host, port: int = DEFAULT_PORT):
        self.brain = host
        self.jobs: queue.Queue[_Job] = queue.Queue()
        self.stats = {"jobs": 0, "peak_concurrency": 0, "active": 0,
                      "max_queue": 0, "by_route": {}}
        self._stats_lock = threading.Lock()
        super().__init__(("127.0.0.1", port), _Handler)
        self._worker = threading.Thread(target=self._work_loop, daemon=True,
                                        name="brainbox-worker")
        self._worker.start()

    def _work_loop(self):
        while True:
            job = self.jobs.get()
            with self._stats_lock:
                self.stats["active"] += 1
                self.stats["peak_concurrency"] = max(
                    self.stats["peak_concurrency"], self.stats["active"])
                self.stats["jobs"] += 1
            try:
                job.result = job.fn()
            except BaseException as e:  # surfaced to the waiting handler
                job.error = e
            finally:
                with self._stats_lock:
                    self.stats["active"] -= 1
                job.event.set()
                self.jobs.task_done()

    def submit(self, route: str, fn, timeout: float = JOB_TIMEOUT_S):
        job = _Job(fn)
        self.jobs.put(job)
        with self._stats_lock:
            self.stats["max_queue"] = max(self.stats["max_queue"], self.jobs.qsize())
            self.stats["by_route"][route] = self.stats["by_route"].get(route, 0) + 1
        if not job.event.wait(timeout):
            raise TimeoutError(f"model job exceeded {timeout:.0f}s")
        if job.error is not None:
            raise job.error
        return job.result


def make_server(host, port: int = DEFAULT_PORT) -> BrainboxServer:
    return BrainboxServer(host, port=port)


class _Handler(BaseHTTPRequestHandler):
    server_version = f"brainbox/{__version__}"
    sys_version = ""

    # ------------------------------------------------------------- plumbing
    def log_message(self, *args):  # keep stdout for the ready line only
        pass

    def _send_json(self, status: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str, **extra):
        self._send_json(status, {"error": message, **extra})

    @property
    def brain(self):
        return self.server.brain  # type: ignore[attr-defined]

    # ------------------------------------------------------------- GET
    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, self.brain.health())
        elif self.path in POST_ROUTES:
            self._error(405, "method not allowed, use POST")
        else:
            self._error(404, "not found")

    def do_PUT(self):
        self._error(405, "method not allowed")

    do_DELETE = do_PATCH = do_PUT

    # ------------------------------------------------------------- POST
    def do_POST(self):
        if self.path == "/health":
            return self._error(405, "method not allowed, use GET")
        if self.path not in POST_ROUTES:
            return self._error(404, "not found")

        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            return self._error(413, "body too large")
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("body must be a json object")
        except Exception as e:
            return self._error(400, f"invalid json body: {e}")

        # RAM governor, fast path (the host re-checks before executing).
        gov = self.brain.governor
        if not gov.can_work():
            return self._error(429, "ram budget exceeded",
                               reason=f"rss over cap {gov.cap_mb:.0f} MB; "
                                      f"resumes under {gov.resume_mb:.0f} MB")

        try:
            fn = self._job_for(self.path, body)
            result = self.server.submit(self.path, fn)  # type: ignore[attr-defined]
        except _BadRequest as e:
            return self._error(400, str(e))
        except GovernorRefused as e:
            return self._error(429, "ram budget exceeded", reason=str(e))
        except CheckpointError as e:
            return self._error(503, "model unavailable", reason=str(e))
        except ValueError as e:
            return self._error(400, str(e))
        except TimeoutError as e:
            return self._error(504, str(e))
        except Exception as e:  # never leak a stack trace / hang a socket
            return self._error(500, f"internal error: {type(e).__name__}: {e}")
        self._send_json(200, result)

    def _job_for(self, route: str, body: dict):
        brain = self.brain
        if route == "/generate":
            prompt = _req_str(body, "prompt")
            max_tokens = _opt_int(body, "max_tokens", 64, 1, 1024)
            temperature = body.get("temperature", 0.8)
            top_k = body.get("top_k", 40)
            seed = body.get("seed")
            return lambda: _pick(
                brain.generate(prompt, max_tokens=max_tokens,
                               temperature=float(temperature),
                               top_k=int(top_k), seed=seed),
                ("text", "tokens", "ms"))
        if route == "/score":
            prompt = _req_str(body, "prompt")
            top_k = _opt_int(body, "top_k", 5, 1, 100)
            return lambda: _pick(brain.score(prompt, top_k=top_k),
                                 ("tokens", "logprobs"))
        # /ground
        png_b64 = _req_str(body, "png_b64")
        instruction = _req_str(body, "instruction")
        try:
            png = base64.b64decode(png_b64, validate=True)
        except (binascii.Error, ValueError) as e:
            raise _BadRequest(f"invalid png_b64: {e}")
        if len(png) > MAX_IMAGE_BYTES:
            raise _BadRequest("image too large")
        return lambda: brain.ground(png, instruction)


class _BadRequest(ValueError):
    pass


def _pick(d: dict, keys) -> dict:
    return {k: d[k] for k in keys}


def _req_str(body: dict, name: str) -> str:
    v = body.get(name)
    if not isinstance(v, str):
        raise _BadRequest(f"field '{name}' is required and must be a string")
    return v


def _opt_int(body: dict, name: str, default: int, lo: int, hi: int) -> int:
    v = body.get(name, default)
    if isinstance(v, bool) or not isinstance(v, int):
        raise _BadRequest(f"field '{name}' must be an integer")
    return min(max(v, lo), hi)
