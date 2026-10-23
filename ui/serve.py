#!/usr/bin/env python3
"""Serve the Neurobrain PWA and proxy its fixed local brainbox API routes.

No packages are required. Brainbox stays on loopback; the browser uses same-origin
/api/* requests, avoiding CORS and never exposing model routes beyond this proxy.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import urllib.error
import urllib.parse
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROUTES = {"GET": {"/health"}, "POST": {"/generate", "/score", "/ground"}}
mimetypes.add_type("application/manifest+json", ".webmanifest")


class Handler(SimpleHTTPRequestHandler):
    brainbox = "http://127.0.0.1:8321"

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        if not self.path.startswith("/api/"):
            super().log_message(fmt, *args)

    def do_GET(self):
        if self.path.startswith("/api/"):
            return self.proxy("GET", None)
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/api/"):
            return self.send_error(404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.send_error(400)
        if length > 12 * 1024 * 1024:
            return self.send_error(413)
        self.proxy("POST", self.rfile.read(length) if length else None)

    def proxy(self, method, body):
        path = urllib.parse.urlsplit(self.path).path
        upstream_path = path.removeprefix("/api")
        if upstream_path not in ROUTES[method]:
            return self.json_response(404, {"error": "unknown local API route"})
        request = urllib.request.Request(
            self.brainbox + upstream_path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=125) as response:
                payload = response.read()
                self.send_response(response.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as error:
            payload = error.read()
            self.send_response(error.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            self.json_response(503, {"error": "brainbox unavailable", "reason": str(error)})

    def json_response(self, status, value):
        payload = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="UI bind address; use 0.0.0.0 for LAN/mobile testing")
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--brainbox", default="http://127.0.0.1:8321", help="loopback brainbox URL")
    args = parser.parse_args()
    parsed = urllib.parse.urlparse(args.brainbox)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("--brainbox must use HTTP and a loopback hostname")
    Handler.brainbox = args.brainbox.rstrip("/")
    handler = partial(Handler, directory=str(ROOT))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Neurobrain UI  http://{args.host}:{args.port}  →  {Handler.brainbox}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
