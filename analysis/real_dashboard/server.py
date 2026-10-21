"""The dashboard's HTTP server: the page at `/`, the run's state as JSON at `/api/state`,
and the newest ceiling-camera frame at `/api/camera.jpg`.

Standard library only, read-only: GET and HEAD, nothing else. The page is one static
file with its script inline and polls the JSON; it fetches nothing from the network.
"""

import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar, Final

from .dashboard import RealRunDashboard

PAGE: Final = Path(__file__).with_name("page.html")
# Several tabs polling at once share one read of the run's files.
_MIN_REFRESH_SECONDS: Final = 1.0
_INITIAL_STATE: Final = b"/*INITIAL_STATE*/null"


class DashboardHandler(BaseHTTPRequestHandler):
    dashboard: ClassVar[RealRunDashboard]
    _cached: ClassVar[tuple[float, bytes] | None] = None

    def do_GET(self) -> None:  # noqa: N802 -- http.server's name
        self._answer(body=True)

    def do_HEAD(self) -> None:  # noqa: N802 -- http.server's name
        self._answer(body=False)

    def _answer(self, *, body: bool) -> None:
        path = self.path.split("?", 1)[0]
        if path in {"/", "/index.html"}:
            # `</` would end the inline script early; `<\/` is the same JSON string.
            state = self._state().replace(b"</", b"<\\/")
            page = PAGE.read_bytes().replace(_INITIAL_STATE, state, 1)
            self._send(content=page, kind="text/html; charset=utf-8", body=body)
        elif path == "/api/state":
            self._send(content=self._state(), kind="application/json", body=body)
        elif path == "/api/camera.jpg":
            feed = type(self).dashboard.camera
            frame = feed.frame() if feed is not None else None
            if frame is None:
                self.send_error(HTTPStatus.NOT_FOUND, "no camera frame")
            else:
                self._send(content=frame, kind="image/jpeg", body=body)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _state(self) -> bytes:
        cls = type(self)
        cached = cls._cached
        if cached is not None and time.monotonic() - cached[0] < _MIN_REFRESH_SECONDS:
            return cached[1]
        content = cls.dashboard.refresh().model_dump_json().encode("utf-8")
        cls._cached = (time.monotonic(), content)
        return content

    def _send(self, *, content: bytes, kind: str, body: bool) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002, PLR0917 -- http.server's override
        del format, args  # A poll every 5 s per tab would bury the terminal.


class DashboardServer:
    @staticmethod
    def make(*, dashboard: RealRunDashboard, host: str, port: int) -> ThreadingHTTPServer:
        """A server bound to `host:port` (0 picks a free port), not yet serving."""
        handler = type("BoundDashboardHandler", (DashboardHandler,), {"dashboard": dashboard})
        server = ThreadingHTTPServer((host, port), handler)
        server.daemon_threads = True
        return server
