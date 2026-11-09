"""Subscriber for the webcam server's frame publisher (server.py, port 6004).

Mirrors prpl-tidybot's `CeilingImageClient`: same multiprocessing.connection
protocol and authkey, `get_image()` returns the latest BGR frame (cached, so a
brief stall returns the previous frame rather than None). The payload also carries
the frame's wall-clock capture time and camera sequence number:
`get_image_and_timestamp()` returns all three, and `last_timestamp` / `last_seq`
hold them for the frame `get_image()` last returned.

The authkey is read from a file (the server's `--frame-authkey-file`), never kept here.

    from scripts.tidybot_real.webcam.client import WebcamImageClient
    c = WebcamImageClient(authkey_file="/path/to/webcam_authkey")  # host="agni" elsewhere
    img, t, seq = c.get_image_and_timestamp()
"""

# ruff: noqa: PLR0917 -- mirrors prpl-tidybot's `CeilingImageClient` signature.

import os
from multiprocessing.connection import Client, Connection

import cv2 as cv
import numpy as np

WEBCAM_IMAGE_PORT = 6004


class WebcamImageClient:
    """Returns the webcam's BGR frames as ndarrays, with capture timestamps."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = WEBCAM_IMAGE_PORT,
        poll_timeout_s: float = 1.0,
        authkey_file: str | None = None,
    ) -> None:
        path = authkey_file or os.environ.get("WEBCAM_FRAME_AUTHKEY_FILE")
        if path is None:
            raise ValueError("pass authkey_file or set WEBCAM_FRAME_AUTHKEY_FILE")
        with open(path, "rb") as f:
            authkey = f.read().strip()
        self._conn: Connection = Client((host, port), authkey=authkey)
        self._poll_timeout_s = poll_timeout_s
        self._conn.send(None)
        self._last: np.ndarray | None = None
        self.last_timestamp: float | None = None  # wall-clock capture time, unix s
        self.last_seq: int | None = None  # camera frame number since server start

    def get_image(self) -> np.ndarray | None:
        """Return the most recent decoded frame, or `None` if none has arrived."""
        if self._conn.poll(timeout=self._poll_timeout_s):
            payload = self._conn.recv()
            self._conn.send(None)
            jpeg = payload.get("jpeg")
            if jpeg:
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                self._last = cv.imdecode(arr, cv.IMREAD_COLOR)
                self.last_timestamp = payload.get("timestamp")
                self.last_seq = payload.get("seq")
        return self._last

    def get_image_and_timestamp(
        self,
    ) -> tuple[np.ndarray | None, float | None, int | None]:
        """Return (frame, wall-clock capture time in unix seconds, camera seq)."""
        img = self.get_image()
        return img, self.last_timestamp, self.last_seq

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()
