"""The ceiling camera's frames and calibration, for the dashboard. Runs in prpl-tidybot's
environment, never hitl-pmp's.

hitl-pmp does not depend on `prpl_tidybot`, so the dashboard runs this file as a
subprocess with prpl-tidybot's interpreter (`--camera-python`). Nothing imports it.

    <prpl python> ceiling_helper.py --out-dir DIR --host 192.168.0.131 \\
        --serials AD8A29AE 5A5C29AE --period 2

It writes `DIR/calibration.json` once (each camera's undistorted pinhole model, in
stitched-image order) and then, every `--period` seconds, the newest stitched frame as
`DIR/frame.jpg` with `DIR/frame.json` beside it, each replaced atomically. A failure
is written to `DIR/frame.json` as `error` and retried. It exits when its parent does.

## Read this before turning it on during a run

prpl-tidybot's `Publisher.run` (`marker_detector/publisher.py`) has no handler around
`listener.accept()`: a client that drops during the connection's auth handshake raises
in the accept thread and takes the whole marker-detector daemon down. That happened at
10:19:41 on 2026-10-21, from a bare TCP probe of port 6003, and broke a real run's
launch. This helper completes the handshake, keeps one connection and reconnects at
most once a minute -- but a helper killed *mid-handshake* would do the same, so leave
the camera off during a run until that accept loop is guarded upstream.

## One subscriber, two frames per look

`CeilingImageClient` is a separate subscriber of the perception PC's publisher, which
encodes each frame once and answers every subscriber from its own thread. A client's
first `get_image` after a pause returns the frame queued at its last request, so each
look takes two: the queued one, discarded, then a fresh one.
"""

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Final

_RECONNECT_SECONDS: Final = 60.0


class CeilingHelper:
    @staticmethod
    def export_calibration(*, serials: list[str], path: Path) -> None:
        from prpl_tidybot.cube_localizer import load_ceiling_cameras  # noqa: PLC0415
        from prpl_tidybot.marker_detector.utils import get_camera_params  # noqa: PLC0415

        cameras = []
        for serial, camera in zip(serials, load_ceiling_cameras(serials), strict=True):
            width, height, _, _ = get_camera_params(serial)
            cameras.append({
                "serial": serial,
                "width": width,
                "height": height,
                "camera_matrix": camera.camera_matrix.tolist(),
                "rvec": camera.rvec.reshape(3).tolist(),
                "tvec": camera.tvec.reshape(3).tolist(),
            })
        CeilingHelper._write(path=path, data=json.dumps({"cameras": cameras}, indent=1).encode())

    @staticmethod
    def serve(*, out_dir: Path, host: str, period: float) -> None:
        import cv2 as cv  # noqa: PLC0415
        from prpl_tidybot.marker_detector.ceiling_image_client import (  # noqa: PLC0415
            CeilingImageClient,
        )

        parent = os.getppid()
        client = None
        while os.getppid() == parent:
            started = time.time()
            try:
                if client is None:
                    client = CeilingImageClient(host=host, poll_timeout_s=3.0)
                client.get_image()
                frame = client.get_image()
                if frame is None:
                    raise RuntimeError("no frame from the ceiling image publisher")
                ok, jpeg = cv.imencode(".jpg", frame, [int(cv.IMWRITE_JPEG_QUALITY), 75])
                if not ok:
                    raise RuntimeError("could not encode the frame")
                CeilingHelper._write(path=out_dir / "frame.jpg", data=jpeg.tobytes())
                meta = {
                    "time": time.time(),
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "grab_seconds": time.time() - started,
                }
            except Exception as error:  # noqa: BLE001 -- shown on the page, then retried
                if client is not None:
                    with contextlib.suppress(Exception):
                        client.close()
                client = None
                failure = {"time": time.time(), "error": f"{type(error).__name__}: {error}"}
                CeilingHelper._write(path=out_dir / "frame.json", data=json.dumps(failure).encode())
                # Reconnect rarely: every connection is a handshake with the publisher.
                time.sleep(_RECONNECT_SECONDS)
                continue
            CeilingHelper._write(path=out_dir / "frame.json", data=json.dumps(meta).encode())
            time.sleep(max(0.0, period - (time.time() - started)))

    @staticmethod
    def _write(*, path: Path, data: bytes) -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--host", default="192.168.0.131")
    parser.add_argument("--serials", nargs="+", default=["AD8A29AE", "5A5C29AE"])
    parser.add_argument("--period", type=float, default=2.0)
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        CeilingHelper.export_calibration(
            serials=args.serials, path=args.out_dir / "calibration.json"
        )
    except Exception as error:  # noqa: BLE001 -- the page falls back to the plain map
        message = {"time": time.time(), "error": f"calibration: {type(error).__name__}: {error}"}
        CeilingHelper._write(path=args.out_dir / "frame.json", data=json.dumps(message).encode())
        sys.exit(1)
    if not args.calibration_only:
        CeilingHelper.serve(out_dir=args.out_dir, host=args.host, period=args.period)


if __name__ == "__main__":
    main()
