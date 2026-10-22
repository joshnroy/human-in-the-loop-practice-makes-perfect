"""Render smoke tests: the server answers the page, the state and the camera frame."""

import json
import os
import re
import shutil
import threading
import time
import urllib.request
from pathlib import Path

from PIL import Image

from analysis.real_dashboard.camera import CameraFeed
from analysis.real_dashboard.dashboard import RealRunDashboard
from analysis.real_dashboard.server import DashboardServer

FIXTURES = Path(__file__).parent / "fixtures"


class _Serving:
    def __init__(self, *, dashboard: RealRunDashboard) -> None:
        self.server = DashboardServer.make(dashboard=dashboard, host="127.0.0.1", port=0)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self) -> "_Serving":
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_the_page_and_its_state_render_on_run9():
    dashboard = RealRunDashboard(
        run_dir=FIXTURES / "run9", skill_calls=FIXTURES / "skill_calls.jsonl"
    )
    with _Serving(dashboard=dashboard) as serving:
        with urllib.request.urlopen(f"{serving.base}/", timeout=10) as response:
            assert response.headers["Content-Type"].startswith("text/html")
            page = response.read().decode()
        for element in ("status", "eval-curve", "camera", "tosses", "details"):
            assert f'id="{element}"' in page
        # The first paint's state is written into the page, not fetched after it.
        assert "/*INITIAL_STATE*/" not in page
        assert '"in_bin_label":"1/7"' in page
        # Nothing is fetched from the network: no CDN, no external script.
        # (The SVG namespace URI is an identifier, never fetched.)
        assert re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", page) is None
        assert re.search(r"""(fetch\(|url\(|@import)\s*["']?(https?:)?//""", page) is None
        with urllib.request.urlopen(f"{serving.base}/api/state", timeout=10) as response:
            state = json.loads(response.read())
        assert state["status"]["practice_steps"] == 1586
        assert state["tosses"]["in_bin_label"] == "1/7"
        assert state["eval_curve"]["points"][0]["label"] == "0/1"
        assert state["camera"] is None
        for path in ("/nope", "/api/camera.jpg"):
            try:
                urllib.request.urlopen(f"{serving.base}{path}", timeout=10)
            except urllib.error.HTTPError as error:
                assert error.code == 404
            else:
                raise AssertionError(f"{path} must 404")


def test_the_camera_frame_is_served_with_its_overlay(*, tmp_path):
    camera_dir = tmp_path / "camera"
    camera_dir.mkdir()
    shutil.copy(FIXTURES / "ceiling" / "calibration.json", camera_dir / "calibration.json")
    Image.new("RGB", (1280, 1440), (200, 180, 160)).save(camera_dir / "frame.jpg")
    (camera_dir / "frame.json").write_text(json.dumps({"time": time.time(), "grab_seconds": 0.4}))
    dashboard = RealRunDashboard(
        run_dir=FIXTURES / "run9",
        skill_calls=FIXTURES / "skill_calls.jsonl",
        camera=CameraFeed(directory=camera_dir),
    )
    with _Serving(dashboard=dashboard) as serving:
        with urllib.request.urlopen(f"{serving.base}/api/camera.jpg", timeout=10) as response:
            assert response.headers["Content-Type"] == "image/jpeg"
            assert response.read()[:2] == b"\xff\xd8"
        with urllib.request.urlopen(f"{serving.base}/api/state", timeout=10) as response:
            camera = json.loads(response.read())["camera"]
    assert camera["available"] is True
    kinds = {shape["kind"] for shape in camera["overlay"]["shapes"]}
    assert {"tape", "spot", "bin", "robot", "landing"} <= kinds


def test_an_old_frame_or_none_leaves_the_camera_unavailable(*, tmp_path):
    feed = CameraFeed(directory=tmp_path)
    state = RealRunDashboard(run_dir=FIXTURES / "run9", camera=feed).refresh()
    assert state.camera is not None
    assert state.camera.available is False
    assert state.camera.error is not None
    shutil.copy(FIXTURES / "ceiling" / "calibration.json", tmp_path / "calibration.json")
    Image.new("RGB", (8, 8)).save(tmp_path / "frame.jpg")
    (tmp_path / "frame.json").write_text(json.dumps({"time": time.time() - 600}))
    state = RealRunDashboard(run_dir=FIXTURES / "run9", camera=feed).refresh()
    assert state.camera is not None
    assert state.camera.available is False
    assert state.camera.age_seconds is not None
    assert state.camera.age_seconds > 500


def test_the_logs_are_stale_only_while_a_skill_runs_and_nothing_is_written(*, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(FIXTURES / "run9", run)
    session = run / "robot_session.log"
    session.write_text("2026-10-21 14:00:05.010 running PickCube ...\n")
    old = time.time() - 600
    for path in run.iterdir():
        os.utime(path, (old, old))
    status = RealRunDashboard(run_dir=run).refresh().status
    assert status.logs_age_seconds is not None
    assert status.logs_age_seconds > 500
    assert status.stale is True
    with session.open("a") as stream:
        stream.write("2026-10-21 14:00:40.000 PickCube: ok (35 s); observing ...\n")
    os.utime(session, (old, old))
    status = RealRunDashboard(run_dir=run).refresh().status
    assert status.stale is False  # Old, but nothing is running: the robot may be idle.
