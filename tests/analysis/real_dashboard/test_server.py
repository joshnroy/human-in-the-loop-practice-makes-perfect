"""Render smoke test: the server answers the page and the state on run9's files."""

import json
import re
import threading
import urllib.request
from pathlib import Path

from analysis.real_dashboard.dashboard import RealRunDashboard
from analysis.real_dashboard.server import DashboardServer

FIXTURES = Path(__file__).parent / "fixtures"


def test_the_page_and_its_state_render_on_run9():
    dashboard = RealRunDashboard(
        run_dir=FIXTURES / "run9", skill_calls=FIXTURES / "skill_calls.jsonl"
    )
    server = DashboardServer.make(dashboard=dashboard, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(f"{base}/", timeout=10) as response:
            assert response.headers["Content-Type"].startswith("text/html")
            page = response.read().decode()
        for element in ("status", "eval-curve", "tosses", "room", "timing"):
            assert f'id="{element}"' in page
        # The first paint's state is written into the page, not fetched after it.
        assert "/*INITIAL_STATE*/" not in page
        assert '"in_bin_label":"1/7"' in page
        # Nothing is fetched from the network: no CDN, no external script.
        # (The SVG namespace URI is an identifier, never fetched.)
        assert re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", page) is None
        assert re.search(r"""(fetch\(|url\(|@import)\s*["']?(https?:)?//""", page) is None
        with urllib.request.urlopen(f"{base}/api/state", timeout=10) as response:
            state = json.loads(response.read())
        assert state["status"]["practice_steps"] == 1586
        assert state["tosses"]["in_bin_label"] == "1/7"
        assert state["eval_curve"]["points"][0]["label"] == "0/1"
        missing = urllib.request.Request(f"{base}/nope")
        try:
            urllib.request.urlopen(missing, timeout=10)
        except urllib.error.HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError("an unknown path must 404")
    finally:
        server.shutdown()
        server.server_close()
