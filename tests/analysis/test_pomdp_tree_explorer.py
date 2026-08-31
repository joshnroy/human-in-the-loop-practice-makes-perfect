"""Read-only indexing and HTTP transport for full decision traces."""

import gzip
import json
import shutil
import subprocess
import threading
from http.server import HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from analysis.pomdp_tree_explorer.server import DecisionStore, ExplorerHandler


def test_browser_logic() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    subprocess.run(
        [node, str(Path(__file__).with_name("pomdp_tree_explorer_ui.cjs"))],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_offsets_and_metadata(*, tmp_path: Path) -> None:
    (tmp_path / "0").mkdir()
    path = tmp_path / "0" / "pomdp_decisions.jsonl"
    first = {
        "event": "decision",
        "cycle": 0,
        "decision": 1,
        "action": {"name": "PickCube"},
        "value": 0.4,
        "horizon": 10,
        "search": [{"event": "node", "node": 0, "summed_cost": 2}],
    }
    second = {**first, "decision": 2, "action": "STOP"}
    records = [{"event": "session_start"}, first, {"event": "outcome"}, second]
    lines = [json.dumps(record).encode() + b"\n" for record in records]
    path.write_bytes(b"".join(lines))
    store = DecisionStore(root=tmp_path)
    assert store.seeds() == [0]
    index = store.index(seed=0)
    assert len(index) == 2
    assert index[0].metadata["summed_cost"] == 2
    assert index[0].metadata["event_count"] == 1
    assert store.decision(seed=0, index=0) == lines[1]
    assert store.decision(seed=0, index=1) == lines[3]
    assert store.index(seed=0) is index
    for invalid in [-1, 2]:
        with pytest.raises(FileNotFoundError):
            store.decision(seed=0, index=invalid)
    with path.open("ab") as out:
        out.write(b'{"event": "decision"')
    assert len(store.index(seed=0)) == 2
    with path.open("ab") as out:
        out.write(b', "search": []}\n')
    assert len(store.index(seed=0)) == 3


@pytest.mark.parametrize(
    "query", ["seed=../0", "seed=-1", "seed=0&seed=1", "seed=", "", "seed=0&x=1"]
)
def test_invalid_query(*, query: str) -> None:
    with pytest.raises(ValueError):
        ExplorerHandler.parse_query(query=query, expected=("seed",))


def test_valid_queries() -> None:
    assert ExplorerHandler.parse_query(query="", expected=()) == {}
    assert ExplorerHandler.parse_query(query="seed=9&index=0", expected=("seed", "index")) == {
        "seed": 9,
        "index": 0,
    }


def test_symlink_escape(*, tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "pomdp_decisions.jsonl").write_text("{}\n")
    (root / "1").symlink_to(outside, target_is_directory=True)
    assert DecisionStore(root=root).seeds() == []


@pytest.mark.parametrize(
    "line", [b"\xff\n", b"bad json\n", b"[]\n", b'{"event":"decision","search":1}\n']
)
def test_bad_log_records(*, line: bytes) -> None:
    with pytest.raises(ValueError, match="byte offset 12"):
        DecisionStore.parse_entry(line=line, offset=12, index=0)


def test_http_endpoints(*, tmp_path: Path) -> None:
    (tmp_path / "0").mkdir()
    record = {"event": "decision", "action": "STOP", "search": []}
    original = json.dumps(record).encode() + b"\n"
    (tmp_path / "0" / "pomdp_decisions.jsonl").write_bytes(original)
    (tmp_path / "index.html").write_text("test html")
    (tmp_path / "explorer.js").write_text("test js")
    (tmp_path / "explorer.css").write_text("test css")

    class FixtureHandler(ExplorerHandler):
        store = DecisionStore(root=tmp_path)
        static_root = tmp_path

    with HTTPServer(("127.0.0.1", 0), FixtureHandler) as http:
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{http.server_port}"
        try:
            with urlopen(base + "/api/seeds") as response:
                assert json.load(response) == {"seeds": [0]}
            for hostile in [
                "attacker.example",
                "localhost@attacker.example",
                "localhost/path",
                "localhost:bad",
            ]:
                with pytest.raises(HTTPError) as caught:
                    urlopen(Request(base + "/api/seeds", headers={"Host": hostile}))
                assert caught.value.code == 403
            for allowed in ["localhost", "localhost:8766", "127.0.0.1", "[::1]:8766"]:
                with urlopen(Request(base + "/api/seeds", headers={"Host": allowed})) as response:
                    assert json.load(response) == {"seeds": [0]}
            with urlopen(base + "/api/index?seed=0") as response:
                assert json.load(response)["decisions"][0]["action"] == "STOP"
            with urlopen(base + "/api/decision?seed=0&index=0") as response:
                assert response.headers["Content-Encoding"] == "gzip"
                assert gzip.decompress(response.read()) == original
            for asset in ["/", "/index.html", "/explorer.js", "/explorer.css"]:
                with urlopen(base + asset) as response:
                    assert response.status == 200
            for endpoint, code in [
                ("/api/index", 400),
                ("/api/seeds?seed=0", 400),
                ("/api/decision?seed=0&index=99", 404),
                ("/../server.py", 404),
            ]:
                with pytest.raises(HTTPError) as caught:
                    urlopen(base + endpoint)
                assert caught.value.code == code
        finally:
            http.shutdown()
            thread.join(timeout=2)
