"""Read-only loopback explorer for saved POMDP decision logs."""

import argparse
import gzip
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import parse_qs, urlsplit

from pydantic import BaseModel, PrivateAttr

from .types import DecisionEntry, DecisionIndex


class ExplorerCli:
    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--results-root", type=Path, required=True)
        parser.add_argument("--port", type=int, default=8766)
        args = parser.parse_args()
        ExplorerHandler.store = DecisionStore(root=args.results_root)
        # Serial requests bound trace parsing to one decision, not N clients.
        with HTTPServer(("127.0.0.1", args.port), ExplorerHandler) as server:
            print(f"Explorer: http://127.0.0.1:{args.port}/", flush=True)
            server.serve_forever()


class ExplorerHandler(BaseHTTPRequestHandler):
    store: ClassVar["DecisionStore"]
    static_root: ClassVar[Path] = Path(__file__).parent
    assets: ClassVar[dict[str, tuple[str, str]]] = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/index.html": ("index.html", "text/html; charset=utf-8"),
        "/explorer.js": ("explorer.js", "application/javascript; charset=utf-8"),
        "/explorer.css": ("explorer.css", "text/css; charset=utf-8"),
    }

    def do_GET(self) -> None:
        try:
            if not self.allowed_host():
                self.send_error(403, "Only loopback hosts are allowed")
                return
            self.serve_request()
        except ValueError as exc:
            self.send_error(400, str(exc))
        except FileNotFoundError:
            self.send_error(404, "Requested artifact not found")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except OSError:
            self.send_error(500, "Unable to read requested artifact")

    def allowed_host(self) -> bool:
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or any(character.isspace() for character in hosts[0]):
            return False
        try:
            host = urlsplit("//" + hosts[0])
            return (
                host.hostname in {"localhost", "127.0.0.1", "::1"}
                and host.username is None
                and host.password is None
                and not host.path
                and not host.query
                and not host.fragment
                and (host.port is None or 0 < host.port <= 65535)
            )
        except ValueError:
            return False

    def serve_request(self) -> None:
        url = urlsplit(self.path)
        if url.path in self.assets:
            self.parse_query(query=url.query, expected=())
            filename, content_type = self.assets[url.path]
            self.send_payload(
                payload=(self.static_root / filename).read_bytes(), content_type=content_type
            )
            return
        result: dict[str, Any]
        if url.path == "/api/seeds":
            self.parse_query(query=url.query, expected=())
            result = {"seeds": self.store.seeds()}
        elif url.path == "/api/index":
            params = self.parse_query(query=url.query, expected=("seed",))
            result = {"decisions": [entry.metadata for entry in self.store.index(**params)]}
        elif url.path == "/api/decision":
            params = self.parse_query(query=url.query, expected=("seed", "index"))
            self.send_payload(
                payload=self.store.decision(**params),
                content_type="application/json; charset=utf-8",
                compress=True,
            )
            return
        else:
            self.send_error(404, "Unknown endpoint")
            return
        self.send_payload(
            payload=json.dumps(result).encode(), content_type="application/json; charset=utf-8"
        )

    @staticmethod
    def parse_query(*, query: str, expected: tuple[str, ...]) -> dict[str, int]:
        params = parse_qs(query, keep_blank_values=True, strict_parsing=True) if query else {}
        if set(params) != set(expected):
            raise ValueError("Unexpected or missing query parameter")
        values = {}
        for key, raw in params.items():
            if len(raw) != 1 or not re.fullmatch(r"0|[1-9][0-9]*", raw[0]):
                raise ValueError("Query parameters must be nonnegative integers")
            values[key] = int(raw[0])
        return values

    def send_payload(self, *, payload: bytes, content_type: str, compress: bool = False) -> None:
        if compress:
            payload = gzip.compress(payload, compresslevel=3, mtime=0)
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if compress:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(payload)


class DecisionStore(BaseModel):
    root: Path
    _cache: dict[int, DecisionIndex] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self.root = self.root.resolve(strict=True)

    def seeds(self) -> list[int]:
        return sorted(
            int(p.name)
            for p in self.root.iterdir()
            if re.fullmatch(r"0|[1-9][0-9]*", p.name)
            and p.is_dir()
            and (p / "pomdp_decisions.jsonl").is_file()
            and (p / "pomdp_decisions.jsonl").resolve().is_relative_to(self.root)
        )

    def path(self, *, seed: int) -> Path:
        if seed not in self.seeds():
            raise FileNotFoundError("Unknown seed")
        return self.root / str(seed) / "pomdp_decisions.jsonl"

    def index(self, *, seed: int) -> list[DecisionEntry]:
        path = self.path(seed=seed)
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        cached = self._cache.get(seed)
        if cached is not None and cached.signature == signature:
            return cached.entries
        entries: list[DecisionEntry] = []
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line or not line.endswith(b"\n"):
                    # An in-flight append is not a completed decision yet.
                    break
                entry = self.parse_entry(line=line, offset=offset, index=len(entries))
                if entry is not None:
                    entries.append(entry)
        self._cache[seed] = DecisionIndex(signature=signature, entries=entries)
        return self._cache[seed].entries

    @staticmethod
    def parse_entry(*, line: bytes, offset: int, index: int) -> DecisionEntry | None:
        try:
            record = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"Invalid JSON log record at byte offset {offset}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected a JSON object at byte offset {offset}")
        if record.get("event") != "decision":
            return None
        events = record.get("search", [])
        if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
            raise ValueError(f"Invalid search trace at byte offset {offset}")
        root: dict[str, Any] = next(
            (event for event in events if event.get("event") == "node" and event.get("node") == 0),
            {},
        )
        return DecisionEntry(
            offset=offset,
            length=len(line),
            metadata={
                "index": index,
                "cycle": record.get("cycle"),
                "decision": record.get("decision"),
                "action": record.get("action"),
                "value": record.get("value"),
                "search_duration_seconds": record.get("search_duration_seconds"),
                "summed_cost": root.get("summed_cost"),
                "horizon": record.get("horizon", root.get("horizon")),
                "event_count": len(events),
            },
        )

    def decision(self, *, seed: int, index: int) -> bytes:
        entries = self.index(seed=seed)
        if index < 0 or index >= len(entries):
            raise FileNotFoundError("Unknown decision index")
        entry = entries[index]
        with self.path(seed=seed).open("rb") as stream:
            stream.seek(entry.offset)
            return stream.read(entry.length)


if __name__ == "__main__":
    ExplorerCli.main()
