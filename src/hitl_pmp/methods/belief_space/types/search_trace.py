"""Recorded expectimax search events."""

import gzip
import json
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel, Field, PrivateAttr

from hitl_pmp.core.log_timing import LogTiming


@contextmanager
def open_trace(*, path: Path) -> Iterator[TextIO]:
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as stream:
            yield stream
    else:
        with path.open("w", encoding="utf-8") as stream:
            yield stream


class SearchTrace(BaseModel):
    """Actual search events, recorded without additional model calls or RNG draws."""

    events: list[dict[str, Any]] = Field(default_factory=list)
    path: Path | None = Field(default=None, exclude=True)
    retain_events: bool = Field(default=True, exclude=True)
    _stream: TextIO | None = PrivateAttr(default=None)
    _stream_stack: ExitStack = PrivateAttr(default_factory=ExitStack)
    _interned: dict[tuple[str, str], int] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self._stream_stack.enter_context(open_trace(path=self.path))

    def record(self, *, event: str, **fields: Any) -> None:
        record = {"event": event, **fields, **LogTiming.fields()}
        if self.retain_events or fields.get("node") == 0:
            self.events.append(record)
        if self._stream is not None:
            streamed = dict(record)
            for field, kind in {
                "environment_state": "state",
                "successor": "state",
                "belief_state": "belief",
                "action": "action",
            }.items():
                value = streamed.get(field)
                if not isinstance(value, (dict, list)):
                    continue
                encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
                key = (kind, encoded)
                identifier = self._interned.get(key)
                if identifier is None:
                    identifier = len(self._interned)
                    self._interned[key] = identifier
                    self._stream.write(
                        LogTiming.encode(
                            record={
                                "event": "intern",
                                "kind": kind,
                                "id": identifier,
                                "value": value,
                            }
                        )
                    )
                streamed[field] = {"$ref": identifier, "$kind": kind}
            self._stream.write(LogTiming.encode(record=streamed))

    def close(self) -> None:
        if self._stream is not None:
            self._stream_stack.close()
            self._stream = None
