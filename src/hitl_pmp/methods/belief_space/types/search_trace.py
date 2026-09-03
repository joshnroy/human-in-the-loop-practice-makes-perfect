"""Recorded expectimax search events."""

from __future__ import annotations

import gzip
import json
import multiprocessing
import os
import queue
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr

from hitl_pmp.core.log_timing import LogTiming

_QUEUE_CAPACITY = 256
_QUEUE_POLL_SECONDS = 0.25


@contextmanager
def open_trace(*, path: Path) -> Iterator[TextIO]:
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as stream:
            yield stream
    else:
        with path.open("w", encoding="utf-8") as stream:
            yield stream


def _write_record(
    *, stream: TextIO, interned: dict[tuple[str, str], int], record: dict[str, Any]
) -> None:
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
        identifier = interned.get(key)
        if identifier is None:
            identifier = len(interned)
            interned[key] = identifier
            stream.write(
                LogTiming.encode(
                    record={"event": "intern", "kind": kind, "id": identifier, "value": value}
                )
            )
        streamed[field] = {"$ref": identifier, "$kind": kind}
    stream.write(json.dumps(streamed, allow_nan=False) + "\n")


def _write_trace(*, records: Any, result: Any, temporary_path: Path, final_path: Path) -> None:
    """Drain trace records in a process that does not share the search's GIL."""
    try:
        interned: dict[tuple[str, str], int] = {}
        with open_trace(path=temporary_path) as stream:
            while (record := records.get()) is not None:
                _write_record(stream=stream, interned=interned, record=record)
        os.replace(temporary_path, final_path)
        result.put(("ok", None))
    except BaseException:  # pragma: no cover - observed and raised in the parent
        temporary_path.unlink(missing_ok=True)
        result.put(("error", traceback.format_exc()))


class SearchTrace(BaseModel):
    """Actual search events, encoded and compressed in a separate process."""

    events: list[dict[str, Any]] = Field(default_factory=list)
    path: Path | None = Field(default=None, exclude=True)
    retain_events: bool = Field(default=True, exclude=True)
    _records: Any = PrivateAttr(default=None)
    _result: Any = PrivateAttr(default=None)
    _writer: Any = PrivateAttr(default=None)
    _temporary_path: Path | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        suffix = ".tmp.gz" if self.path.suffix == ".gz" else ".tmp"
        self._temporary_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}{suffix}")
        context = multiprocessing.get_context("spawn")
        self._records = context.Queue(maxsize=_QUEUE_CAPACITY)
        self._result = context.Queue(maxsize=1)
        self._writer = context.Process(
            target=_write_trace,
            kwargs={
                "records": self._records,
                "result": self._result,
                "temporary_path": self._temporary_path,
                "final_path": self.path,
            },
            daemon=True,
            name="pomdp-search-trace-writer",
        )
        self._writer.start()

    def record(self, *, event: str, **fields: Any) -> None:
        record = {"event": event, **fields, **LogTiming.fields()}
        if self.retain_events or fields.get("node") == 0:
            self.events.append(record)
        if self._writer is not None:
            self._put(record=record)

    def _put(self, *, record: dict[str, Any] | None) -> None:
        while True:
            self._raise_if_writer_failed()
            try:
                self._records.put(record, timeout=_QUEUE_POLL_SECONDS)
                return
            except queue.Full:
                continue

    def _raise_if_writer_failed(self) -> None:
        if self._writer is not None and not self._writer.is_alive():
            self._finish_writer()

    def _finish_writer(self) -> None:
        status, detail = self._result.get(timeout=1.0)
        if status == "error":
            raise RuntimeError(f"search trace writer failed:\n{detail}")

    def close(self) -> None:
        if self._writer is None:
            return
        try:
            self._put(record=None)
            self._writer.join()
            self._finish_writer()
        finally:
            self._records.close()
            self._result.close()
            self._writer = None
