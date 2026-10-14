import json
from collections.abc import Iterator
from pathlib import Path

from pydantic import TypeAdapter

from .types import SCHEMA_VERSION, RecordEntry, RecordingHeader


class RunRecordReader:
    """Reads a `recording.jsonl` back. `entries` streams, so a recording far larger
    than memory replays in constant space."""

    _entry: TypeAdapter[RecordEntry] = TypeAdapter(RecordEntry)

    @staticmethod
    def header(*, path: Path) -> RecordingHeader:
        with path.open(encoding="utf-8") as stream:
            first = json.loads(stream.readline())
        return RunRecordReader.parse_header(record=first, source=path)

    @staticmethod
    def parse_header(*, record: dict, source: Path) -> RecordingHeader:
        if record.get("kind") != "header":
            raise ValueError(f"{source} does not begin with a recording header")
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{source} has schema_version {record.get('schema_version')!r}; "
                f"this reader understands {SCHEMA_VERSION}"
            )
        return RecordingHeader.model_validate(record)

    @staticmethod
    def entries(*, path: Path) -> Iterator[RecordEntry]:
        RunRecordReader.header(path=path)
        with path.open(encoding="utf-8") as stream:
            next(stream)
            for line in stream:
                if line.strip():
                    yield RunRecordReader._entry.validate_json(line)
