import json
import random
import tracemalloc
from pathlib import Path

import pytest
import zstandard

from hitl_pmp.results_writer.jsonl_log import JsonlLog

ROWS = [{"event": "dispatch", "cycle": 0, "cost": 1.0}, {"event": "refit", "cycle": 0}]


def _write_plain(*, path: Path) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in ROWS), encoding="utf-8")


def _compress(*, path: Path, long_window_log: int = 27, level: int = 19) -> Path:
    """Compress the way the disk cleanup does (`zstd --long=27`), then drop the original."""
    params = zstandard.ZstdCompressionParameters.from_level(
        level, window_log=long_window_log, enable_ldm=True
    )
    target = path.with_name(path.name + ".zst")
    target.write_bytes(
        zstandard.ZstdCompressor(compression_params=params).compress(path.read_bytes())
    )
    path.unlink()
    return target


def test_a_plain_log_is_read_as_before(*, tmp_path: Path) -> None:
    path = tmp_path / "pomdp_decisions.jsonl"
    _write_plain(path=path)
    assert JsonlLog.exists(path=path)
    assert JsonlLog.resolve(path=path) == path
    with JsonlLog.open_text(path=path) as stream:
        assert [json.loads(line) for line in stream] == ROWS


def test_the_plain_name_finds_a_compressed_sibling(*, tmp_path: Path) -> None:
    path = tmp_path / "pomdp_decisions.jsonl"
    _write_plain(path=path)
    compressed = _compress(path=path)
    assert not path.exists()
    assert JsonlLog.exists(path=path)
    assert JsonlLog.resolve(path=path) == compressed
    with JsonlLog.open_text(path=path) as stream:
        assert [json.loads(line) for line in stream] == ROWS


def test_a_compressed_path_can_be_named_directly(*, tmp_path: Path) -> None:
    path = tmp_path / "sweep_decisions.jsonl"
    _write_plain(path=path)
    compressed = _compress(path=path)
    with JsonlLog.open_text(path=compressed) as stream:
        assert [json.loads(line) for line in stream] == ROWS


def test_a_log_written_one_zstd_frame_per_record_reads_every_record(*, tmp_path: Path) -> None:
    """A streaming writer appends one frame per record; reading must not stop at the first."""
    compressed = tmp_path / "pomdp_decisions.jsonl.zst"
    compressor = zstandard.ZstdCompressor()
    with compressed.open("ab") as stream:
        for row in ROWS:
            stream.write(compressor.compress((json.dumps(row) + "\n").encode("utf-8")))
    with JsonlLog.open_text(path=tmp_path / "pomdp_decisions.jsonl") as stream:
        assert [json.loads(line) for line in stream] == ROWS


def test_a_long_window_beyond_the_default_limit_still_decodes(*, tmp_path: Path) -> None:
    # zstd's default decoder refuses windows above 2**27; `--long=31` needs the limit lifted.
    path = tmp_path / "pomdp_decisions.jsonl"
    _write_plain(path=path)
    _compress(path=path, long_window_log=31)
    with JsonlLog.open_text(path=path) as stream:
        assert [json.loads(line) for line in stream] == ROWS


def test_the_plain_file_wins_when_both_exist(*, tmp_path: Path) -> None:
    # A run still appending writes the plain name; a stale compressed copy must not shadow it.
    path = tmp_path / "pomdp_decisions.jsonl"
    _write_plain(path=path)
    _compress(path=path)
    path.write_text(json.dumps({"event": "fresh"}) + "\n", encoding="utf-8")
    with JsonlLog.open_text(path=path) as stream:
        assert [json.loads(line) for line in stream] == [{"event": "fresh"}]


def test_a_missing_log_raises_on_the_name_the_caller_gave(*, tmp_path: Path) -> None:
    path = tmp_path / "pomdp_decisions.jsonl"
    assert not JsonlLog.exists(path=path)
    with pytest.raises(FileNotFoundError, match="pomdp_decisions.jsonl"):
        JsonlLog.open_text(path=path)


def test_lines_stream_without_holding_the_whole_log_in_memory(*, tmp_path: Path) -> None:
    rng = random.Random(0)
    path = tmp_path / "pomdp_decisions.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for index in range(20_000):
            stream.write(json.dumps({"i": index, "blob": rng.randbytes(500).hex()}) + "\n")
    decompressed_size = path.stat().st_size
    _compress(path=path, level=1)
    tracemalloc.start()
    try:
        with JsonlLog.open_text(path=path) as stream:
            count = sum(1 for _ in stream)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert count == 20_000
    assert peak < decompressed_size / 4


def _frames(*, data: bytes) -> list[bytes]:
    """Split concatenated zstd frames, decompressing each one on its own."""
    frames = []
    while data:
        decoder = zstandard.ZstdDecompressor(max_window_size=2**31).decompressobj()
        frames.append(decoder.decompress(data))
        data = decoder.unused_data
    return frames


def test_appending_to_a_zst_path_writes_one_frame_per_record(*, tmp_path: Path) -> None:
    # Each record is its own frame, so a run killed mid-write loses only its last record.
    path = tmp_path / "pomdp_decisions.jsonl.zst"
    lines = [json.dumps(row) + "\n" for row in ROWS]
    for line in lines:
        JsonlLog.append(path=path, line=line)
    assert _frames(data=path.read_bytes()) == [line.encode("utf-8") for line in lines]


def test_appending_to_a_plain_path_writes_plain_text(*, tmp_path: Path) -> None:
    path = tmp_path / "pomdp_decisions.jsonl"
    for row in ROWS:
        JsonlLog.append(path=path, line=json.dumps(row) + "\n")
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == ROWS


def test_a_log_appended_through_jsonl_log_round_trips(*, tmp_path: Path) -> None:
    # The writer and the reader together: what a run appends live is what analysis reads.
    path = tmp_path / "pomdp_decisions.jsonl"
    rows = [{"event": "decision", "i": index, "belief": "x" * 70_000} for index in range(5)]
    for row in rows:
        JsonlLog.append(path=path.with_name(path.name + ".zst"), line=json.dumps(row) + "\n")
    with JsonlLog.open_text(path=path) as stream:
        assert [json.loads(line) for line in stream] == rows
