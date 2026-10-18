"""Read a run's JSONL logs whether or not they have been compressed with zstd.

The decision logs (`pomdp_decisions.jsonl`, `sweep_decisions.jsonl`) reach tens of GB per
run, so finished runs are compressed in place to `<name>.jsonl.zst` to free disk. Every
reader asks for the plain name and gets whichever exists, so neither the readers nor the
run layout `analysis/` globs for has to know which runs were compressed.

**The plain file wins when both exist.** A run that is still appending writes the plain
name, so a compressed copy beside it is the older one.

**Streamed, never read whole.** A decompressed log can exceed RAM; `open_text` yields
lines from a bounded buffer, as reading the plain file line by line always did.

**The decoder's window limit is lifted.** `zstd --long=N` for `N > 27` writes frames the
default decoder rejects; reading them must not depend on how a file was compressed."""

import io
from pathlib import Path
from typing import TextIO

import zstandard

COMPRESSED_SUFFIX = ".zst"
# The format's own maximum window (`ZSTD_WINDOWLOG_MAX` on 64-bit builds).
_MAX_WINDOW_SIZE = 2**31


class JsonlLog:
    """Resolves and opens a JSONL log stored plain or as a `.zst` sibling."""

    @staticmethod
    def resolve(*, path: Path) -> Path:
        """The file to read for `path`: itself if present, else its `.zst` sibling."""
        if path.suffix == COMPRESSED_SUFFIX or path.exists():
            return path
        compressed = path.with_name(path.name + COMPRESSED_SUFFIX)
        return compressed if compressed.exists() else path

    @staticmethod
    def exists(*, path: Path) -> bool:
        return JsonlLog.resolve(path=path).exists()

    @staticmethod
    def open_text(*, path: Path) -> TextIO:
        """A text stream over the log's lines; close it (or use it as a context manager)."""
        resolved = JsonlLog.resolve(path=path)
        if resolved.suffix != COMPRESSED_SUFFIX:
            return resolved.open(encoding="utf-8")
        raw = resolved.open("rb")
        try:
            reader = zstandard.ZstdDecompressor(max_window_size=_MAX_WINDOW_SIZE).stream_reader(
                raw, closefd=True
            )
        except BaseException:
            raw.close()
            raise
        return io.TextIOWrapper(io.BufferedReader(reader), encoding="utf-8")
