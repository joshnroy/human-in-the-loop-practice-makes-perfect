"""Follow a log another process is still appending to, reading only what is new.

The dashboard polls every few seconds while a run writes; a decision log grows to
hundreds of MB, so each poll reads from where the last one stopped and returns only
complete lines. A line still being written waits for its newline, and a `.jsonl.zst` log
(one zstd frame per record, `JsonlLog.append`) waits for its frame to be whole. A file
that shrank, or was replaced, is read again from the start and says so (`restarted`), so
the reader can drop what it had.
"""

from pathlib import Path

import zstandard
from pydantic import BaseModel, PrivateAttr

from hitl_pmp.results_writer.jsonl_log import COMPRESSED_SUFFIX, JsonlLog

# The format's own maximum window, as `JsonlLog` reads with.
_MAX_WINDOW_SIZE = 2**31
# One poll reads at most this much, so a first look at a long run stays bounded.
_CHUNK_BYTES = 64 * 2**20


class TailRead(BaseModel):
    lines: list[str]
    restarted: bool = False


class LogTail(BaseModel):
    """The new complete lines of `path` (or its `.zst` sibling) since the last `read`."""

    path: Path

    _resolved: Path | None = PrivateAttr(default=None)
    _inode: int | None = PrivateAttr(default=None)
    _offset: int = PrivateAttr(default=0)
    _pending: bytes = PrivateAttr(default=b"")

    def read(self) -> TailRead:
        resolved = JsonlLog.resolve(path=self.path)
        try:
            stat = resolved.stat()
        except FileNotFoundError:
            return TailRead(lines=[])
        restarted = False
        if resolved != self._resolved or stat.st_ino != self._inode or stat.st_size < self._offset:
            restarted = self._resolved is not None
            self._resolved, self._inode, self._offset, self._pending = (
                resolved,
                stat.st_ino,
                0,
                b"",
            )
        lines: list[str] = []
        size = _CHUNK_BYTES
        while self._offset < stat.st_size:
            with resolved.open("rb") as stream:
                stream.seek(self._offset)
                chunk = stream.read(min(size, stat.st_size - self._offset))
            if not chunk:
                break
            if resolved.suffix == COMPRESSED_SUFFIX:
                consumed, text = LogTail._frames(data=chunk)
                if consumed == 0:
                    if len(chunk) < size:
                        break  # The last frame is still being written.
                    size *= 2  # One frame longer than a chunk: read more at once.
                    continue
                self._offset += consumed
                lines.extend(self._split(data=text))
            else:
                self._offset += len(chunk)
                lines.extend(self._split(data=chunk))
        return TailRead(lines=lines, restarted=restarted)

    def _split(self, *, data: bytes) -> list[str]:
        data = self._pending + data
        *complete, self._pending = data.split(b"\n")
        return [line.decode("utf-8", errors="replace") for line in complete if line.strip()]

    @staticmethod
    def _frames(*, data: bytes) -> tuple[int, bytes]:
        """Decompress every whole frame at the front of `data`: bytes used, text out."""
        consumed = 0
        out = bytearray()
        decompressor = zstandard.ZstdDecompressor(max_window_size=_MAX_WINDOW_SIZE)
        while consumed < len(data):
            frame = decompressor.decompressobj()
            try:
                text = frame.decompress(data[consumed:])
            except zstandard.ZstdError:
                break
            if not frame.eof:
                break
            out += text
            consumed = len(data) - len(frame.unused_data)
        return consumed, bytes(out)
