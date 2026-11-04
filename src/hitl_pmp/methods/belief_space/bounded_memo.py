"""A pure function's memo table with a size ceiling."""

from collections.abc import Hashable
from typing import Generic, TypeVar

KeyT = TypeVar("KeyT", bound=Hashable)
ValueT = TypeVar("ValueT")


class BoundedMemo(Generic[KeyT, ValueT]):
    """Emptied whole when it reaches either ceiling, rather than evicting one entry.

    Search revisits the same few hundred inputs within one decision and rarely across
    decisions, so a full table is mostly stale: dropping it all costs one decision's
    warm-up, and needs no recency bookkeeping on the hot path. The byte ceiling counts
    what each caller reports for its value, since posteriors vary by orders of
    magnitude in size (a grid's weights from kilobytes to hundreds of kilobytes).
    """

    def __init__(self, *, max_entries: int, max_bytes: int | None = None) -> None:
        assert max_entries >= 1
        assert max_bytes is None or max_bytes >= 1
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._table: dict[KeyT, ValueT] = {}
        self._bytes = 0

    def __len__(self) -> int:
        return len(self._table)

    def get(self, *, key: KeyT) -> ValueT | None:
        return self._table.get(key)

    def put(self, *, key: KeyT, value: ValueT, nbytes: int = 0) -> None:
        if len(self._table) >= self.max_entries or (
            self.max_bytes is not None and self._bytes + nbytes > self.max_bytes
        ):
            self.clear()
        self._table[key] = value
        self._bytes += nbytes

    def clear(self) -> None:
        self._table.clear()
        self._bytes = 0
