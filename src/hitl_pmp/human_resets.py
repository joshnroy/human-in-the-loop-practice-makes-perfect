"""`human_resets.jsonl`: one line per charged human reset, written beside `stats.json`.

`stats.json` carries only `num_human_interventions_recorded` and
`summed_human_cost_recorded`, so it cannot say when a reset happened or where the human
put things. Reconstructing that from the state log undercounts. This records it at the
one place every practice-loop rescue is charged, `PracticeLoop._grant_movables_reset`,
so every Method that asks through `HumanCubeBinResetRequested` is covered alike.

A sibling file rather than a `Metrics` field because `stats.json`'s byte-stability is
how this repo proves a change did not alter results, and wall-clock timestamps would
break it on every run. A raw per-event log rather than a per-cycle summary because the
summary is derivable (`HumanResetLog.counts_by_cycle`) and the events are not
derivable from the summary.

| field | meaning |
| --- | --- |
| `cycle` | 0-based practice cycle, the same index `pomdp_decisions.jsonl` uses |
| `step_index` | 0-based action slot within that cycle's practice period |
| `num_online_transitions` | robot transitions taken so far, the learning-curve x-axis |
| `destination` | `HumanCubeBinResetRequested.destination`, as handed to the environment |
| `skill` | the ground skill that asked, which separates human from automatic resets |
| `cost` | the cost banked into `summed_human_cost_recorded` |
| `timestamp`, `elapsed_seconds` | `LogTiming`'s, comparable with the other event logs |

Written whenever there is an `--output-dir`, and created empty at open: an empty file
means no reset happened, and a missing file means the run predates this log. Flushed
per record, so a crashed run keeps every reset up to the crash.

The step protocol (`step_protocol.py`) handles resets itself and keeps its own
`human_resets_by_destination`; it does not write this file."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, ConfigDict, PrivateAttr

from hitl_pmp.core.log_timing import LogTiming

HUMAN_RESETS_FILENAME = "human_resets.jsonl"


class HumanResetLog(BaseModel):
    """Appends one `HumanReset` per charged rescue. Holds an open file handle, so it
    is an instance rather than a static-method container."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_path: Path

    _handle: TextIO | None = PrivateAttr(default=None)

    @staticmethod
    def open_if_requested(*, args: argparse.Namespace) -> "HumanResetLog | None":
        output_dir = getattr(args, "output_dir", None)
        if output_dir is None:
            return None
        return HumanResetLog.open(output_path=Path(output_dir) / HUMAN_RESETS_FILENAME)

    @staticmethod
    def open(*, output_path: Path) -> "HumanResetLog":
        return HumanResetLog(output_path=output_path)

    def model_post_init(self, context: object, /) -> None:  # noqa: PLR0917
        """Opened, and truncated, at construction rather than at the first reset: that
        is what makes an empty file mean "none happened", and a re-run into the same
        directory overwrites this exactly as it overwrites `stats.json`."""
        del context
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.output_path.open("w", encoding="utf-8")

    def record(self, *, reset: "HumanReset") -> None:
        if self._handle is None:
            raise ValueError(f"{self.output_path} was already closed")
        self._handle.write(LogTiming.encode(record=reset.model_dump(mode="json")))
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    @staticmethod
    def read(*, path: Path) -> list["HumanReset"]:
        """The records, without the timing fields: those describe the write, not the
        reset, and are not part of what two runs should agree on."""
        fields = set(HumanReset.model_fields)
        return [
            HumanReset.model_validate({
                key: value for key, value in json.loads(line).items() if key in fields
            })
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @staticmethod
    def counts_by_cycle(
        *, records: list["HumanReset"], num_cycles: int
    ) -> list[dict[str | None, int]]:
        """Resets per cycle, split by destination. One entry per cycle, `{}` included,
        so a cycle with no reset is a row rather than a gap."""
        counts: list[Counter[str | None]] = [Counter() for _ in range(num_cycles)]
        for record in records:
            counts[record.cycle][record.destination] += 1
        return [dict(count) for count in counts]


class HumanReset(BaseModel):
    """One charged reset. Frozen: a record of something that already happened."""

    model_config = ConfigDict(frozen=True)

    cycle: int
    step_index: int
    num_online_transitions: int
    destination: str | None
    skill: str | None
    cost: float
