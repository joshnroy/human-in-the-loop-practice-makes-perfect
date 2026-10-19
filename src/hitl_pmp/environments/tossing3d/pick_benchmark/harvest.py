"""Benchmark states harvested from real runs: the scene just before a human reset the
planners asked for because `PickPlannable` was false.

The reset attribution (`artifacts/tossing3d-reset-attribution/attribution.json`) lists
every reset with its `pp` verdict and `anchor_t`, the timestamp of the last robot
skill's own completion before it. The run's `tossing3d_state_log.jsonl` logs KINDER's
whole object-centric state at every control step, so the last tick at or before
`anchor_t` is the pre-reset scene exactly -- no reconstruction. Resets across the
barrier are left out: there the dry run is not made at all (`pick_plannable.py`), and
no pick is applicable.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

ACROSS_BARRIER = "across barrier"
TIMESTAMP = re.compile(r'"timestamp": "([^"]+)"')


class Harvest(BaseModel):
    source: str
    d4: str
    state: dict[str, list[float]]


class PickHarvest:
    """A static-method container, never instantiated."""

    @staticmethod
    def refusals(*, attribution: dict[str, Any], run_prefix: str) -> list[Harvest]:
        out: list[Harvest] = []
        for label, run in sorted(attribution.items()):
            if not label.startswith(run_prefix):
                continue
            wanted = [
                r
                for r in run["resets"]
                if r.get("pp") is False
                and r.get("anchor_t")
                and not str(r.get("d4", "")).startswith(ACROSS_BARRIER)
            ]
            if not wanted:
                continue
            log = Path(run["run_dir"]) / "tossing3d_state_log.jsonl"
            states = PickHarvest.ticks_before(
                path=log, times=[PickHarvest.seconds(stamp=r["anchor_t"]) + 1e-3 for r in wanted]
            )
            for reset, state in zip(wanted, states, strict=True):
                if state is not None:
                    out.append(
                        Harvest(
                            source=f"{label}#cycle{reset.get('cycle')}",
                            d4=str(reset.get("d4")),
                            state=state,
                        )
                    )
        return out

    @staticmethod
    def seconds(*, stamp: str) -> float:
        return datetime.fromisoformat(stamp).timestamp()

    @staticmethod
    def ticks_before(*, path: Path, times: list[float]) -> list[dict[str, list[float]] | None]:
        """For each time, the last tick logged at or before it. One streaming pass; a
        tick line is parsed only when it is the one a time needs."""
        order = sorted(range(len(times)), key=lambda i: times[i])
        found: list[dict[str, list[float]] | None] = [None] * len(times)
        cursor = 0
        last_line: str | None = None
        with path.open() as stream:
            for line in stream:
                if '"kind": "tick"' not in line:
                    continue
                match = TIMESTAMP.search(line)
                if match is None:
                    continue
                stamp = PickHarvest.seconds(stamp=match.group(1))
                while cursor < len(order) and times[order[cursor]] < stamp:
                    if last_line is not None:
                        found[order[cursor]] = json.loads(last_line)["state"]
                    cursor += 1
                if cursor == len(order):
                    break
                last_line = line
        while cursor < len(order) and last_line is not None:
            found[order[cursor]] = json.loads(last_line)["state"]
            cursor += 1
        return found
