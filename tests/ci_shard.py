"""Split one pytest run across several CI jobs (`--shard-count N --shard-index i`).

Every shard collects the whole suite and keeps a deterministic share of it, so the shares
partition the run exactly. The unit of assignment is a test, except that an
`xdist_group` moves as one unit: its tests share an expensive module-scoped fixture, which
every extra shard holding some of them builds again. The one exception is a group heavier
than one worker's fair share, which would otherwise set the floor of whichever shard got
it: it is cut into contiguous chunks on distinct shards (`split`). Units are weighted by
`ci_shard_durations.json` (seconds per node id, measured on CI) and dealt out
longest-first to the least-loaded shard. A test missing from that file, being new or
renamed, weighs the median: a stale file costs balance, never coverage.

Regenerate the file from a CI run's `junit-tossing3d-*` artifacts:
`python -m tests.ci_shard junit-*.xml`.
"""

import itertools
import json
import math
import statistics
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest
from pydantic import BaseModel


class ShardUnit(BaseModel):
    """What `select` places on one shard: a test, or an `xdist_group` (or a chunk of one)."""

    key: str
    nodeids: list[str]
    weight: float
    group: str | None = None
    # Per-test weights, parallel to `nodeids`; empty means the weight is spread evenly.
    test_weights: list[float] = []

    @property
    def chunk_weights(self) -> list[float]:
        return self.test_weights or [self.weight / len(self.nodeids)] * len(self.nodeids)


class CiShard:
    DURATIONS = Path(__file__).with_name("ci_shard_durations.json")
    # xdist workers per shard: `-n auto` counts the 2 physical cores of a 4-vCPU runner.
    WORKERS_PER_SHARD = 2
    # Most tests round to 0.00 s, so the median default is often 0; a floor keeps ties
    # among them broken by unit count rather than all landing on shard 0.
    MIN_WEIGHT = 0.01

    @staticmethod
    def add_options(*, parser: pytest.Parser) -> None:
        group = parser.getgroup("ci-shard")
        group.addoption("--shard-count", type=int, default=1, help="split the run this many ways")
        group.addoption("--shard-index", type=int, default=0, help="0-based share to keep")

    @staticmethod
    def apply(*, config: pytest.Config, items: list[pytest.Item]) -> None:
        count = config.getoption("--shard-count")
        index = config.getoption("--shard-index")
        if count < 1 or not 0 <= index < count:
            raise pytest.UsageError(f"--shard-index {index} is outside 0..{count - 1}")
        if count == 1:
            return
        durations = json.loads(CiShard.DURATIONS.read_text(encoding="utf-8"))
        units = CiShard.units(
            items=[(item.nodeid, CiShard.group(item=item)) for item in items], durations=durations
        )
        keep = {
            nodeid
            for unit in CiShard.select(units=units, index=index, count=count)
            for nodeid in unit.nodeids
        }
        selected = [item for item in items if item.nodeid in keep]
        deselected = [item for item in items if item.nodeid not in keep]
        if deselected:
            config.hook.pytest_deselected(items=deselected)
        items[:] = selected

    @staticmethod
    def group(*, item: pytest.Item) -> str | None:
        marker = item.get_closest_marker("xdist_group")
        if marker is None:
            return None
        return str(marker.args[0] if marker.args else marker.kwargs.get("name", "default"))

    @staticmethod
    def units(
        *, items: list[tuple[str, str | None]], durations: dict[str, float]
    ) -> list[ShardUnit]:
        default = statistics.median(durations.values()) if durations else 1.0
        units: dict[str, ShardUnit] = {}
        for nodeid, group in items:
            key = f"group:{group}" if group is not None else nodeid
            unit = units.setdefault(
                key, ShardUnit(key=group or nodeid, nodeids=[], weight=0.0, group=group)
            )
            weight = max(durations.get(nodeid.split("@")[0], default), CiShard.MIN_WEIGHT)
            unit.nodeids.append(nodeid)
            unit.test_weights.append(weight)
            unit.weight += weight
        return list(units.values())

    @staticmethod
    def durations_from_junit(*, paths: list[Path], repo: Path) -> dict[str, float]:
        """Mean seconds per node id over junit reports (`--junitxml`) of earlier runs.

        A junit `classname` is the node id's path with `/` -> `.` and `.py` dropped, then
        any class names, so the module is the longest dotted prefix that is a real file.
        xdist's `@group` suffix on a name is stripped, since collection never carries it.
        """
        times: dict[str, list[float]] = {}
        for path in paths:
            for case in ElementTree.parse(path).iter("testcase"):
                parts = case.attrib["classname"].split(".")
                cut = next(
                    cut
                    for cut in range(len(parts), 0, -1)
                    if (repo / Path(*parts[:cut])).with_suffix(".py").is_file()
                )
                module = "/".join(parts[:cut]) + ".py"
                name = case.attrib["name"].split("@")[0]
                nodeid = "::".join([module, *parts[cut:], name])
                times.setdefault(nodeid, []).append(float(case.attrib.get("time", 0.0)))
        return {nodeid: round(statistics.fmean(ts), 2) for nodeid, ts in sorted(times.items())}

    @staticmethod
    def main(*, argv: list[str]) -> None:
        """`python -m tests.ci_shard junit.xml ...` rewrites `ci_shard_durations.json`."""
        durations = CiShard.durations_from_junit(
            paths=[Path(arg) for arg in argv], repo=Path(__file__).resolve().parents[1]
        )
        CiShard.DURATIONS.write_text(json.dumps(durations, indent=0) + "\n", encoding="utf-8")
        print(f"{len(durations)} tests, {sum(durations.values()) / 60:.1f} min in total")

    @staticmethod
    def split(*, units: list[ShardUnit], count: int) -> list[ShardUnit]:
        """Cut each group heavier than one worker's fair share into contiguous chunks.

        The fair share is the whole run's weight over `count * WORKERS_PER_SHARD`. A group
        becomes at most `count` chunks, so `select` can give each its own shard; each chunk
        then costs one more build of the group's fixture, which is cheap beside the tests.
        """
        share = sum(unit.weight for unit in units) / (count * CiShard.WORKERS_PER_SHARD)
        out: list[ShardUnit] = []
        for unit in units:
            pieces = min(count, len(unit.nodeids), math.ceil(unit.weight / share) if share else 1)
            if unit.group is None or pieces <= 1:
                out.append(unit)
                continue
            weights = unit.chunk_weights
            bounds, cumulative = [0], 0.0
            for position, weight in enumerate(weights[:-1], start=1):
                cumulative += weight
                if cumulative >= unit.weight * len(bounds) / pieces and len(bounds) < pieces:
                    bounds.append(position)
            bounds.append(len(weights))
            for chunk, (start, stop) in enumerate(itertools.pairwise(bounds)):
                out.append(
                    ShardUnit(
                        key=f"{unit.key}#{chunk}",
                        nodeids=unit.nodeids[start:stop],
                        weight=sum(weights[start:stop]),
                        group=unit.group,
                    )
                )
        return out

    @staticmethod
    def select(*, units: list[ShardUnit], index: int, count: int) -> list[ShardUnit]:
        """Longest first onto the least-loaded shard (ties: fewest units, then lowest index).

        A chunk of a split group avoids shards already holding a chunk of that group.
        """
        loads = [0.0] * count
        sizes = [0] * count
        groups: list[set[str]] = [set() for _ in range(count)]
        mine: list[ShardUnit] = []
        for unit in sorted(
            CiShard.split(units=units, count=count), key=lambda u: (-u.weight, u.key)
        ):
            free = [s for s in range(count) if unit.group is None or unit.group not in groups[s]]
            target = min(free or range(count), key=lambda s: (loads[s], sizes[s], s))
            loads[target] += unit.weight
            sizes[target] += 1
            if unit.group is not None:
                groups[target].add(unit.group)
            if target == index:
                mine.append(unit)
        return mine


if __name__ == "__main__":
    CiShard.main(argv=sys.argv[1:])
