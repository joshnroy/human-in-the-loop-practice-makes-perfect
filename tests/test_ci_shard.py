"""`--shard-count N --shard-index i` splits one pytest run across N CI jobs, losing nothing.

`test / tossing3d` was a single ~44 min job holding 87.5 min of measured test time. Each
shard collects the full suite and keeps its own share, so the one thing that must hold is
that the shares partition the run: every test lands on exactly one shard. An
`xdist_group` (a module sharing an expensive fixture) stays whole unless it alone would
set a shard's floor, and then each chunk gets a shard of its own.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from .ci_shard import CiShard, ShardUnit

REPO = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"


def _units(*weights: float) -> list[ShardUnit]:
    return [
        ShardUnit(key=f"t{index}", nodeids=[f"t{index}"], weight=weight)
        for index, weight in enumerate(weights)
    ]


def test_every_unit_lands_on_exactly_one_shard() -> None:
    units = _units(5, 1, 1, 3, 2, 2, 8, 1)
    shards = [CiShard.select(units=units, index=index, count=3) for index in range(3)]
    flat = [unit.key for shard in shards for unit in shard]
    assert sorted(flat) == sorted(unit.key for unit in units)


def test_the_heaviest_units_are_spread_before_the_light_ones() -> None:
    loads = [
        sum(unit.weight for unit in CiShard.select(units=_units(9, 9, 1, 1), index=i, count=2))
        for i in range(2)
    ]
    assert loads == [10, 10]


def test_the_split_does_not_depend_on_collection_order() -> None:
    units = _units(5, 1, 1, 3, 2, 2, 8, 1)
    forward = [unit.key for unit in CiShard.select(units=units, index=1, count=3)]
    backward = [unit.key for unit in CiShard.select(units=units[::-1], index=1, count=3)]
    assert sorted(forward) == sorted(backward)


def test_an_xdist_group_is_one_unit_weighted_by_all_its_tests() -> None:
    units = CiShard.units(
        items=[("a.py::x", "grp"), ("a.py::y", "grp"), ("b.py::z", None)],
        durations={"a.py::x": 4.0, "a.py::y": 6.0, "b.py::z": 1.0},
    )
    by_key = {unit.key: unit for unit in units}
    assert by_key["grp"].nodeids == ["a.py::x", "a.py::y"]
    assert by_key["grp"].weight == 10.0
    assert by_key["b.py::z"].weight == 1.0


def test_a_group_heavier_than_one_workers_share_splits_into_contiguous_chunks() -> None:
    """`test_pick_plannable` alone was 18/87.5 min of the job and, as one group, the floor
    of any shard it landed on. Split, each chunk rebuilds the fixture once, on its own
    shard, and stays one xdist group there."""
    heavy = ShardUnit(key="grp", nodeids=[f"g{i}" for i in range(8)], weight=80.0, group="grp")
    light = _units(*[2.0] * 10)
    chunks = [u for u in CiShard.split(units=[heavy, *light], count=4) if u.group == "grp"]
    assert len(chunks) == 4
    assert [nodeid for chunk in chunks for nodeid in chunk.nodeids] == heavy.nodeids
    homes = [
        index
        for index in range(4)
        for unit in CiShard.select(units=[heavy, *light], index=index, count=4)
        if unit.group == "grp"
    ]
    assert sorted(homes) == [0, 1, 2, 3]


def test_a_light_group_is_never_split() -> None:
    group = ShardUnit(key="grp", nodeids=["a", "b"], weight=2.0, group="grp")
    assert CiShard.split(units=[group, *_units(*[2.0] * 20)], count=4) == [
        group,
        *_units(*[2.0] * 20),
    ]


def test_a_test_with_no_recorded_duration_weighs_the_median() -> None:
    units = CiShard.units(
        items=[("a::x", None), ("a::y", None), ("a::z", None), ("new::t", None)],
        durations={"a::x": 1.0, "a::y": 2.0, "a::z": 30.0},
    )
    assert {unit.key: unit.weight for unit in units}["new::t"] == 2.0


def test_the_shipped_durations_file_parses_and_is_keyed_by_node_id() -> None:
    durations = json.loads(CiShard.DURATIONS.read_text(encoding="utf-8"))
    assert durations, "empty durations file"
    assert all("::" in key and "@" not in key for key in durations)
    assert all(isinstance(value, float | int) and value >= 0 for value in durations.values())


class Collected:
    @staticmethod
    def nodeids(*extra: str) -> list[str]:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
            + [*extra, "tests/test_ci_caches.py", "tests/test_ci_coverage.py"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return [line for line in out.splitlines() if "::" in line]


def test_the_real_shards_partition_a_real_collection() -> None:
    whole = Collected.nodeids()
    shards = [Collected.nodeids("--shard-count", "3", "--shard-index", str(i)) for i in range(3)]
    assert all(shards), "a shard selected nothing"
    assert sorted(nodeid for shard in shards for nodeid in shard) == sorted(whole)


def test_an_out_of_range_shard_index_is_a_usage_error() -> None:
    with pytest.raises(subprocess.CalledProcessError):
        Collected.nodeids("--shard-count", "2", "--shard-index", "2")


@pytest.mark.parametrize("job_id", ["test-tossing3d", "test-unit"])
def test_every_shard_the_matrix_runs_is_one_the_command_asks_for(*, job_id: str) -> None:
    job = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"][job_id]
    shards = job["strategy"]["matrix"]["shard"]
    assert len(shards) > 1
    assert job["strategy"]["fail-fast"] is False
    assert shards == list(range(len(shards)))
    (run,) = [step["run"] for step in job["steps"] if "pytest" in step.get("run", "")]
    assert f"--shard-count {len(shards)}" in run
    assert "--shard-index ${{ matrix.shard }}" in run


def test_junit_timings_map_back_to_node_ids_without_the_xdist_suffix(*, tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuites><testsuite name="pytest">'
        '<testcase classname="tests.test_ci_shard" name="test_a[x, y]@grp" time="2.5"/>'
        '<testcase classname="tests.test_ci_shard.TestK" name="test_b" time="1.0"/>'
        '<testcase classname="tests.test_ci_shard" name="test_a[x, y]@grp" time="3.5"/>'
        # A pull-request run tests the merge with `main`, so it can time files this
        # checkout does not have yet; they are dropped rather than guessed at.
        '<testcase classname="tests.test_not_here_yet" name="test_c" time="9.0"/>'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    assert CiShard.durations_from_junit(paths=[junit], repo=REPO) == {
        "tests/test_ci_shard.py::test_a[x, y]": 3.0,
        "tests/test_ci_shard.py::TestK::test_b": 1.0,
    }
