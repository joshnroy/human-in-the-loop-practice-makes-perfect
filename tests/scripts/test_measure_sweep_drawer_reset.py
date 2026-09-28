"""The aggregation over seeds: what counts as retrieved, and what as rescued."""

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("shapely") is None, reason="shapely ships with KINDER"
)

CUBES = tuple(f"cube_{i}" for i in range(5))


def _measurement():
    from scripts.measure_sweep_drawer_reset import ResetMeasurement

    return ResetMeasurement


def _retrieval(*, origin: str, blocked: bool, assists: tuple[str, ...]) -> dict:
    from hitl_pmp.environments.sweep_drawer3d.types import Retrieval

    r = Retrieval.model_validate({
        "origin": origin,
        "blocked": blocked,
        "assists": assists,
        "grasp": "single",
    })
    return r.model_dump() | {"pathway": r.pathway, "rescued_by": list(r.rescued_by)}


def _cycle(
    *,
    seed: int,
    after: dict[str, str],
    piled_after: set[str],
    piled_end: set[str],
    retrievals: dict[str, dict],
    steps: list[dict] | None = None,
    second: bool = False,
) -> dict:
    record = {
        "seed": seed,
        "after_attempt_1": after,
        "in_pile_after_attempt_1": {c: c in piled_after for c in CUBES},
        "reset": {
            "locations": {c: "counter" if c in piled_end else after[c] for c in CUBES},
            "in_pile": {c: c in piled_end for c in CUBES},
            "drawer_pos": 0.0,
            "wiper_xy_error": 0.002,
            "wiper_on_counter": True,
            "robot_actions": 10,
            "ticks": 3000,
            "wall_s": 80.0,
            "success": piled_end == set(CUBES),
            "retrievals": retrievals,
        },
        "steps": steps or [],
    }
    if second:
        record["after_attempt_2"] = dict.fromkeys(CUBES, "counter")
    return record


def _write(*, root: Path, cycles: list[dict]) -> None:
    for c in cycles:
        folder = root / str(c["seed"])
        folder.mkdir(parents=True)
        (folder / "cycle.json").write_text(json.dumps(c))


def _two_seeds(*, root: Path) -> dict:
    """Seed 2: everything in the drawer, all five retrieved, one of them graspable where
    it lay. Seed 10: two cubes never left the pile, one on the counter outside it, two on
    the floor of which one is left there."""
    drawer = dict.fromkeys(CUBES, "drawer")
    mixed = {
        "cube_0": "counter",
        "cube_1": "counter",
        "cube_2": "counter",
        "cube_3": "floor",
        "cube_4": "floor",
    }
    _write(
        root=root,
        cycles=[
            _cycle(
                seed=10,
                after=mixed,
                piled_after={"cube_0", "cube_1"},
                piled_end={"cube_0", "cube_1", "cube_2", "cube_3"},
                retrievals={
                    "cube_2": _retrieval(origin="counter", blocked=False, assists=()),
                    "cube_3": _retrieval(origin="floor", blocked=True, assists=("nudge",)),
                },
            ),
            _cycle(
                seed=2,
                after=drawer,
                piled_after=set(),
                piled_end=set(CUBES),
                retrievals={
                    "cube_0": _retrieval(origin="drawer", blocked=False, assists=("wiggle",)),
                    **{
                        c: _retrieval(origin="drawer", blocked=True, assists=("wiggle", "nudge"))
                        for c in CUBES[1:3]
                    },
                    **{
                        c: _retrieval(origin="drawer", blocked=True, assists=("wiggle",))
                        for c in CUBES[3:]
                    },
                },
                second=True,
            ),
        ],
    )
    return _measurement().summarize(root=root)


def test_seeds_are_reported_in_numeric_order_not_as_text(*, tmp_path: Path) -> None:
    assert _two_seeds(root=tmp_path)["seeds"] == [2, 10]


def test_a_cube_the_sweep_never_moved_is_not_counted_as_retrieved_from_anywhere(
    *, tmp_path: Path
) -> None:
    got = _two_seeds(root=tmp_path)["cubes_back_in_pile_by_origin"]
    assert got["pile (undisturbed)"] == "2/2"
    assert got["counter"] == "1/1"
    assert got["floor"] == "1/2"
    assert got["drawer"] == "5/5"


def test_the_headline_retrieval_leaves_the_undisturbed_cubes_out(*, tmp_path: Path) -> None:
    assert _two_seeds(root=tmp_path)["cubes_moved_by_the_sweep_back_in_pile"] == "7/8"


def test_a_strategy_rescues_only_cubes_that_had_no_grasp(*, tmp_path: Path) -> None:
    """The wiggle shifted all five of seed 2's cubes; four of them needed it."""
    got = _two_seeds(root=tmp_path)["rescued_by_strategy"]
    assert got == {"wiggle": 4, "nudge": 3}


def test_pathways_are_counted_per_origin(*, tmp_path: Path) -> None:
    got = _two_seeds(root=tmp_path)["pathways"]
    assert got["drawer: wiggle > nudge > pick"] == 2
    assert got["drawer: wiggle > pick"] == 2
    assert got["drawer: pick"] == 1
    assert got["floor: nudge > pick"] == 1
    assert got["counter: pick"] == 1


def test_the_cube_left_behind_is_named_with_where_it_lies(*, tmp_path: Path) -> None:
    summary = _two_seeds(root=tmp_path)
    assert summary["cubes_not_retrieved"] == {"left at floor, from floor": 1}
    assert summary["full_reset"] == "1/2"
    assert summary["seeds_not_fully_reset"] == [10]


def test_a_second_attempt_is_counted_only_among_full_resets(*, tmp_path: Path) -> None:
    assert _two_seeds(root=tmp_path)["second_attempt_ran"] == "1/1"
