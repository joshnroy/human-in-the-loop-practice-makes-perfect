"""The state set file, case ids, applicability and the container-agnostic placements --
everything about benchmark states that needs no simulator."""

from pathlib import Path

import pytest

from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.states import (
    CupboardGeometry,
    FloorSupport,
    Placements,
    StateCapture,
    StateSet,
)
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.types import (
    BenchmarkCase,
    BenchmarkState,
)

START_ATOMS = sorted([
    "AnyCubeOnShelf()",
    "HandEmpty(robot)",
    "WiperHome(wiper_0)",
    *(f"OnShelf(cube_{i})" for i in range(5)),
])


def _state(*, state_id: str, cases: list[BenchmarkCase] | None = None) -> BenchmarkState:
    return BenchmarkState(
        state_id=state_id,
        family="start",
        seed=0,
        scene_fingerprint="scene",
        qpos=[0.1, 0.2],
        qvel=[0.0, 0.0],
        ctrl=[0.0],
        grip=0.0,
        atoms=START_ATOMS,
        cases=cases or [BenchmarkCase(skill="PickWiper")],
    )


def test_appending_adds_new_states_and_never_rewrites_an_existing_one(*, tmp_path: Path) -> None:
    path = tmp_path / "states.jsonl"
    assert StateSet.append(path=path, states=[_state(state_id="a"), _state(state_id="b")]) == [
        "a",
        "b",
    ]
    before = path.read_bytes()
    changed = _state(state_id="a", cases=[BenchmarkCase(skill="PickCubeFromShelf", cube="cube_1")])
    assert StateSet.append(path=path, states=[changed, _state(state_id="c")]) == ["c"]
    assert path.read_bytes().startswith(before)
    loaded = StateSet.load(path=path)
    assert [s.state_id for s in loaded] == ["a", "b", "c"]
    assert loaded[0].cases == [BenchmarkCase(skill="PickWiper")]
    assert loaded[0].qpos == [0.1, 0.2]


def test_case_ids_name_the_state_skill_cube_and_setting() -> None:
    assert BenchmarkCase(skill="PickWiper").case_id(state_id="start-s0") == "start-s0-PickWiper"
    assert (
        BenchmarkCase(skill="PickCubeFromShelf", cube="cube_3").case_id(state_id="layout-7")
        == "layout-7-PickCubeFromShelf-cube_3"
    )
    assert (
        BenchmarkCase(skill="SweepShelf", cube="cube_1", params=[-0.05, 30.0]).case_id(state_id="w")
        == "w-SweepShelf-cube_1-p-0.05_30"
    )


def test_a_case_is_run_only_where_its_operator_preconditions_hold() -> None:
    assert StateCapture.applicable(case=BenchmarkCase(skill="PickWiper"), atoms=START_ATOMS)
    assert StateCapture.applicable(
        case=BenchmarkCase(skill="PickCubeFromShelf", cube="cube_2"), atoms=START_ATOMS
    )
    assert not StateCapture.applicable(
        case=BenchmarkCase(skill="PlaceCubeInBox", cube="cube_2"), atoms=START_ATOMS
    )
    # A sweep is aimed at one cube, and applies only while that cube is on the shelf.
    holding = [a for a in START_ATOMS if a not in {"HandEmpty(robot)", "WiperHome(wiper_0)"}]
    holding += ["HoldingWiper(robot,wiper_0)", "WiperAway(wiper_0)"]
    assert StateCapture.applicable(
        case=BenchmarkCase(skill="SweepShelf", cube="cube_3", params=[0.025, 15.0]), atoms=holding
    )
    swept = [a for a in holding if a != "OnShelf(cube_3)"]
    assert not StateCapture.applicable(
        case=BenchmarkCase(skill="SweepShelf", cube="cube_3", params=[0.025, 15.0]), atoms=swept
    )


@pytest.mark.parametrize(
    "bounds",
    [
        (1.12, -0.24, 1.35, 0.24),  # the box
        (1.30, -0.28, 1.49, 0.28),  # a 56 x 19 cm container elsewhere
    ],
)
def test_wall_placements_follow_whatever_container_the_scene_has(*, bounds) -> None:
    half = 0.01
    inside = Placements.inside(bounds=bounds, half=half)
    assert len(inside) == 9
    x0, y0, x1, y1 = bounds
    for _, x, y in inside:
        assert x0 + half <= x <= x1 - half and y0 + half <= y <= y1 - half
    assert min(x for _, x, _ in inside) == pytest.approx(x0 + half + 0.004)
    around = Placements.around(bounds=bounds, wall=0.01, half=half)
    for _, x, y in around:
        assert not (
            x0 - 0.01 - half < x < x1 + 0.01 + half and y0 - 0.01 - half < y < y1 + 0.01 + half
        )


def test_panel_and_shelf_edge_placements_sit_outside_and_inside_the_panels() -> None:
    cupboard = CupboardGeometry(
        front_x=1.373,
        back_inner_x=1.617,
        panel_inner_y=0.291,
        panel_outer_y=0.301,
        board_top_z=0.279,
    )
    for _, x, y in Placements.beside_panels(cupboard=cupboard, half=0.01):
        assert abs(y) > 0.301 + 0.01
        assert 1.373 <= x <= 1.617
    for _, x, y in Placements.shelf_edges(cupboard=cupboard, half=0.01):
        assert abs(y) + 0.01 < 0.291
        assert x - 0.01 > 1.373 and x + 0.01 < 1.617


@pytest.mark.parametrize(
    ("supports", "valid"),
    [
        ({"floor"}, True),
        ({"cupboard_1_shelf1"}, False),  # over the far rim, on the cupboard's bottom board
        ({"cube_1"}, False),  # stacked on another cube
        ({"floor", "cube_1"}, False),  # tipped against another cube
        ({"floor", "box_0_wall"}, False),  # leaning on the box's rim
        (set(), False),  # resting on nothing: not settled
    ],
)
def test_a_floor_case_needs_its_cube_resting_on_the_floor_alone(*, supports, valid) -> None:
    reason = FloorSupport.judge(name="cube_2", supports=supports)
    assert (reason is None) is valid
    if reason is not None:
        assert "cube_2" in reason


def test_the_committed_sweep_cases_are_every_shelf_cube_on_the_whole_grid() -> None:
    """Per (state, target cube): each wiper-in-hand state sweeps every cube it has on
    the shelf at every setting of `SWEEP_GRID`, and nothing else."""
    from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.verdicts import SWEEP_GRID

    states = StateSet.load(
        path=Path(__file__).resolve().parents[4] / "benchmarks/sweepoffshelf3d/states.jsonl"
    )
    swept = [s for s in states if any(c.skill == "SweepShelf" for c in s.cases)]
    assert len(swept) == 8
    for state in swept:
        on_shelf = [
            a.removeprefix("OnShelf(")[:-1] for a in state.atoms if a.startswith("OnShelf(")
        ]
        cases = sorted((c.cube, tuple(c.params)) for c in state.cases if c.skill == "SweepShelf")
        assert cases == sorted((cube, tuple(p)) for cube in on_shelf for p in SWEEP_GRID)
