"""A floor case starts with its cube on the floor, on the live scene.

`OnFloor` is any unheld cube that is neither in the box nor on the cube shelf, so it
also holds for a cube that went over the box's far rim onto the cupboard's bottom board,
or one resting on another cube. A floor skill attempted there is not a floor case, so
the generator drops it instead of storing it."""

import importlib.util
from pathlib import Path

import pytest

from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects as O
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.states import (
    FLOOR_SKILLS,
    FloorSupport,
    Setups,
    StateCapture,
    StateFamilies,
    StateSet,
    StateSpec,
)
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.types import BenchmarkCase

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("sweepoffshelf3d_skill_benchmark_floor_states"),
]

SHIPPED = Path(__file__).resolve().parents[4] / "benchmarks/sweepoffshelf3d/states.jsonl"
HALF = O.cube_half_size
LIFT = 0.0005


@pytest.fixture(name="env", scope="module")
def _env():
    env = SweepOffShelf3DEnvironment(scene_bg=True)
    try:
        yield env
    finally:
        env.close()


@pytest.fixture(name="specs", scope="module")
def _specs(*, env) -> dict[str, StateSpec]:
    return {spec.state_id: spec for spec in StateFamilies.specs(env=env)}


def _floor_cases(*, cube: str) -> list[BenchmarkCase]:
    return [BenchmarkCase(skill=skill, cube=cube) for skill in sorted(FLOOR_SKILLS)]


def test_the_cube_where_the_failure_images_left_it_rests_on_the_cupboard_board(*, env):
    """The pose from shelf-pick-failure-images-20261022's floor-out-far-centre facts."""
    env.reset_to_seed(seed=7)
    Setups.cube_at(x=1.3901, y=0.0001, z=0.0227, cube="cube_2")(env=env)
    Setups.settle(env=env)
    assert "OnFloor(cube_2)" in StateCapture.atoms(env=env)
    assert FloorSupport.supports(env=env, name="cube_2") == {"cupboard_1_shelf1"}
    assert FloorSupport.reason(env=env, name="cube_2") is not None


def test_the_generated_far_rim_floor_state_is_rejected(*, env, specs):
    spec = specs["floor-out-far-centre"]
    state, note = StateFamilies.build(env=env, spec=spec)
    assert state is None
    assert "cupboard_1_shelf1" in note


def test_a_cube_stacked_on_another_is_not_a_floor_case_but_the_one_under_it_is(*, env):
    spec = StateSpec(
        state_id="floor-stacked-test",
        family="near_walls",
        seed=0,
        cases=[*_floor_cases(cube="cube_0"), *_floor_cases(cube="cube_1")],
        provenance="test: cube_0 on top of cube_1 on the open floor",
        setup=Setups.chain(
            Setups.cube_at(x=0.95, y=0.45, z=HALF + LIFT, cube="cube_1"),
            Setups.cube_at(x=0.95, y=0.45, z=3 * HALF + 2 * LIFT, cube="cube_0"),
        ),
    )
    state, note = StateFamilies.build(env=env, spec=spec)
    assert state is not None, note
    assert "OnFloor(cube_0)" in state.atoms
    assert {c.cube for c in state.cases} == {"cube_1"}
    assert len(state.cases) == len(FLOOR_SKILLS)
    assert "cube_0" in note and "cube_1" in note


@pytest.mark.parametrize(
    "state_id", ["floor-out-near-centre", "floor-out-left-mid", "floor-panel-left-front"]
)
def test_a_cube_on_the_open_floor_keeps_both_floor_cases(*, env, specs, state_id):
    state, note = StateFamilies.build(env=env, spec=specs[state_id])
    assert state is not None, note
    assert {c.skill for c in state.cases} == FLOOR_SKILLS


def test_every_floor_case_in_the_shipped_state_set_starts_on_the_floor(*, env):
    invalid = []
    for state in StateSet.load(path=SHIPPED):
        cubes = {c.cube for c in state.cases if c.skill in FLOOR_SKILLS}
        if not cubes:
            continue
        StateCapture.restore(env=env, state=state)
        invalid += [
            f"{state.state_id}/{cube}: {reason}"
            for cube in sorted(c for c in cubes if c is not None)
            if (reason := FloorSupport.reason(env=env, name=cube)) is not None
        ]
    assert not invalid
