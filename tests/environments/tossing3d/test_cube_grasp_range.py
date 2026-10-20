"""`CubeAtSide(cube, barrier, robot_side)` is the footprint test, with no grasp band.

In EXP-21 seed 0 two evaluation picks were refused ("No collision-free cube grasp") with
the cube's face 2.4 and 3.0 cm from the barrier's face, so `CubeAtSide` used to exclude
a 6.25 cm band on the robot's side (`BARRIER_GRASP_CLEARANCE_M`), fitted to where that
pick controller refused. The partly opened, tilted pick (kinder-baselines #178) picks
cubes inside that band, so the band would now hide picks the controller makes. Whether
the pick applies beside the barrier is `PickPlannable`'s -- the controller's own dry
run -- and nothing else's.
"""

import importlib.util

import pytest

from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d import predicates
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    BIN_AT_SIDE,
    CUBE_AT_SIDE,
    HOLDING,
    PICK_PLANNABLE,
)
from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.planning.grounding import SkillGrounder

from .observations import INITIAL_ATOMS, PICK_PLANNABLE_ATOM, state

_BARRIER_X = 1.3
_BARRIER_HALF = 0.03
_CUBE_HALF = 0.025
_RESET_NAMES = {"ask_for_reset_cube_bin_only", "non_human_reset_cube_bin_only"}
# Inside the old band: EXP-21's 3.0 cm refusal.
_BESIDE_THE_BARRIER = 0.03


def _cube_x_at_gap(*, gap: float, robot_side: bool = True) -> float:
    """The cube centre whose face is `gap` from the barrier's face."""
    offset = _BARRIER_HALF + _CUBE_HALF + gap
    return _BARRIER_X - offset if robot_side else _BARRIER_X + offset


def _scene(*, cube_x: float, cube_y: float = 0.0, bin_x: float = 2.0):
    env = Tossing3DEnvironment()
    return env, state(env=env, base_x=0.0, cube_x=cube_x, cube_y=cube_y, bin_x=bin_x)


def _sides(*, cube_x: float) -> set[str]:
    env, scene = _scene(cube_x=cube_x)
    return {
        side.name
        for side in Tossing3DSides.objects()
        if CUBE_AT_SIDE.holds(scene, (env.cube, env.barrier, side))
    }


def _beside_the_barrier_atoms(*, plannable: bool):
    """A cube 3.0 cm from the barrier's face, with or without `PickPlannable`."""
    env, scene = _scene(cube_x=_cube_x_at_gap(gap=_BESIDE_THE_BARRIER))
    scene.abstract_atoms = INITIAL_ATOMS if plannable else INITIAL_ATOMS - {PICK_PLANNABLE_ATOM}
    provider = Tossing3DSkillProvider(env=env, offer_non_human_reset=True)
    atoms = SkillGrounder.abstract_state(
        state=scene, objects=provider.objects(), predicates=provider.predicates()
    )
    return env, provider, atoms


def test_the_grasp_band_constant_is_gone() -> None:
    """Nothing may fit a pick refusal into the side test again by reviving the band."""
    assert not hasattr(predicates, "BARRIER_GRASP_CLEARANCE_M")


@pytest.mark.parametrize(
    ("label", "gap"),
    [
        ("EXP-21 seed 0 eval 11, cube x = 1.221", 1.3 - 0.03 - 1.221 - 0.025),
        ("EXP-21 seed 0 eval 40, cube x = 1.215", 1.3 - 0.03 - 1.215 - 0.025),
        ("touching the barrier", 0.0005),
        ("the last refused/failed gap measured on the old pick", 0.06),
    ],
)
def test_a_cube_inside_the_old_grasp_band_is_on_the_robots_side(*, label: str, gap: float) -> None:
    del label
    assert _sides(cube_x=_cube_x_at_gap(gap=gap)) == {"robot_side"}


@pytest.mark.parametrize("gap", [0.0625, 0.08, 0.5])
def test_a_cube_beyond_the_old_band_is_on_the_robots_side(*, gap: float) -> None:
    assert _sides(cube_x=_cube_x_at_gap(gap=gap)) == {"robot_side"}


@pytest.mark.parametrize("gap", [0.0005, 0.03, 0.06, 0.5])
def test_the_two_sides_are_symmetric(*, gap: float) -> None:
    """Both sides use the footprint test alone, so a gap means the same on either."""
    assert _sides(cube_x=_cube_x_at_gap(gap=gap, robot_side=False)) == {"opposite_side"}
    assert _sides(cube_x=_cube_x_at_gap(gap=gap)) == {"robot_side"}


def test_straddling_is_still_neither_side() -> None:
    for cube_x in (1.3, 1.3 - 0.054, 1.3 + 0.054, 1.3 - 0.055, 1.3 + 0.055):
        assert _sides(cube_x=cube_x) == set()


def test_bin_at_side_is_unchanged() -> None:
    """The bin's side keeps the footprint test."""
    env, scene = _scene(cube_x=0.6, bin_x=_BARRIER_X - _BARRIER_HALF - 0.01)
    assert BIN_AT_SIDE.holds(scene, (env.bin, env.barrier, Tossing3DSides.robot))


def test_beside_the_barrier_pick_plannable_alone_decides_the_pick() -> None:
    """Planned: the pick applies. Refused by the dry run: only the resets do."""
    _, provider, atoms = _beside_the_barrier_atoms(plannable=True)
    assert {
        ground.skill.name
        for ground in SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
        )
    } == {"PickCube"}

    _, provider, refused = _beside_the_barrier_atoms(plannable=False)
    assert (
        SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=provider.objects(), true_atoms=refused
        )
        == []
    )
    for reset in provider.movables_reset_skills():
        assert reset.preconditions <= refused


@pytest.mark.parametrize("plannable", [True, False])
def test_both_planners_defer_to_pick_plannable_beside_the_barrier(*, plannable: bool) -> None:
    from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
    from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
        make_tossing3d_search_state,
    )
    from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

    env, provider, atoms = _beside_the_barrier_atoms(plannable=plannable)
    pomdp = Tossing3DPomdpMethod(env=env, skill_provider=provider, seed=0, pomdp_num_particles=32)
    valid = pomdp._pomdp_model.get_valid_actions(  # noqa: SLF001
        environment_state=make_tossing3d_search_state(state=pomdp.pomdp_state, true_atoms=atoms)
    )
    names = {ground.skill.name for ground in valid}
    if plannable:
        assert "PickCube" in names
    else:
        assert valid
        assert names <= _RESET_NAMES

    ees = EesMethod(
        env=env, skill_provider=Tossing3DSkillProvider(env=env), seed=0, reset_cost_gate=False
    )
    goal = frozenset({GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube))})
    plan = ees.plan_to(init_atoms=atoms, goal=goal, costs={}, practicing=True)
    expected = ["PickCube"] if plannable else ["ask_for_reset_cube_bin_only", "PickCube"]
    assert [ground.skill.name for ground in plan] == expected


@pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KINDER")
def test_beside_the_barrier_the_dry_run_predicts_the_live_pick() -> None:
    """At the live controller, from the seed-125 scene with the cube moved toward the
    barrier: the cube is on the robot's side at every gap, `PickPlannable` holds exactly
    where the dispatched pick is not refused, at 5 mm the pick is still refused, and at
    3.0 cm (EXP-21's refusal on the old pick) it plans and holds the cube. Measured on
    kinder-baselines 734c1a8: refused at gaps <= 0.01 m, planned from 0.015 m."""
    import numpy as np

    env = Tossing3DEnvironment()
    try:
        env.reset_to_seed(seed=125)
        backend = env.backend()
        base = backend.snapshot_to_plain(snapshot=backend.snapshot())
        snapshot = backend.snapshot()
        features = list(snapshot.type_features[snapshot.get_object_from_name("cube_0").type])
        outcomes = {}
        for gap in (0.005, _BESIDE_THE_BARRIER, 0.0625):
            plain = {name: list(values) for name, values in base.items()}
            plain["cube_0"][features.index("x")] = _cube_x_at_gap(gap=gap)
            plain["cube_0"][features.index("y")] = -0.615
            restored = env.restore_plain_snapshot(plain=plain)
            robot_side = CUBE_AT_SIDE.holds(restored, (env.cube, env.barrier, Tossing3DSides.robot))
            plannable = PICK_PLANNABLE.holds(restored, (env.robot, env.cube))
            after = env.take_action(action=np.array([0, 0, 0, 0, 0], dtype=float))
            error = env.last_skill_error()
            refused = bool(error and "No collision-free" in error)
            assert plannable == (not refused), (gap, error)
            outcomes[gap] = (robot_side, plannable, HOLDING.holds(after, (env.robot, env.cube)))
        assert outcomes[0.005] == (True, False, False)
        assert outcomes[_BESIDE_THE_BARRIER] == (True, True, True)
        assert outcomes[0.0625] == (True, True, True)
    finally:
        env.close()
