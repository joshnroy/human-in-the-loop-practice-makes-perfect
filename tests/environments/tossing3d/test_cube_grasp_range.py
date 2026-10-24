"""`CubeAtSide(cube, barrier, robot_side)` is footprint contact; `PickPlannable` decides.

`CubeAtSide` used to exclude a 6.25 cm grasp band on the robot's side of the barrier,
fitted to an earlier pick that refused there (EXP-21 seed 0: "No collision-free cube
grasp" at 2.4 and 3.0 cm). The near-wall pick (kinder-baselines #187) takes most of those
cubes, and a re-scan at kinder-baselines a69febf held 257/265 plannable picks inside the
old band, so the band is gone: a cube clear of the barrier's footprint is on the robot's
side, and the pick's own dry run (`PickPlannable`) says whether it can be taken.
"""

import importlib.util

import pytest

from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d import predicates
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import BIN_AT_SIDE, CUBE_AT_SIDE, HOLDING
from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.planning.grounding import SkillGrounder

from .observations import INITIAL_ATOMS, PICK_PLANNABLE_ATOM, state

_BARRIER_X = 1.3
_BARRIER_HALF = 0.03
_CUBE_HALF = 0.025
_RESET_NAMES = {"ask_for_reset_cube_bin_only", "non_human_reset_cube_bin_only"}
# The re-scan's residual stalls sit at 4.50-4.96 cm; 5.5 cm is the brief's example of a
# gap the old band excluded.
_NEAR_GAPS = [0.0005, 0.024, 0.03, 0.045, 0.055, 0.06]


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


def _near_barrier_atoms(*, plannable: bool):
    env, scene = _scene(cube_x=_cube_x_at_gap(gap=0.03))
    scene.abstract_atoms = INITIAL_ATOMS if plannable else INITIAL_ATOMS - {PICK_PLANNABLE_ATOM}
    provider = Tossing3DSkillProvider(env=env, offer_non_human_reset=True)
    atoms = SkillGrounder.abstract_state(
        state=scene, objects=provider.objects(), predicates=provider.predicates()
    )
    return env, provider, atoms


def test_the_grasp_band_constant_is_gone() -> None:
    assert not hasattr(predicates, "BARRIER_GRASP_CLEARANCE_M")


@pytest.mark.parametrize("gap", [*_NEAR_GAPS, 0.0625, 0.08, 0.5])
def test_a_cube_clear_of_the_footprint_is_on_the_robots_side(*, gap: float) -> None:
    assert _sides(cube_x=_cube_x_at_gap(gap=gap)) == {"robot_side"}


@pytest.mark.parametrize("gap", [0.0005, 0.03, 0.06, 0.5])
def test_the_opposite_side_is_unchanged(*, gap: float) -> None:
    assert _sides(cube_x=_cube_x_at_gap(gap=gap, robot_side=False)) == {"opposite_side"}


def test_straddling_is_still_neither_side() -> None:
    for cube_x in (1.3, 1.3 - 0.054, 1.3 + 0.054, 1.3 - 0.055, 1.3 + 0.055):
        assert _sides(cube_x=cube_x) == set()


def test_bin_at_side_is_unchanged() -> None:
    env, scene = _scene(cube_x=0.6, bin_x=_BARRIER_X - _BARRIER_HALF - 0.01)
    assert BIN_AT_SIDE.holds(scene, (env.bin, env.barrier, Tossing3DSides.robot))


def test_near_the_barrier_the_dry_run_decides_the_pick() -> None:
    _, provider, atoms = _near_barrier_atoms(plannable=True)
    assert {
        ground.skill.name
        for ground in SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
        )
    } == {"PickCube"}

    _, provider, refused = _near_barrier_atoms(plannable=False)
    assert (
        SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=provider.objects(), true_atoms=refused
        )
        == []
    )
    for reset in provider.movables_reset_skills():
        assert reset.preconditions <= refused


@pytest.mark.parametrize("plannable", [True, False])
def test_both_planners_follow_the_dry_run_near_the_barrier(*, plannable: bool) -> None:
    from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
    from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
        make_tossing3d_search_state,
    )
    from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

    env, provider, atoms = _near_barrier_atoms(plannable=plannable)
    pomdp = Tossing3DPomdpMethod(env=env, skill_provider=provider, seed=0, pomdp_num_particles=32)
    valid = {
        ground.skill.name
        for ground in pomdp._pomdp_model.get_valid_actions(  # noqa: SLF001
            environment_state=make_tossing3d_search_state(state=pomdp.pomdp_state, true_atoms=atoms)
        )
    }
    assert valid
    if plannable:
        assert "PickCube" in valid
    else:
        assert valid <= _RESET_NAMES

    ees = EesMethod(
        env=env, skill_provider=Tossing3DSkillProvider(env=env), seed=0, reset_cost_gate=False
    )
    goal = frozenset({GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube))})
    plan = ees.plan_to(init_atoms=atoms, goal=goal, costs={}, practicing=True)
    expected = ["PickCube"] if plannable else ["ask_for_reset_cube_bin_only", "PickCube"]
    assert [ground.skill.name for ground in plan] == expected


@pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KINDER")
def test_the_live_pick_holds_inside_the_old_band() -> None:
    """At the live controller, from the seed-125 scene with the cube moved: at 3.0 cm
    (EXP-21's refusal) and at 6.25 cm (the old band's edge) the cube is on the robot's
    side, and the dispatched pick takes and holds it."""
    import numpy as np

    env = Tossing3DEnvironment()
    try:
        env.reset_to_seed(seed=125)
        backend = env.backend()
        base = backend.snapshot_to_plain(snapshot=backend.snapshot())
        snapshot = backend.snapshot()
        features = list(snapshot.type_features[snapshot.get_object_from_name("cube_0").type])
        outcomes = {}
        for gap in (0.03, 0.0625):
            plain = {name: list(values) for name, values in base.items()}
            plain["cube_0"][features.index("x")] = _cube_x_at_gap(gap=gap)
            plain["cube_0"][features.index("y")] = -0.615
            restored = env.restore_plain_snapshot(plain=plain)
            robot_side = CUBE_AT_SIDE.holds(restored, (env.cube, env.barrier, Tossing3DSides.robot))
            after = env.take_action(action=np.array([0, 0, 0, 0, 0], dtype=float))
            outcomes[gap] = (robot_side, HOLDING.holds(after, (env.robot, env.cube)))
        assert outcomes == {0.03: (True, True), 0.0625: (True, True)}
    finally:
        env.close()
