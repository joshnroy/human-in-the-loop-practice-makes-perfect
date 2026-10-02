"""`CubeAtSide(cube, barrier, robot_side)` holds only within grasping range of the barrier.

In EXP-21 seed 0 two evaluation picks were refused ("No collision-free cube grasp") with
the cube's face 2.4 and 3.0 cm from the barrier's face: only the approach from -x has a
base pose, and on the descent the Robotiq palm overlaps the barrier. The footprint-only
side test still called that cube "on the robot's side", so `PickCube` stayed applicable.
The band is measured at the live pick controller; see `predicates.BARRIER_GRASP_CLEARANCE_M`.
"""

import importlib.util

import pytest

from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    BARRIER_GRASP_CLEARANCE_M,
    BIN_AT_SIDE,
    CUBE_AT_SIDE,
    HOLDING,
)
from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.planning.grounding import SkillGrounder

from .observations import INITIAL_ATOMS, state

_BARRIER_X = 1.3
_BARRIER_HALF = 0.03
_CUBE_HALF = 0.025
_RESET_NAMES = {"ask_for_reset_cube_bin_only", "non_human_reset_cube_bin_only"}


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


def test_the_measured_constant() -> None:
    assert pytest.approx(0.0625) == BARRIER_GRASP_CLEARANCE_M


@pytest.mark.parametrize(
    ("label", "gap"),
    [
        ("EXP-21 seed 0 eval 11, cube x = 1.221", 1.3 - 0.03 - 1.221 - 0.025),
        ("EXP-21 seed 0 eval 40, cube x = 1.215", 1.3 - 0.03 - 1.215 - 0.025),
        ("touching the barrier", 0.0005),
        ("the last refused/failed gap measured", 0.06),
    ],
)
def test_a_cube_inside_the_grasp_band_is_on_neither_side(*, label: str, gap: float) -> None:
    del label
    assert _sides(cube_x=_cube_x_at_gap(gap=gap)) == set()


@pytest.mark.parametrize("gap", [0.0625, 0.08, 0.5])
def test_a_cube_beyond_the_band_is_on_the_robots_side(*, gap: float) -> None:
    assert _sides(cube_x=_cube_x_at_gap(gap=gap)) == {"robot_side"}


@pytest.mark.parametrize("gap", [0.0005, 0.03, 0.06, 0.5])
def test_the_opposite_side_is_unchanged(*, gap: float) -> None:
    """Only the robot's side gets the band: past the barrier nothing is graspable anyway,
    and the opposite side keeps the footprint-only test."""
    assert _sides(cube_x=_cube_x_at_gap(gap=gap, robot_side=False)) == {"opposite_side"}


def test_straddling_is_still_neither_side() -> None:
    for cube_x in (1.3, 1.3 - 0.054, 1.3 + 0.054, 1.3 - 0.055, 1.3 + 0.055):
        assert _sides(cube_x=cube_x) == set()


def test_bin_at_side_is_unchanged() -> None:
    """The band is about grasping the cube; the bin's side keeps the footprint test."""
    env, scene = _scene(cube_x=0.6, bin_x=_BARRIER_X - _BARRIER_HALF - 0.01)
    assert BIN_AT_SIDE.holds(scene, (env.bin, env.barrier, Tossing3DSides.robot))


def test_a_cube_in_the_band_leaves_only_the_resets() -> None:
    env, scene = _scene(cube_x=_cube_x_at_gap(gap=0.03))
    scene.abstract_atoms = INITIAL_ATOMS
    provider = Tossing3DSkillProvider(env=env, offer_non_human_reset=True)
    atoms = SkillGrounder.abstract_state(
        state=scene, objects=provider.objects(), predicates=provider.predicates()
    )
    assert (
        SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
        )
        == []
    )
    for reset in provider.movables_reset_skills():
        assert reset.preconditions <= atoms
    # Just past the band the pick applies again.
    _, clear = _scene(cube_x=_cube_x_at_gap(gap=BARRIER_GRASP_CLEARANCE_M))
    clear.abstract_atoms = INITIAL_ATOMS
    clear_atoms = SkillGrounder.abstract_state(
        state=clear, objects=provider.objects(), predicates=provider.predicates()
    )
    assert {
        ground.skill.name
        for ground in SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=provider.objects(), true_atoms=clear_atoms
        )
    } == {"PickCube"}


def test_both_planners_see_the_band() -> None:
    from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
    from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
        make_tossing3d_search_state,
    )
    from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

    env, scene = _scene(cube_x=_cube_x_at_gap(gap=0.03))
    scene.abstract_atoms = INITIAL_ATOMS
    provider = Tossing3DSkillProvider(env=env, offer_non_human_reset=True)
    atoms = SkillGrounder.abstract_state(
        state=scene, objects=provider.objects(), predicates=provider.predicates()
    )
    pomdp = Tossing3DPomdpMethod(env=env, skill_provider=provider, seed=0, pomdp_num_particles=32)
    valid = pomdp._pomdp_model.get_valid_actions(  # noqa: SLF001
        environment_state=make_tossing3d_search_state(state=pomdp.pomdp_state, true_atoms=atoms)
    )
    assert valid
    assert {ground.skill.name for ground in valid} <= _RESET_NAMES

    ees = EesMethod(
        env=env, skill_provider=Tossing3DSkillProvider(env=env), seed=0, reset_cost_gate=False
    )
    goal = frozenset({GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube))})
    plan = ees.plan_to(init_atoms=atoms, goal=goal, costs={}, practicing=True)
    assert [ground.skill.name for ground in plan] == ["ask_for_reset_cube_bin_only", "PickCube"]


@pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KINDER")
def test_the_live_pick_brackets_the_band() -> None:
    """At the live controller, from the seed-125 scene with the cube moved: a 3.0 cm gap
    (EXP-21's refusal) is refused, and a gap of exactly the constant is picked and held."""
    import numpy as np

    env = Tossing3DEnvironment()
    try:
        env.reset_to_seed(seed=125)
        backend = env.backend()
        base = backend.snapshot_to_plain(snapshot=backend.snapshot())
        snapshot = backend.snapshot()
        features = list(snapshot.type_features[snapshot.get_object_from_name("cube_0").type])
        outcomes = {}
        for gap in (0.03, BARRIER_GRASP_CLEARANCE_M):
            plain = {name: list(values) for name, values in base.items()}
            plain["cube_0"][features.index("x")] = _cube_x_at_gap(gap=gap)
            plain["cube_0"][features.index("y")] = -0.615
            restored = env.restore_plain_snapshot(plain=plain)
            robot_side = CUBE_AT_SIDE.holds(restored, (env.cube, env.barrier, Tossing3DSides.robot))
            after = env.take_action(action=np.array([0, 0, 0, 0, 0], dtype=float))
            outcomes[gap] = (robot_side, HOLDING.holds(after, (env.robot, env.cube)))
        assert outcomes[0.03] == (False, False)
        assert outcomes[BARRIER_GRASP_CLEARANCE_M] == (True, True)
    finally:
        env.close()
