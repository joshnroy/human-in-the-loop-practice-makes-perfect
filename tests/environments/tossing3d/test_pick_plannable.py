"""`PickPlannable(robot, cube)`: the pick controller's own planner finds a plan.

Since PickCube stopped requiring `PickupUnblocked`, a state that satisfies every pick
precondition while the controller refuses at dispatch ("No collision-free cube grasp",
zero controller steps) is retried for the rest of a run. Four such geometries were found
one experiment at a time. The frozen states below are those, restored from the state
logs of the runs that stuck in them (`tossing3d_state_log.jsonl`, the last tick before
the first refused pick), and in every one the dispatched pick really is refused -- which
is the fact the predicate has to predict.
"""

import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
from hitl_pmp.environments.tossing3d.pick_plannable import KB_PICK_PLANNABLE, PickPlannableGate
from hitl_pmp.environments.tossing3d.predicates import (
    BIN_AT_SIDE,
    BIN_ON_GROUND,
    CUBE_AT_SIDE,
    GRASP_CLEAR,
    HAND_EMPTY,
    HOLDING,
    IN_BIN,
    NOT_HOLDING,
    ON_GROUND,
    PICK_PLANNABLE,
    PICKUP_UNBLOCKED,
    ROBOT_AT_SIDE,
)
from hitl_pmp.environments.tossing3d.recovery_skills import SameSideSkills
from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.environments.tossing3d.state_log import StateLogHeader, StateLogWriter
from hitl_pmp.planning.grounding import SkillGrounder

from .observations import INITIAL_ATOMS, state

_FIXTURES = Path(__file__).parent / "fixtures"
_RESET_NAMES = {"ask_for_reset_cube_bin_only", "non_human_reset_cube_bin_only"}
_PICK_NAMES = {"PickCube", "PickCubeFromFloor", "PickCubeFromBin", "PickCubeFromRim"}
_PLANNABLE_ATOM = (KB_PICK_PLANNABLE, ("robot", "cube_0"))
_NEEDS_KINDER = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="needs KINDER"
)

# Every known state the planners stuck in, by the experiment that found it. In each the
# logged run dispatched a pick that the controller refused.
_STUCK_STATES = {
    # EXP-22b: a toss overshot the robot-side bin and the cube rests between the bin
    # and the room's south wall. Every base standing spot 0.55 m from the cube collides
    # with the bin or the wall, or lies outside the base planner's sampling box.
    "exp22b gpoff local_trend 3e-6 seed 0": "pick_plannable/exp22b_s0.json",
    "exp22b gpoff local_trend 3e-6 seed 1": "pick_plannable/exp22b_s1.json",
    "exp22b gpoff local_trend 3e-6 seed 2": "pick_plannable/exp22b_s2.json",
    "exp22b gpon local_trend 3e-6 seed 0": "pick_plannable/exp22b_gp0.json",
    "exp22b ees gpon cost 5 seed 0": "pick_plannable/exp22b_ees0.json",
    # EXP-22: the cube rests inside the upright bin 0.035-0.043 m from an inner wall.
    "exp22 gpoff local_trend 3e-6": "pick_plannable/exp22_A.json",
    "exp22 gpon global_curve 3e-6": "pick_plannable/exp22_C.json",
    "exp22 gpon local_trend 3e-6": "pick_plannable/exp22_D.json",
    "exp22 gpon global_curve 0": "pick_plannable/exp22_F.json",
    # EXP-21: the bin tipped onto its side, or came to rest on top of the cube.
    "exp21 seed 0, bin on its side": "exp21_tipped_bin_state.json",
    "exp21 seed 1, bin resting on the cube": "pick_plannable/exp21_bin_on_cube_s1.json",
    "exp21 seed 2, bin level on the cube": "pick_plannable/exp21_bin_on_cube_s2.json",
    # EXP-21: evaluation cubes 0.024 and 0.030 m from the barrier's face.
    "exp21 seed 0 evaluation, checkpoint 11": "pick_plannable/exp21_barrier_eval11.json",
    "exp21 seed 0 evaluation, checkpoint 40": "pick_plannable/exp21_barrier_eval40.json",
}


def _plain(*, fixture: str) -> dict[str, list[float]]:
    return json.loads((_FIXTURES / fixture).read_text())


# ------------------------------------------------------------------------ the gate


def test_a_cube_beyond_the_barriers_centre_plane_is_across_it() -> None:
    barrier = {"x": 1.3}
    assert PickPlannableGate.cube_across_barrier(robot_x=0.2, cube={"x": 2.0}, barrier=barrier)
    assert not PickPlannableGate.cube_across_barrier(robot_x=0.2, cube={"x": 0.7}, barrier=barrier)
    # Robot-relative, like the side atoms: a robot standing past the barrier sees it
    # the other way round.
    assert PickPlannableGate.cube_across_barrier(robot_x=2.0, cube={"x": 0.7}, barrier=barrier)
    assert not PickPlannableGate.cube_across_barrier(robot_x=2.0, cube={"x": 2.0}, barrier=barrier)


def test_a_cube_touching_the_barrier_from_the_robots_side_is_left_to_the_planner() -> None:
    """The gate carries none of `CubeAtSide`'s margins: inside the grasp band the verdict
    is the dry run's, not a constant's."""
    barrier = {"x": 1.3}
    assert not PickPlannableGate.cube_across_barrier(
        robot_x=0.2, cube={"x": 1.2999}, barrier=barrier
    )
    assert PickPlannableGate.cube_across_barrier(robot_x=0.2, cube={"x": 1.3}, barrier=barrier)


# -------------------------------------------------------------------- the predicate


def test_the_predicate_is_a_lookup_of_the_boundary_atom() -> None:
    env = Tossing3DEnvironment()
    plannable = state(abstract_atoms=INITIAL_ATOMS)
    refused = state(abstract_atoms=INITIAL_ATOMS - {_PLANNABLE_ATOM})
    assert PICK_PLANNABLE.holds(plannable, (env.robot, env.cube))
    assert not PICK_PLANNABLE.holds(refused, (env.robot, env.cube))
    with pytest.raises(ValueError, match="no abstraction"):
        PICK_PLANNABLE.holds(state(), (env.robot, env.cube))


# ------------------------------------------------------------- the operator models


def _provider(*, layout: Tossing3DLayout) -> Tossing3DSkillProvider:
    env = Tossing3DEnvironment(layout=layout, scene_bg=False)
    return Tossing3DSkillProvider(env=env, offer_non_human_reset=True)


def _plannable(*, skills: type[Tossing3DSkills] = Tossing3DSkills) -> LiftedAtom:
    return LiftedAtom(predicate=PICK_PLANNABLE, variables=(skills._robot, skills._cube))


@pytest.mark.parametrize("layout", list(Tossing3DLayout))
def test_every_pick_requires_pick_plannable_and_no_other_skill_does(
    *, layout: Tossing3DLayout
) -> None:
    provider = _provider(layout=layout)
    assert PICK_PLANNABLE in provider.predicates()
    picks = [skill for skill in provider.skills() if skill.name in _PICK_NAMES]
    assert picks
    for skill in provider.skills():
        requires = any(atom.predicate == PICK_PLANNABLE for atom in skill.preconditions)
        assert requires == (skill in picks), skill.name
    for pick in picks:
        robot, cube = pick.parameters[0], pick.parameters[1]
        assert LiftedAtom(predicate=PICK_PLANNABLE, variables=(robot, cube)) in pick.preconditions


def test_pick_plannable_replaces_grasp_clear_in_the_pick() -> None:
    """GraspClear is a fitted clearance and blocks cubes the controller can pick; the
    barrier band and BinOnGround stay."""
    pick = Tossing3DSkills.PICK_CUBE
    assert not any(atom.predicate == GRASP_CLEAR for atom in pick.preconditions)
    assert not any(atom.predicate == PICKUP_UNBLOCKED for atom in pick.preconditions)
    assert {atom.predicate for atom in pick.preconditions} == {
        HAND_EMPTY,
        ON_GROUND,
        ROBOT_AT_SIDE,
        CUBE_AT_SIDE,
        BIN_ON_GROUND,
        PICK_PLANNABLE,
    }
    # The bin stays a parameter: BinOnGround names it.
    assert pick.parameters[-1] == Tossing3DSkills._bin


def test_no_skill_promises_or_retracts_pick_plannable_but_the_reset() -> None:
    for skill in (*_barrier_skills(), *SameSideSkills.skills()):
        assert not any(atom.predicate == PICK_PLANNABLE for atom in skill.add_effects), skill.name
        assert not any(atom.predicate == PICK_PLANNABLE for atom in skill.delete_effects), (
            skill.name
        )


def _barrier_skills() -> tuple:
    return (
        Tossing3DSkills.PICK_CUBE,
        Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        Tossing3DSkills.OPEN_GRIPPER,
    )


def test_the_toss_leaves_pick_plannable_to_observation_like_grasp_clear() -> None:
    """The same mechanism, not a new one: listed in `ignore_effects`, re-asserted by
    nothing."""
    toss = Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS
    assert PICK_PLANNABLE in toss.ignore_effects
    assert GRASP_CLEAR in toss.ignore_effects
    assert not any(atom.predicate == PICK_PLANNABLE for atom in toss.add_effects)


@pytest.mark.parametrize("layout", list(Tossing3DLayout))
def test_every_reset_makes_the_pick_plannable_without_requiring_it(
    *, layout: Tossing3DLayout
) -> None:
    provider = _provider(layout=layout)
    resets = provider.movables_reset_skills()
    assert {reset.skill.name for reset in resets} == _RESET_NAMES
    env = provider.env
    for reset in resets:
        assert GroundAtom(predicate=PICK_PLANNABLE, objects=(env.robot, env.cube)) in (
            reset.add_effects
        )
        assert not any(atom.predicate == PICK_PLANNABLE for atom in reset.preconditions)


def _pickable_atoms(*, provider: Tossing3DSkillProvider) -> frozenset[GroundAtom]:
    """Every PickCube precondition but PickPlannable: a hand-empty cube on the floor on
    the robot's side, bin upright on the same side."""
    env = provider.env
    robot, cube, bin_, barrier = env.robot, env.cube, env.bin, env.barrier
    return frozenset({
        GroundAtom(predicate=HAND_EMPTY, objects=(robot,)),
        GroundAtom(predicate=ON_GROUND, objects=(cube,)),
        GroundAtom(predicate=NOT_HOLDING, objects=(robot, cube)),
        GroundAtom(predicate=ROBOT_AT_SIDE, objects=(robot, barrier, Tossing3DSides.robot)),
        GroundAtom(predicate=CUBE_AT_SIDE, objects=(cube, barrier, Tossing3DSides.robot)),
        GroundAtom(predicate=BIN_AT_SIDE, objects=(bin_, barrier, Tossing3DSides.robot)),
        GroundAtom(predicate=PICKUP_UNBLOCKED, objects=(cube,)),
        GroundAtom(predicate=BIN_ON_GROUND, objects=(bin_,)),
    })


def _pick_plannable(*, provider: Tossing3DSkillProvider) -> GroundAtom:
    return GroundAtom(predicate=PICK_PLANNABLE, objects=(provider.env.robot, provider.env.cube))


def _holding_atoms(*, provider: Tossing3DSkillProvider) -> frozenset[GroundAtom]:
    env = provider.env
    atoms = _pickable_atoms(provider=provider) - {
        GroundAtom(predicate=HAND_EMPTY, objects=(env.robot,)),
        GroundAtom(predicate=ON_GROUND, objects=(env.cube,)),
        GroundAtom(predicate=NOT_HOLDING, objects=(env.robot, env.cube)),
    }
    return atoms | {GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube))}


def test_the_pick_applies_without_grasp_clear_and_not_without_pick_plannable() -> None:
    provider = _provider(layout=Tossing3DLayout.BARRIER)
    atoms = _pickable_atoms(provider=provider)
    assert not any(atom.predicate == GRASP_CLEAR for atom in atoms)
    plannable = SkillGrounder.applicable_ground_skills(
        skills=provider.skills(),
        objects=provider.objects(),
        true_atoms=atoms | {_pick_plannable(provider=provider)},
    )
    assert [ground.skill.name for ground in plannable] == ["PickCube"]
    unplannable = SkillGrounder.applicable_ground_skills(
        skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
    )
    assert unplannable == []
    for reset in provider.movables_reset_skills():
        assert reset.preconditions <= atoms


# ---------------------------------------------------------------- both planners


def _pomdp_method(*, env: Tossing3DEnvironment | None = None) -> Any:
    from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod

    env = Tossing3DEnvironment(scene_bg=False) if env is None else env
    return Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env, offer_non_human_reset=True),
        seed=0,
        pomdp_num_particles=32,
    )


def _root_actions(*, method: Any, atoms: frozenset[GroundAtom]) -> list[GroundSkill]:
    from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
        make_tossing3d_search_state,
    )

    return method._pomdp_model.get_valid_actions(  # noqa: SLF001
        environment_state=make_tossing3d_search_state(state=method.pomdp_state, true_atoms=atoms)
    )


def test_the_belief_space_planner_offers_only_resets_when_the_pick_is_unplannable() -> None:
    method = _pomdp_method()
    provider = method.skill_provider
    atoms = _pickable_atoms(provider=provider)
    offered = _root_actions(method=method, atoms=atoms | {_pick_plannable(provider=provider)})
    assert "PickCube" in {ground.skill.name for ground in offered}
    stuck = _root_actions(method=method, atoms=atoms)
    assert stuck
    assert {ground.skill.name for ground in stuck} <= _RESET_NAMES


def test_an_imagined_reset_makes_the_pick_applicable() -> None:
    from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
        make_tossing3d_search_state,
    )

    method = _pomdp_method()
    model = method._pomdp_model  # noqa: SLF001
    provider = method.skill_provider
    atoms = _pickable_atoms(provider=provider)
    search_state = make_tossing3d_search_state(state=method.pomdp_state, true_atoms=atoms)
    for reset in model.get_valid_actions(environment_state=search_state):
        from hitl_pmp.methods.belief_space.tossing3d_constants import RESET_SKILL
        from hitl_pmp.methods.belief_space.tossing3d_observation_model import mean_competence

        branches = model.outcomes(
            environment_state=search_state, state=method.pomdp_state, action=reset
        )
        successful = [
            branch for branch in branches if _pick_plannable(provider=provider) in branch[2]
        ]
        assert len(successful) == 1
        probability, next_state, after = successful[0]
        expected = (
            mean_competence(belief=method.pomdp_state.skill_beliefs[RESET_SKILL])
            if reset.skill.name == RESET_SKILL
            else 1.0
        )
        assert probability == pytest.approx(expected)
        assert sum(branch[0] for branch in branches) == pytest.approx(1.0)
        for _, _, failed_after in branches:
            if _pick_plannable(provider=provider) not in failed_after:
                assert failed_after == atoms
        following = model.get_valid_actions(
            environment_state=make_tossing3d_search_state(state=next_state, true_atoms=after)
        )
        assert "PickCube" in {ground.skill.name for ground in following}


@pytest.mark.parametrize("plannable_before", [True, False])
def test_imagined_pick_and_toss_outcomes_leave_pick_plannable_as_it_was(
    *, plannable_before: bool
) -> None:
    """The toss lists PickPlannable in `ignore_effects` and re-asserts nothing, so the
    belief-space forecast carries it through unchanged, exactly as it carries
    GraspClear. No imagined branch may drop or invent it."""
    from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
        make_tossing3d_search_state,
    )

    method = _pomdp_method()
    model = method._pomdp_model  # noqa: SLF001
    provider = method.skill_provider
    marker = _pick_plannable(provider=provider)
    starts = [_holding_atoms(provider=provider)]
    if plannable_before:
        starts.append(_pickable_atoms(provider=provider))
    for atoms in starts:
        before = atoms | {marker} if plannable_before else atoms
        search_state = make_tossing3d_search_state(state=method.pomdp_state, true_atoms=before)
        actions = [
            ground
            for ground in model.get_valid_actions(environment_state=search_state)
            if ground.skill.name not in _RESET_NAMES
        ]
        assert actions
        for action in actions:
            for _probability, _next_state, after in model.outcomes(
                environment_state=search_state, state=method.pomdp_state, action=action
            ):
                assert (marker in after) == plannable_before, action.skill.name


def _ees_method(*, env: Tossing3DEnvironment | None = None) -> Any:
    from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

    env = Tossing3DEnvironment(scene_bg=False) if env is None else env
    provider = Tossing3DSkillProvider(env=env, human_reset_practice_cost=5.0)
    return EesMethod(env=env, skill_provider=provider, seed=0, reset_cost_gate=False)


def test_ees_plans_through_the_reset_when_the_pick_is_unplannable() -> None:
    method = _ees_method()
    provider = method.skill_provider
    env = provider.env
    goal = frozenset({GroundAtom(predicate=IN_BIN, objects=(env.cube, env.bin))})
    atoms = _pickable_atoms(provider=provider)

    plan = method.plan_to(init_atoms=atoms, goal=goal, costs={}, practicing=True)
    assert [ground.skill.name for ground in plan] == [
        "ask_for_reset_cube_bin_only",
        "PickCube",
        "MoveToTossLocationAndToss",
    ]
    direct = method.plan_to(
        init_atoms=atoms | {_pick_plannable(provider=provider)}, goal=goal, costs={}
    )
    assert [ground.skill.name for ground in direct] == ["PickCube", "MoveToTossLocationAndToss"]


def test_fast_downward_forgets_pick_plannable_across_a_toss_like_grasp_clear() -> None:
    """In the PDDL an ignore effect is a universally quantified delete, so Fast Downward
    reaches a second pick after an imagined toss only through the reset -- the treatment
    GraspClear has, by the same line of the operator."""
    from hitl_pmp.planning.pddl import PddlWriter

    provider = _provider(layout=Tossing3DLayout.BARRIER)
    domain = PddlWriter.domain_str(
        skills=provider.skills(), predicates=provider.predicates(), types=provider.types()
    )
    toss = domain.split("(:action MoveToTossLocationAndToss")[1].split("(:action")[0]
    assert (
        "(forall (?x0 - tossing3d_robot ?x1 - tossing3d_cube) (not (PickPlannable ?x0 ?x1)))"
    ) in toss
    assert (
        "(forall (?x0 - tossing3d_cube ?x1 - tossing3d_bin) (not (GraspClear ?x0 ?x1)))"
    ) in toss
    pick = domain.split("(:action PickCube")[1].split("(:action")[0]
    assert "(PickPlannable ?robot ?cube)" in pick.split(":effect")[0]
    assert "PickPlannable" not in pick.split(":effect")[1]


# ------------------------------------------------------------------- live KINDER


@pytest.fixture(scope="module")
def live_env() -> Iterator[Tossing3DEnvironment]:
    env = Tossing3DEnvironment()
    env.reset_to_seed(seed=125)
    yield env
    env.close()


def _atoms(*, env: Tossing3DEnvironment, observed: Any) -> frozenset[GroundAtom]:
    provider = Tossing3DSkillProvider(env=env, offer_non_human_reset=True)
    return SkillGrounder.abstract_state(
        state=observed, objects=provider.objects(), predicates=provider.predicates()
    )


def _applicable(*, env: Tossing3DEnvironment, atoms: frozenset[GroundAtom]) -> list[str]:
    provider = Tossing3DSkillProvider(env=env, offer_non_human_reset=True)
    return [
        ground.skill.name
        for ground in SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
        )
    ]


@_NEEDS_KINDER
@pytest.mark.parametrize("label", list(_STUCK_STATES))
def test_a_known_stuck_state_is_unplannable_and_leaves_only_the_resets(
    *, live_env: Tossing3DEnvironment, label: str
) -> None:
    env = live_env
    observed = env.restore_plain_snapshot(plain=_plain(fixture=_STUCK_STATES[label]))

    assert not PICK_PLANNABLE.holds(observed, (env.robot, env.cube))
    atoms = _atoms(env=env, observed=observed)
    assert _applicable(env=env, atoms=atoms) == []
    provider = Tossing3DSkillProvider(env=env, offer_non_human_reset=True)
    for reset in provider.movables_reset_skills():
        assert reset.preconditions <= atoms

    root = _root_actions(method=_pomdp_method(env=env), atoms=atoms)
    assert root
    assert {ground.skill.name for ground in root} <= _RESET_NAMES

    goal = frozenset({GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube))})
    plan = _ees_method(env=env).plan_to(init_atoms=atoms, goal=goal, costs={}, practicing=True)
    assert [ground.skill.name for ground in plan] == ["ask_for_reset_cube_bin_only", "PickCube"]

    # What the predicate predicts: the dispatched pick is refused before it moves.
    run = env.backend().run_pick_cube()
    assert run.steps == 0
    assert run.error is not None
    assert "No collision-free cube grasp" in run.error


def _centred_in_bin_cube() -> dict[str, list[float]]:
    """EXP-22's frozen scene with the cube on the bin floor at the bin's centre, as the
    in-bin calibration scan placed it (pose A, offset 0, yaw 0): picked there."""
    plain = _plain(fixture="pick_plannable/exp22_A.json")
    bin_x, bin_y = plain["bin_0"][0], plain["bin_0"][1]
    cube = plain["cube_0"]
    cube[0], cube[1], cube[2] = bin_x, bin_y, 0.045
    cube[3:7] = [1.0, 0.0, 0.0, 0.0]
    cube[7:13] = [0.0] * 6
    return plain


def _ordinary_state(*, env: Tossing3DEnvironment, label: str) -> Any:
    if label == "cube in its spawn strip":
        return env.reset_to_seed(seed=125)
    return env.restore_plain_snapshot(plain=_centred_in_bin_cube())


@_NEEDS_KINDER
@pytest.mark.parametrize("label", ["cube in its spawn strip", "cube centred in the bin"])
def test_an_ordinary_state_is_plannable_and_the_pick_executes(
    *, live_env: Tossing3DEnvironment, label: str
) -> None:
    env = live_env
    observed = _ordinary_state(env=env, label=label)
    assert PICK_PLANNABLE.holds(observed, (env.robot, env.cube))
    assert "PickCube" in _applicable(env=env, atoms=_atoms(env=env, observed=observed))

    held = env.take_action(action=np.array([float(env.pick_cube_id), 0.0, 0.0, 0.0, 0.0]))
    assert env.last_skill_error() is None
    assert sum(env.last_controller_steps()) > 0
    assert HOLDING.holds(held, (env.robot, env.cube))


@_NEEDS_KINDER
def test_the_exp22_in_bin_pick_that_executed_is_plannable_though_grasp_clear_refuses_it(
    *, live_env: Tossing3DEnvironment
) -> None:
    """EXP-22 gpon global_curve 3e-6 seed 0, skill 434: the cube lay in the bin with
    inner gaps of 0.080 and 0.045 m and the logged pick ran 160 steps. GraspClear's
    0.0625 m in-bin threshold calls that cube unpickable; the dry run does not."""
    env = live_env
    observed = env.restore_plain_snapshot(
        plain=_plain(fixture="pick_plannable/exp22_inbin_pick_434.json")
    )
    assert not GRASP_CLEAR.holds(observed, (env.cube, env.bin))
    assert PICK_PLANNABLE.holds(observed, (env.robot, env.cube))
    assert "PickCube" in _applicable(env=env, atoms=_atoms(env=env, observed=observed))
    run = env.backend().run_pick_cube()
    assert run.error is None
    assert run.steps > 0


@_NEEDS_KINDER
def test_the_dry_run_is_not_made_while_the_cube_is_held_or_across_the_barrier(
    *, live_env: Tossing3DEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = live_env
    env.reset_to_seed(seed=125)
    spawn = env.backend().snapshot_to_plain(snapshot=env.backend().snapshot())
    held = env.take_action(action=np.array([float(env.pick_cube_id), 0.0, 0.0, 0.0, 0.0]))
    assert HOLDING.holds(held, (env.robot, env.cube))
    holding = env.backend().snapshot_to_plain(snapshot=env.backend().snapshot())
    across = {name: list(values) for name, values in spawn.items()}
    across["cube_0"][0] = 2.0

    def refuse(  # noqa: PLR0917 - stands in for a bound method
        self: KinderBackend, *, state: Any = None
    ) -> str | None:
        raise AssertionError("the dry run was made in a state where it cannot matter")

    monkeypatch.setattr(KinderBackend, "pick_cube_plan_failure", refuse)
    # Held: carried as present, so an imagined toss does not forecast every landing
    # as unpickable.
    observed = env.restore_plain_snapshot(plain=holding)
    assert HOLDING.holds(observed, (env.robot, env.cube))
    assert PICK_PLANNABLE.holds(observed, (env.robot, env.cube))
    # Across the barrier: the base has no path, and nothing but a reset changes that.
    observed = env.restore_plain_snapshot(plain=across)
    assert not PICK_PLANNABLE.holds(observed, (env.robot, env.cube))


@_NEEDS_KINDER
def test_a_cube_across_the_barrier_really_is_unplannable(*, live_env: Tossing3DEnvironment) -> None:
    """The gate's one shortcut, checked against the planner it stands in for."""
    env = live_env
    env.reset_to_seed(seed=125)
    backend = env.backend()
    across = backend.snapshot_to_plain(snapshot=backend.snapshot())
    across["cube_0"][0] = 2.0
    failure = backend.pick_cube_plan_failure(state=backend.plain_to_snapshot(plain=across))
    assert failure is not None
    assert "No collision-free cube grasp" in failure


@_NEEDS_KINDER
def test_the_verdict_is_planned_once_per_state_and_by_the_dispatchs_own_construction(
    *, live_env: Tossing3DEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = live_env
    backend = env.backend()
    backend._pick_plan_failures.clear()  # noqa: SLF001
    grounded: list[tuple[str, tuple[str, ...]]] = []
    original = KinderBackend._ground_controller  # noqa: SLF001

    def spy(self: KinderBackend, **kwargs: Any) -> Any:  # noqa: PLR0917 - a bound method
        grounded.append((kwargs["key"], tuple(kwargs["object_names"])))
        return original(self, **kwargs)

    monkeypatch.setattr(KinderBackend, "_ground_controller", spy)
    plain = _plain(fixture="pick_plannable/exp22b_s1.json")
    first = env.restore_plain_snapshot(plain=plain)
    assert len(grounded) == 1
    second = env.restore_plain_snapshot(plain=plain)
    assert len(grounded) == 1, "the same state was planned twice"
    assert first.abstract_atoms == second.abstract_atoms

    # The dispatch builds its controller through the same door, with the same objects.
    run = backend.run_pick_cube()
    assert run.steps == 0
    assert len(grounded) == 2
    assert grounded[0] == grounded[1]
    # A refused pick leaves the state as it was, so observing it again plans nothing.
    env._observed_state(seed=0, steps_taken=0)  # noqa: SLF001
    assert len(grounded) == 2


def _trajectory(*, tmp_path: Path, evaluate: bool, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every physics tick of a pick, a toss and a second pick from the seed-125 scene,
    with the dry run either made or skipped."""
    if not evaluate:
        monkeypatch.setattr(
            KinderBackend, "pick_cube_plan_failure", lambda self, *, state=None: None
        )
    env = Tossing3DEnvironment()
    log = tmp_path / f"evaluate-{evaluate}.jsonl"
    writer = StateLogWriter(
        output_path=log,
        header=StateLogHeader(
            variant="o1", scene_bg=True, canonical_seed=125, seed=0, test_env_seed_offset=10000
        ),
    )
    try:
        env.attach_state_log_writer(writer=writer)
        env.reset_to_seed(seed=125)
        pick = np.array([float(env.pick_cube_id), 0.0, 0.0, 0.0, 0.0])
        toss = np.array([float(env.move_to_toss_location_and_toss_id), 1.31, -0.007, 128.5, 733.0])
        for action in (pick, toss, pick):
            env.take_action(action=action)
        final = env.backend().snapshot_to_plain(snapshot=env.backend().snapshot())
    finally:
        writer.close()
        env.close()
    events = []
    for line in log.read_text().splitlines():
        event = json.loads(line)
        for stamp in ("timestamp", "elapsed_seconds"):
            event.pop(stamp, None)
        events.append(event)
    return [*events, final]


@_NEEDS_KINDER
def test_evaluating_the_predicate_does_not_change_the_trajectory(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two runs of the same three skills from the same seeded scene, one making the dry
    run at every observed state and one skipping it, must log the same ticks exactly:
    the dry run steps no simulator and advances no random stream the run uses."""
    with monkeypatch.context() as patched:
        skipped = _trajectory(tmp_path=tmp_path, evaluate=False, monkeypatch=patched)
    evaluated = _trajectory(tmp_path=tmp_path, evaluate=True, monkeypatch=monkeypatch)
    ticks = [event for event in evaluated[:-1] if event.get("kind") == "tick"]
    assert len(ticks) > 100, "sanity: the skills really ran"
    assert evaluated == skipped


@_NEEDS_KINDER
def test_the_dry_run_leaves_the_simulator_and_the_abstraction_untouched(
    *, live_env: Tossing3DEnvironment
) -> None:
    env = live_env
    env.restore_plain_snapshot(plain=_plain(fixture="pick_plannable/exp22_inbin_pick_434.json"))
    backend = env.backend()
    sim = backend._object_centric()._robot_env.sim  # noqa: SLF001
    data = sim.data.mj_data if hasattr(sim.data, "mj_data") else sim.data

    def simulator() -> tuple[Any, ...]:
        return (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), float(data.time))

    import pybullet

    def planning_clients() -> int:
        return sum(pybullet.getConnectionInfo(i)["isConnected"] for i in range(64))

    before = simulator()
    state_before = backend.snapshot_to_plain(snapshot=backend.snapshot())
    atoms_before = backend.abstract_atoms()
    clients_before = planning_clients()
    backend._pick_plan_failures.clear()  # noqa: SLF001
    assert backend.pick_cube_plan_failure() is None
    after = simulator()
    # The dry run's own planning scene is released with the verdict, not left to the
    # collector: a run makes thousands of these.
    assert planning_clients() == clients_before
    assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))
    assert backend.snapshot_to_plain(snapshot=backend.snapshot()) == state_before
    assert backend.abstract_atoms() == atoms_before
