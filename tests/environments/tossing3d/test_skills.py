"""Offline tests for the four lifted skills, their operator models and their samplers."""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    BARRIER_COLLISION_MARGIN,
    GRIPPER_COMMANDED_CLOSED,
    HAND_EMPTY,
    HOLDING,
    IN_BIN,
    ON_GROUND,
    REACHABLE,
    ROBOT_AT_SUCCESSFUL_THROW_POSE,
    TOSS_RELEASE_MS_BOUNDS,
    TOSS_SPEED_BOUNDS,
    UPSTREAM_DEFAULT_RELEASE_SPEED_DEG_S,
    WORST_BARRIER_COLLISION_STANDOFF,
    RobotAtSuccessfulThrowPoseClassifier,
)
from hitl_pmp.environments.tossing3d.predicates import (
    THROW_STANDOFF_BOUNDS as PREDICATE_THROW_STANDOFF_BOUNDS,
)
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import (
    PICK_DISTANCE_BOUNDS,
    PICK_ROTATION_BOUNDS,
    THROW_STANDOFF_BOUNDS,
    Tossing3DSkills,
)
from hitl_pmp.planning.fast_downward import FastDownwardPlanner
from hitl_pmp.planning.grounding import SkillGrounder

from .observations import BIN_X, state

_ENV = Tossing3DEnvironment()
_SKILLS = Tossing3DSkills


# The exact lifted signature of each operator, in declaration order. Pinned as a literal
# rather than derived, because the thing under test is precisely *which* objects an
# operator is allowed to name -- a derivation would restate whatever `skills.py` happens
# to declare and could never fail.
#
# **No operator names a goal region.** This domain assumes the bin's interior *is* the
# scored region (see `predicates.py`'s module docstring), so the region stopped being a
# symbolic object: it is scene geometry the classifiers read out of `State`, not a thing a
# planner binds a variable to. A signature that reintroduced it would put an object into
# every plan step that no skill can act on.
_EXPECTED_PARAMETERS = {
    "Pick": ("robot", "cube", "barrier", "bin"),
    "MoveToThrowPose": ("robot", "cube", "bin"),
    "Toss": ("robot", "cube", "bin", "barrier"),
    # `?cube` only so the operator can require `OnGround`; the controller takes the robot.
    "OpenGripper": ("robot", "cube"),
}

_ALL_SKILLS = (_SKILLS.PICK, _SKILLS.MOVE_TO_THROW_POSE, _SKILLS.TOSS, _SKILLS.OPEN_GRIPPER)


def test_no_operator_names_a_goal_region_object() -> None:
    """The headline invariant of the bin-is-the-goal-region simplification.

    Checked by *type* rather than by variable name, so renaming the variable cannot smuggle
    the dependency back in."""
    for skill in _ALL_SKILLS:
        declared = [variable.type.name for variable in skill.parameters]
        assert "tossing3d_goal_region" not in declared, (
            f"{skill.name} still takes a goal-region parameter: {declared}"
        )


def test_each_operator_declares_exactly_the_objects_it_acts_on() -> None:
    """The positive half of the test above: dropping the goal region must not have
    disturbed the objects that remain, nor their order (`GroundSkill.objects` is
    positional, and the oracle builds its groundings by hand)."""
    actual = {
        skill.name: tuple(variable.name for variable in skill.parameters) for skill in _ALL_SKILLS
    }
    assert actual == _EXPECTED_PARAMETERS


def _pick() -> GroundSkill:
    return GroundSkill(
        skill=_SKILLS.PICK,
        objects=(_ENV.robot, _ENV.cube, _ENV.barrier, _ENV.bin),
    )


def _move() -> GroundSkill:
    return GroundSkill(
        skill=_SKILLS.MOVE_TO_THROW_POSE,
        objects=(_ENV.robot, _ENV.cube, _ENV.bin),
    )


def _toss() -> GroundSkill:
    return GroundSkill(
        skill=_SKILLS.TOSS,
        objects=(_ENV.robot, _ENV.cube, _ENV.bin, _ENV.barrier),
    )


def test_pick_requires_reachable_so_no_plan_retrieves_a_tossed_cube() -> None:
    """The one precondition that encodes the domain's irreversibility. Without it a
    planner emits "toss, then pick it back up and try again", which the dynamics can
    never execute -- exactly the over-permissive-model defect class that
    tests/environments/test_operator_dynamics_fidelity.py exists for."""
    assert (
        LiftedAtom(predicate=REACHABLE, variables=(_SKILLS._cube, _SKILLS._barrier))
        in _SKILLS.PICK.preconditions
    )


def test_pick_deletes_the_throw_pose_because_the_grasp_drives_the_base_to_the_cube() -> None:
    """`pick_shelf` navigates to the cube, which is on the near side of the barrier and
    therefore far too short of the bin to throw from. A model that let the predicate
    survive a pick would let the planner skip `MoveToThrowPose` entirely."""
    assert (
        LiftedAtom(
            predicate=ROBOT_AT_SUCCESSFUL_THROW_POSE,
            variables=(_SKILLS._robot, _SKILLS._bin),
        )
        in _SKILLS.PICK.delete_effects
    )


def test_toss_deletes_reachable_unconditionally_hit_or_miss() -> None:
    """A toss makes the cube unreachable whether or not it lands in the region. Deleting
    it only on success would be a model in which a missed throw costs nothing -- which is
    precisely the cost this domain exists to represent."""
    assert (
        LiftedAtom(predicate=REACHABLE, variables=(_SKILLS._cube, _SKILLS._barrier))
        in _SKILLS.TOSS.delete_effects
    )


def test_the_four_operator_models_are_exactly_as_declared() -> None:
    """Pinned field by field, so a change to the symbolic layer is a deliberate edit to
    this list rather than a silent drift."""
    assert _SKILLS.PICK.preconditions == frozenset({
        LiftedAtom(predicate=HAND_EMPTY, variables=(_SKILLS._robot,)),
        LiftedAtom(predicate=ON_GROUND, variables=(_SKILLS._cube,)),
        LiftedAtom(predicate=REACHABLE, variables=(_SKILLS._cube, _SKILLS._barrier)),
    })
    assert _SKILLS.PICK.add_effects == frozenset({
        LiftedAtom(predicate=HOLDING, variables=(_SKILLS._robot, _SKILLS._cube)),
        LiftedAtom(predicate=GRIPPER_COMMANDED_CLOSED, variables=(_SKILLS._robot,)),
    })
    assert _SKILLS.PICK.delete_effects == frozenset({
        LiftedAtom(predicate=HAND_EMPTY, variables=(_SKILLS._robot,)),
        LiftedAtom(predicate=ON_GROUND, variables=(_SKILLS._cube,)),
        LiftedAtom(
            predicate=ROBOT_AT_SUCCESSFUL_THROW_POSE,
            variables=(_SKILLS._robot, _SKILLS._bin),
        ),
    })

    assert _SKILLS.MOVE_TO_THROW_POSE.preconditions == frozenset({
        LiftedAtom(predicate=HOLDING, variables=(_SKILLS._robot, _SKILLS._cube))
    })
    assert _SKILLS.MOVE_TO_THROW_POSE.add_effects == frozenset({
        LiftedAtom(
            predicate=ROBOT_AT_SUCCESSFUL_THROW_POSE,
            variables=(_SKILLS._robot, _SKILLS._bin),
        )
    })
    assert _SKILLS.MOVE_TO_THROW_POSE.delete_effects == frozenset()

    assert _SKILLS.TOSS.preconditions == frozenset({
        LiftedAtom(predicate=HOLDING, variables=(_SKILLS._robot, _SKILLS._cube)),
        LiftedAtom(
            predicate=ROBOT_AT_SUCCESSFUL_THROW_POSE,
            variables=(_SKILLS._robot, _SKILLS._bin),
        ),
    })
    assert _SKILLS.TOSS.add_effects == frozenset({
        LiftedAtom(predicate=IN_BIN, variables=(_SKILLS._cube, _SKILLS._bin)),
        LiftedAtom(predicate=HAND_EMPTY, variables=(_SKILLS._robot,)),
    })
    assert _SKILLS.TOSS.delete_effects == frozenset({
        LiftedAtom(predicate=HOLDING, variables=(_SKILLS._robot, _SKILLS._cube)),
        LiftedAtom(predicate=GRIPPER_COMMANDED_CLOSED, variables=(_SKILLS._robot,)),
        LiftedAtom(predicate=REACHABLE, variables=(_SKILLS._cube, _SKILLS._barrier)),
    })

    assert _SKILLS.OPEN_GRIPPER.preconditions == frozenset({
        LiftedAtom(predicate=GRIPPER_COMMANDED_CLOSED, variables=(_SKILLS._robot,)),
        LiftedAtom(predicate=ON_GROUND, variables=(_SKILLS._cube,)),
    })
    assert _SKILLS.OPEN_GRIPPER.add_effects == frozenset({
        LiftedAtom(predicate=HAND_EMPTY, variables=(_SKILLS._robot,))
    })
    assert _SKILLS.OPEN_GRIPPER.delete_effects == frozenset({
        LiftedAtom(predicate=GRIPPER_COMMANDED_CLOSED, variables=(_SKILLS._robot,))
    })


@pytest.mark.parametrize("cube_z", [0.0, 0.025, 0.075])
def test_on_ground_and_holding_cannot_both_hold_of_this_scenes_cube(*, cube_z: float) -> None:
    """What licenses `OpenGripper` to omit a `Holding` delete effect, and therefore what
    keeps it free of the conditional effect `seq-opt-lmcut` refuses.

    Arithmetic rather than convention: `OnGround` admits `z <= bb_z/2 + ON_GROUND_TOL`,
    which for a 0.05 m cube is 0.075, strictly below `Holding`'s `z > 0.1`. A cube whose
    `bb_z` exceeded 0.1 would break this, which is why it is checked against the scene's
    own cube."""
    on_ground = state(cube_z=cube_z, gripper=1.0)
    assert ON_GROUND.holds(on_ground, (_ENV.cube,))
    assert not HOLDING.holds(on_ground, (_ENV.robot, _ENV.cube))


def test_no_skill_declares_ignore_effects() -> None:
    """Unlike Ball-Ring's navigations and Tossing Room's Press, nothing here wipes a whole
    predicate: there is one cube, one bin and one region, so every effect is expressible
    as a plain add or delete."""
    for skill in _ALL_SKILLS:
        assert skill.ignore_effects == frozenset(), skill.name


# A grasp that closes on nothing, as recorded on 4/10 `never`-reset EES seeds. `Pick`
# leaves the gripper *commanded* shut while the cube is still on the floor, and
# `pos_gripper` is that command rather than the finger configuration, so `HandEmpty` and
# `Holding` are both false at once.
_SHUT_ON_NOTHING = {"gripper": 1.0, "cube_z": 0.0249}


def _applicable_in_the_shut_gripper_state() -> set[str]:
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    shut = state(env=env, **_SHUT_ON_NOTHING)
    atoms = SkillGrounder.abstract_state(
        state=shut, objects=provider.objects(), predicates=provider.predicates()
    )
    return {
        ground_skill.skill.name
        for ground_skill in SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
        )
    }


def test_a_gripper_shut_on_nothing_is_neither_hand_empty_nor_holding() -> None:
    """The state that strands the robot, characterised at the predicate level."""
    shut = state(**_SHUT_ON_NOTHING)
    assert not HAND_EMPTY.holds(shut, (_ENV.robot,))
    assert not HOLDING.holds(shut, (_ENV.robot, _ENV.cube))


def test_the_grasping_and_throwing_skills_are_all_inapplicable_with_a_shut_gripper() -> None:
    """`Pick` needs `HandEmpty`; `MoveToThrowPose` and `Toss` both need `Holding`. Neither
    holds here, so none of the three can recover -- which is the defect, not the fix."""
    assert not {"Pick", "MoveToThrowPose", "Toss"} & _applicable_in_the_shut_gripper_state()


def test_some_ground_skill_is_applicable_with_a_gripper_shut_on_nothing() -> None:
    """THE property: no reachable state may leave the robot with nothing to do.

    The cube is still on the robot's side of the barrier and still resting flat, so
    nothing about the world is unrecoverable -- only the symbolic model was. A real
    TidyBot reopens its fingers, and `open_gripper` is a controller upstream ships."""
    assert _applicable_in_the_shut_gripper_state()


def test_no_variable_carries_the_question_mark_the_pddl_writer_adds() -> None:
    """`PddlWriter.variable_str` (planning/pddl.py) documents the convention explicitly:
    our `Variable.name` is plain and the writer prepends the "?" at write time, because
    predicators' `Variable.name` already carries it and ours deliberately does not. A
    name declared as "?robot" therefore renders as "??robot", which Fast Downward's
    translator splits into two tokens -- and the failure is *silent*, because
    `EesMethod._next_plan` catches `PlanningFailure` and degrades to a no-op. So this is
    checked structurally as well as end-to-end below: a run can otherwise exit 0 and
    write a full `stats.json` in which the method never took a single action."""
    for skill in _ALL_SKILLS:
        for variable in skill.parameters:
            assert not variable.name.startswith("?"), f"{skill.name}: {variable.name}"


def test_integration_fast_downward_plans_the_three_skill_solve() -> None:
    """An INTEGRATION test, deliberately not skipped, in the style of
    `tests/planning/test_fast_downward.py`: it shells out to a real Fast Downward on this
    domain's own PDDL. It needs no simulator -- the whole symbolic layer is built from a
    hand-written `KinderObservation` -- so it runs on CI, where the `tossing3d` extra is
    never installed but Fast Downward is.

    The structural test above would not have caught a second way of emitting unparseable
    PDDL; this one catches any of them, and is the check that was missing when
    `--env tossing3d --method ees` ran to completion planning nothing."""
    env = Tossing3DEnvironment()
    # Read off the provider rather than restated here, so a predicate that reaches an
    # operator without reaching the domain declaration fails as an unparseable domain
    # rather than passing against a list that quietly matches.
    provider = Tossing3DSkillProvider(env=env)
    objects = provider.objects()
    predicates = provider.predicates()
    init_atoms = SkillGrounder.abstract_state(state=state(), objects=objects, predicates=predicates)
    plan = FastDownwardPlanner.plan(
        skills=provider.skills(),
        predicates=predicates,
        types=provider.types(),
        objects=objects,
        init_atoms=init_atoms,
        goal=frozenset({GroundAtom(predicate=IN_BIN, objects=(env.cube, env.bin))}),
    )
    assert [step.skill.name for step in plan] == ["Pick", "MoveToThrowPose", "Toss"]


def test_param_dims_give_the_throw_a_release_speed_of_its_own() -> None:
    """`Toss`'s dials are upstream's own `TossController.reset` parameters rather than
    quantities this package synthesised, so both arm configurations remain upstream's.
    """
    assert _SKILLS.PICK.param_dim == 2
    assert _SKILLS.MOVE_TO_THROW_POSE.param_dim == 1
    assert _SKILLS.TOSS.param_dim == 2
    # Nothing to draw: `OpenGripperController.sample_parameters` raises.
    assert _SKILLS.OPEN_GRIPPER.param_dim == 0


def test_compute_action_encodes_the_skill_id_in_slot_zero() -> None:
    assert Tossing3DSkills.compute_action(
        ground_skill=_pick(), params=np.array([0.55, 0.1]), state=state()
    ) == pytest.approx([Tossing3DEnvironment.pick_id, 0.55, 0.1])
    assert Tossing3DSkills.compute_action(
        ground_skill=_move(), params=np.array([1.35]), state=state()
    ) == pytest.approx([Tossing3DEnvironment.move_to_throw_pose_id, 1.35, 0.0])
    assert Tossing3DSkills.compute_action(
        ground_skill=_toss(), params=np.array([140.0, 720.0]), state=state()
    ) == pytest.approx([Tossing3DEnvironment.toss_id, 140.0, 720.0])


def test_every_action_matches_the_declared_action_space() -> None:
    for ground_skill, params in (
        (_pick(), np.array([0.55, 0.1])),
        (_move(), np.array([1.35])),
        (_toss(), np.array([140.0, 720.0])),
    ):
        action = Tossing3DSkills.compute_action(
            ground_skill=ground_skill, params=params, state=state()
        )
        assert action.shape == Tossing3DEnvironment.action_space.shape


def test_sample_params_draws_pick_parameters_inside_upstreams_own_bounds() -> None:
    """Upstream's `PickShelfController.sample_parameters` draws uniformly from exactly
    these two constants and then rejection-tests against *other* cubes; with one cube
    there is nothing to reject, so a plain uniform draw is what it reduces to here."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        distance, rotation = Tossing3DSkills.sample_params(ground_skill=_pick(), rng=rng)
        assert PICK_DISTANCE_BOUNDS[0] <= distance <= PICK_DISTANCE_BOUNDS[1]
        assert PICK_ROTATION_BOUNDS[0] <= rotation <= PICK_ROTATION_BOUNDS[1]


def test_the_sampler_draws_both_satisfying_and_unsatisfying_standoffs() -> None:
    """**The sampler must be able to miss.** `MoveToThrowPose`'s only add effect is
    `RobotAtSuccessfulThrowPose`, and EES trains that skill's success classifier on
    exactly that label. This test used to assert the opposite -- that *every* sampled
    standoff satisfies the add effect -- which is a tautology dressed as a fidelity check:
    it holds precisely when the label is constant-true and the sampler cannot learn.

    Both outcomes must occur, and every draw must still be in bounds."""
    rng = np.random.default_rng(0)
    outcomes = []
    for _ in range(200):
        (standoff,) = Tossing3DSkills.sample_params(ground_skill=_move(), rng=rng)
        assert THROW_STANDOFF_BOUNDS[0] <= standoff <= THROW_STANDOFF_BOUNDS[1]
        outcomes.append(
            RobotAtSuccessfulThrowPoseClassifier.holds(
                state=state(base_x=BIN_X - standoff),
                robot=_ENV.robot,
                target=_ENV.bin,
            )
        )
    assert any(outcomes), "no draw ever succeeds: the skill could never achieve its effect"
    assert not all(outcomes), (
        "every draw succeeds: the add effect is constant-true and no sampler can learn"
    )


def test_the_sampler_range_is_the_measured_feasible_range() -> None:
    """The measured feasible range `[0.40, 2.06]`, inset at both ends: below 0.40 m the
    base shoves the bin across the floor, above 2.06 m `Toss`'s windup fails to
    motion-plan, and above ~1.79 m the predicate would start accepting the pose `Pick`
    leaves the base in.

    **The lower bound is no longer that 0.40 m inset.** A separate, tighter hazard sits
    inside it: `cuboid_barrier` is a real dynamic MuJoCo body `move_to_target`'s base
    motion planner does not collision-check against, so a standoff up to 1.00 m still
    drives the base through it (see `predicates.THROW_STANDOFF_BOUNDS`'s docstring for
    the three-ways-confirmed measurement). `BARRIER_COLLISION_MARGIN` moves the floor to
    1.10 m, well above the bin-shove threshold this docstring's first paragraph
    describes.

    `skills.py` imports the interval from `predicates.py` rather than declaring its own,
    so there is one place it is measured. That import is **not** the same thing as the
    predicate's acceptance band, which is derived per call -- see the test below."""
    assert THROW_STANDOFF_BOUNDS == (1.10, 1.75)
    assert THROW_STANDOFF_BOUNDS is PREDICATE_THROW_STANDOFF_BOUNDS


def test_the_sampler_range_excludes_the_measured_barrier_collision_range() -> None:
    """**The safety property, pinned so it survives a future retune.** Unlike the test
    above -- which pins the current bound to a literal and fails on any change at all,
    including a safe widening -- this one states *why* the lower bound has to be where
    it is: clear of every standoff `test_move_to_throw_pose_at_the_lower_standoff_bound_
    does_not_disturb_the_barrier` (in `test_kinder_fidelity.py`, which needs a live KINDER
    install and so skips on CI) confirmed drives the base through `cuboid_barrier`.

    This is the one offline check of that safety property CI can actually run. If someone
    re-measures `WORST_BARRIER_COLLISION_STANDOFF` higher, or shrinks
    `BARRIER_COLLISION_MARGIN`, without widening `THROW_STANDOFF_BOUNDS` to match, this
    fails here rather than only in the live-KINDER test that skips everywhere but a
    KINDER-installed machine."""
    assert THROW_STANDOFF_BOUNDS[0] >= WORST_BARRIER_COLLISION_STANDOFF + BARRIER_COLLISION_MARGIN


def test_the_sampler_range_is_not_the_predicates_acceptance_band() -> None:
    """The defect this whole change exists to fix, pinned as a property.

    `predicates.py` used to hand `THROW_STANDOFF_BOUNDS` straight to the classifier as its
    acceptance interval, so widening the sampler's range widened the acceptance region in
    lockstep and the add effect stayed constant-true. The acceptance band is now derived
    from the live goal region and `THROW_RANGE`, and is strictly narrower than the range
    the sampler draws from."""
    low, high = THROW_STANDOFF_BOUNDS
    accepted = [
        standoff / 1000
        for standoff in range(int(low * 1000), int(high * 1000))
        if RobotAtSuccessfulThrowPoseClassifier.holds(
            state=state(base_x=BIN_X - standoff / 1000),
            robot=_ENV.robot,
            target=_ENV.bin,
        )
    ]
    assert accepted
    assert max(accepted) - min(accepted) < (high - low) / 2


def test_toss_samples_one_release_speed_inside_the_measured_bounds() -> None:
    """The dial is in joint-path deg/s. `TOSS_SPEED_BOUNDS` stays inside PR #213's
    measured grid, so a draw is never an extrapolation.
    """
    rng = np.random.default_rng(0)
    draws = [
        float(Tossing3DSkills.sample_params(ground_skill=_toss(), rng=rng)[0]) for _ in range(200)
    ]
    assert all(TOSS_SPEED_BOUNDS[0] <= speed <= TOSS_SPEED_BOUNDS[1] for speed in draws)
    # A real draw, not a constant dressed as one.
    assert max(draws) - min(draws) > (TOSS_SPEED_BOUNDS[1] - TOSS_SPEED_BOUNDS[0]) / 2


def test_the_shipped_default_speed_is_the_upper_edge_of_the_samplers_range() -> None:
    """140 deg/s is upstream's default and exactly the top of the sampler's range: it must
    be reachable so the oracle's throw is learnable, but it is the edge rather than an
    interior point because it is what the real TidyBot primitive commands
    (`movej_primitive.execute(..., max_vel=140, ...)`).

    Asserted as equality rather than containment, so widening the range fails here.
    """
    assert TOSS_SPEED_BOUNDS[0] < UPSTREAM_DEFAULT_RELEASE_SPEED_DEG_S
    assert TOSS_SPEED_BOUNDS[1] == UPSTREAM_DEFAULT_RELEASE_SPEED_DEG_S


def test_toss_samples_a_gripper_release_ms_inside_the_measured_bounds() -> None:
    """The second dial, in milliseconds from the start of the swing. Slot 1 is drawn
    separately and tested above.
    """
    rng = np.random.default_rng(0)
    draws = [
        float(Tossing3DSkills.sample_params(ground_skill=_toss(), rng=rng)[1]) for _ in range(200)
    ]
    assert all(TOSS_RELEASE_MS_BOUNDS[0] <= ms <= TOSS_RELEASE_MS_BOUNDS[1] for ms in draws)
    # A real draw, not a constant dressed as one.
    assert max(draws) - min(draws) > (TOSS_RELEASE_MS_BOUNDS[1] - TOSS_RELEASE_MS_BOUNDS[0]) / 2


def test_the_two_toss_dials_are_drawn_independently() -> None:
    """A sampler that wrote one draw into both slots, or drew the second from the first,
    would pass each single-slot test above while collapsing the space onto a line. Pinned
    as near-zero rank correlation.
    """
    rng = np.random.default_rng(0)
    draws = np.array([
        Tossing3DSkills.sample_params(ground_skill=_toss(), rng=rng) for _ in range(500)
    ])
    speeds, release_ms = draws[:, 0], draws[:, 1]
    speed_ranks = np.argsort(np.argsort(speeds))
    ms_ranks = np.argsort(np.argsort(release_ms))
    assert abs(float(np.corrcoef(speed_ranks, ms_ranks)[0, 1])) < 0.15


def test_every_release_ms_the_sampler_can_draw_still_opens_the_gripper() -> None:
    """The bounds' upper edge is set by the *shortest* swing, not the longest.
    `gripper_release_ms` is unclamped upstream: past the end of the swing the gripper
    never opens and the cube is never thrown.

    Recomputed from upstream's own profile rather than against a copied number, so it
    fails if a pin bump changes the swing's timing.
    """
    kinder_models = pytest.importorskip("kinder_models")
    del kinder_models
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        TOSS_RELEASE_ARM_CONFIGURATION,
        TOSS_SLICES_PER_CONTROL_STEP,
        TOSS_WINDUP_ARM_CONFIGURATION,
        toss_profile_limits,
    )
    from kinder_models.dynamic3d.utils import _CONTROL_TIMESTEP, _trapezoidal_motion_profile

    s_total = float(np.linalg.norm(TOSS_RELEASE_ARM_CONFIGURATION - TOSS_WINDUP_ARM_CONFIGURATION))
    shortest_ms = min(
        (
            len(
                _trapezoidal_motion_profile(
                    s_total,
                    max_vel=limits[0],
                    max_accel=limits[1],
                    max_decel=limits[2],
                    step_size=_CONTROL_TIMESTEP,
                )
            )
            - 1
        )
        * TOSS_SLICES_PER_CONTROL_STEP
        for limits in (
            toss_profile_limits(np.deg2rad(deg))
            for deg in np.linspace(TOSS_SPEED_BOUNDS[0], TOSS_SPEED_BOUNDS[1], 37)
        )
    )
    assert TOSS_RELEASE_MS_BOUNDS[1] <= shortest_ms


def test_an_unknown_skill_raises_from_both_sampler_and_encoder() -> None:
    stray = GroundSkill(
        skill=_SKILLS.PICK.model_copy(update={"name": "NotASkill"}),
        objects=(_ENV.robot, _ENV.cube, _ENV.barrier, _ENV.bin),
    )
    with pytest.raises(ValueError, match="Unknown skill"):
        Tossing3DSkills.sample_params(ground_skill=stray, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="Unknown skill"):
        Tossing3DSkills.compute_action(ground_skill=stray, params=np.zeros(2), state=state())
