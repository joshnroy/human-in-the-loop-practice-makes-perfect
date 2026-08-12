"""Offline tests for the three lifted skills, their operator models and their samplers."""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    BARRIER_COLLISION_MARGIN,
    HAND_EMPTY,
    HOLDING,
    IN_GOAL_REGION,
    ON_GROUND,
    REACHABLE,
    ROBOT_AT_SUCCESSFUL_THROW_POSE,
    TOSS_SPEED_BOUNDS,
    UPSTREAM_DEFAULT_RELEASE_SPEED_DEG_S,
    WORST_BARRIER_COLLISION_STANDOFF,
    RobotAtSuccessfulThrowPoseClassifier,
)
from hitl_pmp.environments.tossing3d.predicates import (
    THROW_STANDOFF_BOUNDS as PREDICATE_THROW_STANDOFF_BOUNDS,
)
from hitl_pmp.environments.tossing3d.skills import (
    PICK_DISTANCE_BOUNDS,
    PICK_ROTATION_BOUNDS,
    THROW_STANDOFF_BOUNDS,
    Tossing3DSkills,
)
from hitl_pmp.planning.fast_downward import FastDownwardPlanner
from hitl_pmp.planning.grounding import SkillGrounder

from .observations import COINCIDENT_BIN_X, state

_ENV = Tossing3DEnvironment()
_SKILLS = Tossing3DSkills


def _pick() -> GroundSkill:
    return GroundSkill(
        skill=_SKILLS.PICK,
        objects=(_ENV.robot, _ENV.cube, _ENV.barrier, _ENV.bin, _ENV.goal_region),
    )


def _move() -> GroundSkill:
    return GroundSkill(
        skill=_SKILLS.MOVE_TO_THROW_POSE,
        objects=(_ENV.robot, _ENV.cube, _ENV.bin, _ENV.goal_region),
    )


def _toss() -> GroundSkill:
    return GroundSkill(
        skill=_SKILLS.TOSS,
        objects=(_ENV.robot, _ENV.cube, _ENV.bin, _ENV.barrier, _ENV.goal_region),
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
            variables=(_SKILLS._robot, _SKILLS._bin, _SKILLS._goal_region),
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


def test_the_three_operator_models_are_exactly_as_declared() -> None:
    """Pinned field by field, so a change to the symbolic layer is a deliberate edit to
    this list rather than a silent drift."""
    assert _SKILLS.PICK.preconditions == frozenset({
        LiftedAtom(predicate=HAND_EMPTY, variables=(_SKILLS._robot,)),
        LiftedAtom(predicate=ON_GROUND, variables=(_SKILLS._cube,)),
        LiftedAtom(predicate=REACHABLE, variables=(_SKILLS._cube, _SKILLS._barrier)),
    })
    assert _SKILLS.PICK.add_effects == frozenset({
        LiftedAtom(predicate=HOLDING, variables=(_SKILLS._robot, _SKILLS._cube))
    })
    assert _SKILLS.PICK.delete_effects == frozenset({
        LiftedAtom(predicate=HAND_EMPTY, variables=(_SKILLS._robot,)),
        LiftedAtom(predicate=ON_GROUND, variables=(_SKILLS._cube,)),
        LiftedAtom(
            predicate=ROBOT_AT_SUCCESSFUL_THROW_POSE,
            variables=(_SKILLS._robot, _SKILLS._bin, _SKILLS._goal_region),
        ),
    })

    assert _SKILLS.MOVE_TO_THROW_POSE.preconditions == frozenset({
        LiftedAtom(predicate=HOLDING, variables=(_SKILLS._robot, _SKILLS._cube))
    })
    assert _SKILLS.MOVE_TO_THROW_POSE.add_effects == frozenset({
        LiftedAtom(
            predicate=ROBOT_AT_SUCCESSFUL_THROW_POSE,
            variables=(_SKILLS._robot, _SKILLS._bin, _SKILLS._goal_region),
        )
    })
    assert _SKILLS.MOVE_TO_THROW_POSE.delete_effects == frozenset()

    assert _SKILLS.TOSS.preconditions == frozenset({
        LiftedAtom(predicate=HOLDING, variables=(_SKILLS._robot, _SKILLS._cube)),
        LiftedAtom(
            predicate=ROBOT_AT_SUCCESSFUL_THROW_POSE,
            variables=(_SKILLS._robot, _SKILLS._bin, _SKILLS._goal_region),
        ),
    })
    assert _SKILLS.TOSS.add_effects == frozenset({
        LiftedAtom(predicate=IN_GOAL_REGION, variables=(_SKILLS._cube, _SKILLS._goal_region)),
        LiftedAtom(predicate=HAND_EMPTY, variables=(_SKILLS._robot,)),
    })
    assert _SKILLS.TOSS.delete_effects == frozenset({
        LiftedAtom(predicate=HOLDING, variables=(_SKILLS._robot, _SKILLS._cube)),
        LiftedAtom(predicate=REACHABLE, variables=(_SKILLS._cube, _SKILLS._barrier)),
    })


def test_no_skill_declares_ignore_effects() -> None:
    """Unlike Ball-Ring's navigations and Tossing Room's Press, nothing here wipes a whole
    predicate: there is one cube, one bin and one region, so every effect is expressible
    as a plain add or delete."""
    for skill in (_SKILLS.PICK, _SKILLS.MOVE_TO_THROW_POSE, _SKILLS.TOSS):
        assert skill.ignore_effects == frozenset(), skill.name


def test_no_variable_carries_the_question_mark_the_pddl_writer_adds() -> None:
    """`PddlWriter.variable_str` (planning/pddl.py) documents the convention explicitly:
    our `Variable.name` is plain and the writer prepends the "?" at write time, because
    predicators' `Variable.name` already carries it and ours deliberately does not. A
    name declared as "?robot" therefore renders as "??robot", which Fast Downward's
    translator splits into two tokens -- and the failure is *silent*, because
    `EesMethod._next_plan` catches `PlanningFailure` and degrades to a no-op. So this is
    checked structurally as well as end-to-end below: a run can otherwise exit 0 and
    write a full `stats.json` in which the method never took a single action."""
    for skill in (_SKILLS.PICK, _SKILLS.MOVE_TO_THROW_POSE, _SKILLS.TOSS):
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
    objects = (env.robot, env.cube, env.bin, env.barrier, env.goal_region)
    predicates = (
        IN_GOAL_REGION,
        HAND_EMPTY,
        HOLDING,
        ON_GROUND,
        REACHABLE,
        ROBOT_AT_SUCCESSFUL_THROW_POSE,
    )
    init_atoms = SkillGrounder.abstract_state(state=state(), objects=objects, predicates=predicates)
    plan = FastDownwardPlanner.plan(
        skills=(_SKILLS.PICK, _SKILLS.MOVE_TO_THROW_POSE, _SKILLS.TOSS),
        predicates=predicates,
        types=(
            env.robot_type,
            env.cube_type,
            env.bin_type,
            env.barrier_type,
            env.goal_region_type,
        ),
        objects=objects,
        init_atoms=init_atoms,
        goal=frozenset({GroundAtom(predicate=IN_GOAL_REGION, objects=(env.cube, env.goal_region))}),
    )
    assert [step.skill.name for step in plan] == ["Pick", "MoveToThrowPose", "Toss"]


def test_param_dims_give_the_throw_a_release_speed_of_its_own() -> None:
    """`Toss` used to have zero parameters, because the only dial this package could
    offer without inventing one was an interpolation between upstream's two arm
    configurations -- and that interpolation would have been ours.

    It is a real dial now, and still not an invented one: upstream's `TossController.
    reset` takes `release_speed` as of `joynroy/kinder-baselines` PR #8, so the throw's
    energy is upstream's own parameter rather than a quantity this package synthesised.
    Both arm configurations remain upstream's, untouched.
    """
    assert _SKILLS.PICK.param_dim == 2
    assert _SKILLS.MOVE_TO_THROW_POSE.param_dim == 1
    assert _SKILLS.TOSS.param_dim == 1


def test_compute_action_encodes_the_skill_id_in_slot_zero() -> None:
    assert Tossing3DSkills.compute_action(
        ground_skill=_pick(), params=np.array([0.55, 0.1]), state=state()
    ) == pytest.approx([Tossing3DEnvironment.pick_id, 0.55, 0.1])
    assert Tossing3DSkills.compute_action(
        ground_skill=_move(), params=np.array([1.35]), state=state()
    ) == pytest.approx([Tossing3DEnvironment.move_to_throw_pose_id, 1.35, 0.0])
    assert Tossing3DSkills.compute_action(
        ground_skill=_toss(), params=np.array([140.0]), state=state()
    ) == pytest.approx([Tossing3DEnvironment.toss_id, 140.0, 0.0])


def test_every_action_matches_the_declared_action_space() -> None:
    for ground_skill, params in (
        (_pick(), np.array([0.55, 0.1])),
        (_move(), np.array([1.35])),
        (_toss(), np.array([140.0])),
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
                state=state(base_x=COINCIDENT_BIN_X - standoff),
                robot=_ENV.robot,
                target=_ENV.bin,
                goal_region=_ENV.goal_region,
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
            state=state(base_x=COINCIDENT_BIN_X - standoff / 1000),
            robot=_ENV.robot,
            target=_ENV.bin,
            goal_region=_ENV.goal_region,
        )
    ]
    assert accepted
    assert max(accepted) - min(accepted) < (high - low) / 2


def test_toss_samples_one_release_speed_inside_the_measured_bounds() -> None:
    """The dial is in **joint-path deg/s**, the same unit every measurement of it is in.

    `TOSS_SPEED_BOUNDS` is `(60, 240)`, the span PR #213's grid actually drove, so a
    draw is always a speed something has been observed at rather than an extrapolation.
    """
    rng = np.random.default_rng(0)
    draws = [
        float(Tossing3DSkills.sample_params(ground_skill=_toss(), rng=rng)[0]) for _ in range(200)
    ]
    assert all(TOSS_SPEED_BOUNDS[0] <= speed <= TOSS_SPEED_BOUNDS[1] for speed in draws)
    # A real draw, not a constant dressed as one: the sampler has to be able to move the
    # dial or `Toss` is back to having no learnable parameter.
    assert max(draws) - min(draws) > (TOSS_SPEED_BOUNDS[1] - TOSS_SPEED_BOUNDS[0]) / 2


def test_the_shipped_default_speed_is_inside_the_bounds_the_sampler_draws_from() -> None:
    """140 deg/s is upstream's own default and the speed every committed Tossing3D
    number was measured at. A sampler whose range excluded it would make the oracle's
    throw unreachable by learning, which is the one thing this dial exists to allow."""
    assert TOSS_SPEED_BOUNDS[0] < UPSTREAM_DEFAULT_RELEASE_SPEED_DEG_S < TOSS_SPEED_BOUNDS[1]


def test_an_unknown_skill_raises_from_both_sampler_and_encoder() -> None:
    stray = GroundSkill(
        skill=_SKILLS.PICK.model_copy(update={"name": "NotASkill"}),
        objects=(_ENV.robot, _ENV.cube, _ENV.barrier, _ENV.bin, _ENV.goal_region),
    )
    with pytest.raises(ValueError, match="Unknown skill"):
        Tossing3DSkills.sample_params(ground_skill=stray, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="Unknown skill"):
        Tossing3DSkills.compute_action(ground_skill=stray, params=np.zeros(2), state=state())
