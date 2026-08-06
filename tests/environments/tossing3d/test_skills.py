"""Offline tests for the three lifted skills, their operator models and their samplers."""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    HAND_EMPTY,
    HOLDING,
    IN_GOAL_REGION,
    NEAR_BIN,
    ON_GROUND,
    REACHABLE,
    THROW_SOLVING_BAND,
    NearBinClassifier,
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
    return GroundSkill(skill=_SKILLS.PICK, objects=(_ENV.robot, _ENV.cube, _ENV.barrier, _ENV.bin))


def _move() -> GroundSkill:
    return GroundSkill(skill=_SKILLS.MOVE_TO_THROW_POSE, objects=(_ENV.robot, _ENV.cube, _ENV.bin))


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


def test_pick_deletes_near_bin_because_the_grasp_drives_the_base_to_the_cube() -> None:
    """`pick_shelf` navigates to the cube, which is on the near side of the barrier and
    therefore nowhere near the bin. A model that let `NearBin` survive a pick would let
    the planner skip `MoveToThrowPose` entirely."""
    assert (
        LiftedAtom(predicate=NEAR_BIN, variables=(_SKILLS._robot, _SKILLS._bin))
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
        LiftedAtom(predicate=NEAR_BIN, variables=(_SKILLS._robot, _SKILLS._bin)),
    })

    assert _SKILLS.MOVE_TO_THROW_POSE.preconditions == frozenset({
        LiftedAtom(predicate=HOLDING, variables=(_SKILLS._robot, _SKILLS._cube))
    })
    assert _SKILLS.MOVE_TO_THROW_POSE.add_effects == frozenset({
        LiftedAtom(predicate=NEAR_BIN, variables=(_SKILLS._robot, _SKILLS._bin))
    })
    assert _SKILLS.MOVE_TO_THROW_POSE.delete_effects == frozenset()

    assert _SKILLS.TOSS.preconditions == frozenset({
        LiftedAtom(predicate=HOLDING, variables=(_SKILLS._robot, _SKILLS._cube)),
        LiftedAtom(predicate=NEAR_BIN, variables=(_SKILLS._robot, _SKILLS._bin)),
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
    predicates = (IN_GOAL_REGION, HAND_EMPTY, HOLDING, ON_GROUND, REACHABLE, NEAR_BIN)
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


def test_param_dims_put_the_only_learnable_dial_on_the_walk_not_the_throw() -> None:
    """`Toss` has zero parameters on purpose: both arm configurations are upstream's own
    and this package interpolates nothing. The standoff is the dial the coincident
    scene's own sweep actually resolves, so that is where the parameter lives."""
    assert _SKILLS.PICK.param_dim == 2
    assert _SKILLS.MOVE_TO_THROW_POSE.param_dim == 1
    assert _SKILLS.TOSS.param_dim == 0


def test_compute_action_encodes_the_skill_id_in_slot_zero() -> None:
    assert Tossing3DSkills.compute_action(
        ground_skill=_pick(), params=np.array([0.55, 0.1]), state=state()
    ) == pytest.approx([Tossing3DEnvironment.pick_id, 0.55, 0.1])
    assert Tossing3DSkills.compute_action(
        ground_skill=_move(), params=np.array([1.35]), state=state()
    ) == pytest.approx([Tossing3DEnvironment.move_to_throw_pose_id, 1.35, 0.0])
    assert Tossing3DSkills.compute_action(
        ground_skill=_toss(), params=np.zeros(0), state=state()
    ) == pytest.approx([Tossing3DEnvironment.toss_id, 0.0, 0.0])


def test_every_action_matches_the_declared_action_space() -> None:
    for ground_skill, params in (
        (_pick(), np.array([0.55, 0.1])),
        (_move(), np.array([1.35])),
        (_toss(), np.zeros(0)),
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


def test_every_sampled_throw_standoff_satisfies_near_bin() -> None:
    """A sampler that could draw a standoff at which its own skill's add effect is false
    would hand a planner an operator that provably cannot achieve what it claims. Checked
    against the predicate itself, not against the constants, so widening either one
    without the other fails here."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        (standoff,) = Tossing3DSkills.sample_params(ground_skill=_move(), rng=rng)
        assert THROW_STANDOFF_BOUNDS[0] <= standoff <= THROW_STANDOFF_BOUNDS[1]
        assert NearBinClassifier.holds(
            state=state(base_x=COINCIDENT_BIN_X - standoff),
            robot=_ENV.robot,
            target=_ENV.bin,
        )


def test_the_sampler_range_is_the_measured_feasible_range_and_is_shared_with_the_predicate() -> (
    None
):
    """Both endpoints are measured, and neither is the endpoint of a solving band.

    The feasible range is `[0.40, 2.06]`: below 0.40 m the base drives into the bin and
    shoves it across the floor rather than failing to reach it, and above 2.06 m the
    `Toss` windup stops being motion-plannable. The bounds inset that at both ends -- at
    the bottom by `NEAR_BIN_TOLERANCE`, so the predicate never admits a pose inside the
    collision regime, and at the top to 1.75 so the widened `NearBin` does not start
    admitting the pose `Pick` leaves the base in, whereupon the oracle would skip
    `MoveToThrowPose` and throw from wherever it stood. Every number here, and the seeds
    it was measured over, is in `predicates.THROW_STANDOFF_BOUNDS`'s own comment.

    The identity check is the important half: `skills.py` imports this interval from
    `predicates.py` rather than declaring its own, so "what the sampler draws" and "what
    NearBin admits" cannot drift apart. They were briefly two separate constants, and that
    gap is what let an over-permissive NearBin ship."""
    assert THROW_STANDOFF_BOUNDS == (0.45, 1.75)
    assert THROW_STANDOFF_BOUNDS is PREDICATE_THROW_STANDOFF_BOUNDS


def test_the_solving_band_is_a_small_fraction_of_the_range_the_sampler_draws_from() -> None:
    """The whole point of the bounds: a uniform draw has to be wrong most of the time.

    The standoff is effectively a *constant* here -- the bin comes from a 1 mm-wide
    region, so the correct answer is the same every episode -- and a constant is only a
    learning problem if the prior does not already find it. The bounds were previously
    `(1.20, 1.65)`, barely wider than the solving band; pooled over that range the oracle
    solved 155/330, so a learned sampler had almost no headroom to beat its own prior.

    `THROW_SOLVING_BAND` is the 5/5-seeds core, so this ratio deliberately ignores the
    soft edges (2/5 at 1.125, 3/5 at 1.400, 2/5 at 1.425) and is therefore an
    *under*-estimate of how often a uniform draw succeeds. The threshold is set with that
    slack in mind rather than tuned to the current numbers.

    Pinned as a ratio rather than as two more literals, because it is the ratio that
    carries the design intent: narrowing the bounds back toward the band, or widening the
    band, both make the domain unable to show sampler learning and both fail here."""
    band_low, band_high = THROW_SOLVING_BAND
    range_low, range_high = THROW_STANDOFF_BOUNDS
    assert range_low <= band_low < band_high <= range_high
    fraction = (band_high - band_low) / (range_high - range_low)
    assert fraction <= 0.20


def test_toss_samples_no_parameters_at_all() -> None:
    assert Tossing3DSkills.sample_params(
        ground_skill=_toss(), rng=np.random.default_rng(0)
    ).shape == (0,)


def test_an_unknown_skill_raises_from_both_sampler_and_encoder() -> None:
    stray = GroundSkill(
        skill=_SKILLS.PICK.model_copy(update={"name": "NotASkill"}),
        objects=(_ENV.robot, _ENV.cube, _ENV.barrier, _ENV.bin),
    )
    with pytest.raises(ValueError, match="Unknown skill"):
        Tossing3DSkills.sample_params(ground_skill=stray, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="Unknown skill"):
        Tossing3DSkills.compute_action(ground_skill=stray, params=np.zeros(2), state=state())
