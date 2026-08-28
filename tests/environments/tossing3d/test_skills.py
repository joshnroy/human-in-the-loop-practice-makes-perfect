"""Offline tests for the two lifted skills, their operator models and their samplers.

**It was three, and the middle one is gone.** `Pick` -> `MoveToThrowPose` -> `Toss`
became `PickCube` -> `MoveToTossLocationAndToss`, because upstream composed the base move
and the throw into one controller and derived the pick's standoff and grasp rotation
internally. Two whole families of test went with them and are not ported:

- **`MoveToThrowPose`'s standoff sampler**, and the pair of properties it existed to
  carry: that a draw could both satisfy and fail `RobotAtSuccessfulThrowPose` (so EES saw
  two classes), and that `THROW_STANDOFF_BOUNDS`' floor cleared
  `WORST_BARRIER_COLLISION_STANDOFF + BARRIER_COLLISION_MARGIN`. There is no standoff
  skill to sample for and no predicate to satisfy; the standoff is now the composed
  toss's first parameter, drawn from upstream's own `TARGET_DISTANCE_BOUNDS`.
- **`Pick`'s `(distance, rotation)` sampler.** `PickCube` has `param_dim=0`.

What replaces both, and is the stronger statement, is that every continuous bound this
module draws from is now *upstream's own*, pinned against the installed controller in
`test_kinder_pin.py` -- so there is one place these numbers are measured and it is not
this repo.
"""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    HAND_EMPTY,
    HOLDING,
    IN_BIN,
    ON_GROUND,
    REACHABLE,
)
from hitl_pmp.environments.tossing3d.skills import (
    MAX_TOSS_ROTATION,
    TOSS_DISTANCE_BOUNDS,
    TOSS_RELEASE_MS_BOUNDS,
    TOSS_ROTATION_BOUNDS,
    TOSS_SPEED_BOUNDS,
    WAYPOINT_TOLERANCE,
    Tossing3DSkills,
)
from hitl_pmp.planning.fast_downward import FastDownwardPlanner
from hitl_pmp.planning.grounding import SkillGrounder

from .observations import INITIAL_ATOMS, state

_ENV = Tossing3DEnvironment()
_SKILLS = Tossing3DSkills

# The four bounds the composed toss draws from, in slot order, so the sampler tests can
# be written once over all four rather than once per dial.
_TOSS_BOUNDS = (
    TOSS_DISTANCE_BOUNDS,
    TOSS_ROTATION_BOUNDS,
    TOSS_SPEED_BOUNDS,
    TOSS_RELEASE_MS_BOUNDS,
)


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
#
# Both orders are upstream's own -- `(robot, cube, barrier)` for the pick and
# `(robot, target, held, barrier)` for the composed toss -- so a ground skill built here
# can be handed to upstream's controller unpermuted.
_EXPECTED_PARAMETERS = {
    "PickCube": ("robot", "cube", "barrier"),
    "MoveToTossLocationAndToss": ("robot", "bin", "cube", "barrier"),
}


def _every_skill() -> tuple:
    return (_SKILLS.PICK_CUBE, _SKILLS.MOVE_TO_TOSS_LOCATION_AND_TOSS)


def _pick_cube() -> GroundSkill:
    return GroundSkill(skill=_SKILLS.PICK_CUBE, objects=(_ENV.robot, _ENV.cube, _ENV.barrier))


def _toss() -> GroundSkill:
    return GroundSkill(
        skill=_SKILLS.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(_ENV.robot, _ENV.bin, _ENV.cube, _ENV.barrier),
    )


def test_the_domain_declares_exactly_two_skills() -> None:
    """The headline of the migration. `MoveToThrowPose` is not a skill any more, and a
    reintroduced third one would mean the operator layer had drifted from the controllers
    `kinder_backend.py` can actually drive."""
    assert [skill.name for skill in _every_skill()] == [
        "PickCube",
        "MoveToTossLocationAndToss",
    ]


def test_no_operator_names_a_goal_region_object() -> None:
    """The headline invariant of the bin-is-the-goal-region simplification.

    Checked by *type* rather than by variable name, so renaming the variable cannot smuggle
    the dependency back in."""
    for skill in _every_skill():
        declared = [variable.type.name for variable in skill.parameters]
        assert "tossing3d_goal_region" not in declared, (
            f"{skill.name} still takes a goal-region parameter: {declared}"
        )


def test_each_operator_declares_exactly_the_objects_it_acts_on() -> None:
    """The positive half of the test above, and the check that both signatures are still
    upstream's own object order (`GroundSkill.objects` is positional, the oracle builds its
    groundings by hand, and `KinderBackend` hands the names to upstream unpermuted)."""
    actual = {
        skill.name: tuple(variable.name for variable in skill.parameters)
        for skill in _every_skill()
    }
    assert actual == _EXPECTED_PARAMETERS


def test_pick_requires_reachable_so_no_plan_retrieves_a_tossed_cube() -> None:
    """The one precondition that encodes the domain's irreversibility. Without it a
    planner emits "toss, then pick it back up and try again", which the dynamics can
    never execute -- exactly the over-permissive-model defect class that
    tests/environments/test_operator_dynamics_fidelity.py exists for."""
    assert (
        LiftedAtom(predicate=REACHABLE, variables=(_SKILLS._cube, _SKILLS._barrier))
        in _SKILLS.PICK_CUBE.preconditions
    )


def test_the_toss_deletes_reachable_unconditionally_hit_or_miss() -> None:
    """A toss makes the cube unreachable whether or not it lands in the region. Deleting
    it only on success would be a model in which a missed throw costs nothing -- which is
    precisely the cost this domain exists to represent."""
    assert (
        LiftedAtom(predicate=REACHABLE, variables=(_SKILLS._cube, _SKILLS._barrier))
        in _SKILLS.MOVE_TO_TOSS_LOCATION_AND_TOSS.delete_effects
    )


def test_the_toss_lands_the_cube_flat_as_well_as_in_the_bin() -> None:
    """**The add effect that forced `OnGround` to become face-interchangeable.**

    kb#113's operator model records that `15/15` scoring throws left the cube resting on a
    face, so the composed toss adds `OnGround`. Under the old "flat on the face it started
    on" classifier that effect would read false on most scoring throws and EES would train
    its sampler against noise -- see `test_predicates.py`'s six-faces tests for the other
    half of this pair."""
    assert (
        LiftedAtom(predicate=ON_GROUND, variables=(_SKILLS._cube,))
        in _SKILLS.MOVE_TO_TOSS_LOCATION_AND_TOSS.add_effects
    )


def test_the_toss_requires_reachable_as_well_as_deleting_it() -> None:
    """An operator whose delete effect names an atom it does not require would be
    describing a state it never established. Upstream keeps this precondition for a
    binding reason the types here already rule out; it stays because it is also true."""
    assert (
        LiftedAtom(predicate=REACHABLE, variables=(_SKILLS._cube, _SKILLS._barrier))
        in _SKILLS.MOVE_TO_TOSS_LOCATION_AND_TOSS.preconditions
    )


def test_the_two_operator_models_are_exactly_as_declared() -> None:
    """Pinned field by field, so a change to the symbolic layer is a deliberate edit to
    this list rather than a silent drift."""
    assert _SKILLS.PICK_CUBE.preconditions == frozenset({
        LiftedAtom(predicate=HAND_EMPTY, variables=(_SKILLS._robot,)),
        LiftedAtom(predicate=ON_GROUND, variables=(_SKILLS._cube,)),
        LiftedAtom(predicate=REACHABLE, variables=(_SKILLS._cube, _SKILLS._barrier)),
    })
    assert _SKILLS.PICK_CUBE.add_effects == frozenset({
        LiftedAtom(predicate=HOLDING, variables=(_SKILLS._robot, _SKILLS._cube))
    })
    assert _SKILLS.PICK_CUBE.delete_effects == frozenset({
        LiftedAtom(predicate=HAND_EMPTY, variables=(_SKILLS._robot,)),
        LiftedAtom(predicate=ON_GROUND, variables=(_SKILLS._cube,)),
    })

    assert _SKILLS.MOVE_TO_TOSS_LOCATION_AND_TOSS.preconditions == frozenset({
        LiftedAtom(predicate=HOLDING, variables=(_SKILLS._robot, _SKILLS._cube)),
        LiftedAtom(predicate=REACHABLE, variables=(_SKILLS._cube, _SKILLS._barrier)),
    })
    assert _SKILLS.MOVE_TO_TOSS_LOCATION_AND_TOSS.add_effects == frozenset({
        LiftedAtom(predicate=HAND_EMPTY, variables=(_SKILLS._robot,)),
        LiftedAtom(predicate=IN_BIN, variables=(_SKILLS._cube, _SKILLS._bin)),
        LiftedAtom(predicate=ON_GROUND, variables=(_SKILLS._cube,)),
    })
    assert _SKILLS.MOVE_TO_TOSS_LOCATION_AND_TOSS.delete_effects == frozenset({
        LiftedAtom(predicate=HOLDING, variables=(_SKILLS._robot, _SKILLS._cube)),
        LiftedAtom(predicate=REACHABLE, variables=(_SKILLS._cube, _SKILLS._barrier)),
    })


def test_no_skill_declares_ignore_effects() -> None:
    """Unlike Ball-Ring's navigations and Tossing Room's Press, nothing here wipes a whole
    predicate: there is one cube, one bin and one barrier, so every effect is expressible
    as a plain add or delete."""
    for skill in _every_skill():
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
    for skill in _every_skill():
        for variable in skill.parameters:
            assert not variable.name.startswith("?"), f"{skill.name}: {variable.name}"


def test_integration_fast_downward_plans_the_two_skill_solve() -> None:
    """An INTEGRATION test, deliberately not skipped, in the style of
    `tests/planning/test_fast_downward.py`: it shells out to a real Fast Downward on this
    domain's own PDDL. It needs no simulator -- the whole symbolic layer is built from a
    hand-written `KinderObservation` -- so it runs on CI, where Fast Downward is present.

    The structural test above would not have caught a second way of emitting unparseable
    PDDL; this one catches any of them, and is the check that was missing when
    `--env tossing3d --method ees` ran to completion planning nothing.

    **Two steps, not three.** That the planner finds the shorter plan at all is the
    end-to-end evidence that the composed operator's preconditions are reachable from the
    initial abstract state without the retired throw-pose predicate in between."""
    env = Tossing3DEnvironment()
    objects = (env.robot, env.cube, env.bin, env.barrier)
    predicates = (IN_BIN, HAND_EMPTY, HOLDING, ON_GROUND, REACHABLE)
    init_atoms = SkillGrounder.abstract_state(
        state=state(abstract_atoms=INITIAL_ATOMS), objects=objects, predicates=predicates
    )
    plan = FastDownwardPlanner.plan(
        skills=_every_skill(),
        predicates=predicates,
        types=(env.robot_type, env.cube_type, env.bin_type, env.barrier_type),
        objects=objects,
        init_atoms=init_atoms,
        goal=frozenset({GroundAtom(predicate=IN_BIN, objects=(env.cube, env.bin))}),
    )
    assert [step.skill.name for step in plan] == ["PickCube", "MoveToTossLocationAndToss"]


def test_the_pick_takes_no_continuous_parameters_at_all() -> None:
    """`PickCubeController` derives its standoff (`PickCubeController.STANDOFF`) and its
    grasp rotation (`upright_grasp_rotations`) internally, so there is nothing for a
    refiner to backtrack over and nothing for EES to learn. This is a real narrowing of
    what a learning run on this domain measures, and it is upstream's choice rather than
    one made here."""
    assert _SKILLS.PICK_CUBE.param_dim == 0


def test_the_composed_toss_carries_all_four_dials() -> None:
    """Standoff and yaw used to belong to `MoveToThrowPose`, speed and millisecond to
    `Toss`. One controller now takes all four, so every learned parameter in this domain
    belongs to this one skill."""
    assert _SKILLS.MOVE_TO_TOSS_LOCATION_AND_TOSS.param_dim == 4


def test_the_rotation_bound_is_computed_from_the_waypoint_tolerance_not_typed() -> None:
    """Upstream derives `MAX_TARGET_ROTATION` from `WAYPOINT_TOLERANCE` and the largest
    standoff -- the widest yaw about the bin that still leaves the base within half the
    tolerance of the bin's axis -- and this module reproduces the derivation rather than
    the number it currently produces. A literal here would go stale silently if upstream
    retuned either input."""
    assert (
        pytest.approx(float(np.arcsin(0.5 * WAYPOINT_TOLERANCE / TOSS_DISTANCE_BOUNDS[1])))
        == MAX_TOSS_ROTATION
    )
    assert TOSS_ROTATION_BOUNDS == (-MAX_TOSS_ROTATION, MAX_TOSS_ROTATION)


def test_compute_action_encodes_the_skill_id_in_slot_zero() -> None:
    assert Tossing3DSkills.compute_action(
        ground_skill=_pick_cube(), params=np.zeros(0), state=state()
    ) == pytest.approx([Tossing3DEnvironment.pick_cube_id, 0.0, 0.0, 0.0, 0.0])
    assert Tossing3DSkills.compute_action(
        ground_skill=_toss(), params=np.array([1.35, 0.01, 140.0, 792.0]), state=state()
    ) == pytest.approx([
        Tossing3DEnvironment.move_to_toss_location_and_toss_id,
        1.35,
        0.01,
        140.0,
        792.0,
    ])


def test_the_toss_parameters_land_in_slots_one_through_four_in_order() -> None:
    """Four dials in four slots is four chances to transpose a pair, and a transposition
    of speed and millisecond typechecks. Encoded from four mutually distinguishable
    values, so any permutation fails."""
    action = Tossing3DSkills.compute_action(
        ground_skill=_toss(), params=np.array([1.31, -0.007, 128.5, 733.0]), state=state()
    )
    assert list(action[1:]) == pytest.approx([1.31, -0.007, 128.5, 733.0])


def test_every_action_matches_the_declared_action_space() -> None:
    for ground_skill, params in (
        (_pick_cube(), np.zeros(0)),
        (_toss(), np.array([1.35, 0.0, 140.0, 792.0])),
    ):
        action = Tossing3DSkills.compute_action(
            ground_skill=ground_skill, params=params, state=state()
        )
        assert action.shape == Tossing3DEnvironment.action_space.shape


def test_sampling_the_pick_returns_an_empty_vector_rather_than_a_zero() -> None:
    """`param_dim=0` has to mean "no parameters", not "one parameter that happens to be
    zero": a length-1 draw would give EES a dial to learn that the controller ignores."""
    drawn = Tossing3DSkills.sample_params(ground_skill=_pick_cube(), rng=np.random.default_rng(0))
    assert drawn.shape == (0,)


@pytest.mark.parametrize("slot", range(4))
def test_every_toss_dial_is_drawn_across_its_own_bounds(*, slot: int) -> None:
    """Each dial in bounds, and each one a real draw rather than a constant dressed as
    one -- a sampler that returned a bound's midpoint in some slot would pass a
    containment-only check on every draw."""
    rng = np.random.default_rng(0)
    draws = [
        float(Tossing3DSkills.sample_params(ground_skill=_toss(), rng=rng)[slot])
        for _ in range(200)
    ]
    low, high = _TOSS_BOUNDS[slot]
    assert all(low <= value <= high for value in draws)
    assert max(draws) - min(draws) > (high - low) / 2


def test_the_four_toss_dials_are_drawn_independently() -> None:
    """A sampler that wrote one draw into several slots, or derived one from another,
    would pass every single-slot test above while collapsing the space onto a line or a
    plane. Pinned as near-zero rank correlation between all six pairs."""
    rng = np.random.default_rng(0)
    draws = np.array([
        Tossing3DSkills.sample_params(ground_skill=_toss(), rng=rng) for _ in range(500)
    ])
    ranks = np.argsort(np.argsort(draws, axis=0), axis=0)
    correlations = np.corrcoef(ranks, rowvar=False)
    off_diagonal = correlations[~np.eye(4, dtype=bool)]
    assert np.max(np.abs(off_diagonal)) < 0.15


def test_an_unknown_skill_raises_from_both_sampler_and_encoder() -> None:
    stray = GroundSkill(
        skill=_SKILLS.PICK_CUBE.model_copy(update={"name": "NotASkill"}),
        objects=(_ENV.robot, _ENV.cube, _ENV.barrier),
    )
    with pytest.raises(ValueError, match="Unknown skill"):
        Tossing3DSkills.sample_params(ground_skill=stray, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="Unknown skill"):
        Tossing3DSkills.compute_action(ground_skill=stray, params=np.zeros(4), state=state())


def test_same_side_plans_bin_retrieval_as_a_throw_prerequisite() -> None:
    from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
    from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider

    env = Tossing3DEnvironment(layout=Tossing3DLayout.SAME_SIDE)
    provider = Tossing3DSkillProvider(env=env)
    skills = {skill.name: skill for skill in provider.skills()}
    assert "PickCubeFromBin" in skills
    retrieval = GroundSkill(
        skill=skills["PickCubeFromBin"], objects=(env.robot, env.cube, env.bin, env.barrier)
    )
    toss = GroundSkill(
        skill=skills["MoveToTossLocationAndToss"],
        objects=(env.robot, env.bin, env.cube, env.barrier),
    )
    in_bin = GroundAtom(predicate=IN_BIN, objects=(env.cube, env.bin))
    holding = GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube))
    reachable = GroundAtom(predicate=REACHABLE, objects=(env.cube, env.barrier))
    assert in_bin in retrieval.preconditions
    assert in_bin in retrieval.delete_effects
    assert holding in retrieval.add_effects
    assert reachable not in toss.delete_effects
    assert reachable in toss.add_effects


@pytest.mark.parametrize(
    "inside,closed", [(False, False), (True, False), (False, True), (True, True)]
)
def test_same_side_planner_recovers_from_each_landing(*, inside: bool, closed: bool) -> None:
    from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
    from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider

    env = Tossing3DEnvironment(layout=Tossing3DLayout.SAME_SIDE)
    provider = Tossing3DSkillProvider(env=env)
    atoms = {("OnGround", ("cube_0",)), ("MovableIsDownX", ("cube_0", "cuboid_barrier"))}
    if inside:
        atoms.add(("MovableInGoalRegion", ("cube_0",)))
    if not closed:
        atoms.add(("HandEmpty", ("robot",)))
    observed = state(env=env, abstract_atoms=frozenset(atoms))
    plan = FastDownwardPlanner.plan(
        skills=provider.skills(),
        predicates=provider.predicates(),
        types=provider.types(),
        objects=provider.objects(),
        init_atoms=SkillGrounder.abstract_state(
            state=observed, objects=provider.objects(), predicates=provider.predicates()
        ),
        goal=frozenset({GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube))}),
    )
    expected = ["OpenGripper"] if closed else []
    expected.append("PickCubeFromBin" if inside else "PickCubeFromFloor")
    assert [step.skill.name for step in plan] == expected
    for step in plan:
        params = provider.sample_params(ground_skill=step, rng=np.random.default_rng(0))
        action = provider.compute_action(ground_skill=step, params=params, state=observed)
        assert action.shape == (5,)
        assert (
            action[0]
            == {"OpenGripper": 2, "PickCubeFromFloor": 0, "PickCubeFromBin": 3}[step.skill.name]
        )


def test_ees_implicitly_retrieves_after_hits_and_misses() -> None:
    """Replay observed atom states through real EES, without injecting a plan/target."""
    from hitl_pmp.core.problem.tasks.types import Goal, Task
    from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
    from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
    from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

    from .observations import HOLDING_ATOMS

    env = Tossing3DEnvironment(layout=Tossing3DLayout.SAME_SIDE)
    provider = Tossing3DSkillProvider(env=env)
    method = EesMethod(env=env, skill_provider=provider, seed=0, goal_pursuit_horizon=2)
    floor = state(env=env, abstract_atoms=INITIAL_ATOMS)
    holding = state(env=env, abstract_atoms=HOLDING_ATOMS)
    inside = state(env=env, abstract_atoms=INITIAL_ATOMS | {("MovableInGoalRegion", ("cube_0",))})
    task = Task(
        initial_state=floor,
        goal=Goal(atoms=frozenset({GroundAtom(predicate=IN_BIN, objects=(env.cube, env.bin))})),
    )
    policy = method.get_practice_policy(task=task)
    actions = [policy(observed) for observed in (floor, holding, inside, holding, floor)]
    assert [int(action.action[0]) for action in actions] == [0, 1, 3, 1, 0]
    outcomes = method.practice_outcomes()
    assert outcomes["PickCubeFromBin"].num_successes == 1
    assert outcomes["MoveToTossLocationAndToss"].num_attempts == 2
    assert outcomes["MoveToTossLocationAndToss"].num_successes == 1


@pytest.mark.parametrize("yaw", [0.0, 0.7, 2.1])
def test_rim_support_uses_bin_frame_and_rejects_non_support(*, yaw: float) -> None:
    from scipy.spatial.transform import Rotation

    from hitl_pmp.environments.tossing3d.rim_geometry import RimGeometry

    rotation = Rotation.from_euler("z", yaw)
    quaternion = dict(zip(("qx", "qy", "qz", "qw"), rotation.as_quat(), strict=True))
    bin_ = dict(
        x=1.0, y=-0.5, z=0.0, bb_x=0.3, bb_y=0.3, bb_z=0.2, vx=0.0, vy=0.0, vz=0.0, **quaternion
    )
    xyz = rotation.apply([0.14, 0.0, 0.225]) + [1.0, -0.5, 0.0]
    cube = dict(zip(("x", "y", "z"), xyz, strict=True)) | dict(
        bb_x=0.05, bb_y=0.05, bb_z=0.05, vx=0.0, vy=0.0, vz=0.0, **quaternion
    )
    assert RimGeometry.supported(cube=cube, bin_=bin_, wall_thickness=0.01)
    for changes in (
        {"z": 0.3},
        {"z": 0.025},
        {"x": 3.0},
        {"vz": 1.0},
        {"x": bin_["x"], "y": bin_["y"]},
    ):
        assert not RimGeometry.supported(cube=cube | changes, bin_=bin_, wall_thickness=0.01)
    tipped = dict(
        zip(("qx", "qy", "qz", "qw"), Rotation.from_euler("x", np.pi).as_quat(), strict=True)
    )
    assert not RimGeometry.supported(cube=cube, bin_=bin_ | tipped, wall_thickness=0.01)
