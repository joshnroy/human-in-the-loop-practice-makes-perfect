"""Simulator-backed tests for the two lifted skills, their operator models and samplers.

## What changed, and why most of the old file is gone

This used to test **three** skills -- `Pick`, `MoveToThrowPose(standoff)`,
`Toss(speed, ms)` -- over six predicates, with nine tests pinning this repo's own sampling
bounds. Upstream fuses the base move and the throw into one controller, *"so that no
predicate has to name the pose between them"*, and the domain now imports upstream's own
two-operator model.

Every test that pinned a `*_BOUNDS` constant is deleted, because the constants are. That
is the substantive fix rather than a tidy-up: hitl declared `TOSS_RELEASE_MS_BOUNDS =
(300, 1400)` beside a controller whose own measured band is `(700, 840)`, so it drew from
a window about nine times too wide and the large majority of its draws could not score.
Nothing detected it, because both numbers were internally consistent -- there was simply
no mechanism by which upstream narrowing its band could narrow hitl's. `sample_params`
now delegates to the controller's own `sample_parameters`, so there is no second number to
keep in step and nothing here to re-assert.

## Why a simulator

`Tossing3DSkills.all` builds `LiftedAtom`s over `core.Predicate`s that close over one live
scene's abstraction, so there is no skill to inspect without a scene. See `conftest.py`.
"""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    HAND_EMPTY,
    HOLDING,
    MOVABLE_IN_GOAL_REGION,
    MOVABLE_IS_DOWN_X,
    ON_GROUND,
    Tossing3DPredicates,
)
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import (
    MOVE_TO_TOSS_LOCATION_AND_TOSS,
    PARAM_DIMS,
    PICK_CUBE,
    SKILL_NAMES,
    Tossing3DSkills,
)
from hitl_pmp.planning.fast_downward import FastDownwardPlanner
from hitl_pmp.planning.grounding import SkillGrounder

from .conftest import requires_kinder

pytestmark = requires_kinder

# The exact lifted signature of each operator, in declaration order. Pinned as a literal
# rather than derived, because the thing under test is precisely *which* objects an
# operator is allowed to name -- a derivation would restate whatever `skills.py` happens
# to declare and could never fail.
_EXPECTED_PARAMETERS = {
    PICK_CUBE: ("robot", "cube", "barrier"),
    MOVE_TO_TOSS_LOCATION_AND_TOSS: ("robot", "held", "barrier"),
}


def _skills(*, env: Tossing3DEnvironment) -> dict[str, object]:
    return {skill.name: skill for skill in Tossing3DSkills.all(abstraction=env.abstraction())}


def _atom(*, env: Tossing3DEnvironment, name: str, variables: tuple) -> LiftedAtom:
    return LiftedAtom(
        predicate=Tossing3DPredicates.get(abstraction=env.abstraction(), name=name),
        variables=variables,
    )


def _pick(*, env: Tossing3DEnvironment) -> GroundSkill:
    return GroundSkill(
        skill=Tossing3DSkills.pick_cube(abstraction=env.abstraction()),
        objects=(env.robot, env.cube, env.barrier),
    )


def _toss(*, env: Tossing3DEnvironment) -> GroundSkill:
    return GroundSkill(
        skill=Tossing3DSkills.move_to_toss_location_and_toss(abstraction=env.abstraction()),
        objects=(env.robot, env.cube, env.barrier),
    )


def test_the_domain_declares_exactly_two_skills(*, live_env: Tossing3DEnvironment) -> None:
    """Three became two when upstream fused the base move and the throw. A third
    reappearing would mean this repo had started re-splitting upstream's controllers."""
    assert SKILL_NAMES == (PICK_CUBE, MOVE_TO_TOSS_LOCATION_AND_TOSS)
    assert tuple(_skills(env=live_env)) == SKILL_NAMES


def test_each_operator_declares_exactly_the_objects_it_acts_on(
    *, live_env: Tossing3DEnvironment
) -> None:
    """`GroundSkill.objects` is positional and the oracle builds its groundings by hand,
    so both the membership and the order are part of the interface."""
    actual = {
        name: tuple(variable.name for variable in skill.parameters)
        for name, skill in _skills(env=live_env).items()
    }
    assert actual == _EXPECTED_PARAMETERS


def test_no_operator_names_an_object_outside_the_two_declared_types(
    *, live_env: Tossing3DEnvironment
) -> None:
    """The goal region stopped being a symbolic object when `MovableInGoalRegion` became a
    unary predicate reading the live simulator. Checked by *type* rather than by variable
    name, so renaming a variable cannot smuggle a dependency back in."""
    allowed = {Tossing3DEnvironment.robot_type.name, Tossing3DEnvironment.movable_type.name}
    for name, skill in _skills(env=live_env).items():
        declared = {variable.type.name for variable in skill.parameters}
        assert declared <= allowed, f"{name} binds a type outside the domain's two: {declared}"


def test_pick_requires_the_cube_is_still_on_the_robots_side_of_the_barrier(
    *, live_env: Tossing3DEnvironment
) -> None:
    """The one precondition that encodes the domain's irreversibility. Without it a
    planner emits "toss, then pick it back up and try again", which the dynamics can never
    execute -- exactly the over-permissive-model defect class that
    `tests/environments/test_operator_dynamics_fidelity.py` exists for."""
    skill = Tossing3DSkills.pick_cube(abstraction=live_env.abstraction())
    cube, barrier = Tossing3DSkills.cube, Tossing3DSkills.barrier
    assert (
        _atom(env=live_env, name=MOVABLE_IS_DOWN_X, variables=(cube, barrier))
        in skill.preconditions
    )


def test_the_toss_deletes_that_precondition_unconditionally_hit_or_miss(
    *, live_env: Tossing3DEnvironment
) -> None:
    """The one-way door. A toss makes the cube unreachable whether or not it lands in the
    region; deleting the atom only on success would be a model in which a missed throw
    costs nothing -- which is precisely the cost this domain exists to represent."""
    skill = Tossing3DSkills.move_to_toss_location_and_toss(abstraction=live_env.abstraction())
    held, barrier = Tossing3DSkills.held, Tossing3DSkills.barrier
    assert (
        _atom(env=live_env, name=MOVABLE_IS_DOWN_X, variables=(held, barrier))
        in skill.delete_effects
    )


def test_the_two_operator_models_are_exactly_as_declared(*, live_env: Tossing3DEnvironment) -> None:
    """Pinned field by field against upstream's own bilevel model, so a change to the
    symbolic layer is a deliberate edit to this list rather than a silent drift."""
    env = live_env
    robot, cube, held, barrier = (
        Tossing3DSkills.robot,
        Tossing3DSkills.cube,
        Tossing3DSkills.held,
        Tossing3DSkills.barrier,
    )
    pick = Tossing3DSkills.pick_cube(abstraction=env.abstraction())
    toss = Tossing3DSkills.move_to_toss_location_and_toss(abstraction=env.abstraction())

    assert pick.preconditions == frozenset({
        _atom(env=env, name=HAND_EMPTY, variables=(robot,)),
        _atom(env=env, name=ON_GROUND, variables=(cube,)),
        _atom(env=env, name=MOVABLE_IS_DOWN_X, variables=(cube, barrier)),
    })
    assert pick.add_effects == frozenset({
        _atom(env=env, name=HOLDING, variables=(robot, cube)),
    })
    assert pick.delete_effects == frozenset({
        _atom(env=env, name=HAND_EMPTY, variables=(robot,)),
        _atom(env=env, name=ON_GROUND, variables=(cube,)),
    })

    assert toss.preconditions == frozenset({
        _atom(env=env, name=HOLDING, variables=(robot, held)),
        _atom(env=env, name=MOVABLE_IS_DOWN_X, variables=(held, barrier)),
    })
    assert toss.add_effects == frozenset({
        _atom(env=env, name=HAND_EMPTY, variables=(robot,)),
        _atom(env=env, name=MOVABLE_IN_GOAL_REGION, variables=(held,)),
        _atom(env=env, name=ON_GROUND, variables=(held,)),
    })
    assert toss.delete_effects == frozenset({
        _atom(env=env, name=HOLDING, variables=(robot, held)),
        _atom(env=env, name=MOVABLE_IS_DOWN_X, variables=(held, barrier)),
    })


def test_no_skill_declares_ignore_effects(*, live_env: Tossing3DEnvironment) -> None:
    """Unlike Ball-Ring's navigations and Tossing Room's Press, nothing here wipes a whole
    predicate: there is one cube, one bin and one barrier, so every effect is expressible
    as a plain add or delete."""
    for name, skill in _skills(env=live_env).items():
        assert skill.ignore_effects == frozenset(), name


def test_no_variable_carries_the_question_mark_the_pddl_writer_adds(
    *, live_env: Tossing3DEnvironment
) -> None:
    """`PddlWriter._variable_str` documents the convention explicitly: our `Variable.name`
    is plain and the writer prepends the "?" at write time, because predicators'
    `Variable.name` already carries it and ours deliberately does not. A name declared as
    "?robot" therefore renders as "??robot", which Fast Downward's translator splits into
    two tokens -- and the failure is **silent**, because `EesMethod._next_plan` catches
    `PlanningFailure` and degrades to a no-op, so a run exits 0 and writes a full
    `stats.json` in which the method never took a single action.

    This matters more since the skills started coming from upstream: KINDER follows PDDL
    convention and names its variables `?robot`, so the sigil is now genuinely present one
    layer down and `KinderControllers.variables` strips it on the way across."""
    for name, skill in _skills(env=live_env).items():
        for variable in skill.parameters:
            assert not variable.name.startswith("?"), f"{name}: {variable.name}"


def test_the_controllers_own_variables_arrive_without_the_sigil(
    *, live_env: Tossing3DEnvironment
) -> None:
    """The other half of the test above, one layer down: upstream's own variable names
    really do carry the `?`, so the strip in `KinderControllers.variables` is load-bearing
    rather than defensive. If upstream ever dropped the sigil this would still pass; if
    the strip were removed, this fails while the structural test above also fails."""
    controllers = live_env.controllers()
    for key in SKILL_NAMES:
        for variable in controllers.variables(key=key):
            assert not variable.name.startswith("?"), f"{key}: {variable.name}"


def test_the_declared_param_dims_match_the_controllers_own_samplers(
    *, live_env: Tossing3DEnvironment
) -> None:
    """**`params_space` is `None` on every Tossing3D controller**, so a domain cannot read
    a controller's parameter arity off it and `skills.py` has to declare `PARAM_DIMS` by
    hand. A hand-declared number is exactly the kind of second copy this refactor set out
    to remove, so it is checked against what the controller actually draws.

    Both halves are asserted: that `params_space` really is `None` (otherwise the
    declaration should be deleted in favour of reading it), and that each declared arity
    equals the width of a real draw."""
    controllers = live_env.controllers()
    state = live_env.get_current_state()
    rng = np.random.default_rng(0)

    for key in SKILL_NAMES:
        controller = controllers.lifted_controllers[key]
        assert controller.params_space is None, (
            f"{key} now declares a params_space; read the arity off it instead of "
            "hand-declaring PARAM_DIMS"
        )
        drawn = controllers.sample_params(
            key=key,
            object_names=("robot", "cube_0", "cuboid_barrier"),
            state=state,
            rng=rng,
        )
        assert drawn.size == PARAM_DIMS[key], key


def test_the_throw_draws_four_parameters_where_the_pick_draws_two(
    *, live_env: Tossing3DEnvironment
) -> None:
    """The arity is the visible trace of the fusion: standoff and rotation for the base
    move, plus the two toss dials, all drawn in one `sample_parameters` call. Holding the
    standoff fixed while sweeping a dial is no longer expressible, which is why the sweep
    drivers that did so were deleted rather than adapted."""
    assert PARAM_DIMS[PICK_CUBE] == 2
    assert PARAM_DIMS[MOVE_TO_TOSS_LOCATION_AND_TOSS] == 4
    del live_env


def test_sampling_delegates_to_the_controller_rather_than_a_local_range(
    *, live_env: Tossing3DEnvironment
) -> None:
    """The whole point of the change. Asserted as reproducibility against the controller's
    own sampler at the same seed: a local uniform over invented bounds could not match it
    except by coincidence."""
    ground_skill = _toss(env=live_env)
    state = live_env.get_current_state()
    through_skills = Tossing3DSkills.sample_params(
        ground_skill=ground_skill,
        rng=np.random.default_rng(7),
        controllers=live_env.controllers(),
        state=state,
    )
    direct = live_env.controllers().sample_params(
        key=MOVE_TO_TOSS_LOCATION_AND_TOSS,
        object_names=("robot", "cube_0", "cuboid_barrier"),
        state=state,
        rng=np.random.default_rng(7),
    )
    assert through_skills == pytest.approx(direct)


def test_pick_sampling_is_state_independent_on_the_single_cube_scene(
    *, live_env: Tossing3DEnvironment
) -> None:
    """`Tossing3DSkills.sample_params` deviates from `SkillProvider`'s state-*independent*
    contract on purpose, because `PickCubeController.sample_parameters` reads the target's
    pose and rejection-tests the base pose against other cubes.

    On the shipped `o1` scene that deviation has no practical effect: the rejection loop
    looks for other objects whose name contains "cube", and there is exactly one, so it
    accepts its first draw. Pinned here so that a scene gaining a second cube surfaces the
    deviation rather than silently changing what the oracle draws."""
    ground_skill = _pick(env=live_env)
    first = Tossing3DSkills.sample_params(
        ground_skill=ground_skill,
        rng=np.random.default_rng(11),
        controllers=live_env.controllers(),
        state=live_env.get_current_state(),
    )

    # A state whose cube has moved a long way. If the draw depended on it, this differs.
    moved = live_env.get_current_state().copy()
    moved.set(obj=live_env.cube, feature_name="x", feature_val=1.9)
    second = Tossing3DSkills.sample_params(
        ground_skill=ground_skill,
        rng=np.random.default_rng(11),
        controllers=live_env.controllers(),
        state=moved,
    )
    assert first == pytest.approx(second)


def test_compute_action_encodes_the_skill_id_in_slot_zero(
    *, live_env: Tossing3DEnvironment
) -> None:
    state = live_env.get_current_state()
    pick_action = Tossing3DSkills.compute_action(
        ground_skill=_pick(env=live_env), params=np.array([0.55, 0.1]), state=state
    )
    assert pick_action == pytest.approx([
        Tossing3DEnvironment.pick_cube_id,
        0.55,
        0.1,
        0.0,
        0.0,
    ])
    toss_action = Tossing3DSkills.compute_action(
        ground_skill=_toss(env=live_env),
        params=np.array([1.35, 0.0, 140.0, 720.0]),
        state=state,
    )
    assert toss_action == pytest.approx([
        Tossing3DEnvironment.move_to_toss_location_and_toss_id,
        1.35,
        0.0,
        140.0,
        720.0,
    ])


def test_every_action_matches_the_declared_action_space(*, live_env: Tossing3DEnvironment) -> None:
    """Five slots now rather than three: one skill id and four parameters, because the
    fused throw draws four. `pick_cube` uses two and the rest are zero-padded."""
    state = live_env.get_current_state()
    for ground_skill, params in (
        (_pick(env=live_env), np.array([0.55, 0.1])),
        (_toss(env=live_env), np.array([1.35, 0.0, 140.0, 720.0])),
    ):
        action = Tossing3DSkills.compute_action(
            ground_skill=ground_skill, params=params, state=state
        )
        assert action.shape == Tossing3DEnvironment.action_space.shape


def test_an_unknown_skill_raises_from_the_encoder(*, live_env: Tossing3DEnvironment) -> None:
    """A skill name this domain does not encode must be loud: `compute_action` picks the
    slot-zero id off the name, so an unrecognised one would otherwise silently become
    whichever branch the `if` falls through to."""
    real = Tossing3DSkills.pick_cube(abstraction=live_env.abstraction())
    stray = GroundSkill(
        skill=real.model_copy(update={"name": "NotASkill"}),
        objects=(live_env.robot, live_env.cube, live_env.barrier),
    )
    with pytest.raises(ValueError, match="Unknown skill"):
        Tossing3DSkills.compute_action(
            ground_skill=stray, params=np.zeros(2), state=live_env.get_current_state()
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Fast Downward reports the task provably unsolvable because OnGround grounds onto "
        "nothing (typed over upstream's base MujocoObjectType, which no o1 object carries), "
        "so no pick_cube grounding is ever applicable. Strict, so this fails loudly (XPASS) "
        "the moment the typing is fixed. See test_predicates.py's OnGround test."
    ),
)
def test_integration_fast_downward_plans_the_two_skill_solve(
    *, live_env: Tossing3DEnvironment
) -> None:
    """An INTEGRATION test, deliberately not skipped, in the style of
    `tests/planning/test_fast_downward.py`: it shells out to a real Fast Downward on this
    domain's own PDDL.

    The structural `??robot` test would not have caught a second way of emitting
    unparseable PDDL; this one catches any of them, and is the check that was missing when
    `--env tossing3d --method ees` ran to completion planning nothing.

    **This currently fails, and the failure is a real defect rather than a stale
    expectation** -- see `test_predicates.py`'s
    `test_on_ground_grounds_onto_at_least_one_object_in_this_scene`. `OnGround` is typed
    over upstream's base `mujoco_object`, no object in this scene carries that exact type,
    `SkillGrounder` matches types by equality, so the atom never grounds, `pick_cube` is
    never applicable and Fast Downward reports the task provably unsolvable."""
    provider = Tossing3DSkillProvider(env=live_env)
    state = live_env.get_current_state()
    goal_predicate = Tossing3DPredicates.get(
        abstraction=live_env.abstraction(), name=MOVABLE_IN_GOAL_REGION
    )
    init_atoms = SkillGrounder.abstract_state(
        state=state, objects=provider.objects(), predicates=provider.predicates()
    )
    plan = FastDownwardPlanner.plan(
        skills=provider.skills(),
        predicates=provider.predicates(),
        types=provider.types(),
        objects=provider.objects(),
        init_atoms=init_atoms,
        goal=frozenset({goal_predicate(state=state, objects=(live_env.cube,))}),
    )
    assert [step.skill.name for step in plan] == [PICK_CUBE, MOVE_TO_TOSS_LOCATION_AND_TOSS]
