"""Simulator-backed tests for Tossing3D's five predicates.

## Why this file is a tenth of its former size

It used to hold 35 tests over six hand-written classifiers -- boundary probes on `InBin`,
threshold checks on `HandEmpty`/`Holding`/`OnGround`, a one-way-door check on `Reachable`,
and eighteen tests pinning `RobotAtSuccessfulThrowPose`'s calibrated acceptance band. None
of those subjects exists any more:

* **`RobotAtSuccessfulThrowPose` is deleted outright.** Upstream fuses the base move and
  the throw into one controller *"so that no predicate has to name the pose between
  them"*, so there is no intermediate state left to characterise and no band to calibrate.
* **`InBin` -> `MovableInGoalRegion`**, **`Reachable` -> `MovableIsDownX`**, and
  `HandEmpty`/`Holding`/`OnGround` keep their names but are now upstream's
  implementations, from `kinder_models.dynamic3d.tossing.state_abstractions`.

So the classifiers this file used to test are gone, and re-testing upstream's internals
here would rebuild exactly the duplication the refactor removed -- six classifiers "kept
in agreement with upstream's by test rather than by construction". What is genuinely
this domain's to state is much smaller, and is all that remains below: **which** five
predicates the operators are written over, in what order, over which types, and that they
really do evaluate against a live scene.

## Why every test here needs a simulator

The predicates are no longer pure functions of a `core.State`. `Holding` runs forward
kinematics off the arm joints and `MovableInGoalRegion` reads the goal region off the live
simulator through the same `check_in_region` call `_check_goals()` makes. There is no dict
of floats that can answer them, which is why the offline `observations.py` fixture is gone
rather than adapted. See `conftest.py` for the shared scene and what it costs.
"""

import pytest

from hitl_pmp.core.problem.tasks.types import Predicate
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    HAND_EMPTY,
    HOLDING,
    MOVABLE_IN_GOAL_REGION,
    MOVABLE_IS_DOWN_X,
    ON_GROUND,
    PREDICATE_NAMES,
    Tossing3DPredicates,
)

from .conftest import requires_kinder

pytestmark = requires_kinder


def test_the_domain_declares_exactly_the_five_predicates_upstream_reports() -> None:
    """The names are written out in `predicates.py` rather than imported, so that the
    module can say what the domain is about without starting MuJoCo. That duplication is
    deliberate and this is what keeps it honest: upstream's abstractor reporting a sixth
    predicate must be a decision someone takes, not a fact the symbolic layer silently
    drops."""
    assert PREDICATE_NAMES == (
        HAND_EMPTY,
        ON_GROUND,
        HOLDING,
        MOVABLE_IN_GOAL_REGION,
        MOVABLE_IS_DOWN_X,
    )


def test_all_returns_the_five_in_declaration_order(*, live_env: Tossing3DEnvironment) -> None:
    """A `SkillProvider` reports predicates in this order and a `PddlWriter` emits them in
    it, so the order is part of the interface rather than an accident of a dict."""
    predicates = Tossing3DPredicates.all(abstraction=live_env.abstraction())
    assert tuple(predicate.name for predicate in predicates) == PREDICATE_NAMES


def test_all_returns_real_core_predicates_rather_than_names(
    *, live_env: Tossing3DEnvironment
) -> None:
    for predicate in Tossing3DPredicates.all(abstraction=live_env.abstraction()):
        assert isinstance(predicate, Predicate)


def test_asking_for_a_predicate_this_domain_does_not_have_names_the_ones_it_does(
    *, live_env: Tossing3DEnvironment
) -> None:
    """`RobotAtSuccessfulThrowPose` is the name most likely to be asked for by stale code,
    since it survived in this repo for months. The error has to say what *is* available,
    or the caller cannot tell a typo from a genuinely retired predicate."""
    with pytest.raises(KeyError) as excinfo:
        Tossing3DPredicates.get(
            abstraction=live_env.abstraction(), name="RobotAtSuccessfulThrowPose"
        )
    message = str(excinfo.value)
    assert "RobotAtSuccessfulThrowPose" in message
    for name in PREDICATE_NAMES:
        assert name in message


def test_every_predicate_declares_the_types_upstream_gives_it(
    *, live_env: Tossing3DEnvironment
) -> None:
    """A `Predicate`'s `types` is what `SkillGrounder` enumerates over, so a wrong entry
    silently grounds a predicate onto objects it was never written for.

    **These are upstream's declarations, not this domain's choices**, so they are pinned
    by name against `state_abstractions.py`'s own five `Predicate(...)` calls. Three
    distinct types appear, not two: `HandEmpty`/`Holding` over the TidyBot robot type,
    `MovableInGoalRegion`/`MovableIsDownX` over the movable type, and **`OnGround` over
    the base `mujoco_object` type** -- which no object in this scene has. See
    `test_on_ground_grounds_onto_no_object_in_this_scene` for what that costs.

    The old four `core.Type`s (`cube_type`/`bin_type`/`barrier_type`/`robot_type`) are
    gone: `bin_0`, `cube_0` and `cuboid_barrier` are all one movable type upstream. That
    makes this test weaker than it was -- it can no longer catch a cube/bin transposition
    -- and that weakening is upstream's typing rather than a choice made here."""
    abstraction = live_env.abstraction()
    expected = {
        HAND_EMPTY: ("mujoco_tidybot_robot",),
        ON_GROUND: ("mujoco_object",),
        HOLDING: ("mujoco_tidybot_robot", "mujoco_movable_object"),
        MOVABLE_IN_GOAL_REGION: ("mujoco_movable_object",),
        MOVABLE_IS_DOWN_X: ("mujoco_movable_object", "mujoco_movable_object"),
    }
    for name, type_names in expected.items():
        predicate = Tossing3DPredicates.get(abstraction=abstraction, name=name)
        assert tuple(declared.name for declared in predicate.types) == type_names, name


def test_every_predicate_answers_against_a_live_scene(*, live_env: Tossing3DEnvironment) -> None:
    """The five are only useful if they evaluate, and evaluating is exactly what they
    could not do before a simulator was required: `Holding` needs forward kinematics and
    `MovableInGoalRegion` needs the live goal region. Asserts a real bool comes back for
    each, without asserting *which* -- upstream owns the classifier, this domain owns only
    the wiring."""
    abstraction = live_env.abstraction()
    state = live_env.get_current_state()
    applied = {
        HAND_EMPTY: (live_env.robot,),
        ON_GROUND: (live_env.cube,),
        HOLDING: (live_env.robot, live_env.cube),
        MOVABLE_IN_GOAL_REGION: (live_env.cube,),
        MOVABLE_IS_DOWN_X: (live_env.cube, live_env.barrier),
    }
    for name, objects in applied.items():
        held = Tossing3DPredicates.get(abstraction=abstraction, name=name).holds(state, objects)
        assert isinstance(held, bool), name


def test_the_initial_scene_is_the_one_the_operators_expect(
    *, live_env: Tossing3DEnvironment
) -> None:
    """The preconditions of `pick_cube` are `HandEmpty`, `OnGround` and `MovableIsDownX`,
    so if any of the three were false at reset the domain would be unsolvable from its own
    initial state and every rollout would be a no-op. Cheap, and it fails loudly on a pin
    bump that moves the cube, the barrier or the gripper's rest position."""
    abstraction = live_env.abstraction()
    state = live_env.get_current_state()

    def holds(*, name: str, objects: tuple) -> bool:
        return Tossing3DPredicates.get(abstraction=abstraction, name=name).holds(state, objects)

    assert holds(name=HAND_EMPTY, objects=(live_env.robot,))
    assert holds(name=ON_GROUND, objects=(live_env.cube,))
    assert holds(name=MOVABLE_IS_DOWN_X, objects=(live_env.cube, live_env.barrier))
    assert not holds(name=HOLDING, objects=(live_env.robot, live_env.cube))
    assert not holds(name=MOVABLE_IN_GOAL_REGION, objects=(live_env.cube,))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "OnGround is typed over upstream's base MujocoObjectType, which no object in the "
        "o1 scene carries, so it grounds onto nothing and pick_cube is never applicable. "
        "Recorded as a strict xfail so the marker fails loudly (XPASS) the moment it is "
        "fixed; the fix is a scope decision, not a test bug. See this file's docstring."
    ),
)
def test_on_ground_grounds_onto_at_least_one_object_in_this_scene(
    *, live_env: Tossing3DEnvironment
) -> None:
    """**A live defect, pinned rather than worked around.** `OnGround` is `pick_cube`'s
    precondition, and `SkillGrounder` grounds a predicate by *exact* type equality
    (`grounding.py`: `[obj for obj in objects if obj.type == object_type]`) because
    `core.Type` deliberately carries no parent and nothing walks ancestors.

    Upstream declares `OnGround = Predicate("OnGround", [MujocoObjectType])` -- the base
    type -- while every object in this scene is `mujoco_movable_object` or
    `mujoco_tidybot_robot`. So `OnGround` grounds onto the empty set: the atom is absent
    from every abstract state even though the classifier itself answers `True`, no
    `pick_cube` grounding is ever applicable, and Fast Downward reports the task
    *provably unsolvable*.

    That failure is silent in a run, which is why it is worth a test of its own rather
    than only the planner integration test: `EesMethod._next_plan` catches
    `PlanningFailure` and degrades to a no-op, so `--env tossing3d --method ees` exits 0
    and writes a full `stats.json` in which the method never took a single action.
    """
    scene_objects = (live_env.robot, live_env.cube, live_env.bin, live_env.barrier)
    grounded = [obj for obj in scene_objects if obj.type.name == "mujoco_object"]
    assert grounded, (
        "OnGround is typed over `mujoco_object`, which no object in this scene has, so "
        "it grounds onto nothing and `pick_cube` can never be applicable. Either the "
        "translator must map upstream's base type onto the movable type, or `core.Type` "
        "needs the ancestor-walking it currently forgoes."
    )


def test_objects_pass_through_in_declaration_order(*, live_env: Tossing3DEnvironment) -> None:
    """`Predicate.holds` is a positional `(state, objects)` callable, and the abstraction
    adapts it onto a membership test over ground atoms. A transposed pair would typecheck
    and then silently ask whether the barrier is on the cube's side of the cube.

    `MovableIsDownX` is the one that can show this now that every movable shares a type:
    it is genuinely asymmetric, and both of its arguments are the same `core.Type`, so
    nothing but this test stands between a swapped pair and a silently inverted one-way
    door."""
    abstraction = live_env.abstraction()
    state = live_env.get_current_state()
    is_down_x = Tossing3DPredicates.get(abstraction=abstraction, name=MOVABLE_IS_DOWN_X)

    # The cube starts at x ~ 0.71 and the barrier at x = 1.3, so the cube is down-x of the
    # barrier and the barrier is emphatically not down-x of the cube.
    assert is_down_x.holds(state, (live_env.cube, live_env.barrier))
    assert not is_down_x.holds(state, (live_env.barrier, live_env.cube))
