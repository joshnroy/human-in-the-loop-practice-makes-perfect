"""Tossing3D's two lifted skills: `pick_cube` and `move_to_toss_location_and_toss`.

These are upstream's operators, not this repo's. `f12326c`'s own bilevel model
(`kinder-bilevel-planning/.../tidybot3d_tossing3D.py`) reduces Tossing3D to exactly two
`LiftedOperator`s over five predicates, and the declarations below are that model
translated into `core.method.types.Skill`. Where the old three-skill decomposition
disagreed with upstream, upstream wins.

```text
pick_cube(?robot, ?cube, ?barrier)
    pre: HandEmpty(?robot), OnGround(?cube), MovableIsDownX(?cube, ?barrier)
    add: Holding(?robot, ?cube)
    del: HandEmpty(?robot), OnGround(?cube)

move_to_toss_location_and_toss(?robot, ?held, ?barrier)
    pre: Holding(?robot, ?held), MovableIsDownX(?held, ?barrier)
    add: HandEmpty(?robot), MovableInGoalRegion(?held), OnGround(?held)
    del: Holding(?robot, ?held), MovableIsDownX(?held, ?barrier)
```

## Two skills, not three, and the fusion is the point

hitl had `Pick` -> `MoveToThrowPose` -> `Toss`. Upstream fuses the base move and the
throw into one controller, and its own docstring gives the reason: *"Composed rather than
split so that no predicate has to name the pose between them."*

That single sentence retires a large amount of this repo's machinery.
`RobotAtSuccessfulThrowPose` existed only to say "the base is standing somewhere a throw
from here scores", and its acceptance band was derived from a calibrated throw range, two
asymmetric margins and a lateral tolerance -- roughly 200 lines of measured constants
whose whole job was to characterise an intermediate state. With the move and the throw
executed together there is no intermediate state to characterise. Upstream names the cost
honestly too: *"a standoff which cannot score is only discovered by throwing from it,
where a separate move could have been rejected first."*

## `MovableIsDownX` is the one-way door

`move_to_toss_location_and_toss` **deletes** `MovableIsDownX(?held, ?barrier)`, and
`pick_cube` **requires** it. That pair is the whole reason this domain is in the project:
past the barrier the cube cannot be picked again, so a planner cannot emit
"toss, then pick it back up and try again", which the dynamics silently refuse.

Deleted unconditionally, hit or miss -- which is what makes the model of a *failed* toss
honest. The alternative, deleting it only on success, is a model in which a missed throw
costs nothing.

## `OnGround` is an add effect, and it is measured rather than assumed

Upstream's own comment on that effect: *"Measured on 20 throws: 15/15 that scored left
the cube resting on a face."* It is upstream's claim, carried across with its evidence,
not one this repo re-measured.

## Where the continuous parameters come from

**Nowhere in this file.** `sample_params` delegates to the controller's own
`sample_parameters` through `adapters.kinder.controllers`. That is the substantive fix,
not a tidy-up: hitl declared `TOSS_RELEASE_MS_BOUNDS = (300, 1400)` beside a controller
whose own measured band is `(700, 840)`, so it sampled a window about nine times too wide
and the large majority of its draws could not score. `PICK_DISTANCE_BOUNDS`,
`PICK_ROTATION_BOUNDS`, `TOSS_SPEED_BOUNDS` and `THROW_STANDOFF_BOUNDS` are all gone for
the same reason: every one was a second copy of a number upstream owns.

`param_dim` is still declared here, because `LiftedParameterizedController.params_space`
is `None` on every Tossing3D controller and there is nothing to read it off.
`test_the_declared_param_dims_match_the_controllers_own_samplers` compares each
declaration against what the controller actually draws.
"""

from typing import ClassVar

import numpy as np

from hitl_pmp.adapters.kinder.abstraction import KinderAbstraction
from hitl_pmp.adapters.kinder.controllers import KinderControllers
from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Action, State

from .environment import Tossing3DEnvironment
from .predicates import (
    HAND_EMPTY,
    HOLDING,
    MOVABLE_IN_GOAL_REGION,
    MOVABLE_IS_DOWN_X,
    ON_GROUND,
    Tossing3DPredicates,
)

# The controller keys in `create_lifted_controllers`' own dict, which are also the
# operator names upstream gives them.
PICK_CUBE = "pick_cube"
MOVE_TO_TOSS_LOCATION_AND_TOSS = "move_to_toss_location_and_toss"

SKILL_NAMES = (PICK_CUBE, MOVE_TO_TOSS_LOCATION_AND_TOSS)

# `distance, rotation` and `distance, rotation, tossing_speed, tossing_ms`. Declared
# because `params_space` is `None`; checked against the real samplers by a fidelity test.
PARAM_DIMS = {PICK_CUBE: 2, MOVE_TO_TOSS_LOCATION_AND_TOSS: 4}


class Tossing3DSkills:
    """This domain's lifted skills and the lifted -> ground -> raw-`Action` pipeline.

    A static-method container, never instantiated, same as every other business-logic
    class in this project. Unlike Light Switch's, the skills cannot be `ClassVar`s: their
    preconditions and effects are `LiftedAtom`s over `Predicate`s that close over one
    live scene's abstraction, so they are built per environment.
    """

    # No leading "?" on any of these. `PddlWriter.variable_str` adds it at write time
    # (predicators' own `Variable.name` carries the sigil and ours deliberately does not).
    # Declaring "?robot" here rendered "??robot", which Fast Downward's translator split
    # into two tokens, so every plan call in the domain failed -- silently, since
    # `EesMethod._next_plan` catches `PlanningFailure` and degrades to a no-op.
    # `KinderControllers.variables` strips the sigil off upstream's own names for the
    # same reason.
    robot: ClassVar[Variable] = Variable(name="robot", type=Tossing3DEnvironment.robot_type)
    cube: ClassVar[Variable] = Variable(name="cube", type=Tossing3DEnvironment.movable_type)
    held: ClassVar[Variable] = Variable(name="held", type=Tossing3DEnvironment.movable_type)
    barrier: ClassVar[Variable] = Variable(name="barrier", type=Tossing3DEnvironment.movable_type)

    @staticmethod
    def all(*, abstraction: KinderAbstraction) -> tuple[Skill, ...]:
        """The two, in `SKILL_NAMES` order."""
        return (
            Tossing3DSkills.pick_cube(abstraction=abstraction),
            Tossing3DSkills.move_to_toss_location_and_toss(abstraction=abstraction),
        )

    @staticmethod
    def pick_cube(*, abstraction: KinderAbstraction) -> Skill:
        """Pick the cube up off the ground. Takes no continuous parameters it uses --
        `PickCubeController.reset` opens with `del params` -- so refinement backtracks
        over the throw alone. It still *samples* two, and they are still drawn from
        upstream's sampler, because a domain deciding a parameter is ignored is exactly
        the kind of local knowledge that goes stale."""
        predicate = Tossing3DPredicates.get
        robot, cube, barrier = (
            Tossing3DSkills.robot,
            Tossing3DSkills.cube,
            Tossing3DSkills.barrier,
        )
        return Skill(
            name=PICK_CUBE,
            parameters=(robot, cube, barrier),
            preconditions=frozenset({
                LiftedAtom(
                    predicate=predicate(abstraction=abstraction, name=HAND_EMPTY),
                    variables=(robot,),
                ),
                LiftedAtom(
                    predicate=predicate(abstraction=abstraction, name=ON_GROUND), variables=(cube,)
                ),
                # Only a cube still on this side of the barrier can be reached.
                LiftedAtom(
                    predicate=predicate(abstraction=abstraction, name=MOVABLE_IS_DOWN_X),
                    variables=(cube, barrier),
                ),
            }),
            add_effects=frozenset({
                LiftedAtom(
                    predicate=predicate(abstraction=abstraction, name=HOLDING),
                    variables=(robot, cube),
                ),
            }),
            delete_effects=frozenset({
                LiftedAtom(
                    predicate=predicate(abstraction=abstraction, name=HAND_EMPTY),
                    variables=(robot,),
                ),
                LiftedAtom(
                    predicate=predicate(abstraction=abstraction, name=ON_GROUND), variables=(cube,)
                ),
            }),
            param_dim=PARAM_DIMS[PICK_CUBE],
        )

    @staticmethod
    def move_to_toss_location_and_toss(*, abstraction: KinderAbstraction) -> Skill:
        """Drive to a pose to throw from, and throw, as one skill. See this module's
        docstring for why the two are fused and what that retires."""
        predicate = Tossing3DPredicates.get
        robot, held, barrier = (
            Tossing3DSkills.robot,
            Tossing3DSkills.held,
            Tossing3DSkills.barrier,
        )
        return Skill(
            name=MOVE_TO_TOSS_LOCATION_AND_TOSS,
            parameters=(robot, held, barrier),
            preconditions=frozenset({
                LiftedAtom(
                    predicate=predicate(abstraction=abstraction, name=HOLDING),
                    variables=(robot, held),
                ),
                # Only toss if the held cube is still on this side of the barrier.
                LiftedAtom(
                    predicate=predicate(abstraction=abstraction, name=MOVABLE_IS_DOWN_X),
                    variables=(held, barrier),
                ),
            }),
            add_effects=frozenset({
                LiftedAtom(
                    predicate=predicate(abstraction=abstraction, name=HAND_EMPTY),
                    variables=(robot,),
                ),
                LiftedAtom(
                    predicate=predicate(abstraction=abstraction, name=MOVABLE_IN_GOAL_REGION),
                    variables=(held,),
                ),
                # Upstream's own measurement: 15/15 throws that scored left the cube
                # resting on a face.
                LiftedAtom(
                    predicate=predicate(abstraction=abstraction, name=ON_GROUND), variables=(held,)
                ),
            }),
            delete_effects=frozenset({
                LiftedAtom(
                    predicate=predicate(abstraction=abstraction, name=HOLDING),
                    variables=(robot, held),
                ),
                # The one-way door: past the barrier the cube cannot be picked again.
                LiftedAtom(
                    predicate=predicate(abstraction=abstraction, name=MOVABLE_IS_DOWN_X),
                    variables=(held, barrier),
                ),
            }),
            param_dim=PARAM_DIMS[MOVE_TO_TOSS_LOCATION_AND_TOSS],
        )

    @staticmethod
    def sample_params(
        *,
        ground_skill: GroundSkill,
        rng: np.random.Generator,
        controllers: KinderControllers,
        state: State,
    ) -> np.ndarray:
        """Draw this skill's continuous parameters -- from the controller's own sampler.

        **A documented deviation from `SkillProvider.sample_params`, which specifies a
        *state-independent* draw.** `MoveToTossLocationAndTossController.sample_parameters`
        genuinely is state-independent (it opens with `del x`), but
        `PickCubeController`'s reads the target's pose and rejection-tests the resulting
        base pose against other cubes, so it needs the live state. Handing it a blank one
        would silently change what upstream samples, which is the exact failure mode this
        whole change exists to remove -- so the state goes in and the deviation is stated
        rather than hidden.

        On the shipped `o1` scene the deviation has no effect: the rejection loop looks
        for other objects whose name contains "cube", and there is exactly one, so it
        accepts its first draw and reduces to two uniforms over upstream's own bounds.
        `test_pick_sampling_is_state_independent_on_the_single_cube_scene` pins that.
        """
        return controllers.sample_params(
            key=ground_skill.skill.name,
            object_names=tuple(obj.name for obj in ground_skill.objects),
            state=state,
            rng=rng,
        )

    @staticmethod
    def compute_action(*, ground_skill: GroundSkill, params: np.ndarray, state: State) -> Action:
        """Realize a (ground skill, parameters) pair as this domain's action vector.

        `[skill_id, p0, p1, p2, p3]`, padded with zeros: `pick_cube` uses two slots and
        `move_to_toss_location_and_toss` four, and the unused slots are ignored rather
        than validated, exactly as Tossing Room does.

        `state` is unused -- every parameter here is absolute and is interpreted by
        upstream's controller against whatever state it is reset from.
        """
        del state
        name = ground_skill.skill.name
        if name not in SKILL_NAMES:
            raise ValueError(f"Unknown skill: {name}")
        skill_id = (
            Tossing3DEnvironment.pick_cube_id
            if name == PICK_CUBE
            else Tossing3DEnvironment.move_to_toss_location_and_toss_id
        )
        slots = np.zeros(Tossing3DEnvironment.action_space.shape[0] - 1, dtype=float)
        values = np.asarray(params, dtype=float).ravel()
        slots[: values.size] = values
        return np.concatenate([[float(skill_id)], slots])
