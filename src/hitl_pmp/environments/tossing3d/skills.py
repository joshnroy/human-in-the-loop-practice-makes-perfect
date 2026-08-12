"""Tossing3D's three lifted skills: `Pick`, `MoveToThrowPose`, `Toss`.

Each one is a `core.method.types.Skill` -- an operator model -- paired to an upstream
KINDER controller that actually executes it. That pairing is the shape
`kinder_bilevel_planning.env_models.dynamic3d.tidybot3d_shelf3D.py` uses, translated to
this project's types: there, `LiftedSkill(PickTargetOperator, LiftedPickShelfController)`
binds a `LiftedOperator` (preconditions / add effects / delete effects) to
`create_lifted_controllers()["pick_shelf"]`. Here the operator half is the `Skill` below
and the controller half is `KinderBackend.run_*`, dispatched by `compute_action` through
`Tossing3DEnvironment.take_action`.

## This package invents no controller parameter

Everything a controller is actually handed is upstream's own value or drawn from
upstream's own bounds:

| skill | controller(s) | parameters | where the numbers come from |
| --- | --- | --- | --- |
| `Pick` | `pick_shelf` | distance, rotation | upstream's `MOVE_TO_TARGET_{DISTANCE,ROT}_BOUNDS` |
| `MoveToThrowPose` | `move_to_target` | standoff | **ours** -- see `THROW_STANDOFF_BOUNDS` |
| `Toss` | `move_arm_to_conf`, then `toss` | release speed | upstream's own `release_speed` |

The one genuinely new range is the throw standoff, and it has to be: upstream's own
`MOVE_TO_TARGET_DISTANCE_BOUNDS` is `(0.5, 0.6)`, which is a *grasping* standoff, and
upstream's tossing test simply hardcodes `1.35` with no range at all. So the interval
below is this repo's, taken from the standoffs it has actually measured rather than
invented.

**`Toss` has exactly one continuous parameter, and it is upstream's, not ours.** An
earlier iteration of this domain interpolated a `swing` dial between upstream's windup
and full-power arm configurations. That interpolation was *ours*, and it made the dial a
quantity no upstream measurement covered; it was removed, leaving `Toss` with no
parameters at all.

The dial is back on a different footing. `joshnroy/kinder-baselines` PR #8 made
`release_speed` a parameter of upstream's own `TossController.reset`, so the throw's
energy is now upstream's knob rather than a quantity this package synthesised, and
**both arm configurations remain upstream's, untouched** -- the property the old `swing`
dial gave up. The range is `predicates.TOSS_SPEED_BOUNDS`, `(60, 240)` joint-path deg/s,
which is the span PR #213's grid actually drove; see that constant's own comment for the
unit, the absence of a feasibility clamp, and the non-monotonicity artifact at the low
end.

## The operator models, and the two choices that are load-bearing

The hard rule this repo enforces (`tests/environments/test_operator_dynamics_fidelity.py`)
is that an operator model must be *exactly as strong* as the dynamics guard it stands
for -- a precondition weaker than reality yields plans that look valid and cannot
execute. Two of the declarations below exist only for that reason:

1. **`Pick` requires `Reachable(?cube, ?barrier)`.** The base cannot cross the barrier,
   so a cube past it can never be grasped. Without this precondition a planner emits
   "toss, then pick it back up and try again", which the dynamics silently refuse.
2. **`Pick` deletes `RobotAtSuccessfulThrowPose(?robot, ?bin)`.**
   `pick_shelf` drives the base to the cube, which is on the near side of the barrier and
   therefore far too short of the bin to throw from. A model that let the predicate
   survive a pick would let the planner skip `MoveToThrowPose`.

And one on the other side:

3. **`Toss` deletes `Reachable(?cube, ?barrier)`.** A toss makes the cube unreachable
   whether or not it lands in the bin. Declaring it unconditionally is what makes
   the planner's model of a *failed* toss honest -- the alternative, deleting it only on
   success, is a model in which a missed throw costs nothing.

## No operator takes a goal region

Every signature below names only objects a controller acts on or is aimed at. The scored
landing box is scene geometry the classifiers read out of `State` -- carried on the bin,
under this domain's stated assumption that the bin's interior *is* that box (see
`predicates.py`'s module docstring, which also names the config where that is false).

It used to be a fifth object, `blocks_goal_region`, threaded through all three operators.
Nothing ever bound it to anything but the single region in the scene, no skill could move
or otherwise affect it, and every grounding was therefore one-to-one -- so dropping it
changes no operator's strength, only what a plan step has to mention. The preconditions
and effects below are the same predicates over the same states they were before.
"""

from typing import ClassVar

import numpy as np

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Action, State

from .environment import Tossing3DEnvironment
from .predicates import (
    HAND_EMPTY,
    HOLDING,
    IN_BIN,
    ON_GROUND,
    REACHABLE,
    ROBOT_AT_SUCCESSFUL_THROW_POSE,
    THROW_STANDOFF_BOUNDS,
    TOSS_SPEED_BOUNDS,
)

# Upstream's own bounds for a `pick_shelf` base standoff and yaw, from
# `kinder_models/dynamic3d/utils.py:57-58`. Upstream's `PickShelfController.
# sample_parameters` draws uniformly from exactly these and then rejection-tests the
# resulting base pose against *other* cubes; with one cube in the scene there is nothing
# to reject against, so a plain uniform draw is what upstream's sampler reduces to here.
PICK_DISTANCE_BOUNDS = (0.5, 0.6)
PICK_ROTATION_BOUNDS = (-np.pi / 4, np.pi / 4)

# `THROW_STANDOFF_BOUNDS` is the *feasible* range of throw standoffs and is what the
# sampler below draws from. It is emphatically **not** the range
# `RobotAtSuccessfulThrowPose` accepts: that band is derived per-call from the live scored
# box carried on the bin and `THROW_RANGE`, and covers 6/13 of this interval (the
# untrimmed geometric band, 0.300/0.65 -- see
# `predicates.RobotAtSuccessfulThrowPoseClassifier`'s docstring for the smaller,
# margin-trimmed band actually accepted). Those two used to be one symbol, which
# made `MoveToThrowPose`'s add effect constant-true and its sampler unlearnable -- see
# `predicates.THROW_STANDOFF_BOUNDS`. It still lives in `predicates.py` rather than here
# only because that is where its measured provenance is written down.


class Tossing3DSkills:
    """This domain's lifted skills and the lifted -> ground -> raw-`Action` pipeline.

    A static-method container, never instantiated, same as every other business-logic
    class in this project.
    """

    # No leading "?" on any of these: `PddlWriter.variable_str` adds it at write time
    # (planning/pddl.py, deviation 1 -- predicators' own `Variable.name` carries the
    # sigil and ours deliberately does not). Declaring "?robot" here rendered "??robot",
    # which Fast Downward's translator split into two tokens, so every plan call in the
    # domain failed -- silently, since `EesMethod._next_plan` catches `PlanningFailure`
    # and degrades to a no-op. Every other domain here declares plain names.
    _robot: ClassVar[Variable] = Variable(name="robot", type=Tossing3DEnvironment.robot_type)
    _cube: ClassVar[Variable] = Variable(name="cube", type=Tossing3DEnvironment.cube_type)
    _bin: ClassVar[Variable] = Variable(name="bin", type=Tossing3DEnvironment.bin_type)
    _barrier: ClassVar[Variable] = Variable(name="barrier", type=Tossing3DEnvironment.barrier_type)

    PICK: ClassVar[Skill] = Skill(
        name="Pick",
        parameters=(_robot, _cube, _barrier, _bin),
        preconditions=frozenset({
            LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,)),
            LiftedAtom(predicate=ON_GROUND, variables=(_cube,)),
            # The barrier is one-way: see this module's docstring, choice 1.
            LiftedAtom(predicate=REACHABLE, variables=(_cube, _barrier)),
        }),
        add_effects=frozenset({LiftedAtom(predicate=HOLDING, variables=(_robot, _cube))}),
        delete_effects=frozenset({
            LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,)),
            LiftedAtom(predicate=ON_GROUND, variables=(_cube,)),
            # pick_shelf drives the base to the cube: see choice 2.
            LiftedAtom(predicate=ROBOT_AT_SUCCESSFUL_THROW_POSE, variables=(_robot, _bin)),
        }),
        param_dim=2,
    )

    MOVE_TO_THROW_POSE: ClassVar[Skill] = Skill(
        name="MoveToThrowPose",
        parameters=(_robot, _cube, _bin),
        # `Holding` rather than nothing: `move_to_target` here passes
        # `disable_collision_objects=["cube_0"]`, upstream's own argument, which is only
        # correct while the cube is in the gripper. Planning the base motion with the cube
        # ignored while it sits on the floor would drive straight through it.
        preconditions=frozenset({LiftedAtom(predicate=HOLDING, variables=(_robot, _cube))}),
        add_effects=frozenset({
            LiftedAtom(predicate=ROBOT_AT_SUCCESSFUL_THROW_POSE, variables=(_robot, _bin))
        }),
        delete_effects=frozenset(),
        param_dim=1,
    )

    TOSS: ClassVar[Skill] = Skill(
        name="Toss",
        parameters=(_robot, _cube, _bin, _barrier),
        preconditions=frozenset({
            LiftedAtom(predicate=HOLDING, variables=(_robot, _cube)),
            LiftedAtom(predicate=ROBOT_AT_SUCCESSFUL_THROW_POSE, variables=(_robot, _bin)),
        }),
        add_effects=frozenset({
            LiftedAtom(predicate=IN_BIN, variables=(_cube, _bin)),
            LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,)),
        }),
        delete_effects=frozenset({
            LiftedAtom(predicate=HOLDING, variables=(_robot, _cube)),
            # Unconditionally, hit or miss: see this module's docstring, choice 3.
            LiftedAtom(predicate=REACHABLE, variables=(_cube, _barrier)),
        }),
        param_dim=1,
    )

    @staticmethod
    def sample_params(*, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        """A state-independent draw of this skill's continuous parameters.

        State-independent by the `SkillProvider` contract: a learned sampler generates
        many candidates from this and then picks among them using the state.
        """
        skill = ground_skill.skill
        if skill == Tossing3DSkills.PICK:
            return np.array([
                rng.uniform(*PICK_DISTANCE_BOUNDS),
                rng.uniform(*PICK_ROTATION_BOUNDS),
            ])
        if skill == Tossing3DSkills.MOVE_TO_THROW_POSE:
            return np.array([rng.uniform(*THROW_STANDOFF_BOUNDS)])
        if skill == Tossing3DSkills.TOSS:
            return np.array([rng.uniform(*TOSS_SPEED_BOUNDS)])
        raise ValueError(f"Unknown skill: {skill.name}")

    @staticmethod
    def compute_action(*, ground_skill: GroundSkill, params: np.ndarray, state: State) -> Action:
        """Realize a (ground skill, parameters) pair as this domain's `[id, p0, p1]` vector.

        `state` is unused: unlike Light Switch, whose skills compute a delta against the
        robot's current position, every parameter here is absolute (a standoff from the
        bin, a yaw about the cube) and is interpreted by upstream's controller against
        whatever state it is reset from.
        """
        del state
        skill = ground_skill.skill
        if skill == Tossing3DSkills.PICK:
            return np.array(
                [Tossing3DEnvironment.pick_id, float(params[0]), float(params[1])], dtype=float
            )
        if skill == Tossing3DSkills.MOVE_TO_THROW_POSE:
            return np.array(
                [Tossing3DEnvironment.move_to_throw_pose_id, float(params[0]), 0.0], dtype=float
            )
        if skill == Tossing3DSkills.TOSS:
            return np.array([Tossing3DEnvironment.toss_id, float(params[0]), 0.0], dtype=float)
        raise ValueError(f"Unknown skill: {skill.name}")
