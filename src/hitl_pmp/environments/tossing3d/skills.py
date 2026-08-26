"""Tossing3D's two lifted skills: `PickCube` and `MoveToTossLocationAndToss`.

Each one is a `core.method.types.Skill` -- an operator model -- paired to an upstream
KINDER controller that actually executes it. That pairing is the shape
`kinder_bilevel_planning.env_models.dynamic3d.tidybot3d_tossing3D.py` uses, translated to
this project's types: there, `LiftedSkill(PickCubeOperator, controllers["pick_cube"])`
binds a `LiftedOperator` (preconditions / add effects / delete effects) to a lifted
controller. Here the operator half is the `Skill` below and the controller half is
`KinderBackend.run_*`, dispatched by `compute_action` through
`Tossing3DEnvironment.take_action`.

## It was three skills, and the middle one is gone

| was | controller(s) | is |
| --- | --- | --- |
| `Pick` (distance, rotation) | `pick_shelf` | `PickCube`, **no parameters** |
| `MoveToThrowPose` (standoff) | `move_to_target` | folded into the toss |
| `Toss` (speed, ms) | `move_arm_to_conf`, `toss` | `MoveToTossLocationAndToss`, four |

Upstream composed the base move and the throw so that **no predicate has to name the pose
between them**. On the pick, upstream's `PickCubeController` exposes `sample_parameters`
(sampling standoff and rotation), but its `reset()` executes `del params` and hardcodes
`TARGET_DISTANCE = 0.55, TARGET_ROTATION = 0.0` -- so `param_dim=0` is declared here to
match upstream's effective execution behavior at this pin. Both changes keep controller
execution identical to upstream's behavior, which is the direction this domain's rules
point: `kinder_backend.py` drives upstream's controllers unmodified, and this package
invents no controller parameter.

Two consequences worth stating rather than discovering:

1. **`PickCube` has `param_dim=0`, so it has no sampler and cannot be learned.** EES
   builds one success classifier and one sampler per skill with `param_dim > 0`; with two
   skills and only one of them parameterised, every learned parameter in this domain now
   belongs to the composed toss. Tossing Room already has `param_dim=0` skills, so nothing
   in the harness is new -- but what a learning run on this domain *measures* is narrower
   than it was.
2. **A standoff that cannot score is now only discovered by throwing from it.** The old
   `MoveToThrowPose` could be rejected before any throw happened, which is what
   `RobotAtSuccessfulThrowPose` was for. The composed skill's only failure signal is the
   landing.

## Every continuous bound below is upstream's

`MoveToTossLocationAndTossController` declares all four on itself, and they are narrower
than the ones this package used to draw from -- upstream measured (480 draws across 16
seeds) that every scoring draw fell in speed [117.5, 140.0] deg/s and release
[710.4, 836.1] ms, and set its bounds a small margin outside that. The old
`TOSS_SPEED_BOUNDS` of `(60, 140)` and `TOSS_RELEASE_MS_BOUNDS` of `(300, 1400)` spent
most of a sampler's budget on combinations that can never score.

**That is a real change to what a learning run measures, in the same direction as the
`param_dim=0` pick**: narrowing the box a sampler draws from raises what an *untrained*
sampler scores, which is the baseline a trained one has to beat. How much headroom is
left is a measurement, not an argument, and is not asserted here.

It is not a choice made in this package -- these are the bounds upstream's own controller
samples from, and `test_kinder_pin.py` pins them against it, so this domain and the
kb-side bilevel planner draw from the same box. Adopting them is what makes the two
comparable at all.

`TOSS_SPEED_BOUNDS` is in joint-path **deg/s** while upstream's `SPEED_BOUNDS` is rad/s;
`KinderBackend.run_move_to_toss_location_and_toss` is the one site that converts. The
degree convention is kept because every measured toss number in this domain's docs is in
deg/s.

## The operator models, and the two choices that are load-bearing

The rule here is that an operator model must never permit *more* than the raw dynamics
allow -- a precondition weaker than reality yields plans that look valid and cannot
execute. Both are upstream's own, for the same reasons upstream gives:

1. **`PickCube` requires `Reachable(?cube, ?barrier)`.** The base cannot cross the
   barrier, so a cube past it can never be grasped. Without this precondition a planner
   emits "toss, then pick it back up and try again", which the dynamics silently refuse.
2. **The toss deletes `Reachable(?cube, ?barrier)` unconditionally**, hit or miss. A toss
   makes the cube unreachable whether or not it scored; deleting it only on success would
   be a model in which a missed throw costs nothing.

And one add effect that is neither of those: **the toss adds `OnGround(?cube)`**, because
upstream measured 15/15 scoring throws leaving the cube resting on a face. Upstream's
`OnGround` is face-interchangeable for exactly that reason: a version reading "flat on the
face it started on" would call most scoring throws a failure, and this add effect could
not be honest against it. That is one of the things this domain gets for free by looking
upstream's atoms up rather than re-implementing them (see `predicates.py`).

**The toss also requires `Reachable(?held, ?barrier)`.** Upstream's own note: without it
the grounder can bind `?barrier` to an object the held cube was never down-x of, and the
delete effect then targets an atom that was never true. Here the types make `?barrier`
bind only to the barrier, so the binding hazard does not arise -- but the precondition
stays, because it is also simply true (the pick requires it and nothing between them
touches it) and because dropping it would make this operator's delete effect describe an
atom the operator does not require.

## No operator takes a goal region

Every signature below names only objects a controller acts on or is aimed at. The scored
landing box is scene geometry the classifiers read out of `State` -- carried on the bin,
under this domain's stated assumption that the bin's interior *is* that box (see
`predicates.py`'s module docstring, which also names the config where that is false).
"""

from typing import ClassVar

import numpy as np

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Action, State

from .environment import Tossing3DEnvironment
from .predicates import HAND_EMPTY, HOLDING, IN_BIN, ON_GROUND, REACHABLE

# Upstream's `MoveToTossLocationAndTossController.TARGET_DISTANCE_BOUNDS`: where a throw
# is possible, in metres from the bin. The upper part of the wider range upstream tried
# does not score.
TOSS_DISTANCE_BOUNDS = (1.25, 1.45)

# Upstream's `WAYPOINT_TOLERANCE` (`kinder_models/dynamic3d/utils.py`), how close
# `_check_robot_is_close_to_pose` requires the base to be to its own planned waypoint.
WAYPOINT_TOLERANCE = 4 * 1e-2

# Upstream's `TARGET_ROTATION_BOUNDS`: the widest yaw about the bin that still leaves the
# base within half of `WAYPOINT_TOLERANCE` of the bin's axis at the largest standoff.
# Computed from the two constants above rather than written as a literal, exactly as
# upstream computes it, so a bump to either cannot silently drift out of sync.
MAX_TOSS_ROTATION = float(np.arcsin(0.5 * WAYPOINT_TOLERANCE / TOSS_DISTANCE_BOUNDS[1]))
TOSS_ROTATION_BOUNDS = (-MAX_TOSS_ROTATION, MAX_TOSS_ROTATION)

# Upstream's `SPEED_BOUNDS`, in joint-path deg/s rather than upstream's rad/s -- see this
# module's docstring for why the degree convention is kept and where it is converted.
TOSS_SPEED_BOUNDS = (115.0, 140.0)

# Upstream's `RELEASE_MS_BOUNDS`: the millisecond from the start of the swing at which
# the gripper opens. Absolute rather than a swing fraction because that is what the real
# TidyBot's `movej_primitive.execute()` takes.
TOSS_RELEASE_MS_BOUNDS = (700.0, 840.0)


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

    PICK_CUBE: ClassVar[Skill] = Skill(
        name="PickCube",
        # Upstream's own object order for `pick_cube`: (robot, cube, barrier). The
        # barrier is unused by the controller and present so the operator can say the
        # cube is still on this side of it.
        parameters=(_robot, _cube, _barrier),
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
        }),
        param_dim=0,
    )

    MOVE_TO_TOSS_LOCATION_AND_TOSS: ClassVar[Skill] = Skill(
        name="MoveToTossLocationAndToss",
        # Upstream's own object order: (robot, target, held, barrier).
        parameters=(_robot, _bin, _cube, _barrier),
        preconditions=frozenset({
            LiftedAtom(predicate=HOLDING, variables=(_robot, _cube)),
            LiftedAtom(predicate=REACHABLE, variables=(_cube, _barrier)),
        }),
        add_effects=frozenset({
            LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,)),
            LiftedAtom(predicate=IN_BIN, variables=(_cube, _bin)),
            # Measured upstream on 20 throws: 15/15 that scored left the cube on a face.
            LiftedAtom(predicate=ON_GROUND, variables=(_cube,)),
        }),
        delete_effects=frozenset({
            LiftedAtom(predicate=HOLDING, variables=(_robot, _cube)),
            # Unconditionally, hit or miss: see this module's docstring, choice 2.
            LiftedAtom(predicate=REACHABLE, variables=(_cube, _barrier)),
        }),
        param_dim=4,
    )

    # A third, robot-executed skill (not the human `ask_for_reset_cube_bin_only`): the
    # robot's own gripper-open primitive, upstream's `open_gripper` controller. Exists to
    # give the planner a way out of a near-miss grasp -- the gripper can end up commanded
    # closed on nothing (`HandEmpty` False, `Holding` False both at once, since the close
    # never actually caught the cube), and `PickCube` is the only skill this leaves
    # reachable, but it requires `HandEmpty`, which is exactly what is missing. No other
    # operator in this domain ever re-adds `HandEmpty`: `MoveToTossLocationAndToss` does,
    # but only as a post-throw effect that requires `Holding`, the other predicate this
    # dead end lacks. No precondition -- opening the gripper is always physically safe,
    # same reasoning as `Tossing3DSkillProvider.human_cube_bin_reset_skill`'s empty
    # precondition -- and unlike that skill, this one's effect is honest against the real
    # simulator: it is a real command sent to the robot, so `HandEmpty` (which reads the
    # command, not contact) is genuinely true on the very next observation, not merely
    # predicted.
    #
    # **No `Holding` delete effect, and no `?cube` parameter -- found the hard way.**
    # Declaring `delete_effects={Holding(?robot, ?cube)}` on an operator whose
    # precondition does not also require `Holding(?robot, ?cube)` is exactly the shape
    # that needs a conditional effect ("delete it if it was there") once Fast Downward's
    # invariant synthesis merges `HandEmpty`/`Holding` into one mutex-tracked variable --
    # `PickCube` and `MoveToTossLocationAndToss` both get this for free because their own
    # preconditions already pin which value the variable had beforehand, but this skill's
    # empty precondition cannot. `astar(lmcut())` (this project's default search alias)
    # does not support conditional effects and aborts outright on every `plan_to` call
    # once this operator is in scope -- confirmed by bisection: dropping just this delete
    # effect is what fixes it, dropping the add effect instead does not. This project's
    # `Skill` type has no conditional-effect construct to reach for (same negation gap as
    # `human_cube_bin_reset_skill`), so the delete effect is dropped rather than
    # expressed. The one state this leaves imprecise: bridging through `OpenGripper`
    # while genuinely `Holding` a cube (not a near-miss) would leave the plan believing
    # `Holding` survives, when the real dynamics drop the cube. Not reachable today --
    # `OnGround(?cube)` stays False in that belief too (nothing here or elsewhere adds
    # it), which keeps `PickCube` symbolically unreachable right after -- but a future
    # skill that adds `OnGround` without going through a pick/toss boundary would need to
    # revisit this.
    OPEN_GRIPPER: ClassVar[Skill] = Skill(
        name="OpenGripper",
        parameters=(_robot,),
        preconditions=frozenset(),
        add_effects=frozenset({LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,))}),
        delete_effects=frozenset(),
        param_dim=0,
    )

    @staticmethod
    def sample_params(*, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        """A state-independent draw of this skill's continuous parameters.

        State-independent by the `SkillProvider` contract: a learned sampler generates
        many candidates from this and then picks among them using the state.

        The four components are drawn independently but are not independent in effect --
        the swing's duration is a function of its speed, so a fixed millisecond is a
        different fraction of a slow swing than of a fast one.
        """
        skill = ground_skill.skill
        if skill == Tossing3DSkills.PICK_CUBE:
            return np.zeros(0)
        if skill == Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS:
            return np.array([
                rng.uniform(*TOSS_DISTANCE_BOUNDS),
                rng.uniform(*TOSS_ROTATION_BOUNDS),
                rng.uniform(*TOSS_SPEED_BOUNDS),
                rng.uniform(*TOSS_RELEASE_MS_BOUNDS),
            ])
        if skill == Tossing3DSkills.OPEN_GRIPPER:
            return np.zeros(0)
        raise ValueError(f"Unknown skill: {skill.name}")

    @staticmethod
    def compute_action(*, ground_skill: GroundSkill, params: np.ndarray, state: State) -> Action:
        """Realize a (ground skill, parameters) pair as this domain's five-slot vector.

        `state` is unused: unlike Light Switch, whose skills compute a delta against the
        robot's current position, every parameter here is absolute (a standoff from the
        bin, a yaw about it, a joint-path speed, a millisecond) and is interpreted by
        upstream's controller against whatever state it is reset from.
        """
        del state
        skill = ground_skill.skill
        if skill == Tossing3DSkills.PICK_CUBE:
            return np.array([Tossing3DEnvironment.pick_cube_id, 0.0, 0.0, 0.0, 0.0], dtype=float)
        if skill == Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS:
            return np.array(
                [
                    Tossing3DEnvironment.move_to_toss_location_and_toss_id,
                    float(params[0]),
                    float(params[1]),
                    float(params[2]),
                    float(params[3]),
                ],
                dtype=float,
            )
        if skill == Tossing3DSkills.OPEN_GRIPPER:
            return np.array([Tossing3DEnvironment.open_gripper_id, 0.0, 0.0, 0.0, 0.0], dtype=float)
        raise ValueError(f"Unknown skill: {skill.name}")
