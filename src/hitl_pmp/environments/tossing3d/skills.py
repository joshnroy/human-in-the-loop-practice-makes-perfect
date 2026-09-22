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

`MoveToTossLocationAndTossController` declares all four on itself. The current simulation
range extends the standoff to 2.6 m, release speed to 420 deg/s, and release timing down
to 400 ms so the farther default receivers have feasible release poses and trajectories.
The low-level toss profile keeps its 140 deg/s default; the composed simulation controller
explicitly permits up to three times that effort.

Earlier bounds came from a shorter-range scene's 480-draw study. Those bounds and success
rates do not describe this expanded candidate space. The live regression tests replay
three certified solutions in the default far-bin scene through this package's wrapper;
they establish reachability, not an untrained sampler's success rate.

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

1. **`PickCube` binds the robot and cube to one typed side.** The base cannot cross the
   barrier, so a cube on the opposite side can never be grasped. This replaces the old,
   redundant ``Reachable`` predicate with the same fact in relational form.
2. **The toss copies the bin's side to the cube.** ``CubeAtSide`` is functional, so the
   old value is cleared before the selected destination side is added.

And one add effect that is neither of those: **the toss adds `OnGround(?cube)`**, because
upstream measured 15/15 scoring throws leaving the cube resting on a face. Upstream's
`OnGround` is face-interchangeable for exactly that reason: a version reading "flat on the
face it started on" would call most scoring throws a failure, and this add effect could
not be honest against it. That is one of the things this domain gets for free by looking
upstream's atoms up rather than re-implementing them (see `predicates.py`).

The typed relation also makes same-side practice honest: tossing into a same-side bin
leaves the cube pickable, while tossing across the barrier does not.

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
from .predicates import (
    BIN_AT_SIDE,
    CLOSED_EMPTY,
    CUBE_AT_SIDE,
    GRASP_CLEAR,
    HAND_EMPTY,
    HOLDING,
    IN_BIN,
    NOT_HOLDING,
    ON_GROUND,
    ROBOT_AT_SIDE,
)
from .sides import Tossing3DSides

# Match upstream's candidate standoffs. Farther receivers require release poses
# that remain on the robot's side of the barrier; keep the lower end for same-side
# practice. These bounds cover candidates, not guaranteed scoring distances.
TOSS_DISTANCE_BOUNDS = (1.25, 2.6)

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
# The simulator controller explicitly permits higher effort for farther receivers.
# The low-level default remains unchanged upstream; this is a simulation range.
TOSS_SPEED_BOUNDS = (115.0, 420.0)

# Upstream's `RELEASE_MS_BOUNDS`: the millisecond from the start of the swing at which
# the gripper opens. Absolute rather than a swing fraction because that is what the real
# TidyBot's `movej_primitive.execute()` takes.
TOSS_RELEASE_MS_BOUNDS = (400.0, 840.0)


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
    _side: ClassVar[Variable] = Variable(name="side", type=Tossing3DSides.type)

    PICK_CUBE: ClassVar[Skill] = Skill(
        name="PickCube",
        # Upstream's own object order for `pick_cube`: (robot, cube, barrier). The
        # barrier is unused by the controller and present so the operator can say the
        # cube is still on this side of it. The bin comes last, appended after #346's
        # side for the same reason the side was: the GraspClear gate has to name the
        # bin whose walls it measures, and appending preserves upstream's prefix order.
        parameters=(_robot, _cube, _barrier, _side, _bin),
        preconditions=frozenset({
            LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,)),
            LiftedAtom(predicate=ON_GROUND, variables=(_cube,)),
            # The barrier is one-way: see this module's docstring, choice 1.
            LiftedAtom(predicate=ROBOT_AT_SIDE, variables=(_robot, _barrier, _side)),
            LiftedAtom(predicate=CUBE_AT_SIDE, variables=(_cube, _barrier, _side)),
            # The grasp planner refuses a cube whose faces sit within
            # GRASP_CLEARANCE_M of the bin's wall band, and retrying such a pick is
            # the degenerate loop the 2026-09-22 trap diagnosis pinned -- gate it
            # symbolically so recovery (the paid reset) becomes the plan instead.
            LiftedAtom(predicate=GRASP_CLEAR, variables=(_cube, _bin)),
        }),
        add_effects=frozenset({LiftedAtom(predicate=HOLDING, variables=(_robot, _cube))}),
        delete_effects=frozenset({
            LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,)),
            LiftedAtom(predicate=NOT_HOLDING, variables=(_robot, _cube)),
            LiftedAtom(predicate=ON_GROUND, variables=(_cube,)),
        }),
        param_dim=0,
        practice_cost=1.0,
    )

    MOVE_TO_TOSS_LOCATION_AND_TOSS: ClassVar[Skill] = Skill(
        name="MoveToTossLocationAndToss",
        # Upstream's own object order: (robot, target, held, barrier).
        parameters=(_robot, _bin, _cube, _barrier, _side),
        preconditions=frozenset({
            LiftedAtom(predicate=HOLDING, variables=(_robot, _cube)),
            LiftedAtom(predicate=BIN_AT_SIDE, variables=(_bin, _barrier, _side)),
        }),
        add_effects=frozenset({
            LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,)),
            LiftedAtom(predicate=NOT_HOLDING, variables=(_robot, _cube)),
            LiftedAtom(predicate=IN_BIN, variables=(_cube, _bin)),
            # Measured upstream on 20 throws: 15/15 that scored left the cube on a face.
            LiftedAtom(predicate=ON_GROUND, variables=(_cube,)),
            LiftedAtom(predicate=CUBE_AT_SIDE, variables=(_cube, _barrier, _side)),
        }),
        delete_effects=frozenset({
            LiftedAtom(predicate=HOLDING, variables=(_robot, _cube)),
        }),
        # GraspClear joins CubeAtSide as a functional update: whether the landed cube
        # is graspable is a fact of the physics, re-read from observation rather than
        # promised by the operator model.
        ignore_effects=frozenset({CUBE_AT_SIDE, GRASP_CLEAR}),
        param_dim=4,
        practice_cost=1.0,
    )

    # A third, robot-executed skill (not the human `ask_for_reset_cube_bin_only`): the
    # robot's own gripper-open primitive, upstream's `open_gripper` controller. Exists to
    # give the planner a way out of a near-miss grasp -- the gripper can end up commanded
    # closed on nothing (`HandEmpty` and `Holding` are both false). `ClosedEmpty` makes
    # that recovery state explicit, so an already-open gripper cannot be practiced as a
    # no-op and opening while genuinely holding a cube is not modeled incorrectly.
    OPEN_GRIPPER: ClassVar[Skill] = Skill(
        name="OpenGripper",
        parameters=(_robot, _cube),
        preconditions=frozenset({LiftedAtom(predicate=CLOSED_EMPTY, variables=(_robot, _cube))}),
        add_effects=frozenset({LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,))}),
        delete_effects=frozenset({LiftedAtom(predicate=CLOSED_EMPTY, variables=(_robot, _cube))}),
        param_dim=0,
        practice_cost=1.0,
    )

    @staticmethod
    def sample_params(*, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        """A state-independent draw of this skill's continuous parameters.

        State-independent by the `SkillProvider` contract: a learned sampler generates
        many candidates from this and then picks among them using the state.

        The toss is deliberately absent: its candidates come from
        `WideLongRangeTossProposal` via `Tossing3DSkillProvider.sample_params` on the
        barrier layout, and from `SameSideSkills.sample_params` on the same-side one,
        so a toss draw reaching this sampler is a routing bug worth failing loudly on.
        """
        skill = ground_skill.skill
        if skill == Tossing3DSkills.PICK_CUBE:
            return np.zeros(0)
        if skill == Tossing3DSkills.OPEN_GRIPPER:
            return np.zeros(0)
        raise ValueError(f"Unknown skill: {skill.name}")

    @staticmethod
    def compute_action(*, ground_skill: GroundSkill, params: np.ndarray, state: State) -> Action:
        """Realize a (ground skill, parameters) pair as this domain's five-slot vector.

        `state` is unused: the four toss parameters are bin-relative standoff, bin-relative
        yaw, joint-path speed, and release time rather than world coordinates.
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
