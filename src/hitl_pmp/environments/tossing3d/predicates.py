"""Tossing3D's symbolic layer: five predicates, all pure arithmetic over `core.State`.

**KINDER ships no symbolic model for Tossing3D.** `kinder_bilevel_planning.env_models.
dynamic3d` has one for base motion, one for `Shelf3D` and one for `Sweep3D`, and that is
all -- so unlike the controllers, which are upstream's verbatim, these are ours. The
shape they follow is `tidybot3d_shelf3D.py`'s: a small set of `Predicate`s over the
scene's objects, consumed by `LiftedOperator`s that are paired to upstream's controllers
(there, `LiftedSkill(PickTargetOperator, LiftedPickShelfController)`; here, `skills.py`'s
`Skill`s plus `skill_provider.py`).

Three of the five are ported from upstream's own `kinder_models.dynamic3d.shelf.
state_abstractions`, whose `HandEmpty`/`Holding`/`OnGround` classify the same TidyBot
state this domain reads -- thresholds included, so they are upstream's numbers rather
than ours. Two are genuinely new, and are called out below.

Where the port deviates from upstream, once, and deliberately: upstream's `Holding` also
requires the end-effector to be within 5 cm of the object, computed by forward kinematics
through a live `PyBulletSim`. A `core.Predicate.holds` is a pure function of `State` with
no simulator handle -- that is the whole reason planning can happen off-simulator -- so
that third conjunct is dropped, leaving upstream's other two (gripper closed past
`GRASP_THRESHOLD`, object lifted past `HOLDING_HEIGHT`). The result is *weaker* than
upstream's: it can call a cube "held" that is closed-gripper and airborne without being
grasped. With one cube and a grasp that either works or drops it on the floor, no
observed state distinguishes them, but it is a real difference and not a restatement.
"""

import numpy as np

from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.core.problem.tasks.types import Predicate

from .environment import Tossing3DEnvironment

# Upstream's own thresholds, from `kinder_models/dynamic3d/shelf/state_abstractions.py`.
# Named here rather than inlined so a future upstream bump has one place to check.
HANDEMPTY_TOL = 1e-3  # `handempty_tol`
GRASP_THRESHOLD = 0.1  # `GraspThreshold`
HOLDING_HEIGHT = 0.1  # the `state.get(target, "z") > 0.1` conjunct
ON_GROUND_TOL = 0.05  # `on_ground_tol`

# Ours: the range of throw standoffs, in metres. There is no upstream number to borrow --
# upstream simply hardcodes 1.35 in its own test, and its own `MOVE_TO_TARGET_DISTANCE_
# BOUNDS` of (0.5, 0.6) is a *grasping* standoff -- so this is the interval
# `scripts/tossing3d_oracle_demo.py --sweep` covers, which is the only range of throw
# standoffs anything in this repo has measured (11 standoffs, 1.20 to 1.65). It lives
# here rather than in `skills.py` so that `NearBin` and the `MoveToThrowPose` sampler are
# the same interval by construction; `skills.py` imports it back.
THROW_STANDOFF_BOUNDS = (1.20, 1.65)

# Upstream's `WAYPOINT_TOL` (`kinder_models/dynamic3d/utils.py:54`), which is how close
# `MoveToTargetGroundController.terminated()` requires the base to be to its own planned
# waypoint. Using upstream's own number means `NearBin` admits exactly the poses that
# controller is willing to stop at, rather than a tolerance invented here.
NEAR_BIN_TOLERANCE = 4 * 1e-2


class InGoalRegionClassifier:
    """The cube's centre lies inside the goal region's box.

    This is the domain's success criterion, and it is written to agree with KINDER's own
    `_check_goals()` **exactly**, not approximately. That is checkable rather than
    aspirational: for `["on", "cube_0", "blocks_goal_region"]` on a ground region,
    `_check_goals` (`envs.py:1053-1167`) builds `[x, y, z]` from the object's own state
    features and hands it to `Region.check_in_region`, which is a plain inclusive
    point-in-box test against `Region.bbox` (`objects/base.py:148-185`). Both halves are
    reproduced here: the box arrives in the `State` as the live `bbox`
    (`KinderBackend.goal_region_bbox`), and the comparison below is the same six
    inclusive bounds.

    Inclusive on purpose (`<=`, not `<`), because upstream's is.

    The box is read from the state rather than re-derived from the task JSON. The JSON's
    range is inflated by `ground_placement_threshold` (0.05 m per side, z clamped at 0)
    before it becomes a region, so the literal in the file is 2/3 of the true width on x
    -- the axis a toss controls -- and a predicate written against it scores KINDER
    successes as failures. That defect has already shipped once in this project's history.
    """

    @staticmethod
    def holds(*, state: State, cube: Object, goal_region: Object) -> bool:
        position = [state.get(obj=cube, feature_name=name) for name in ("x", "y", "z")]
        lower = [
            state.get(obj=goal_region, feature_name=name) for name in ("x_min", "y_min", "z_min")
        ]
        upper = [
            state.get(obj=goal_region, feature_name=name) for name in ("x_max", "y_max", "z_max")
        ]
        return all(
            low <= value <= high for value, low, high in zip(position, lower, upper, strict=True)
        )


class HandEmptyClassifier:
    """The gripper is open. Upstream's `HandEmpty`, verbatim including `handempty_tol`."""

    @staticmethod
    def holds(*, state: State, robot: Object) -> bool:
        gripper = state.get(obj=robot, feature_name="pos_gripper")
        return bool(np.isclose(gripper, 0.0, atol=HANDEMPTY_TOL))


class HoldingClassifier:
    """The gripper is closed and the cube is off the floor.

    Upstream's `Holding` minus its forward-kinematics conjunct -- see this module's
    docstring for why that one cannot be evaluated here and what it costs.
    """

    @staticmethod
    def holds(*, state: State, robot: Object, cube: Object) -> bool:
        gripper = state.get(obj=robot, feature_name="pos_gripper")
        return bool(
            gripper > GRASP_THRESHOLD and state.get(obj=cube, feature_name="z") > HOLDING_HEIGHT
        )


class OnGroundClassifier:
    """The cube is resting flat on the floor. Upstream's `OnGround`, verbatim.

    Flatness (`qx`/`qy` near zero) is upstream's own condition and is load-bearing rather
    than decorative: `pick_shelf` builds its grasp pose from the object's orientation, so
    a cube that came to rest on a corner is not a cube this grasp is modelled on.
    """

    @staticmethod
    def holds(*, state: State, cube: Object) -> bool:
        z = state.get(obj=cube, feature_name="z")
        bb_z = state.get(obj=cube, feature_name="bb_z")
        return bool(
            np.isclose(z - bb_z / 2, 0.0, atol=ON_GROUND_TOL)
            and np.isclose(state.get(obj=cube, feature_name="qx"), 0.0, atol=ON_GROUND_TOL)
            and np.isclose(state.get(obj=cube, feature_name="qy"), 0.0, atol=ON_GROUND_TOL)
        )


class ReachableClassifier:
    """The cube is on the robot's side of the barrier.

    **Ours, and the one that carries this domain's whole point.** The barrier is a 5 m
    immovable wall the base cannot pass, so a cube past it can never be picked up again --
    the irreversibility that makes this domain interesting is exactly "`Reachable` is a
    one-way door". Making `Pick` require it, and `Toss` delete it, is what stops a planner
    emitting the retrieve-and-retry plan the dynamics can never execute.

    Compared against the barrier's own live x rather than a constant: the barrier's pose
    is sampled from `barrier_init_region` per episode, so a literal would be right for one
    seed and quietly wrong for the next.
    """

    @staticmethod
    def holds(*, state: State, cube: Object, barrier: Object) -> bool:
        return bool(
            state.get(obj=cube, feature_name="x") < state.get(obj=barrier, feature_name="x")
        )


class NearBinClassifier:
    """The base is standing where a throw is thrown from: on the bin's axis, at a
    standoff inside the measured band.

    **Ours.** `move_to_target` has no symbolic model upstream, and its own termination
    condition (`_robot_is_close_to_pose`) is about the base having reached *its own
    planned waypoint*, which says nothing about where that waypoint was. So the effect a
    planner needs -- "the robot is now somewhere it can throw from" -- has to be stated
    here.

    It is stated as an exact characterisation of what `MoveToThrowPose` produces, rather
    than a loose proximity test, and that precision is load-bearing rather than
    fastidious. `move_to_target(bin, d, rot)` places the base at
    `(bin_x - d*cos(ang), bin_y - d*sin(ang))` with `ang = bin_yaw + rot`
    (`kinder_models/dynamic3d/utils.py:395`); `MoveToThrowPose` pins `rot = 0` and the
    bin's yaw range is `[[0, 0]]`, so the resulting pose is exactly `(bin_x - d, bin_y)`.
    Hence both conjuncts: **on the bin's axis** (`|dy| <= NEAR_BIN_TOLERANCE`) and **at a
    band standoff** (`|dx|` inside `THROW_STANDOFF_BOUNDS`, widened by the same
    tolerance). The tolerance is upstream's own `WAYPOINT_TOL`, i.e. exactly how far off
    its own waypoint that controller is willing to stop.

    **A plain distance test is not enough, and this was measured.** An earlier version
    tested only `1.0 <= hypot(dx, dy) <= 1.8`. After `Pick` -- which drives the base to
    the *cube*, off to one side -- the base sat at 1.76 m from the bin, inside that band,
    so `NearBin` was already true, the oracle skipped `MoveToThrowPose` entirely and threw
    from a pose facing 40 degrees away from the bin: the cube landed at
    `(0.9969, -0.7196)` and the episode scored a failure. The lateral conjunct rules that
    out by 0.72 m rather than by the 7 cm the distance test had to spare. This is exactly
    the over-permissive-operator-model defect class that
    `tests/environments/test_operator_dynamics_fidelity.py` exists for, caught here by
    `test_the_oracle_reproduces_the_recorded_coincident_landing_and_step_counts`.
    """

    @staticmethod
    def holds(*, state: State, robot: Object, target: Object) -> bool:
        dx = abs(
            state.get(obj=robot, feature_name="pos_base_x")
            - state.get(obj=target, feature_name="x")
        )
        dy = abs(
            state.get(obj=robot, feature_name="pos_base_y")
            - state.get(obj=target, feature_name="y")
        )
        low, high = THROW_STANDOFF_BOUNDS
        return bool(
            dy <= NEAR_BIN_TOLERANCE and low - NEAR_BIN_TOLERANCE <= dx <= high + NEAR_BIN_TOLERANCE
        )


# `Predicate.holds` is a positional `(state, objects)` callable per its interface contract
# (`Goal.is_satisfied` calls it that way), so each lambda below adapts that into a call to
# the relevant class's keyword-only `holds` -- exactly as Light Switch and Tossing Room do.
IN_GOAL_REGION = Predicate(
    name="InGoalRegion",
    types=(Tossing3DEnvironment.cube_type, Tossing3DEnvironment.goal_region_type),
    holds=lambda state, objects: InGoalRegionClassifier.holds(
        state=state, cube=objects[0], goal_region=objects[1]
    ),
)

HAND_EMPTY = Predicate(
    name="HandEmpty",
    types=(Tossing3DEnvironment.robot_type,),
    holds=lambda state, objects: HandEmptyClassifier.holds(state=state, robot=objects[0]),
)

HOLDING = Predicate(
    name="Holding",
    types=(Tossing3DEnvironment.robot_type, Tossing3DEnvironment.cube_type),
    holds=lambda state, objects: HoldingClassifier.holds(
        state=state, robot=objects[0], cube=objects[1]
    ),
)

ON_GROUND = Predicate(
    name="OnGround",
    types=(Tossing3DEnvironment.cube_type,),
    holds=lambda state, objects: OnGroundClassifier.holds(state=state, cube=objects[0]),
)

REACHABLE = Predicate(
    name="Reachable",
    types=(Tossing3DEnvironment.cube_type, Tossing3DEnvironment.barrier_type),
    holds=lambda state, objects: ReachableClassifier.holds(
        state=state, cube=objects[0], barrier=objects[1]
    ),
)

NEAR_BIN = Predicate(
    name="NearBin",
    types=(Tossing3DEnvironment.robot_type, Tossing3DEnvironment.bin_type),
    holds=lambda state, objects: NearBinClassifier.holds(
        state=state, robot=objects[0], target=objects[1]
    ),
)
