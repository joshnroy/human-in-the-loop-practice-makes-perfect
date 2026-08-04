"""Tossing3D's symbolic layer. Every classifier here is a pure function of the feature
vectors in `State` -- no simulator call, no environment config -- which is what lets the
whole symbolic layer be tested in CI on a machine with no MuJoCo installed."""

import numpy as np

from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.core.problem.tasks.types import Predicate

from .environment import Tossing3DEnvironment


class InGoalRegionClassifier:
    """Whether the cube is inside KINDER's `blocks_goal_region` axis-aligned box.

    This is the benchmark's own success criterion: the inclusive per-axis test below is
    the one `Region.check_in_region` performs, and the box it runs against is upstream's
    own computed `Region.bbox`, read by `KinderBackend.goal_region_bounds` and carried
    into the state as `goal_region`'s features.

    That box is **not** the task JSON's `ranges[0]` -- KINDER inflates the range by
    `ground_placement_threshold` (0.05 m) per side before testing anything, so for `o1`
    the real region is x in [1.85, 2.15], y in [-0.15, 0.15], z in [0, 0.15], not the
    JSON's [1.90, 2.10] x [-0.10, 0.10] x [0, 0.10]. This file used to claim verified
    equivalence while reading the raw JSON, which understated every success rate; the
    equivalence is only true now that the bbox is read directly.

    Two tests guard different halves of that, and neither substitutes for the other:
    `test_goal_region_bounds_match_kinders_own_region` pins the box element-wise against
    upstream's `Region.bbox` (the only check that catches a *wrong box*), while the
    offline boundary tests in `test_predicates.py` pin this containment arithmetic
    inside the 5 cm inflation shells (the only checks that catch wrong *comparisons* --
    a simulator random walk lands the cube deep inside or far outside almost every time
    and cannot see those shells).

    Note that the region and the bin overlap: the bin's 0.30 m footprint spans
    x in [2.08, 2.38], so a cube resting on the bin floor at x in [2.08, 2.15] satisfies
    KINDER's goal, while a full-power toss out to x ~ 2.22 overshoots it. This domain
    uses KINDER's criterion verbatim rather than substituting an "in the bin" test, so
    that a number reported here is a number about the benchmark.
    """

    @staticmethod
    def holds(*, state: State, cube: Object, region: Object) -> bool:
        position = np.array([state.get(obj=cube, feature_name=axis) for axis in ("x", "y", "z")])
        low = np.array([
            state.get(obj=region, feature_name=name) for name in ("x_min", "y_min", "z_min")
        ])
        high = np.array([
            state.get(obj=region, feature_name=name) for name in ("x_max", "y_max", "z_max")
        ])
        return bool(np.all(position >= low) and np.all(position <= high))


class HandEmptyClassifier:
    @staticmethod
    def holds(*, state: State, robot: Object) -> bool:
        return int(round(state.get(obj=robot, feature_name="holding"))) == 0


class HoldingClassifier:
    @staticmethod
    def holds(*, state: State, robot: Object, cube: Object) -> bool:
        del cube  # single-cube variant: `holding` cannot be ambiguous about which
        return int(round(state.get(obj=robot, feature_name="holding"))) == 1


class ReachableClassifier:
    """Whether the cube is still on the robot's side of the immovable barrier.

    This is the domain's irreversibility, expressed symbolically. The barrier is 5 m
    wide across y and the base cannot pass it, so a cube whose x exceeds the barrier's
    is gone for good -- successfully (in the goal region) or not (overshot into the
    bin, or short of the region but past the barrier). Making `Pick` require this is
    what stops the planner emitting a retrieve-and-retry plan the dynamics can never
    execute.
    """

    @staticmethod
    def holds(*, state: State, cube: Object, barrier: Object) -> bool:
        return bool(
            state.get(obj=cube, feature_name="x") < state.get(obj=barrier, feature_name="x")
        )


class AtThrowPoseClassifier:
    """Whether the base is standing at the throw pose: roughly `throw_standoff` metres
    from the bin. Both the standoff and its tolerance are `Tossing3DEnvironment`
    ClassVars rather than constructor fields precisely so this classifier -- whose
    signature is only (state, objects) -- can read them; see their definition for why
    they are structural constants of the KINDER scene and not per-run config."""

    @staticmethod
    def holds(*, state: State, robot: Object, bin_object: Object) -> bool:
        base = np.array([state.get(obj=robot, feature_name=name) for name in ("base_x", "base_y")])
        bin_position = np.array([
            state.get(obj=bin_object, feature_name=axis) for axis in ("x", "y")
        ])
        distance = float(np.linalg.norm(base - bin_position))
        return (
            abs(distance - Tossing3DEnvironment.throw_standoff)
            <= Tossing3DEnvironment.throw_pose_tolerance
        )


# Predicate.holds is a positional (state, objects) callable per its interface contract
# (Goal.is_satisfied calls it that way) -- each lambda below just adapts that into a
# call to the relevant class's keyword-only holds, exactly like Tossing Room's
# predicates.py.
IN_GOAL_REGION = Predicate(
    name="InGoalRegion",
    types=(Tossing3DEnvironment.cube_type, Tossing3DEnvironment.region_type),
    holds=lambda state, objects: InGoalRegionClassifier.holds(
        state=state, cube=objects[0], region=objects[1]
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

REACHABLE = Predicate(
    name="Reachable",
    types=(Tossing3DEnvironment.cube_type, Tossing3DEnvironment.barrier_type),
    holds=lambda state, objects: ReachableClassifier.holds(
        state=state, cube=objects[0], barrier=objects[1]
    ),
)

AT_THROW_POSE = Predicate(
    name="AtThrowPose",
    types=(Tossing3DEnvironment.robot_type, Tossing3DEnvironment.bin_type),
    holds=lambda state, objects: AtThrowPoseClassifier.holds(
        state=state, robot=objects[0], bin_object=objects[1]
    ),
)
