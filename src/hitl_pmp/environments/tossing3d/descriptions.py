"""Tossing3D described for a coding agent: feature and parameter semantics, and the task.

Every number here is read from the constant that defines it, so the description moves
with the code and the KINDER pin rather than drifting from them. Only the prose is
written by hand, and each sentence restates a docstring elsewhere in this package.
"""

from typing import ClassVar

from hitl_pmp.core.method.types import FeatureSpec, ParameterSpec

from .environment import Tossing3DEnvironment
from .sides import BIN_RESET_REGION_BY_SIDE, Tossing3DSide, Tossing3DSides
from .wide_long_range_proposal import (
    FAR_STAND_X_LIMIT_M,
    WIDE_TOSS_RELEASE_MS_BOUNDS,
    WIDE_TOSS_SPEED_BOUNDS,
    WIDE_TOSS_STANDOFF_BOUNDS,
)


class Tossing3DDescriptions:
    """A static-method container, never instantiated."""

    # `[standoff, speed, release]`, in `Tossing3DToss.compute_action`'s order.
    TOSS_PARAMETERS: ClassVar[tuple[ParameterSpec, ...]] = (
        ParameterSpec(
            name="standoff",
            unit="m",
            meaning="Horizontal distance from the bin centre at which the mobile base "
            "stops, facing the bin, before throwing. Which of four bin-relative "
            "directions the robot stands in is chosen by the controller per state, "
            "not by this parameter.",
            lower=WIDE_TOSS_STANDOFF_BOUNDS[0],
            upper=WIDE_TOSS_STANDOFF_BOUNDS[1],
        ),
        ParameterSpec(
            name="speed",
            unit="deg/s",
            meaning="Joint-path speed of the arm's throwing swing.",
            lower=WIDE_TOSS_SPEED_BOUNDS[0],
            upper=WIDE_TOSS_SPEED_BOUNDS[1],
        ),
        ParameterSpec(
            name="release",
            unit="ms",
            meaning="Milliseconds after the start of the swing at which the gripper "
            "opens and releases the cube.",
            lower=WIDE_TOSS_RELEASE_MS_BOUNDS[0],
            upper=WIDE_TOSS_RELEASE_MS_BOUNDS[1],
        ),
    )

    # Per type, in each type's own `feature_names` order; a test pins the order. Units
    # and meanings restate `environment.py`'s comments and KINDER's own state readers.
    FEATURES: ClassVar[dict[str, tuple[FeatureSpec, ...]]] = {
        Tossing3DEnvironment.robot_type.name: (
            FeatureSpec(name="pos_base_x", unit="m", meaning="Mobile base x, world frame."),
            FeatureSpec(name="pos_base_y", unit="m", meaning="Mobile base y, world frame."),
            FeatureSpec(name="pos_base_rot", unit="rad", meaning="Mobile base yaw, world frame."),
            FeatureSpec(
                name="pos_gripper",
                unit="fraction [0, 1]",
                meaning="Commanded gripper actuator over its 0-255 range (KINDER's "
                "ctrl/255); the HandEmpty and Holding classifiers read it.",
            ),
        ),
        Tossing3DEnvironment.cube_type.name: (
            FeatureSpec(name="x", unit="m", meaning="Cube centre x, world frame."),
            FeatureSpec(name="y", unit="m", meaning="Cube centre y, world frame."),
            FeatureSpec(name="z", unit="m", meaning="Cube centre height above the floor."),
            FeatureSpec(name="qx", unit="quaternion", meaning="Cube orientation, x."),
            FeatureSpec(name="qy", unit="quaternion", meaning="Cube orientation, y."),
            FeatureSpec(name="qz", unit="quaternion", meaning="Cube orientation, z."),
            FeatureSpec(name="qw", unit="quaternion", meaning="Cube orientation, w."),
            FeatureSpec(name="bb_x", unit="m", meaning="Cube full bounding-box extent, x."),
            FeatureSpec(name="bb_y", unit="m", meaning="Cube full bounding-box extent, y."),
            FeatureSpec(name="bb_z", unit="m", meaning="Cube full bounding-box extent, z."),
        ),
        Tossing3DEnvironment.bin_type.name: (
            FeatureSpec(name="x", unit="m", meaning="Bin centre x, world frame."),
            FeatureSpec(name="y", unit="m", meaning="Bin centre y, world frame."),
            FeatureSpec(name="z", unit="m", meaning="Bin centre height."),
            *(
                FeatureSpec(
                    name=f"{axis}_{bound}",
                    unit="m",
                    meaning=f"{bound.capitalize()} {axis} of the scored goal box, world "
                    "frame: the cube scores when its centre lies inside this box. It "
                    "sits strictly inside the bin's footprint, so a cube can rest in "
                    "the bin without scoring.",
                )
                for bound in ("min", "max")
                for axis in ("x", "y", "z")
            ),
        ),
        Tossing3DEnvironment.barrier_type.name: (
            FeatureSpec(name="x", unit="m", meaning="Barrier centre x, world frame."),
            FeatureSpec(name="y", unit="m", meaning="Barrier centre y, world frame."),
            FeatureSpec(name="z", unit="m", meaning="Barrier centre height."),
            FeatureSpec(name="bb_x", unit="m", meaning="Barrier full thickness along x."),
        ),
        Tossing3DEnvironment.scene_type.name: (
            FeatureSpec(name="seed", unit="integer", meaning="Scene seed of this episode."),
            FeatureSpec(name="steps_taken", unit="count", meaning="Skills executed since reset."),
        ),
        Tossing3DSides.type.name: (),
    }

    @staticmethod
    def task() -> str:
        robot_side = BIN_RESET_REGION_BY_SIDE[Tossing3DSide.ROBOT].ranges[0]
        far_side = BIN_RESET_REGION_BY_SIDE[Tossing3DSide.OPPOSITE].ranges[0]
        return f"""# The Tossing3D task

A mobile manipulator (a TidyBot: wheeled base plus a 7-joint arm and gripper) in a
simulated room must get one cube into one bin by throwing it. A barrier wall runs along
y at fixed x and divides the room into the robot's side and the far side. The robot
cannot cross the barrier.

## Goal and evaluation
- The goal is `InBin(cube_0, bin_0)`: the cube's centre lies inside the scored goal
  box carried on the bin's `x_min..z_max` features. That box is strictly inside the
  bin's footprint, so "the cube is in the bin" does not imply "the cube scores".
- Evaluation tasks are fresh scenes the robot never practised on, with the bin on the
  far side of the barrier. Each evaluation task is solved or not; nothing learned
  during evaluation is kept.
- Practice happens in cycles. Each cycle is one practice period of at most a fixed
  number of skill executions, then your code is refitted (that is this session), then
  an evaluation sweep runs the code version you leave behind.

## Skills
- `PickCube`: picks the cube from the floor on the robot's side. No parameters.
- `MoveToTossLocationAndToss`: drives the base to a stand `standoff` metres from the
  bin, facing it, and throws with the given `speed` and `release`. Three continuous
  parameters; this is the skill whose parameters are learned. Its controller picks
  which of four bin-relative directions to stand in. A far-side bin can only be
  thrown at from a stand west of x = {FAR_STAND_X_LIMIT_M}, so standoffs below
  `bin_x - {FAR_STAND_X_LIMIT_M}` have no legal stand for it.
- `OpenGripper`: opens a gripper that closed on nothing, so the robot can pick again.
- `ask_for_reset_cube_bin_only`: a human puts the cube back on the robot's side and
  re-places the bin upright on the side the planner chose, at a cost. Not available
  during evaluation.

## The one-way door
A thrown cube that lands on the far side of the barrier cannot be picked up again by
the robot. Only the human reset brings it back. A miss on the far side therefore ends
the robot's ability to practise until it pays for a reset.

## What success means per skill
A skill execution succeeds when every add effect its operator promises holds in the
state observed right after it. For the toss that includes `InBin(cube, bin)`, so a toss
succeeds only if it scores; landing in the bin outside the scored box is a failure.

## Placement regions
- Robot-side bin placements: x in [{robot_side[0]}, {robot_side[2]}],
  y in [{robot_side[1]}, {robot_side[3]}].
- Far-side bin placements: x in [{far_side[0]}, {far_side[2]}],
  y in [{far_side[1]}, {far_side[3]}].
"""
