"""Tossing3D's symbolic layer: five predicates, all pure arithmetic over `core.State`.

**KINDER now ships a symbolic model for Tossing3D**, which it did not when this module
was written: `kinder_models.dynamic3d.tossing.state_abstractions` declares
`HandEmpty`/`Holding`/`OnGround`/`MovableInGoalRegion`/`MovableIsDownX`/`IsThrowTarget`,
and `kinder_bilevel_planning.env_models.dynamic3d.tidybot3d_tossing3D` builds two
operators over them. This module is the port of that model into `core` types, so the two
describe the same domain and their results can be compared.

The correspondence, and the two places it is deliberately not one-to-one:

| here | upstream | note |
| --- | --- | --- |
| `InBin` | `MovableInGoalRegion` | same box, read from `State` rather than the sim |
| `HandEmpty` | `HandEmpty` | upstream's threshold verbatim |
| `Holding` | `Holding` | minus the forward-kinematics conjunct; see below |
| `OnGround` | `OnGround` | ported including the cube-symmetry branch |
| `Reachable` | `MovableIsDownX` | same arithmetic, named for what it means here |
| -- | `IsThrowTarget` | **not needed here**; see below |

**`IsThrowTarget` has no counterpart because this domain's types already say it.**
Upstream gives the cube, the bin and the barrier all one `MujocoMovableObjectType`, so a
grounder is free to bind a throw's `?target` to the cube being thrown -- a discrete
mistake no amount of continuous sampling recovers from, measured upstream on 6/9 seeds.
Here `bin_type`, `cube_type` and `barrier_type` are distinct `core.Type`s and the
composed toss's `?target` is typed `bin_type`, so the wrong binding is not constructible.
Adding a static, always-true predicate to re-state a type constraint would be a
tautology of exactly the kind this module removed once already.

**`RobotAtSuccessfulThrowPose` is gone, with the skill it was the effect of.** It named
the pose between `MoveToThrowPose` and `Toss`; upstream composed those two into one
controller, so there is no longer a state between them for any predicate to describe. The
measured calibration that backed it -- `THROW_RANGE`, `THROW_RANGE_MIN`/`MAX`, the
overshoot/shortfall margins, `THROW_STANDOFF_BOUNDS` -- went with it. Those numbers are
not wrong; they simply describe a decomposition this domain no longer runs, and they stay
readable in git history and in the experiment logs that cite them.

## The stated assumption: the bin's interior **is** the scored region

Every predicate below that needs a landing box reads it off the **bin**, and no predicate
and no operator takes a goal region as an argument. That is an assumption this domain
makes, deliberately, and it is **false in general**.

What is true in general is that KINDER scores `["on", "cube_0", "blocks_goal_region"]` --
a ground *region*, which is a box in the scene, entirely independent of where the bin is.
Under upstream's stock `Tossing3D-o1` the two are 23 cm apart: the bin sits at x = 2.2305
while `blocks_goal_region` inflates to x in [1.85, 2.15], so **a cube that lands in the bin
scores a failure** and only one that misses the bin and lands on bare floor scores a
success. There is a commit in this repo's history titled "Tossing3D: the bin is scenery"
for exactly that reason. Our default `scripts/task_configs/Tossing3D-o1-coincident.json`
is *named* for putting the bin back at x = 2.0 so that the two coincide -- measured live,
they then agree to 0.1 mm.

So this is a modelling choice, not an observation. Three consequences, stated rather than
left to be discovered:

1. **Fidelity is preserved, because the number is unchanged.** The box these classifiers
   read is still the live `Region.bbox` of `blocks_goal_region`
   (`KinderBackend.goal_region_bbox`), carried in the `State` on the bin object. Nothing is
   re-derived from the bin's pose or from a literal, so `InBin` still agrees with
   `_check_goals()` exactly, on **both** task configs. What was dropped is the *symbolic*
   dependence -- an object a planner had to bind and no skill could act on -- not the
   classifier's access to scene geometry.
2. **Under `Tossing3DTaskConfig.STOCK` the name lies, and the bin object is internally
   inconsistent.** Its own `x` (2.2305) sits outside its own `x_max` (2.15). `InBin` is then
   true exactly when the cube is *not* in the bin. The predicate remains *correct* -- it
   still scores what KINDER scores -- but it is misnamed there, and any reading of a stock
   run's symbolic trace has to account for that. Stock stays selectable because every
   number in `docs/kinder-environment-validation.md` was measured on it.
3. **Wherever the bin is taken to be movable, moving it moves the goal.** kindergarden#126
   moves the bin, and the rest of this module is written so the accepted band follows live
   scene geometry rather than a literal; under this assumption the scored box is *part of*
   that geometry, so relocating the bin retargets the throw rather than leaving a stale
   target behind. That is the intended reading of a move-the-bin task -- "put the cube in
   the bin, wherever the bin is" -- rather than a problem to work around. It is worth
   stating because the opposite reading, a fixed goal the bin merely decorates, is exactly
   what the stock config implements.

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


class CubeSymmetry:
    """How far a cube is from resting flat on one of its faces.

    A static-method container, never instantiated, same as every other business-logic
    class in this project.

    Upstream's `kinder_models.dynamic3d.cube_symmetry.cube_tilt_from_upright` computes
    this by composing the measured rotation with all 24 rotations that map a cube onto
    itself (`scipy`'s octahedral group) and taking the smallest resulting `qx^2 + qy^2`.
    That value is `(1 - R[2, 2]) / 2` for the composed rotation, and composing with the
    group sweeps the composed matrix's third column over the six signed body axes -- so
    the minimum is `(1 - max_i |R[2, i]|) / 2`, in closed form, with no group and no
    `scipy`.

    Written that way rather than ported literally so this module keeps depending on
    nothing but numpy, which is what lets the whole symbolic layer run on a machine with
    no KINDER. `test_kinder_fidelity.py` checks the two agree to floating point over
    random rotations whenever the simulator is installed, so the algebra is verified
    rather than asserted.
    """

    @staticmethod
    def tilt_from_upright(*, rotation: tuple[float, float, float, float]) -> float:
        """Zero for any face-down rest at any yaw; larger nearer an edge or a corner."""
        x, y, z, w = rotation
        third_row = (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))
        return float((1.0 - max(abs(value) for value in third_row)) / 2.0)


class InBinClassifier:
    """The cube's centre lies inside the bin's interior.

    This is the domain's success criterion, and it is written to agree with KINDER's own
    `_check_goals()` **exactly**, not approximately. That is checkable rather than
    aspirational: for `["on", "cube_0", "blocks_goal_region"]` on a ground region,
    `_check_goals` (`envs.py:1053-1167`) builds `[x, y, z]` from the object's own state
    features and hands it to `Region.check_in_region`, which is a plain inclusive
    point-in-box test against `Region.bbox` (`objects/base.py:148-185`). Both halves are
    reproduced here: the box arrives in the `State` as the live `bbox`
    (`KinderBackend.goal_region_bbox`), carried on the bin, and the comparison below is the
    same six inclusive bounds.

    **"The bin's interior" is this domain's assumption, not KINDER's arrangement** -- see
    this module's docstring for what it buys, and for the stock config where the name is
    wrong while the arithmetic stays right. The name is chosen to read honestly under the
    default config every committed number on this domain is measured at, where the bin and
    the scored region coincide to 0.1 mm and "the cube is in the bin" is exactly what
    KINDER scores. The alternative -- keeping `InGoalRegion` -- would name a thing that no
    longer exists as an object anywhere in the domain, which is worse: it would invite a
    reader to look for a goal-region argument that is gone.

    Inclusive on purpose (`<=`, not `<`), because upstream's is.

    The box is read from the state rather than re-derived from the task JSON, and *never*
    from the bin's own pose. The JSON's range is inflated by `ground_placement_threshold`
    (0.05 m per side, z clamped at 0) before it becomes a region, so the literal in the
    file is 2/3 of the true width on x -- the axis a toss controls -- and a predicate
    written against it scores KINDER successes as failures. That defect has already shipped
    once in this project's history. Deriving the box from the bin's pose plus a half-extent
    would be the same defect in a new costume.
    """

    @staticmethod
    def holds(*, state: State, cube: Object, target: Object) -> bool:
        position = [state.get(obj=cube, feature_name=name) for name in ("x", "y", "z")]
        lower = [state.get(obj=target, feature_name=name) for name in ("x_min", "y_min", "z_min")]
        upper = [state.get(obj=target, feature_name=name) for name in ("x_max", "y_max", "z_max")]
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
    """The cube is resting on the floor, on any one of its faces.

    Upstream's `OnGround`, ported rather than paraphrased -- including the branch that
    decides whether an object *is* a cube by comparing its three bounding-box extents,
    so this and upstream's classifier are the same function of the same features.

    **This is not the test this domain used to run, and the change is load-bearing.**
    It used to require `qx` and `qy` near zero, which is "resting on the face it started
    on". That was right while `OnGround` only ever described the *initial* cube: the old
    `Pick` required it, nothing added it. The composed toss adds it -- kb#113's operator
    model records that 15/15 scoring throws left the cube resting on a face -- and a
    thrown cube lands on whichever face it likes. Under the old test that add effect
    would read false on most successful throws, so EES would label a throw that scored a
    failure and train its sampler against noise.

    Zero tilt for a face-down rest at any yaw is what the symmetry group buys: the cube's
    roll and pitch carry no information once a face is down, so composing the measured
    rotation with the 24 rotations that map a cube onto itself and taking the smallest
    residual `qx^2 + qy^2` is the pose-independent form of "flat".
    """

    @staticmethod
    def holds(*, state: State, cube: Object) -> bool:
        z = state.get(obj=cube, feature_name="z")
        bb_z = state.get(obj=cube, feature_name="bb_z")
        if not np.isclose(z - bb_z / 2, 0.0, atol=ON_GROUND_TOL):
            return False
        rotation = (
            state.get(obj=cube, feature_name="qx"),
            state.get(obj=cube, feature_name="qy"),
            state.get(obj=cube, feature_name="qz"),
            state.get(obj=cube, feature_name="qw"),
        )
        extents = [state.get(obj=cube, feature_name=name) for name in ("bb_x", "bb_y", "bb_z")]
        # Only a cube's faces are interchangeable; anything else keeps the strict test.
        if not np.allclose(extents, extents[0]):
            return bool(
                np.isclose(rotation[0], 0.0, atol=ON_GROUND_TOL)
                and np.isclose(rotation[1], 0.0, atol=ON_GROUND_TOL)
            )
        return bool(CubeSymmetry.tilt_from_upright(rotation=rotation) < ON_GROUND_TOL)


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


# `Predicate.holds` is a positional `(state, objects)` callable per its interface contract
# (`Goal.is_satisfied` calls it that way), so each lambda below adapts that into a call to
# the relevant class's keyword-only `holds` -- exactly as Light Switch and Tossing Room do.
IN_BIN = Predicate(
    name="InBin",
    types=(Tossing3DEnvironment.cube_type, Tossing3DEnvironment.bin_type),
    holds=lambda state, objects: InBinClassifier.holds(
        state=state, cube=objects[0], target=objects[1]
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
