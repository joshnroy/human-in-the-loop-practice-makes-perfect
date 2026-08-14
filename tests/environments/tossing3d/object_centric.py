"""Hand-built KINDER `ObjectCentricState`s, so the offline half of this domain survives.

`ObjectCentricState` is constructible with no MuJoCo -- it is a plain object/feature
container -- and four of kinder-baselines' six Tossing3D classifiers are
`@staticmethod`s, so they can be called with no abstractor instance and therefore no
simulator process. That is what keeps the predicate boundary probes offline after the
swap to upstream's classifiers.

The geometry is the same scene `observations.py` uses, so an offline test and a
simulator-backed one are talking about the same numbers.
"""

from typing import Any

from .observations import BARRIER_X, BIN_X, CUBE_BB_Z, CUBE_START_X

__all__ = [
    "BARRIER_X",
    "BIN_X",
    "CUBE_BB_Z",
    "CUBE_START_X",
    "at_throw_pose",
    "object_centric_state",
]


def _kinder():
    """KINDER's own object types, imported on use rather than at module scope.

    Lazy because several tests assert that merely *constructing* a `Tossing3DEnvironment`
    pulls no simulator, and pytest imports every test module during collection -- so a
    module-scope `import kinder` here would put `mujoco` in `sys.modules` before those
    tests ever run and fail them from another file.
    """
    from kinder.envs.dynamic3d.object_types import (
        MujocoMovableObjectType,
        MujocoObjectTypeFeatures,
        MujocoTidyBotRobotObjectType,
    )
    from relational_structs import Object, ObjectCentricState

    return (
        MujocoObjectTypeFeatures,
        MujocoObjectTypeFeatures[MujocoTidyBotRobotObjectType],
        MujocoObjectTypeFeatures[MujocoMovableObjectType],
        MujocoTidyBotRobotObjectType,
        MujocoMovableObjectType,
        Object,
        ObjectCentricState,
    )


def _vector(*, features: tuple[str, ...], **overrides: float) -> list[float]:
    """A full feature vector, defaulting every unnamed feature to zero.

    Defaulting rather than requiring all 22 robot features keeps a test's call site
    about the two or three numbers it actually varies.
    """
    unknown = set(overrides) - set(features)
    assert not unknown, f"not features of this type: {sorted(unknown)}"
    return [float(overrides.get(name, 0.0)) for name in features]


def object_centric_state(  # noqa: PLR0914
    *,
    cube_x: float = CUBE_START_X,
    cube_y: float = 0.0,
    cube_z: float = 0.025,
    cube_qx: float = 0.0,
    cube_qy: float = 0.0,
    gripper: float = 0.0,
    base_x: float = 0.0,
    base_y: float = 0.0,
    base_rot: float = 0.0,
    bin_x: float = BIN_X,
    barrier_x: float = BARRIER_X,
) -> tuple[Any, dict[str, Any]]:
    """One scene snapshot as KINDER's own state type, plus its objects by name.

    Defaults are the initial state: cube flat on the floor at the near side, gripper
    open, robot at the origin.
    """
    (
        MujocoObjectTypeFeatures,
        _ROBOT_FEATURES,
        _MOVABLE_FEATURES,
        robot_type,
        movable_type,
        Object,
        ObjectCentricState,
    ) = _kinder()
    robot = Object("robot", robot_type)
    cube = Object("cube_0", movable_type)
    target_bin = Object("bin_0", movable_type)
    barrier = Object("cuboid_barrier", movable_type)
    state = ObjectCentricState(
        {
            robot: _vector(
                features=_ROBOT_FEATURES,
                pos_base_x=base_x,
                pos_base_y=base_y,
                pos_base_rot=base_rot,
                pos_gripper=gripper,
            ),
            cube: _vector(
                features=_MOVABLE_FEATURES,
                x=cube_x,
                y=cube_y,
                z=cube_z,
                qx=cube_qx,
                qy=cube_qy,
                bb_z=CUBE_BB_Z,
            ),
            target_bin: _vector(features=_MOVABLE_FEATURES, x=bin_x, y=0.0, z=0.1),
            barrier: _vector(features=_MOVABLE_FEATURES, x=barrier_x, y=0.0, z=0.05),
        },
        MujocoObjectTypeFeatures,
    )
    objects = {"robot": robot, "cube_0": cube, "bin_0": target_bin, "cuboid_barrier": barrier}
    return state, objects


def at_throw_pose(*, standoff: float, base_y: float = 0.0, base_rot: float = 0.0) -> bool:
    """Upstream's own `RobotAtThrowPose`, at an *achieved* standoff, with no simulator.

    The classifier is a `@staticmethod`, so it needs no abstractor instance -- which is
    what keeps the throw-pose band's boundary probes offline after the swap to upstream's
    classifiers.
    """
    from kinder_models.dynamic3d.tossing.state_abstractions import Tossing3DStateAbstractor

    state, objects = object_centric_state(base_x=BIN_X - standoff, base_y=base_y, base_rot=base_rot)
    return bool(
        Tossing3DStateAbstractor._check_at_throw_pose(  # noqa: SLF001
            state, objects["robot"], objects["bin_0"]
        )
    )
