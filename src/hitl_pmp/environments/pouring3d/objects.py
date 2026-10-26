"""The fixed five-cube pouring task's relational object schema."""

from hitl_pmp.core.problem.environment.types import Object, Type


class PouringObjects:
    pose_features = ("x", "y", "z", "qw", "qx", "qy", "qz")
    cube_type = Type(name="pour_cube", feature_names=pose_features)
    vessel_type = Type(name="pour_vessel", feature_names=pose_features)
    robot_type = Type(name="pour_robot", feature_names=("x", "y", "yaw"))
    robot = Object(name="robot", type=robot_type)
    pitcher = Object(name="pitcher", type=vessel_type)
    tray = Object(name="tray", type=vessel_type)
    cubes = (
        Object(name="cube_0", type=cube_type),
        Object(name="cube_1", type=cube_type),
        Object(name="cube_2", type=cube_type),
        Object(name="cube_3", type=cube_type),
        Object(name="cube_4", type=cube_type),
    )
