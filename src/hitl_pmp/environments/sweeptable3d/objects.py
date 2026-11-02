"""SweepTable3D's fixed cast and sampler feature schemas."""

from typing import ClassVar

from hitl_pmp.core.problem.environment.types import Object, Type

NUM_BALLS = 5
NUM_CUBES = NUM_BALLS
_BALL_TYPE = Type(name="table_ball", feature_names=("x", "y", "z"))


class SweepTable3DObjects:
    """Five spheres; cube aliases preserve the inherited recovery and recording IDs."""

    robot_type: ClassVar[Type] = Type(
        name="table_robot",
        feature_names=("pos_base_x", "pos_base_y", "pos_base_rot", "pos_gripper"),
    )
    ball_type: ClassVar[Type] = _BALL_TYPE
    cube_type: ClassVar[Type] = ball_type
    wiper_type: ClassVar[Type] = Type(name="table_wiper", feature_names=("x", "y", "z", "yaw"))
    box_type: ClassVar[Type] = Type(
        name="table_box", feature_names=("x_min", "y_min", "x_max", "y_max", "rim_z")
    )
    scene_type: ClassVar[Type] = Type(name="table_scene", feature_names=("seed", "steps_taken"))

    robot: ClassVar[Object] = Object(name="robot", type=robot_type)
    balls: ClassVar[tuple[Object, ...]] = tuple(
        Object(name=f"cube_{index}", type=_BALL_TYPE) for index in range(NUM_BALLS)
    )
    cubes: ClassVar[tuple[Object, ...]] = balls
    wiper: ClassVar[Object] = Object(name="wiper_0", type=wiper_type)
    box: ClassVar[Object] = Object(name="box_0", type=box_type)
    scene: ClassVar[Object] = Object(name="scene", type=scene_type)
    ball_radius: ClassVar[float] = 0.01
    cube_half_size: ClassVar[float] = ball_radius

    @staticmethod
    def all_types() -> tuple[Type, ...]:
        objects = SweepTable3DObjects
        return (objects.robot_type, objects.cube_type, objects.wiper_type, objects.box_type)

    @staticmethod
    def all_objects() -> tuple[Object, ...]:
        """Objects used in grounding; the scene record is bookkeeping only."""
        objects = SweepTable3DObjects
        return (objects.robot, *objects.cubes, objects.wiper, objects.box)
