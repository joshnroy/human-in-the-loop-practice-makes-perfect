"""SweepOffShelf3D's types and its fixed cast of objects.

The cast is fixed by the KINDER task JSON (`SweepOffShelf3D-o5`): one TidyBot, five
cubes, one wiper and one box. Features are the few the symbolic layer and the samplers
read; every predicate is decided once per state at the simulator boundary (see
`predicates.py`), so no classifier reads these vectors directly.
"""

from typing import ClassVar

from hitl_pmp.core.problem.environment.types import Object, Type

NUM_CUBES = 5
_CUBE_TYPE = Type(name="shelf_cube", feature_names=("x", "y", "z"))


class SweepOffShelf3DObjects:
    """A static-method container, never instantiated."""

    robot_type: ClassVar[Type] = Type(
        name="shelf_robot",
        feature_names=("pos_base_x", "pos_base_y", "pos_base_rot", "pos_gripper"),
    )
    cube_type: ClassVar[Type] = _CUBE_TYPE
    wiper_type: ClassVar[Type] = Type(name="shelf_wiper", feature_names=("x", "y", "z", "yaw"))
    # The box's inner footprint and rim height, as `cube_in_box` scores them.
    box_type: ClassVar[Type] = Type(
        name="shelf_box", feature_names=("x_min", "y_min", "x_max", "y_max", "rim_z")
    )
    # `seed` rebuilds an episode-initial scene; `steps_taken` lets set_state refuse
    # any other state.
    scene_type: ClassVar[Type] = Type(name="shelf_scene", feature_names=("seed", "steps_taken"))

    robot: ClassVar[Object] = Object(name="robot", type=robot_type)
    cubes: ClassVar[tuple[Object, ...]] = tuple(
        Object(name=f"cube_{i}", type=_CUBE_TYPE) for i in range(NUM_CUBES)
    )
    wiper: ClassVar[Object] = Object(name="wiper_0", type=wiper_type)
    box: ClassVar[Object] = Object(name="box_0", type=box_type)
    scene: ClassVar[Object] = Object(name="scene", type=scene_type)

    cube_half_size: ClassVar[float] = 0.01

    @staticmethod
    def all_types() -> tuple[Type, ...]:
        o = SweepOffShelf3DObjects
        return (o.robot_type, o.cube_type, o.wiper_type, o.box_type)

    @staticmethod
    def all_objects() -> tuple[Object, ...]:
        """Every object the symbolic layer grounds over (not the scene record)."""
        o = SweepOffShelf3DObjects
        return (o.robot, *o.cubes, o.wiper, o.box)
