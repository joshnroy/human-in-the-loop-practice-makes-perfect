"""Fixed cooked-dumpling proxy cast, using core Object and Type contracts."""

from typing import ClassVar

from hitl_pmp.core.problem.environment.types import Object, Type

NUM_FOOD = 5
NUM_CUBES = NUM_FOOD
_FOOD_TYPE = Type(name="rack_food", feature_names=("x", "y", "z", "qw", "qx", "qy", "qz"))


class RackServing3DObjects:
    robot_type: ClassVar[Type] = Type(
        name="rack_robot",
        feature_names=("pos_base_x", "pos_base_y", "pos_base_rot", "pos_gripper"),
    )
    food_type: ClassVar[Type] = _FOOD_TYPE
    scraper_type: ClassVar[Type] = Type(name="rack_scraper", feature_names=("x", "y", "z", "yaw"))
    tray_type: ClassVar[Type] = Type(
        name="serving_tray", feature_names=("x_min", "y_min", "x_max", "y_max", "rim_z")
    )
    rack_type: ClassVar[Type] = Type(
        name="cooling_rack", feature_names=("x_min", "y_min", "x_max", "y_max", "top_z")
    )
    scene_type: ClassVar[Type] = Type(
        name="rack_scene", feature_names=("seed", "steps_taken", "elapsed_seconds", "human_cost")
    )
    robot: ClassVar[Object] = Object(name="robot", type=robot_type)
    food: ClassVar[tuple[Object, ...]] = tuple(
        Object(name=f"cube_{index}", type=_FOOD_TYPE) for index in range(NUM_FOOD)
    )
    scraper: ClassVar[Object] = Object(name="wiper_0", type=scraper_type)
    tray: ClassVar[Object] = Object(name="box_0", type=tray_type)
    rack: ClassVar[Object] = Object(name="rack_0", type=rack_type)
    scene: ClassVar[Object] = Object(name="scene", type=scene_type)
    cube_type: ClassVar[Type] = food_type
    wiper_type: ClassVar[Type] = scraper_type
    box_type: ClassVar[Type] = tray_type
    cubes: ClassVar[tuple[Object, ...]] = food
    wiper: ClassVar[Object] = scraper
    box: ClassVar[Object] = tray

    @staticmethod
    def all_types() -> tuple[Type, ...]:
        objects = RackServing3DObjects
        return (
            objects.robot_type,
            objects.food_type,
            objects.scraper_type,
            objects.tray_type,
            objects.rack_type,
        )

    @staticmethod
    def all_objects() -> tuple[Object, ...]:
        objects = RackServing3DObjects
        return (objects.robot, *objects.food, objects.scraper, objects.tray, objects.rack)
