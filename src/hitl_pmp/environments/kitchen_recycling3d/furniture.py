"""Task-local MimicLabs desk and Replica chairs, with conservative collisions."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, ClassVar

from kinder.envs.dynamic3d import envs

FURNITURE_POSITION = (2.5, 1.35, 0.0)
FURNITURE_SCALE = (0.6, 0.6, 0.8)
CHAIR_POSITIONS = ((2.15, 0.67), (2.85, 0.67), (2.15, 2.03), (2.85, 2.03))


def rectangular_kitchen(*, root: Any) -> None:
    world = root.find("worldbody")
    assert world is not None
    for parent in world.iter():
        for geom in list(parent.findall("geom")):
            if geom.get("name") in {
                "living_room",
                "wall_decoration",
                "wall_leftcorner_visual",
                "wall_rightcorner_visual",
            }:
                parent.remove(geom)
    room = world.find("body[@name='tossing_room']")
    assert room is not None
    for child in list(room):
        room.remove(child)
    walls = {
        "kitchen_wall_west": ("-2 0 1.5", "0.01 3 1.5"),
        "kitchen_wall_east": ("4.2 0 1.5", "0.01 3 1.5"),
        "kitchen_wall_south": ("1.1 -3 1.5", "3.1 0.01 1.5"),
        "kitchen_wall_north": ("1.1 3 1.5", "3.1 0.01 1.5"),
    }
    for name, (position, size) in walls.items():
        ET.SubElement(
            room,
            "geom",
            name=name,
            type="box",
            pos=position,
            size=size,
            rgba="0.82 0.82 0.8 1",
            contype="1",
            conaffinity="1",
            group="1",
            solref="0.001 1",
            solimp="0.99 0.99 0.001",
            priority="1",
        )
    stove = ET.SubElement(world, "body", name="kitchen_stove", pos="-1.6 0.6 0")
    parts = {
        "stove_body": ("0 0 0.44", "0.32 0.45 0.44", "0.65 0.67 0.69 1"),
        "stove_cooktop": ("0 0 0.89", "0.33 0.46 0.015", "0.06 0.065 0.07 1"),
        "stove_oven_window": ("0.326 0 0.43", "0.008 0.33 0.23", "0.04 0.055 0.065 1"),
        "stove_handle": ("0.36 0 0.72", "0.028 0.32 0.02", "0.8 0.8 0.82 1"),
        "stove_backguard": ("-0.31 0 1.02", "0.02 0.46 0.13", "0.65 0.67 0.69 1"),
    }
    for name, (position, size, color) in parts.items():
        ET.SubElement(
            stove,
            "geom",
            name=name,
            type="box",
            pos=position,
            size=size,
            rgba=color,
            contype="1",
            conaffinity="1",
            group="1",
        )
    for index, (horizontal, lateral) in enumerate((
        (-0.15, -0.22),
        (-0.15, 0.22),
        (0.15, -0.22),
        (0.15, 0.22),
    )):
        ET.SubElement(
            stove,
            "geom",
            name=f"stove_burner_{index}",
            type="cylinder",
            pos=f"{horizontal} {lateral} 0.914",
            size="0.105 0.009",
            rgba="0.4 0.41 0.42 1",
            contype="0",
            conaffinity="0",
            group="1",
        )
        ET.SubElement(
            stove,
            "geom",
            type="cylinder",
            pos=f"{horizontal} {lateral} 0.925",
            size="0.073 0.006",
            rgba="0.045 0.045 0.05 1",
            contype="0",
            conaffinity="0",
            group="1",
        )
        ET.SubElement(
            stove,
            "geom",
            name=f"stove_knob_{index}",
            type="cylinder",
            pos=f"0.34 {0.27 - index * 0.18} 0.81",
            size="0.032 0.015",
            quat="0.707106781187 0 0.707106781187 0",
            rgba="0.1 0.1 0.1 1",
            contype="0",
            conaffinity="0",
            group="1",
        )
    for index, lateral in enumerate((-0.8, 2.0)):
        counter = ET.SubElement(
            world, "body", name=f"kitchen_counter_{index}", pos=f"-1.6 {lateral} 0"
        )
        ET.SubElement(
            counter,
            "geom",
            name=f"counter_{index}_cabinet",
            type="box",
            pos="0 0 0.43",
            size="0.32 0.55 0.43",
            rgba="0.72 0.63 0.49 1",
            contype="1",
            conaffinity="1",
            group="1",
        )
        ET.SubElement(
            counter,
            "geom",
            name=f"counter_{index}_top",
            type="box",
            pos="0 0 0.88",
            size="0.34 0.57 0.02",
            rgba="0.83 0.82 0.78 1",
            contype="1",
            conaffinity="1",
            group="1",
        )


class VisualFurniture:
    """The desk's and chairs' meshes, or, in a scene that will never be rendered, a
    stand-in for each.

    The meshes are visual only (`contype="0"`, `conaffinity="0"`, on bodies with no
    joint), so no physics reads them, but compiling them costs about 5 s per build and
    every `mj_step` carries them. A stand-in is an invisible, non-colliding 1 mm sphere
    with the mesh geom's name, in the mesh geom's place: MuJoCo numbers geoms in tree
    order, and the scene's collider objects are named by geom index
    (`collider:recycling_chair_0:52`), so dropping the geoms outright would rename them
    in every observation and every recorded state."""

    @staticmethod
    def add(*, body: Any, mesh: str, geom: dict[str, str], visual: bool) -> None:
        if visual:
            named = {"name": geom["name"]} if "name" in geom else {}
            ET.SubElement(body, "geom", {**named, "type": "mesh", "mesh": mesh, **geom})
            return
        stand_in = {key: value for key, value in geom.items() if key in ("name", "pos")}
        ET.SubElement(
            body,
            "geom",
            {
                "type": "sphere",
                "size": "0.001",
                "rgba": "0 0 0 0",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
                **stand_in,
            },
        )

    @staticmethod
    def asset(*, assets: Any, tag: str, visual: bool, **attributes: str) -> None:
        if visual:
            ET.SubElement(assets, tag, dict(attributes))


def add_lab_furniture(*, xml: str, visual: bool = True) -> str:
    root = ET.fromstring(xml)
    rectangular_kitchen(root=root)
    assets = root.find("asset")
    world = root.find("worldbody")
    assert assets is not None and world is not None
    library = Path(envs.__file__).parent / "models/assets"
    source = library / "mimiclabs_scenes/meshes/lab3"
    desk = ET.SubElement(
        world,
        "body",
        name="recycling_lab_desk",
        pos=" ".join(map(str, FURNITURE_POSITION)),
        quat="0.707106781187 0 0 0.707106781187",
    )
    colors = ("0.12 0.12 0.12 1", "0.65 0.42 0.23 1", "0.16 0.14 0.12 1", "0.45 0.3 0.17 1")
    for index, color in enumerate(colors):
        mesh = f"recycling_lab_desk_{index}"
        VisualFurniture.asset(
            assets=assets,
            tag="mesh",
            visual=visual,
            name=mesh,
            file=str(source / f"scene2_table_{index}.obj"),
            scale=" ".join(map(str, FURNITURE_SCALE)),
        )
        VisualFurniture.add(
            body=desk,
            mesh=mesh,
            geom={
                "name": f"{mesh}_visual",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
                "rgba": color,
            },
            visual=visual,
        )
    ET.SubElement(
        desk,
        "geom",
        name="recycling_desk_clearance",
        type="box",
        pos="0 0 0.37",
        size="0.352 0.87 0.37",
        rgba="0 0 0 0",
        contype="1",
        conaffinity="1",
        group="3",
    )
    chair_source = library / "replica_objects/frl_apartment_chair_04"
    VisualFurniture.asset(
        assets=assets,
        tag="mesh",
        visual=visual,
        name="recycling_chair_mesh",
        file=str(chair_source / "frl_apartment_chair_04.obj"),
    )
    VisualFurniture.asset(
        assets=assets,
        tag="texture",
        visual=visual,
        name="recycling_chair_texture",
        type="2d",
        file=str(chair_source / "11gfdgdg.png"),
    )
    VisualFurniture.asset(
        assets=assets,
        tag="material",
        visual=visual,
        name="recycling_chair_material",
        texture="recycling_chair_texture",
    )
    for index, (horizontal, lateral) in enumerate(CHAIR_POSITIONS):
        chair = ET.SubElement(
            world,
            "body",
            name=f"recycling_chair_{index}",
            pos=f"{horizontal} {lateral} 0",
            quat="0 0 0 1" if index < 2 else "1 0 0 0",
        )
        VisualFurniture.add(
            body=chair,
            mesh="recycling_chair_mesh",
            geom={
                "material": "recycling_chair_material",
                "pos": "0 -0.032779485 0.45560272",
                "quat": "0.707106781187 0.707106781187 0 0",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            },
            visual=visual,
        )
        ET.SubElement(
            chair,
            "geom",
            name=f"recycling_chair_{index}_clearance",
            type="box",
            pos="0 0 0.429",
            size="0.246 0.265 0.429",
            rgba="0 0 0 0",
            contype="1",
            conaffinity="1",
            group="3",
        )
    return ET.tostring(root, encoding="unicode")


def add_mats(*, xml: str, bounds: Any) -> str:
    """The Sprout mats as a flat visual: they mark the no-drive line and collide with
    nothing, so they must not be a scene object the planners would see."""
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    assert world is not None
    left, bottom, right, top = bounds
    ET.SubElement(
        world,
        "geom",
        name="sprout_mats",
        type="box",
        pos=f"{(left + right) / 2} {(bottom + top) / 2} 0.0006",
        size=f"{(right - left) / 2} {(top - bottom) / 2} 0.0005",
        rgba="0.45 0.55 0.58 1",
        contype="0",
        conaffinity="0",
        group="1",
    )
    return ET.tostring(root, encoding="unicode")


class RecyclingObjectCentricEnv(envs.ObjectCentricTidyBot3DEnv):
    visual_furniture: ClassVar[bool] = True

    def _create_scene_xml(self) -> str:
        xml = add_lab_furniture(xml=super()._create_scene_xml(), visual=self.visual_furniture)
        bounds = self.task_config.get("kitchen_mats")
        return xml if bounds is None else add_mats(xml=xml, bounds=bounds)


class PhysicsOnlyRecyclingObjectCentricEnv(RecyclingObjectCentricEnv):
    """The same scene without the visual-only furniture meshes: never to be rendered."""

    visual_furniture: ClassVar[bool] = False


class RecyclingEnv(envs.TidyBot3DEnv):
    def _create_object_centric_env(self, *args: Any, **kwargs: Any) -> Any:
        return RecyclingObjectCentricEnv(*args, **kwargs)


class PhysicsOnlyRecyclingEnv(envs.TidyBot3DEnv):
    def _create_object_centric_env(self, *args: Any, **kwargs: Any) -> Any:
        return PhysicsOnlyRecyclingObjectCentricEnv(*args, **kwargs)
