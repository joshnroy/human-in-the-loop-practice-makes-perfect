"""Task-local MimicLabs desk and Replica chairs, with conservative collisions."""

import xml.etree.ElementTree as ET
from pathlib import Path

from kinder.envs.dynamic3d import envs

FURNITURE_POSITION = (2.5, 1.35, 0.0)
FURNITURE_SCALE = (0.6, 0.6, 0.8)
CHAIR_POSITIONS = ((2.15, 0.67), (2.85, 0.67), (2.15, 2.03), (2.85, 2.03))


def rectangular_kitchen(*, root):
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


def add_lab_furniture(*, xml: str) -> str:
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
        ET.SubElement(
            assets,
            "mesh",
            name=mesh,
            file=str(source / f"scene2_table_{index}.obj"),
            scale=" ".join(map(str, FURNITURE_SCALE)),
        )
        ET.SubElement(
            desk,
            "geom",
            name=f"{mesh}_visual",
            type="mesh",
            mesh=mesh,
            contype="0",
            conaffinity="0",
            group="1",
            rgba=color,
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
    ET.SubElement(
        assets,
        "mesh",
        name="recycling_chair_mesh",
        file=str(chair_source / "frl_apartment_chair_04.obj"),
    )
    ET.SubElement(
        assets,
        "texture",
        name="recycling_chair_texture",
        type="2d",
        file=str(chair_source / "11gfdgdg.png"),
    )
    ET.SubElement(
        assets, "material", name="recycling_chair_material", texture="recycling_chair_texture"
    )
    for index, (horizontal, lateral) in enumerate(CHAIR_POSITIONS):
        chair = ET.SubElement(
            world,
            "body",
            name=f"recycling_chair_{index}",
            pos=f"{horizontal} {lateral} 0",
            quat="0 0 0 1" if index < 2 else "1 0 0 0",
        )
        ET.SubElement(
            chair,
            "geom",
            type="mesh",
            mesh="recycling_chair_mesh",
            material="recycling_chair_material",
            pos="0 -0.032779485 0.45560272",
            quat="0.707106781187 0.707106781187 0 0",
            contype="0",
            conaffinity="0",
            group="1",
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


class RecyclingObjectCentricEnv(envs.ObjectCentricTidyBot3DEnv):
    def _create_scene_xml(self) -> str:
        return add_lab_furniture(xml=super()._create_scene_xml())


class RecyclingEnv(envs.TidyBot3DEnv):
    def _create_object_centric_env(self, *args, **kwargs):
        return RecyclingObjectCentricEnv(*args, **kwargs)
