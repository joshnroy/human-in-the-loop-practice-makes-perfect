"""Contact-dynamic cubes and vessels, with scripted idealized manipulation.

TidyBot's arm is visualized with numerical IK and prescribed joint positions.
Robot collisions and grasp mechanics are deliberately outside this prototype.
Cube transfer during both pouring actions is entirely MuJoCo contact dynamics.
"""

import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial.transform import Rotation, Slerp


class PouringSimulation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any
    data: Any
    arm_indices: np.ndarray
    arm_dofs: np.ndarray
    robot_nq: int
    robot_nv: int
    cube_addresses: list[int]
    cube_dofs: list[int]
    pitcher_mocap: int
    tray_mocap: int
    anchor_mocap: int
    site: int
    joint_pose: np.ndarray
    held: str | None = None
    gripper_closed: bool = False
    elapsed_seconds: float = 0.0
    human_interventions: int = 0
    events: list[dict[str, Any]] = Field(default_factory=list)
    frame_callback: Callable[..., None] | None = None
    phase: str = "Initial setup"
    frame_stride: int = 16
    tick: int = 0
    renderer: Any = None
    max_ik_error: float = 0.0
    source_xml: str = ""

    @classmethod
    def create(cls, *, seed: int = 0) -> "PouringSimulation":
        import mujoco

        xml = PouringScene.build(seed=seed)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        arm_joints = [model.joint(f"joint_{i}").id for i in range(1, 8)]
        indices = model.jnt_qposadr[arm_joints]
        dofs = model.jnt_dofadr[arm_joints]
        addresses = [model.joint(f"cube{i}_free").qposadr.item() for i in range(5)]
        cube_dofs = [model.joint(f"cube{i}_free").dofadr.item() for i in range(5)]
        pose = np.array([0.0, 0.26, np.pi, -2.27, 0.0, 0.96, 1.57])
        data.qpos[indices] = pose
        mujoco.mj_forward(model, data)
        sim = cls(
            model=model,
            data=data,
            arm_indices=indices,
            arm_dofs=dofs,
            robot_nq=min(addresses),
            robot_nv=min(cube_dofs),
            cube_addresses=addresses,
            cube_dofs=cube_dofs,
            pitcher_mocap=model.body("pitcher").mocapid.item(),
            tray_mocap=model.body("tray").mocapid.item(),
            anchor_mocap=model.body("grasp_anchor").mocapid.item(),
            site=model.site("pinch_site").id,
            joint_pose=pose,
            source_xml=xml,
        )
        sim.solve_ik(target=np.array([0.40, -0.03, 0.48]))
        sim.advance(seconds=1.0)
        sim.elapsed_seconds = 0.0
        return sim

    @staticmethod
    def pitcher_home() -> np.ndarray:
        return np.array([0.62, -0.25, 0.012])

    @staticmethod
    def tray_home() -> np.ndarray:
        return np.array([0.68, 0.20, 0.012])

    def cube_positions(self) -> np.ndarray:
        return np.array([self.data.qpos[a : a + 3].copy() for a in self.cube_addresses])

    def locations(self) -> list[str]:
        result = []
        for i, pos in enumerate(self.cube_positions()):
            if self.held == f"cube{i}":
                result.append("held")
                continue
            location = "floor" if pos[2] < 0.045 else "air"
            for name, mid, half, height in [
                ("pitcher", self.pitcher_mocap, np.array([0.105, 0.105]), 0.21),
                ("tray", self.tray_mocap, np.array([0.225, 0.195]), 0.075),
            ]:
                rot = Rotation.from_quat(self.data.mocap_quat[mid][[1, 2, 3, 0]])
                local = rot.inv().apply(pos - self.data.mocap_pos[mid])
                # Whole-cube containment: rotate all eight corners into vessel coordinates.
                addr = self.cube_addresses[i]
                crot = Rotation.from_quat(self.data.qpos[addr + 3 : addr + 7][[1, 2, 3, 0]])
                corners = np.array([
                    [x, y, z]
                    for x in [-0.015, 0.015]
                    for y in [-0.015, 0.015]
                    for z in [-0.015, 0.015]
                ])
                bounds = rot.inv().apply(crot.apply(corners)) + local
                if (
                    np.all(np.abs(bounds[:, :2]) < half + 0.0015)
                    and bounds[:, 2].min() > 0.006
                    and bounds[:, 2].max() < height + 0.0015
                ):
                    location = name
                    break
            result.append(location)
        return result

    def settled(self) -> bool:
        return all(
            np.linalg.norm(self.data.qvel[d : d + 3]) < 0.025
            and np.linalg.norm(self.data.qvel[d + 3 : d + 6]) < 0.25
            for d in self.cube_dofs
        )

    def ready(self) -> bool:
        return (
            self.held is None
            and self.locations() == ["pitcher"] * 5
            and self.settled()
            and np.allclose(
                self.data.mocap_pos[self.pitcher_mocap], self.pitcher_home(), atol=0.005
            )
            and np.allclose(self.data.mocap_pos[self.tray_mocap], self.tray_home(), atol=0.005)
            and np.allclose(
                self.data.mocap_quat[[self.pitcher_mocap, self.tray_mocap]],
                [[1, 0, 0, 0]] * 2,
                atol=0.01,
            )
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "qpos": self.data.qpos.tolist(),
            "qvel": self.data.qvel.tolist(),
            "mocap_pos": self.data.mocap_pos.tolist(),
            "mocap_quat": self.data.mocap_quat.tolist(),
            "held": self.held,
            "gripper_closed": self.gripper_closed,
            "locations": self.locations(),
            "settled": self.settled(),
            "ready": self.ready(),
        }

    def execute(
        self,
        *,
        skill: str,
        cube: int | None = None,
        parameters: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        parameters = parameters or {}
        self.check_preconditions(skill=skill, cube=cube, parameters=parameters)
        before = self.snapshot()
        start = self.elapsed_seconds
        self.phase = skill
        if skill == "PickPitcher":
            self.move_hand(target=self.pitcher_home() + [-0.15, 0, 0.14], seconds=1.5)
            self.gripper_closed = True
            self.held = "pitcher"
            self.move_vessel(
                mid=self.pitcher_mocap, target=self.pitcher_home() + [0, 0, 0.28], seconds=1.8
            )
        elif skill == "Pour":
            target = self.tray_home() + [
                parameters.get("offset_x", 0.0),
                parameters.get("offset_y", -0.055),
                0.34,
            ]
            self.move_vessel(mid=self.pitcher_mocap, target=target, seconds=2.0)
            quat = Rotation.from_euler(
                "x", -parameters.get("angle", 130.0), degrees=True
            ).as_quat()[[3, 0, 1, 2]]
            self.move_vessel(
                mid=self.pitcher_mocap,
                target=target,
                quat=quat,
                seconds=parameters.get("tilt_seconds", 2.0),
            )
            self.advance(seconds=parameters.get("hold_seconds", 2.0))
            self.move_vessel(
                mid=self.pitcher_mocap,
                target=target,
                quat=np.array([1.0, 0.0, 0.0, 0.0]),
                seconds=1.5,
            )
            self.advance(seconds=1.0)
        elif skill == "PlacePitcherAtHome":
            self.move_vessel(
                mid=self.pitcher_mocap, target=self.pitcher_home() + [0, 0, 0.28], seconds=1.8
            )
            self.move_vessel(mid=self.pitcher_mocap, target=self.pitcher_home(), seconds=1.8)
            self.held = None
            self.gripper_closed = False
            self.move_hand(target=np.array([0.40, -0.03, 0.48]), seconds=1.3)
            self.advance(seconds=0.5)
        elif skill in {"PickCubeFromTray", "PickCubeFromFloor"}:
            assert cube is not None
            pos = self.cube_positions()[cube]
            self.move_hand(target=pos + [0, 0, 0.15], seconds=1.2)
            self.move_hand(target=pos, seconds=0.8)
            self.data.mocap_pos[self.anchor_mocap] = pos
            addr = self.cube_addresses[cube]
            self.data.mocap_quat[self.anchor_mocap] = self.data.qpos[addr + 3 : addr + 7]
            self.data.eq_active[self.model.equality(f"hold_cube{cube}").id] = 1
            self.held = f"cube{cube}"
            self.gripper_closed = True
            self.move_hand(target=pos + [0, 0, 0.25], seconds=1.2)
        elif skill == "DropCubeIntoPitcher":
            assert cube is not None
            # Distinct release sites keep subsequent idealized grasps visually separate.
            offset = np.array([(-0.045 if cube % 2 == 0 else 0.045), (cube // 2 - 1) * 0.045, 0.32])
            self.move_hand(target=self.pitcher_home() + offset, seconds=1.8)
            self.data.eq_active[self.model.equality(f"hold_cube{cube}").id] = 0
            self.held = None
            self.gripper_closed = False
            self.advance(seconds=1.0)
        elif skill == "HumanPourBack":
            self.human_interventions += 1
            self.advance(seconds=1.5)
            corner = np.array([0.225, 0.195, 0.045])
            pivot = self.pitcher_home() + [-0.045, -0.065, 0.255]
            self.move_vessel(
                mid=self.tray_mocap,
                target=self.tray_home() + [0, 0, 0.40],
                seconds=1.5,
                robot_tracks=False,
            )
            self.move_vessel(
                mid=self.tray_mocap, target=pivot - corner, seconds=1.8, robot_tracks=False
            )
            for angle in np.linspace(0, 125, 76)[1:]:
                rot = Rotation.from_rotvec(np.array([-1, 1, 0]) / np.sqrt(2) * np.deg2rad(angle))
                self.move_vessel(
                    mid=self.tray_mocap,
                    target=pivot - rot.apply(corner),
                    quat=rot.as_quat()[[3, 0, 1, 2]],
                    seconds=0.04,
                    robot_tracks=False,
                )
            self.advance(seconds=2.0)
            self.move_vessel(
                mid=self.tray_mocap,
                target=self.tray_home() + [0, 0, 0.40],
                quat=np.array([1.0, 0.0, 0.0, 0.0]),
                seconds=1.8,
                robot_tracks=False,
            )
            self.move_vessel(
                mid=self.tray_mocap, target=self.tray_home(), seconds=1.5, robot_tracks=False
            )
            self.advance(seconds=1.0)
        elif skill == "Wait":
            self.advance(seconds=1.0)
        success = self.skill_success(skill=skill, cube=cube)
        event = {
            "skill": skill,
            "cube": cube,
            "parameters": parameters,
            "before": before,
            "after": self.snapshot(),
            "success": success,
            "cost_seconds": self.elapsed_seconds - start,
            "human_interventions": int(skill == "HumanPourBack"),
        }
        self.events.append(event)
        return event

    def check_preconditions(
        self,
        *,
        skill: str,
        cube: int | None,
        parameters: dict[str, float],
    ) -> None:
        locations = self.locations()
        if skill == "PickPitcher":
            valid = self.held is None and np.allclose(
                self.data.mocap_pos[self.pitcher_mocap], self.pitcher_home(), atol=0.005
            )
        elif skill == "Pour":
            valid = self.held == "pitcher" and locations == ["pitcher"] * 5
            bounds = {
                "angle": (0.0, 150.0),
                "offset_x": (-0.4, 0.4),
                "offset_y": (-0.4, 0.4),
                "tilt_seconds": (0.5, 4.0),
                "hold_seconds": (0.2, 4.0),
            }
            if any(
                k not in bounds or not np.isfinite(v) or not bounds[k][0] <= v <= bounds[k][1]
                for k, v in parameters.items()
            ):
                raise ValueError("Pour parameter outside declared bounds")
        elif skill == "PlacePitcherAtHome":
            valid = self.held == "pitcher"
        elif skill in {"PickCubeFromTray", "PickCubeFromFloor", "DropCubeIntoPitcher"}:
            if cube is None or cube not in range(5):
                raise ValueError("A cube index in 0..4 is required")
            if skill == "DropCubeIntoPitcher":
                valid = self.held == f"cube{cube}" and np.allclose(
                    self.data.mocap_pos[self.pitcher_mocap], self.pitcher_home(), atol=0.005
                )
            else:
                source = "tray" if skill == "PickCubeFromTray" else "floor"
                valid = self.held is None and locations[cube] == source
        elif skill == "HumanPourBack":
            valid = (
                self.held is None
                and "tray" in locations
                and all(x in {"pitcher", "tray"} for x in locations)
                and np.allclose(
                    self.data.mocap_pos[self.pitcher_mocap], self.pitcher_home(), atol=0.005
                )
            )
        elif skill == "Wait":
            valid = True
        else:
            raise ValueError(f"Unknown skill: {skill}")
        if not valid:
            raise ValueError(f"Preconditions unmet for {skill}: {locations}, held={self.held}")

    def skill_success(self, *, skill: str, cube: int | None) -> bool:
        if skill == "Pour":
            return self.locations() == ["tray"] * 5 and self.settled()
        if skill == "HumanPourBack":
            return self.ready()
        if skill == "DropCubeIntoPitcher":
            assert cube is not None
            return self.locations()[cube] == "pitcher" and self.settled()
        if skill.startswith("PickCube"):
            return self.held == f"cube{cube}"
        if skill == "PickPitcher":
            return self.held == "pitcher"
        return self.held is None if skill == "PlacePitcherAtHome" else True

    def solve_ik(self, *, target: np.ndarray) -> None:
        import mujoco

        jac = np.zeros((3, self.model.nv))
        for _ in range(24):
            mujoco.mj_forward(self.model, self.data)
            error = target - self.data.site_xpos[self.site]
            if np.linalg.norm(error) < 0.002:
                break
            mujoco.mj_jacSite(self.model, self.data, jac, None, self.site)
            j = jac[:, self.arm_dofs]
            delta = j.T @ np.linalg.solve(j @ j.T + np.eye(3) * 0.0004, error)
            self.data.qpos[self.arm_indices] += np.clip(delta, -0.12, 0.12)
            for joint in range(1, 8):
                jid = self.model.joint(f"joint_{joint}").id
                if self.model.jnt_limited[jid]:
                    a = self.model.jnt_qposadr[jid]
                    self.data.qpos[a] = np.clip(self.data.qpos[a], *self.model.jnt_range[jid])
        mujoco.mj_forward(self.model, self.data)
        self.max_ik_error = max(
            self.max_ik_error, float(np.linalg.norm(target - self.data.site_xpos[self.site]))
        )
        self.joint_pose = self.data.qpos[self.arm_indices].copy()

    def advance(self, *, seconds: float) -> None:
        import mujoco

        steps = max(1, round(seconds / self.model.opt.timestep))
        robot_pose = self.data.qpos[: self.robot_nq].copy()
        robot_pose[self.arm_indices] = self.joint_pose
        for _ in range(steps):
            self.data.qpos[: self.robot_nq] = robot_pose
            self.data.qvel[: self.robot_nv] = 0
            self.data.qfrc_applied[: self.robot_nv] = self.data.qfrc_bias[: self.robot_nv]
            mujoco.mj_step(self.model, self.data)
            self.data.qpos[: self.robot_nq] = robot_pose
            self.data.qvel[: self.robot_nv] = 0
            self.tick += 1
            if self.frame_callback is not None and self.tick % self.frame_stride == 0:
                mujoco.mj_forward(self.model, self.data)
                self.frame_callback(sim=self)
        self.elapsed_seconds += steps * self.model.opt.timestep
        mujoco.mj_forward(self.model, self.data)

    def move_hand(self, *, target: np.ndarray, seconds: float) -> None:
        import mujoco

        mujoco.mj_forward(self.model, self.data)
        start = self.data.site_xpos[self.site].copy()
        frames = max(1, round(seconds * 30))
        for alpha in np.linspace(0, 1, frames + 1)[1:]:
            s = alpha * alpha * (3 - 2 * alpha)
            pos = start * (1 - s) + target * s
            self.solve_ik(target=pos)
            if self.held is not None and self.held.startswith("cube"):
                self.data.mocap_pos[self.anchor_mocap] = pos
            self.advance(seconds=seconds / frames)

    def move_vessel(
        self,
        *,
        mid: int,
        target: np.ndarray,
        seconds: float,
        quat: np.ndarray | None = None,
        robot_tracks: bool = True,
    ) -> None:
        start = self.data.mocap_pos[mid].copy()
        first_quat = self.data.mocap_quat[mid].copy()
        quat = first_quat if quat is None else quat
        rotations = Rotation.from_quat(np.array([first_quat, quat])[:, [1, 2, 3, 0]])
        slerp = Slerp([0, 1], rotations)
        frames = max(1, round(seconds / self.model.opt.timestep))
        for alpha in np.linspace(0, 1, frames + 1)[1:]:
            s = alpha * alpha * (3 - 2 * alpha)
            rot = slerp(s)
            pos = start * (1 - s) + target * s
            self.data.mocap_pos[mid] = pos
            self.data.mocap_quat[mid] = rot.as_quat()[[3, 0, 1, 2]]
            if robot_tracks and self.tick % 16 == 0:
                self.solve_ik(target=pos + rot.apply([-0.15, 0, 0.14]))
            self.advance(seconds=seconds / frames)

    def render(self, *, overhead: bool = False) -> np.ndarray:
        import mujoco

        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=640, width=960)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.43, 0.0, 0.35]
        camera.distance = 1.95 if overhead else 2.35
        camera.azimuth = 90 if overhead else 130
        camera.elevation = -89 if overhead else -35
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class PouringScene:
    @staticmethod
    def build(*, seed: int) -> str:
        import kinder

        models = Path(kinder.__file__).parent / "envs/dynamic3d/models"
        root = ET.parse(models / "stanford_tidybot/tidybot.xml").getroot()
        root.set("model", "Pouring3D proof of concept")
        compiler = root.find("compiler")
        actuators = root.find("actuator")
        option = root.find("option")
        assets = root.find("asset")
        world = root.find("worldbody")
        equality = root.find("equality")
        assert all(x is not None for x in [compiler, actuators, option, assets, world, equality])
        assert compiler is not None and actuators is not None and option is not None
        assert assets is not None and world is not None and equality is not None
        compiler.set("meshdir", str(models / "assets"))
        for geom in root.findall(".//geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
        for actuator in list(actuators):
            actuators.remove(actuator)
        option.set("timestep", "0.002")
        option.set("gravity", "0 0 -9.81")
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "global", offwidth="960", offheight="640")
        ET.SubElement(visual, "headlight", ambient=".4 .4 .4", diffuse=".7 .7 .7")
        ET.SubElement(
            assets,
            "texture",
            type="skybox",
            builtin="gradient",
            rgb1=".78 .84 .89",
            rgb2=".93 .94 .95",
            width="512",
            height="512",
        )
        ET.SubElement(world, "light", pos="0.4 -1 2.5", diffuse="0.8 0.8 0.8")
        ET.SubElement(world, "light", pos="1.2 1.5 2", diffuse="0.5 0.5 0.5")
        ET.SubElement(
            world,
            "geom",
            name="floor",
            type="plane",
            size="3 3 .05",
            rgba=".91 .92 .91 1",
            friction=".7 .01 .001",
        )
        for name, home, half, height, rgba in [
            ("pitcher", PouringSimulation.pitcher_home(), [0.113, 0.113], 0.21, ".2 .57 .72 .45"),
            ("tray", PouringSimulation.tray_home(), [0.233, 0.203], 0.075, ".75 .45 .21 1"),
        ]:
            body = ET.SubElement(
                world, "body", name=name, mocap="true", pos=PouringScene.vec(v=home)
            )
            PouringScene.box(
                body=body, name=f"{name}_bottom", pos=[0, 0, 0.006], size=[*half, 0.006], rgba=rgba
            )
            for axis in range(2):
                for sign in [-1, 1]:
                    pos = [0.0, 0.0, height / 2]
                    size = [*half, height / 2]
                    pos[axis] = sign * (half[axis] - 0.004)
                    size[axis] = 0.004
                    PouringScene.box(
                        body=body, name=f"{name}_wall{axis}_{sign}", pos=pos, size=size, rgba=rgba
                    )
            if name == "pitcher":
                for z in [0.065, 0.175]:
                    PouringScene.box(
                        body=body,
                        name=f"handle{z}",
                        pos=[-0.143, 0, z],
                        size=[0.03, 0.014, 0.01],
                        rgba=".12 .34 .44 1",
                    )
                PouringScene.box(
                    body=body,
                    name="handle_grip",
                    pos=[-0.17, 0, 0.12],
                    size=[0.012, 0.014, 0.065],
                    rgba=".12 .34 .44 1",
                )
            # Flat markings do not add obstacles to physical cube recovery.
            ET.SubElement(
                world,
                "geom",
                type="box",
                pos=PouringScene.vec(v=[*home[:2], 0.0005]),
                size=PouringScene.vec(v=[half[0] + 0.02, half[1] + 0.02, 0.0005]),
                rgba=".44 .52 .48 .25",
                contype="0",
                conaffinity="0",
            )
        ET.SubElement(world, "body", name="grasp_anchor", mocap="true", pos=".5 0 .4")
        rng = np.random.default_rng(seed)
        colors = [
            ".85 .24 .18 1",
            ".91 .65 .12 1",
            ".28 .64 .35 1",
            ".33 .45 .82 1",
            ".69 .31 .68 1",
        ]
        for i in range(5):
            offset = np.array([[-0.05, -0.05], [0.05, -0.05], [-0.05, 0.05], [0.05, 0.05], [0, 0]])[
                i
            ]
            offset += rng.uniform(-0.006, 0.006, 2)
            pos = PouringSimulation.pitcher_home() + [*offset, 0.035]
            body = ET.SubElement(world, "body", name=f"cube{i}", pos=PouringScene.vec(v=pos))
            ET.SubElement(body, "freejoint", name=f"cube{i}_free")
            ET.SubElement(
                body,
                "geom",
                name=f"cube{i}_geom",
                type="box",
                size=".015 .015 .015",
                mass=".025",
                rgba=colors[i],
                friction=".45 .005 .0001",
                solref=".008 1",
                solimp=".95 .99 .001",
            )
            ET.SubElement(
                equality,
                "weld",
                name=f"hold_cube{i}",
                body1="grasp_anchor",
                body2=f"cube{i}",
                active="false",
                relpose="0 0 0 1 0 0 0",
                solref=".004 1",
            )
        return ET.tostring(root, encoding="unicode")

    @staticmethod
    def box(*, body: Any, name: str, pos: list[float], size: list[float], rgba: str) -> None:
        ET.SubElement(
            body,
            "geom",
            name=name,
            type="box",
            pos=PouringScene.vec(v=pos),
            size=PouringScene.vec(v=size),
            rgba=rgba,
            friction=".4 .005 .0001",
            solref=".008 1",
            solimp=".95 .99 .001",
        )

    @staticmethod
    def vec(*, v: Any) -> str:
        return " ".join(str(float(x)) for x in v)
