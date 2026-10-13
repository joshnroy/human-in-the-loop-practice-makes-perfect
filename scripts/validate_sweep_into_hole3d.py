"""TEMPORARY scripted validation of SweepIntoHole3D: two captioned videos in lab2.

Not a planner skill, not a `Method`, not a `HumanOracle`. This drives the new
`SweepIntoHole3D-o5` task with the classical controllers the frozen Dustpan3D fork
validated on the identical robot, broom and cubes (`dustpan_skills/motion.py`,
`broom.py`), edited only where the target changed: a pit at x in [0.80, 1.25] with
no ramp instead of a pan at x = 0.9. Its purpose is to record what the physics does,
so a failed sweep is a finding this script reports, never something it patches.

Two episodes, each a bounded sequence of control steps rendered from `task_view`:

1. `sweep_then_reset_cubes`: pick the broom, sweep the cube strip toward the pit,
   count `x/5` in the hole by the environment's own `cube_in_hole`, then the human
   `reset_cubes` prototype fires (k = 0: every cube returns to `cubes_start`).
2. `drive_in_then_reset_robot`: pick the broom (so it is away from its start),
   drive the base straight over the rim into the pit, then the human `reset_robot`
   prototype fires and restores the robot to `robot_start` and the broom to
   `wiper_start`.

The human prototypes (`HumanResets.reset_cubes`, `reset_broom`, `reset_robot`) are
scripted teleports through the environment's own `reset_ground_objects_to_regions`
and `set_robot_base_pos_yaw`, pure apart from the simulator, and deliberately not
wired into `hitl_pmp`.

Every KINDER import lives inside `import_kinder`; argument parsing and the
caption/sheet helpers test without a simulator, the episodes only where `kinder`
and the MimicLabs assets are installed.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict
from scipy.spatial.transform import Rotation, Slerp

from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend

ENV_ID = "kinder/SweepIntoHole3D-o5-v0"
PIT_X = (0.80, 1.25)
CUBES = tuple(f"cube_{i}" for i in range(5))
FRAME_SIZE = (640, 480)
EPISODES = ("sweep_then_reset_cubes", "drive_in_then_reset_robot")


class StepBudgetExhausted(RuntimeError):
    """The episode's hard control-step cap was reached; nothing here loops open."""


class Phase(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    start_tick: int
    end_tick: int
    cubes_in_hole: int
    error: str | None = None


class EpisodeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    episode: str
    seed: int
    scene: str
    ticks: int
    phases: tuple[Phase, ...]
    cubes_in_hole_after_sweep: int | None
    cubes_in_hole_after_reset: int
    base_z_min: float
    base_z_max: float
    broom_head_z_min: float
    cubes_bounced_out: int
    lowest_robot_geom_z: float
    video: Path
    contact_sheet: Path
    encoder: str


class ValidationPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_dir: Path
    seed: int = 0
    scene: str = "mimiclabs-lab2"
    episodes: tuple[str, ...] = EPISODES
    fps: int = 30
    # Hard cap on control steps per episode; the Dustpan fork's skills budgeted
    # 6000 per skill, so this allows a pick plus a sweep plus the reset holds.
    tick_limit: int = 9000


class Captions:
    @staticmethod
    def font(*, size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
        for candidate in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            if Path(candidate).is_file():
                return ImageFont.truetype(candidate, size)
        return ImageFont.load_default()

    @staticmethod
    def annotate(*, frame: np.ndarray, header: str, banner: str | None) -> np.ndarray:
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle([0, 0, image.width, 26], fill=(0, 0, 0, 170))
        draw.text((8, 5), header, fill=(255, 255, 255, 255), font=Captions.font(size=15))
        if banner:
            draw.rectangle(
                [0, image.height - 44, image.width, image.height], fill=(190, 30, 30, 200)
            )
            draw.text(
                (8, image.height - 36),
                banner,
                fill=(255, 255, 255, 255),
                font=Captions.font(size=20),
            )
        return np.asarray(image)

    @staticmethod
    def contact_sheet(
        *, keyframes: list[tuple[str, np.ndarray]], path: Path, columns: int = 4
    ) -> Path:
        rows = (len(keyframes) + columns - 1) // columns
        width, height = FRAME_SIZE
        sheet = Image.new("RGB", (columns * width, rows * (height + 30)), (20, 20, 20))
        draw = ImageDraw.Draw(sheet)
        font = Captions.font(size=18)
        for index, (caption, frame) in enumerate(keyframes):
            x = (index % columns) * width
            y = (index // columns) * (height + 30)
            sheet.paste(Image.fromarray(frame), (x, y + 30))
            draw.text((x + 8, y + 6), caption, fill=(255, 255, 255), font=font)
        sheet.save(path)
        return path


class VideoWriter:
    """Stream RGB frames to ffmpeg; NVENC when the binary has it, else libx264."""

    def __init__(self, *, path: Path, fps: int) -> None:
        import imageio_ffmpeg

        binary = imageio_ffmpeg.get_ffmpeg_exe()
        encoders = subprocess.run(
            [binary, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False
        ).stdout
        self.encoder = "h264_nvenc" if "h264_nvenc" in encoders else "libx264"
        width, height = FRAME_SIZE
        self.process = subprocess.Popen(
            [
                binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                self.encoder,
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        self.frames = 0

    def write(self, *, frame: np.ndarray) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError(f"ffmpeg ({self.encoder}) exited with {self.process.returncode}")


class HumanResets:
    """Prototypes of the human skills, as teleports through the env's own machinery."""

    @staticmethod
    def release_tool(*, core: Any) -> None:
        sim = core._robot_env.sim  # noqa: SLF001
        model, data = sim.model.mj_model, sim.data.mj_data
        data.eq_active[model.equality("broom_grasp").id] = False
        core._robot_env.broken_tool_grasps.clear()  # noqa: SLF001

    @staticmethod
    def reset_cubes(*, core: Any) -> None:
        core.reset_ground_objects_to_regions({name: "cubes_start" for name in CUBES})

    @staticmethod
    def reset_broom(*, core: Any) -> None:
        """Broom back to the declared `wiper_start` pose.

        Not through `reset_ground_objects_to_regions`: that sampler models the wiper
        as an AABB centred on its origin and, with the cubes already on the strip,
        finds no collision-free pose in the point-sized region (observed: 100 failed
        attempts). The task's own initial placement only succeeds because it samples
        the wiper before the cubes. A direct set_pose at the region's centre is the
        same pose that placement produces.
        """
        HumanResets.release_tool(core=core)
        region = core.task_config["regions"]["wiper_start"]
        x0, y0, x1, y1 = region["ranges"][0]
        yaw = float(np.deg2rad(region["yaw_ranges"][0][0]))
        wiper = core.get_object("wiper_0")
        bounds = type(wiper).get_bounding_box_from_config(
            np.zeros(3, dtype=np.float32), core.task_config["objects"]["wiper"]["wiper_0"]
        )
        wiper.set_pose(
            [(x0 + x1) / 2, (y0 + y1) / 2, -bounds[2]],
            [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))],
        )
        wiper.set_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        core._robot_env.sim.forward()  # noqa: SLF001

    @staticmethod
    def reset_robot(*, core: Any, home_qpos: np.ndarray, robot_qpos_indices: np.ndarray) -> None:
        """Robot back to `robot_start` with its reset arm pose; the broom back too."""
        HumanResets.reset_broom(core=core)
        sim = core._robot_env.sim  # noqa: SLF001
        data = sim.data.mj_data
        data.qpos[robot_qpos_indices] = home_qpos[robot_qpos_indices]
        data.qvel[:] = 0.0
        x, y, _, yaw = core._ground_fixture.sample_pose_in_region("robot_start", core.np_random)  # noqa: SLF001
        core._robot_env.set_robot_base_pos_yaw(x, y, yaw)  # noqa: SLF001
        sim.forward()


class HoleProbe:
    """The Dustpan fork's classical controllers, pointed at the pit."""

    def __init__(self, *, plan: ValidationPlan, episode: str) -> None:
        kinder = Validator.import_kinder()
        from kinder.envs.dynamic3d.envs import TidyBot3DConfig

        self.plan = plan
        self.episode = episode
        self.env = kinder.make(
            ENV_ID,
            allow_state_access=True,
            render_mode="rgb_array",
            scene_bg=plan.scene,
            scene_render_camera="task_view",
            config=TidyBot3DConfig(use_arm_velocities=False),
        )
        self.env.reset(seed=plan.seed)
        self.core = self.env.unwrapped._object_centric_env  # noqa: SLF001
        self.sim = self.core._robot_env.sim  # noqa: SLF001
        self.model = self.sim.model.mj_model
        self.data = self.sim.data.mj_data
        self.ik_data = self.mujoco.MjData(self.model)
        self.site = self.model.site("robot_pinch_site").id
        self.joints = [self.model.joint(f"robot_joint_{i}").id for i in range(1, 8)]
        self.qa = self.model.jnt_qposadr[self.joints]
        self.va = self.model.jnt_dofadr[self.joints]
        self.ba = [
            self.model.jnt_qposadr[self.model.joint(f"robot_joint_{a}").id]
            for a in ("x", "y", "th")
        ]
        self.robot_qpos_indices = np.array([
            self.model.jnt_qposadr[j]
            for j in range(self.model.njnt)
            if (self.model.joint(j).name or "").startswith("robot_")
        ])
        self.home_qpos = self.data.qpos.copy()
        self.base_target = self.data.qpos[self.ba].copy()
        self.arm_reference = self.data.qpos[self.qa].copy()
        self.base_step_limit = 0.015
        self.grip = 0.0
        self.ticks = 0
        self.phase = "start"
        self.banner: str | None = None
        self.banner_until = -1
        self.phases: list[Phase] = []
        self.keyframes: list[tuple[str, np.ndarray]] = []
        self.base_body = self.model.body("robot_base_link").id
        self.robot_geoms = sorted(self.core.collision_groups()["robot"])
        self.wiper_geoms = sorted(self.core.collision_groups()["wiper"])
        self.base_z: list[float] = []
        self.head_z: list[float] = []
        self.lowest_robot_geom_z = float("inf")
        self.in_hole_history: list[int] = []
        plan.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = plan.output_dir / f"sweep_into_hole3d_{episode}.mp4"
        self.writer = VideoWriter(path=self.video_path, fps=plan.fps)
        self.capture(label="reset")

    @property
    def mujoco(self) -> Any:
        import mujoco

        return mujoco

    # --- measurement -------------------------------------------------------

    def in_hole(self) -> int:
        return sum(self.core.cube_in_hole(name) for name in CUBES)

    def header(self) -> str:
        return (
            f"SweepIntoHole3D | {self.episode} | step {self.ticks} | {self.phase} | "
            f"in hole {self.in_hole()}/5"
        )

    def capture(self, *, label: str, keyframe: bool = False) -> None:
        frame = self.sim.render(
            width=FRAME_SIZE[0], height=FRAME_SIZE[1], camera_name="task_view", mode="offscreen"
        )
        if isinstance(frame, tuple):
            frame = frame[0]
        banner = self.banner if self.ticks <= self.banner_until else None
        annotated = Captions.annotate(frame=np.asarray(frame), header=self.header(), banner=banner)
        self.writer.write(frame=annotated)
        if keyframe:
            self.keyframes.append((f"step {self.ticks}: {label}", annotated))
        self.base_z.append(float(self.data.xpos[self.base_body][2]))
        self.head_z.append(
            float(min(self.data.geom_xpos[g][2] for g in self.wiper_geoms))
            if self.wiper_geoms
            else float("nan")
        )
        self.lowest_robot_geom_z = min(
            self.lowest_robot_geom_z,
            float(min(self.data.geom_xpos[g][2] for g in self.robot_geoms)),
        )
        self.in_hole_history.append(self.in_hole())

    def show_banner(self, *, text: str, hold_ticks: int) -> None:
        self.banner = text
        self.banner_until = self.ticks + hold_ticks

    # --- control (Dustpan fork, verbatim apart from the step budget) ---------

    def step(self, *, arm: np.ndarray, base: np.ndarray | None = None, label: str = "move") -> None:
        if self.ticks >= self.plan.tick_limit:
            raise StepBudgetExhausted(
                f"{self.plan.tick_limit} control steps reached during {label}"
            )
        self.arm_reference = np.asarray(arm).copy()
        action = np.zeros(11)
        error = (arm - self.data.qpos[self.qa] + np.pi) % (2 * np.pi) - np.pi
        action[3:10] = np.clip(error, -0.1, 0.1)
        base = self.base_target if base is None else base
        action[:3] = np.clip(
            base - self.data.qpos[self.ba], -self.base_step_limit, self.base_step_limit
        )
        action[10] = self.grip
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        self.env.step(action)
        self.ticks += 1
        self.capture(label=label)

    def hold(self, *, count: int = 15, label: str = "hold") -> None:
        arm = self.arm_reference.copy()
        for _ in range(count):
            self.step(arm=arm, label=label)

    def move_base(self, *, target: Any, label: str = "move_base") -> None:
        self.base_target = np.asarray(target, dtype=float)
        arm = self.arm_reference.copy()
        distance = np.max(np.abs(self.base_target - self.data.qpos[self.ba]))
        for _ in range(max(150, int(np.ceil(distance / self.base_step_limit)) + 50)):
            self.step(arm=arm, base=self.base_target, label=label)
            if np.max(np.abs(self.data.qpos[self.ba] - self.base_target)) < 0.003:
                return
        raise RuntimeError(f"Base failed to converge during {label}")

    def ik(self, *, position: Any, rotation: np.ndarray) -> np.ndarray:
        mujoco = self.mujoco
        d, m = self.ik_data, self.model
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        target = Rotation.from_matrix(rotation)
        rng = np.random.default_rng(0)
        initial = self.data.qpos[self.qa].copy()
        candidates = []
        robot_geoms = set(self.robot_geoms)
        err = np.full(6, np.inf)
        for attempt in range(16):
            d.qpos[:] = self.data.qpos
            if attempt:
                d.qpos[self.qa] += rng.normal(0, 0.9, 7)
            for _ in range(200):
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                current = Rotation.from_matrix(d.site_xmat[self.site].reshape(3, 3))
                err = np.r_[
                    np.asarray(position) - d.site_xpos[self.site],
                    (target * current.inv()).as_rotvec() * 0.4,
                ]
                if np.linalg.norm(err) < 0.001:
                    break
                mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
                j = np.vstack([jacp[:, self.va], 0.4 * jacr[:, self.va]])
                dq = j.T @ np.linalg.solve(j @ j.T + 0.001 * np.eye(6), err)
                d.qpos[self.qa] += np.clip(dq, -0.12, 0.12)
                for i, joint in enumerate(self.joints):
                    if m.jnt_limited[joint]:
                        d.qpos[self.qa[i]] = np.clip(d.qpos[self.qa[i]], *m.jnt_range[joint])
            if np.linalg.norm(err) < 0.003:
                mujoco.mj_collision(m, d)
                penetrations = sum(
                    max(0, -c.dist - 0.001)
                    for c in d.contact
                    if c.geom1 in robot_geoms and c.geom2 in robot_geoms
                )
                q = d.qpos[self.qa].copy()
                score = (
                    np.linalg.norm((q - initial + np.pi) % (2 * np.pi) - np.pi)
                    + 1000 * penetrations
                )
                candidates.append((score, q))
                if attempt == 0 and penetrations == 0:
                    return q
        if candidates:
            return min(candidates, key=lambda c: c[0])[1]
        raise RuntimeError(f"IK failed at {position}: residual {np.linalg.norm(err):.3f}")

    def move(
        self,
        *,
        position: Any,
        rotation: np.ndarray,
        label: str = "move",
        limit: int = 250,
        position_tolerance: float = 0.008,
        orientation_tolerance: float = 0.05,
    ) -> None:
        arm = self.ik(position=position, rotation=rotation)
        for i in range(limit):
            if i and i % 8 == 0:
                arm = self.ik(position=position, rotation=rotation)
            self.step(arm=arm, label=label)
            actual_rotation = Rotation.from_matrix(self.data.site_xmat[self.site].reshape(3, 3))
            angle = (Rotation.from_matrix(rotation) * actual_rotation.inv()).magnitude()
            if (
                np.linalg.norm(self.data.site_xpos[self.site] - position) < position_tolerance
                and angle < orientation_tolerance
            ):
                return
        raise RuntimeError(
            f"Physical move failed: {label}, target={np.round(position, 3)}, "
            f"actual={np.round(self.data.site_xpos[self.site], 3)}"
        )

    def broom_pose(self) -> np.ndarray:
        wiper = self.core.get_object("wiper_0")
        p, q = wiper.get_pose()
        obj = np.eye(4)
        obj[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        obj[:3, 3] = p
        if not wiper.upright:
            virtual = np.eye(4)
            virtual[:3, :3] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
            virtual[:3, 3] = [
                -(wiper.handle_height + wiper.head_height) / 2,
                0,
                wiper.handle_width / 2,
            ]
            obj = obj @ virtual
        return obj

    def pick_floor_broom(self) -> None:
        obj = self.broom_pose()
        point = obj[:3, 3] + obj[:3, :3] @ [0, 0, 0.7]
        shaft = obj[:2, 2]
        normal = np.array([shaft[1], -shaft[0]])
        normal /= np.linalg.norm(normal)
        heading = np.arctan2(normal[1], normal[0])
        r = Rotation.from_euler("z", heading).as_matrix() @ np.diag([1.0, -1.0, -1.0])
        self.grip = 0.0
        self.move_base(target=[*(point[:2] - 0.65 * normal), heading])
        self.move(position=point + [0, 0, 0.18], rotation=r, label="approach_floor_broom")
        for z in np.linspace(0.18, 0.012, 18):
            self.move(position=point + [0, 0, z], rotation=r, label="lower_to_floor_handle")
        self.grip = 1.0
        self.hold(count=20, label="close_on_floor_handle")
        if not self.data.eq_active[self.model.equality("broom_grasp").id]:
            raise RuntimeError("Floor grasp failed: broom_grasp weld did not engage")
        for z in np.linspace(0.03, 0.60, 30):
            self.move_base(
                target=[
                    *(point[:2] - (0.7 + 0.05 * np.clip((z - 0.2) / 0.3, 0, 1)) * normal),
                    heading,
                ],
                label="lift_base",
            )
            self.move(
                position=point + [0, 0, z],
                rotation=r,
                label="lift_floor_broom",
                position_tolerance=0.025,
            )
        if self.core.get_object("wiper_0").get_pose()[0][2] <= 0.05:
            raise RuntimeError("Broom was not lifted")

    def sweep_into_hole(self) -> None:
        """The fork's `sweep_motion` with the pan's ramp and lip replaced by the rim.

        Edits versus the fork: the sweep frame is the world frame (the pit does not
        move); the ramp-height term is gone (no ramp); the stroke ends with the head
        0.15 m past the near rim instead of at the pan's lip.
        """
        ee = np.eye(4)
        ee[:3, :3] = self.data.site_xmat[self.site].reshape(3, 3)
        ee[:3, 3] = self.data.site_xpos[self.site]
        obj = self.broom_pose()
        relative = np.linalg.inv(ee) @ obj
        yaw = Rotation.from_euler("z", 270, degrees=True).as_matrix()
        outside = [
            self.core.get_object(name).get_pose()[0]
            for name in CUBES
            if not self.core.cube_in_hole(name)
        ]
        center_y = (
            float(
                np.clip(
                    (min(p[1] for p in outside) + max(p[1] for p in outside)) / 2, -0.025, 0.025
                )
            )
            if outside
            else 0.0
        )
        sweep_y = center_y
        start_x = min(0.25, min((p[0] for p in outside), default=0.45) - 0.20)

        def move_broom(
            *,
            x: float,
            z: float,
            angle: float = 0.0,
            rotation: np.ndarray | None = None,
            label: str,
        ) -> None:
            obj[:3, :3] = (
                Rotation.from_euler("y", angle, degrees=True).as_matrix() @ yaw
                if rotation is None
                else rotation
            )
            obj[:3, 3] = [x, sweep_y, z]
            target = obj @ np.linalg.inv(relative)
            distance = 0.7 + 0.05 * np.clip((target[2, 3] - 0.2) / 0.3, 0, 1)
            shaft_xy = obj[:2, 2]
            normal = np.array([shaft_xy[1], -shaft_xy[0]])
            normal /= np.linalg.norm(normal)
            base_xy = target[:2, 3] - distance * normal
            self.move_base(
                target=[*base_xy, np.arctan2(normal[1], normal[0])], label="position_base"
            )
            for _ in range(20):
                self.move(
                    position=target[:3, 3],
                    rotation=target[:3, :3],
                    label=label,
                    position_tolerance=0.04,
                    orientation_tolerance=0.10,
                )
                self.hold(count=1, label=label)
                actual = self.broom_pose()
                correction = obj @ np.linalg.inv(actual)
                if (
                    np.linalg.norm(obj[:3, 3] - actual[:3, 3]) < 0.015
                    and abs(obj[2, 3] - actual[2, 3]) < 0.006
                    and Rotation.from_matrix(correction[:3, :3]).magnitude() < 0.04
                ):
                    return
                corrected = correction @ target
                target[:3, 3] = 0.5 * (target[:3, 3] + corrected[:3, 3])
                target[:3, :3] = Slerp(
                    [0, 1], Rotation.from_matrix([target[:3, :3], corrected[:3, :3]])
                )(0.5).as_matrix()
            raise RuntimeError(
                f"Broom pose did not converge during {label}: desired={np.round(obj[:3, 3], 3)}, "
                f"actual={np.round(actual[:3, 3], 3)}"
            )

        start = self.broom_pose()
        end_rotation = Rotation.from_euler("y", -80, degrees=True).as_matrix() @ yaw
        rotations = Slerp([0, 1], Rotation.from_matrix([start[:3, :3], end_rotation]))
        for t in np.linspace(0, 1, 16):
            xyz = (1 - t) * start[:3, 3] + t * np.array([start_x, sweep_y, 0.50])
            sweep_y_saved = sweep_y
            sweep_y = float(xyz[1])
            move_broom(
                x=float(xyz[0]),
                z=float(xyz[2]),
                rotation=rotations(t).as_matrix(),
                label="align_broom",
            )
            sweep_y = sweep_y_saved
        self.capture(label="broom aligned", keyframe=True)
        angle = -80.0
        clearance = 0.01 * np.sin(np.deg2rad(80)) + 0.001
        for z in np.linspace(0.50, clearance, 17):
            move_broom(x=start_x, z=float(z), angle=angle, label="lower_broom")
        self.capture(label="broom lowered", keyframe=True)
        stroke = np.linspace(
            start_x, PIT_X[0] + 0.15, max(2, int((PIT_X[0] + 0.15 - start_x) / 0.03) + 1)
        )
        for x in stroke:
            move_broom(x=float(x), z=float(clearance), angle=angle, label="push_to_rim")
        self.capture(label="stroke end past rim", keyframe=True)
        self.hold(count=15, label="settle_at_rim")
        for x in np.linspace(PIT_X[0] + 0.15, 0.60, 8):
            move_broom(x=float(x), z=float(clearance), angle=angle, label="withdraw_broom")
        move_broom(x=0.60, z=0.15, angle=angle, label="lift_clear")
        self.hold(count=30, label="settle_after_withdrawal")

    # --- episodes ----------------------------------------------------------

    def run_phase(self, *, name: str, action: Any) -> Phase:
        self.phase = name
        start = self.ticks
        error = None
        try:
            action()
        except StepBudgetExhausted as exc:
            error = f"step budget: {exc}"
        except RuntimeError as exc:
            error = repr(exc)
        self.capture(label=f"{name} ended" + (" with error" if error else ""), keyframe=True)
        record = Phase(
            name=name,
            start_tick=start,
            end_tick=self.ticks,
            cubes_in_hole=self.in_hole(),
            error=error,
        )
        self.phases.append(record)
        print(json.dumps(record.model_dump()), flush=True)
        return record

    def human_fires(self, *, name: str, action: Any) -> None:
        self.phase = f"human {name}"
        self.show_banner(text=f"HUMAN {name} fires", hold_ticks=self.plan.fps)
        self.capture(label=f"before {name}", keyframe=True)
        action()
        self.capture(label=f"{name} fired", keyframe=True)
        # Hold the arm reference so the robot neither drifts nor fights its reset.
        self.arm_reference = self.data.qpos[self.qa].copy()
        self.base_target = self.data.qpos[self.ba].copy()
        self.grip = 0.0
        self.hold(count=self.plan.fps, label=f"after {name}")

    def run(self) -> EpisodeResult:
        self.capture(label="reset", keyframe=True)
        swept: int | None = None
        if self.episode == "sweep_then_reset_cubes":
            pick = self.run_phase(name="pick_broom", action=self.pick_floor_broom)
            if pick.error is None:
                sweep = self.run_phase(name="sweep_into_hole", action=self.sweep_into_hole)
                swept = sweep.cubes_in_hole
            self.human_fires(
                name="reset_cubes", action=lambda: HumanResets.reset_cubes(core=self.core)
            )
        elif self.episode == "drive_in_then_reset_robot":
            pick = self.run_phase(name="pick_broom", action=self.pick_floor_broom)
            if pick.error is None:
                self.run_phase(name="raise_broom", action=self.raise_broom)
            self.run_phase(name="drive_into_pit", action=self.drive_into_pit)
            self.human_fires(
                name="reset_robot",
                action=lambda: HumanResets.reset_robot(
                    core=self.core,
                    home_qpos=self.home_qpos,
                    robot_qpos_indices=self.robot_qpos_indices,
                ),
            )
        else:
            raise ValueError(f"unknown episode {self.episode!r}")
        self.phase = "end"
        self.capture(label="end", keyframe=True)
        self.writer.close()
        sheet = Captions.contact_sheet(
            keyframes=self.keyframes[:8] if len(self.keyframes) <= 8 else self.thin_keyframes(),
            path=self.plan.output_dir / f"sweep_into_hole3d_{self.episode}_contact_sheet.png",
        )
        peak = max(self.in_hole_history)
        return EpisodeResult(
            episode=self.episode,
            seed=self.plan.seed,
            scene=self.plan.scene,
            ticks=self.ticks,
            phases=tuple(self.phases),
            cubes_in_hole_after_sweep=swept,
            cubes_in_hole_after_reset=self.in_hole(),
            base_z_min=min(self.base_z),
            base_z_max=max(self.base_z),
            broom_head_z_min=float(np.nanmin(self.head_z)),
            cubes_bounced_out=max(0, peak - (swept if swept is not None else peak)),
            lowest_robot_geom_z=self.lowest_robot_geom_z,
            video=self.video_path,
            contact_sheet=sheet,
            encoder=self.writer.encoder,
        )

    def thin_keyframes(self) -> list[tuple[str, np.ndarray]]:
        picks = np.linspace(0, len(self.keyframes) - 1, 8).round().astype(int)
        return [self.keyframes[i] for i in picks]

    def raise_broom(self) -> None:
        position = self.data.site_xpos[self.site].copy() + [0.0, 0.0, 0.15]
        rotation = self.data.site_xmat[self.site].reshape(3, 3).copy()
        self.move(
            position=position, rotation=rotation, label="raise_broom", position_tolerance=0.03
        )

    def drive_into_pit(self) -> None:
        """Base straight along +x through the near rim to the pit centre, arm held."""
        here = self.data.qpos[self.ba].copy()
        self.move_base(target=[here[0], 0.0, 0.0], label="line_up_with_pit")
        self.capture(label="lined up with pit", keyframe=True)
        centre_x = (PIT_X[0] + PIT_X[1]) / 2
        self.move_base(target=[centre_x, 0.0, 0.0], label="drive_over_rim")
        self.hold(count=self.plan.fps, label="in_pit")

    def close(self) -> None:
        self.env.close()


class Validator:
    @staticmethod
    def import_kinder() -> Any:
        os.environ.setdefault("DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD", "1")
        KinderBackend.configure_headless_rendering()
        import kinder
        import kinder.envs.dynamic3d.envs  # noqa: F401  (the MODULE, not the package)

        kinder.register_all_environments()
        KinderBackend.configure_headless_rendering()
        return kinder

    @staticmethod
    def run(*, plan: ValidationPlan) -> list[EpisodeResult]:
        results = []
        for episode in plan.episodes:
            probe = HoleProbe(plan=plan, episode=episode)
            try:
                results.append(probe.run())
            finally:
                probe.close()
        (plan.output_dir / "sweep_into_hole3d_validation.json").write_text(
            json.dumps([r.model_dump(mode="json") for r in results], indent=2) + "\n"
        )
        return results

    @staticmethod
    def parse_args(*, argv: list[str] | None = None) -> ValidationPlan:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument(
            "--scene", default="mimiclabs-lab2", choices=["mimiclabs-lab2", "simple"]
        )
        parser.add_argument("--episodes", nargs="+", choices=EPISODES, default=list(EPISODES))
        parser.add_argument("--fps", type=int, default=30)
        parser.add_argument("--tick-limit", type=int, default=9000)
        args = parser.parse_args(argv)
        return ValidationPlan(
            output_dir=args.output_dir,
            seed=args.seed,
            scene=args.scene,
            episodes=tuple(args.episodes),
            fps=args.fps,
            tick_limit=args.tick_limit,
        )

    @staticmethod
    def main() -> None:
        plan = Validator.parse_args()
        for result in Validator.run(plan=plan):
            print(json.dumps(result.model_dump(mode="json"), indent=2), flush=True)


if __name__ == "__main__":
    Validator.main()
