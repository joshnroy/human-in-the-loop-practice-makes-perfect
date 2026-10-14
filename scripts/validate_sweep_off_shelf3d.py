"""TEMPORARY scripted solvability check of SweepOffShelf3D: captioned videos in lab2.

Not a planner skill, not a `Method`, not a `HumanOracle`. This drives the draft
`SweepOffShelf3D-o5` task with the classical IK/move/base controllers of
`validate_sweep_into_hole3d.py` (same TidyBot), to record whether the physics as built
lets a scripted robot do the task at all. A failure is a finding this script reports,
never something it patches: nothing here edits the environment, and nothing teleports
inside an episode except the documented scene setup of `cube_out_of_box_to_shelf`.

Episodes, each one video rendered from the task's `front_view` and `side_view`
cameras side by side:

1. `pick_wiper`: grasp the wiper on shelf index 2 by friction, lift it out of the
   cupboard, then drive and turn the base while holding it.
2. `sweep_cubes_into_box`: pick the wiper, then pull the cubes on shelf index 1 over
   the open front edge with the wiper head; `x/5` by the env's own `cube_in_box`.
3. `cube_out_of_box_to_shelf`: setup drops `cube_0` into the box; the robot picks it
   out and places it back on shelf index 1.

The cupboard slots are 0.254 m tall, too short for a vertical gripper (the wrist hits
the board above), so every grasp leans the gripper about its closing axis. A lean
keeps the pads on the same two faces, which is why the cube is picked and placed at
one lean: it stays upright only if the gripper's orientation is the same at both ends.
The wiper is grasped on its handle as close to its centre of mass as the pads allow,
because a friction grasp resists rotation about the closing axis worst of all.

Every KINDER import lives inside `import_kinder`; argument parsing, stroke planning
and the caption/sheet helpers test without a simulator.
"""

import argparse
import contextlib
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

ENV_ID = "kinder/SweepOffShelf3D-o5-v0"
CUBES = tuple(f"cube_{i}" for i in range(5))
# The task's front_view, and a free camera added here (not in the task) that looks
# into the cube slot from the front left, so the head, the cubes and the front edge show.
CAMERAS = ("front_view", "slot_view")
SLOT_LOOKAT = (1.45, 0.0, 0.29)
SLOT_DISTANCE = 0.85
SLOT_AZIMUTH = 40.0
SLOT_ELEVATION = -24.0
PANEL_SIZE = (640, 480)
FRAME_SIZE = (PANEL_SIZE[0] * len(CAMERAS), PANEL_SIZE[1])
EPISODES = ("pick_wiper", "sweep_cubes_into_box", "cube_out_of_box_to_shelf")
CUBE_SHELF = 1
WIPER_SHELF = 2
# The base stays this far back so its front casters clear the box's near wall (x 1.11).
BASE_X = 0.80
# Lean of the gripper from vertical, about its closing axis, for the wiper grasp, and
# how far the pinch site stands back from the handle's centre line along the approach
# so the pad tips clear the board (kinematic probe: pads 2-4 mm above it).
WIPER_LEAN_DEG = 60.0
WIPER_STANDOFF = 0.035
CUBE_STANDOFF = 0.035
CUBE_LEANS_DEG = (30.0, 40.0, 50.0, 60.0)
PAD_HALF_LENGTH = 0.011
# Shelf pick on the handle's middle; set-down spot on the cupboard top for the regrasp.
PICK_ALONG = 0.14
# At the centre of mass: a friction grasp 9 cm from it lets the head droop and creep.
REGRASP_ALONG = None
# Head centre when the wiper is laid down between strokes: its centre of mass on the board.
RESEAT_HEAD_X = 1.50
SLOT_ENTRY_RISE = 0.02
REGRASP_HEAD_XY = (0.95, 0.55)
REGRASP_BASE_XS = (0.45, 0.55)
SWEEP_STEM = (-1.0, 0.0, 0.0)
CARRY_Z = 0.65
HEAD_AHEAD = 0.80
HEAD_AHEADS = (0.80, 0.75, 0.85, 0.70, 0.90, 0.65)
# Stroke geometry: the head's centre starts behind the deepest cube and stops with
# the head just past the shelf's front edge; reach is the head half-length less a
# margin, limit keeps the head off the cupboard's side walls.
STROKE_START_X = 1.60
STROKE_END_X = 1.355
STROKE_STEP = 0.02
STROKE_REACH = 0.08
STROKE_LIMIT = 0.17
HOVER = 0.08
# The head skims the board rather than resting on it: board friction on a loaded
# head is a moment about the closing axis, which a friction grasp barely resists
# (measured: 10-12 deg of tilt and the head riding over the cubes); the cubes'
# horizontal push on the head's face is not a moment about that axis at all.
HEAD_ON_BOARD = 0.004
LOWER_STEP = 0.002
LOWER_STEPS = 50
CONTACT_STEP_LIMIT = 40
CONTACT_PENETRATION = 0.015
SWEEP_PASSES = 2
PLACE_XY = (1.45, 0.0)
DROP_XY = (1.235, 0.0)


class StepBudgetExhausted(RuntimeError):
    """The episode's hard control-step cap was reached; nothing here loops open."""


class Phase(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    start_tick: int
    end_tick: int
    in_box: int
    on_floor: int
    on_shelf: int
    error: str | None = None


class EpisodeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    episode: str
    seed: int
    scene: str
    ticks: int
    phases: tuple[Phase, ...]
    in_box: int
    on_floor: int
    on_shelf: int
    success: bool
    wiper_slip_m: float | None
    wiper_slip_deg: float
    wiper_lift_m: float
    robot_cupboard_contact_steps: int
    arm_cupboard_contact_steps: int
    robot_box_contact_steps: int
    notes: tuple[str, ...]
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
    tick_limit: int = 9000


class Strokes:
    @staticmethod
    def cover(*, ys: list[float], reach: float, limit: float) -> list[float]:
        """Fewest stroke centres, each covering +/- reach, that cover every y."""
        centres: list[float] = []
        for y in sorted(ys):
            if centres and y <= centres[-1] + reach:
                continue
            centres.append(float(np.clip(y + reach, -limit, limit)))
        return centres


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
    def stack(*, panels: list[np.ndarray]) -> np.ndarray:
        return np.concatenate([np.asarray(p, dtype=np.uint8) for p in panels], axis=1)

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
        *, keyframes: list[tuple[str, np.ndarray]], path: Path, columns: int = 2
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
        if self.process.stdin.closed:
            return
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError(f"ffmpeg ({self.encoder}) exited with {self.process.returncode}")


class Frames:
    @staticmethod
    def pose(*, rotation: np.ndarray, position: Any) -> np.ndarray:
        out = np.eye(4)
        out[:3, :3] = rotation
        out[:3, 3] = position
        return out

    @staticmethod
    def leaned(*, closing: np.ndarray, lean_toward: np.ndarray, lean_deg: float) -> np.ndarray:
        """Gripper rotation: pads close along `closing`, approach leans from straight
        down toward the horizontal unit vector `lean_toward` (local z is the approach,
        local x the closing axis, as in the hole script's top-down grasp)."""
        lean = np.deg2rad(lean_deg)
        approach = np.sin(lean) * lean_toward + np.array([0.0, 0.0, -np.cos(lean)])
        return np.column_stack([closing, np.cross(approach, closing), approach])


class ShelfProbe:
    """The hole script's controllers, pointed at the cupboard."""

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
        owner = {g: self.root_name(body=self.model.geom_bodyid[g]) for g in range(self.model.ngeom)}
        self.robot_geoms = {g for g, r in owner.items() if r == "robot_base_link"}
        # The 2F-85's own links (pads, fingers, linkage, palm), as opposed to the arm.
        self.gripper_geoms = {
            g
            for g in self.robot_geoms
            if any(
                part in str(self.model.body(self.model.geom_bodyid[g]).name)
                for part in ("pad", "follower", "driver", "coupler", "spring", "robot_base")
            )
            and self.model.body(self.model.geom_bodyid[g]).name != "robot_base_link"
        }
        self.pad_geoms = {g for g in self.robot_geoms if "pad" in (self.model.geom(g).name or "")}
        self.cupboard_geoms = {g for g, r in owner.items() if r == "cupboard_1"}
        self.box_geoms = {g for g, r in owner.items() if r == "box_0"}
        self.wiper_geoms = sorted(g for g, r in owner.items() if r == "wiper_0")
        sizes = self.model.geom_size[self.wiper_geoms]
        self.handle_geom = self.wiper_geoms[int(np.argmax(sizes[:, 0]))]
        self.head_geom = self.wiper_geoms[int(np.argmax(sizes[:, 1]))]
        self.wiper_body = self.model.body("wiper_0").id
        self.board = {
            shelf: self.core.shelf_top_z(shelf=shelf) for shelf in (CUBE_SHELF, WIPER_SHELF)
        }
        self.front_x = self.core.shelf_front_x()
        self.wiper_start = self.wiper_pose()
        self.head_start = self.data.geom_xpos[self.head_geom].copy()
        self.stem_start = self.stem_direction()
        self.base_target = self.data.qpos[self.ba].copy()
        self.arm_reference = self.data.qpos[self.qa].copy()
        self.home_arm = self.arm_reference.copy()
        self.base_step_limit = 0.015
        self.grip = 0.0
        self.ticks = 0
        self.phase = "start"
        self.banner: str | None = None
        self.banner_until = -1
        self.phases: list[Phase] = []
        self.keyframes: list[tuple[str, np.ndarray]] = []
        self.notes: list[str] = []
        self.cupboard_contact_steps = 0
        self.arm_cupboard_contact_steps = 0
        self.box_contact_steps = 0
        self.in_hand: np.ndarray | None = None
        self.slip: float | None = None
        self.slip_deg = 0.0
        self.worst_wiper_error = 0.0
        self.contact_shortfalls = 0
        self.slot_renderer = self.mujoco.Renderer(
            self.model, height=PANEL_SIZE[1], width=PANEL_SIZE[0]
        )
        self.slot_camera = self.mujoco.MjvCamera()
        self.slot_camera.lookat[:] = SLOT_LOOKAT
        self.slot_camera.distance = SLOT_DISTANCE
        self.slot_camera.azimuth = SLOT_AZIMUTH
        self.slot_camera.elevation = SLOT_ELEVATION
        plan.output_dir.mkdir(parents=True, exist_ok=True)
        self.stem = f"sweep_off_shelf3d_{episode}_seed{plan.seed}"
        self.video_path = plan.output_dir / f"{self.stem}.mp4"
        self.writer = VideoWriter(path=self.video_path, fps=plan.fps)

    @property
    def mujoco(self) -> Any:
        import mujoco

        return mujoco

    def root_name(self, *, body: int) -> str:
        while self.model.body_parentid[body] != 0:
            body = self.model.body_parentid[body]
        return str(self.model.body(body).name)

    # --- measurement -------------------------------------------------------

    def counts(self) -> dict[str, int]:
        in_box = on_floor = on_shelf = 0
        for name in CUBES:
            x, _, z = self.core.get_object(name).get_pose()[0]
            if self.core.cube_in_box(name):
                in_box += 1
            elif z < 0.05:
                on_floor += 1
            elif x > self.front_x and abs(z - self.board[CUBE_SHELF] - 0.01) < 0.01:
                on_shelf += 1
        return {"in_box": in_box, "on_floor": on_floor, "on_shelf": on_shelf}

    def wiper_pose(self) -> np.ndarray:
        return Frames.pose(
            rotation=self.data.xmat[self.wiper_body].reshape(3, 3),
            position=self.data.xpos[self.wiper_body],
        )

    def site_pose(self) -> np.ndarray:
        return Frames.pose(
            rotation=self.data.site_xmat[self.site].reshape(3, 3),
            position=self.data.site_xpos[self.site],
        )

    def stem_direction(self) -> np.ndarray:
        """Unit vector along the handle, from the head toward the handle's free end."""
        axis = self.data.geom_xmat[self.handle_geom].reshape(3, 3)[:, 0].copy()
        offset = self.data.geom_xpos[self.handle_geom] - self.data.geom_xpos[self.head_geom]
        return axis if float(offset @ axis) > 0 else -axis

    def wiper_lift(self) -> float:
        return float(self.data.xpos[self.wiper_body][2] - self.wiper_start[2, 3])

    def header(self) -> str:
        c = self.counts()
        return (
            f"SweepOffShelf3D seed {self.plan.seed} | {self.episode} | step {self.ticks} | "
            f"{self.phase} | in box {c['in_box']}/5 | floor {c['on_floor']} | "
            f"shelf {c['on_shelf']}"
            + (f" | grasp slip {self.slip_deg:.0f} deg" if self.in_hand is not None else "")
        )

    def render(self) -> np.ndarray:
        panels = []
        for camera in CAMERAS:
            if camera == "slot_view":
                self.slot_renderer.update_scene(self.data, camera=self.slot_camera)
                panels.append(self.slot_renderer.render())
                continue
            frame = self.sim.render(
                width=PANEL_SIZE[0], height=PANEL_SIZE[1], camera_name=camera, mode="offscreen"
            )
            panels.append(np.asarray(frame[0] if isinstance(frame, tuple) else frame))
        return Captions.stack(panels=panels)

    def capture(self, *, label: str, keyframe: bool = False) -> None:
        banner = self.banner if self.ticks <= self.banner_until else None
        annotated = Captions.annotate(frame=self.render(), header=self.header(), banner=banner)
        self.writer.write(frame=annotated)
        if keyframe:
            self.keyframes.append((f"step {self.ticks}: {label}", annotated))

    def tally_contacts(self) -> None:
        cupboard = box = arm = False
        for contact in self.data.contact[: self.data.ncon]:
            a, b = contact.geom1, contact.geom2
            if a not in self.robot_geoms and b not in self.robot_geoms:
                continue
            mine, other = (a, b) if a in self.robot_geoms else (b, a)
            cupboard |= other in self.cupboard_geoms
            box |= other in self.box_geoms
            arm |= other in self.cupboard_geoms and mine not in self.gripper_geoms
        self.cupboard_contact_steps += int(cupboard)
        self.arm_cupboard_contact_steps += int(arm)
        self.box_contact_steps += int(box)

    def track_slip(self) -> None:
        if self.in_hand is None:
            return
        now = np.linalg.inv(self.site_pose()) @ self.wiper_pose()
        drift = float(np.linalg.norm(now[:3, 3] - self.in_hand[:3, 3]))
        turn = float(
            np.rad2deg(Rotation.from_matrix(now[:3, :3] @ self.in_hand[:3, :3].T).magnitude())
        )
        self.slip = drift if self.slip is None else max(self.slip, drift)
        self.slip_deg = max(self.slip_deg, turn)

    # --- control (the hole script's, with collision-aware IK) ---------------

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
        self.tally_contacts()
        self.track_slip()
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

    def touching(self, *, d: Any) -> list[str]:
        """Robot links in contact with the cupboard or box in `d`, for error messages."""
        self.mujoco.mj_collision(self.model, d)
        fixed = self.cupboard_geoms | self.box_geoms
        names = set()
        for contact in d.contact[: d.ncon]:
            for mine, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                if mine in self.robot_geoms and other in fixed:
                    names.add(
                        f"{self.model.body(self.model.geom_bodyid[mine]).name}"
                        f"/{self.model.geom(other).name or 'box'}"
                    )
        return sorted(names)

    def collision_cost(self, *, d: Any, holding: bool) -> float:
        """Penetration of the robot into itself, the cupboard or the box, and (unless
        the wiper is held) of non-pad robot links into the wiper."""
        self.mujoco.mj_collision(self.model, d)
        cost = 0.0
        wiper = set(self.wiper_geoms)
        fixed = self.cupboard_geoms | self.box_geoms
        for contact in d.contact[: d.ncon]:
            a, b = contact.geom1, contact.geom2
            depth = max(0.0, -contact.dist)
            if a in self.robot_geoms and b in self.robot_geoms:
                cost += max(0.0, depth - 0.001)
            elif (a in self.robot_geoms and b in fixed) or (b in self.robot_geoms and a in fixed):
                cost += depth + 0.001
            elif not holding and (
                (a in self.robot_geoms and b in wiper and a not in self.pad_geoms)
                or (b in self.robot_geoms and a in wiper and b not in self.pad_geoms)
            ):
                cost += depth
        return cost

    def ik(
        self,
        *,
        position: Any,
        rotation: np.ndarray,
        base: Any = None,
        holding: bool = False,
        attempts: int = 16,
    ) -> tuple[np.ndarray, float]:
        mujoco = self.mujoco
        d, m = self.ik_data, self.model
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        target = Rotation.from_matrix(rotation)
        rng = np.random.default_rng(0)
        initial = self.data.qpos[self.qa].copy()
        candidates = []
        err = np.full(6, np.inf)
        for attempt in range(attempts):
            d.qpos[:] = self.data.qpos
            if base is not None:
                d.qpos[self.ba] = base
            # Odd restarts start near the home arm pose: seeding only from the current
            # pose gets stuck in local minima once the arm is stretched into the slot.
            if attempt % 2:
                d.qpos[self.qa] = self.home_arm + rng.normal(0, 0.3 * (attempt > 1), 7)
            elif attempt:
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
            if np.linalg.norm(err) < 0.006:
                collision = self.collision_cost(d=d, holding=holding)
                q = d.qpos[self.qa].copy()
                score = np.linalg.norm((q - initial + np.pi) % (2 * np.pi) - np.pi)
                candidates.append((score + 1000 * collision, q, collision))
                # Holding the wiper, a jump to another IK branch swings it into the
                # box; stay on the current branch unless it really penetrates.
                if attempt == 0 and (collision == 0 or (holding and collision < 0.01)):
                    return q, float(collision)
        if candidates:
            best = min(candidates, key=lambda c: c[0])
            return best[1], best[2]
        raise RuntimeError(
            f"IK failed at {np.round(position, 3)}: residual {np.linalg.norm(err):.3f}"
        )

    def move(
        self,
        *,
        position: Any,
        rotation: np.ndarray,
        label: str = "move",
        limit: int = 250,
        position_tolerance: float = 0.008,
        orientation_tolerance: float = 0.05,
        holding: bool = False,
    ) -> None:
        try:
            arm, _ = self.ik(position=position, rotation=rotation, holding=holding)
        except RuntimeError as exc:
            raise RuntimeError(f"{label}: {exc}") from exc
        for i in range(limit):
            if i and i % 8 == 0:
                # Keep the last reachable target; convergence is checked below.
                with contextlib.suppress(RuntimeError):
                    arm, _ = self.ik(position=position, rotation=rotation, holding=holding)
            self.step(arm=arm, label=label)
            actual = Rotation.from_matrix(self.data.site_xmat[self.site].reshape(3, 3))
            angle = (Rotation.from_matrix(rotation) * actual.inv()).magnitude()
            if (
                np.linalg.norm(self.data.site_xpos[self.site] - position) < position_tolerance
                and angle < orientation_tolerance
            ):
                return
        raise RuntimeError(
            f"Physical move failed: {label}, target={np.round(position, 3)}, "
            f"actual={np.round(self.data.site_xpos[self.site], 3)}"
        )

    def path(
        self,
        *,
        position: Any,
        rotation: np.ndarray,
        label: str,
        count: int = 6,
        holding: bool = False,
        position_tolerance: float = 0.012,
    ) -> None:
        """Cartesian straight line with a slerped orientation, so the joint-space
        controller cannot swing the arm through the cupboard between waypoints."""
        start = self.site_pose()
        rotations = Slerp([0, 1], Rotation.from_matrix([start[:3, :3], rotation]))
        for t in np.linspace(0, 1, count + 1)[1:]:
            self.move(
                position=(1 - t) * start[:3, 3] + t * np.asarray(position),
                rotation=rotations(t).as_matrix(),
                label=label,
                holding=holding,
                position_tolerance=position_tolerance if t < 1 else 0.008,
                orientation_tolerance=0.08 if t < 1 else 0.05,
            )

    def choose_base(
        self,
        *,
        targets: list[tuple[np.ndarray, np.ndarray]],
        ys: list[float],
        holding: bool,
        xs: tuple[float, ...] = (BASE_X,),
    ) -> np.ndarray:
        """Base pose (facing +x) whose IK reaches every target with the least
        penetration; ties go to the first (x, y) listed."""
        best: tuple[float, np.ndarray] | None = None
        for x, y in [(x, y) for x in xs for y in ys]:
            base = np.array([x, y, 0.0])
            cost = 0.0
            for position, rotation in targets:
                try:
                    _, collision = self.ik(
                        position=position, rotation=rotation, base=base, holding=holding
                    )
                except RuntimeError:
                    collision = 1.0
                cost += collision
            if best is None or cost < best[0] - 1e-6:
                best = (cost, base)
        assert best is not None
        if best[0] > 0:
            self.notes.append(f"best base y={best[1][1]:.2f} still penetrates {best[0]:.4f} m")
        return best[1]

    # --- wiper ---------------------------------------------------------------

    def wiper_grasp(
        self, *, along: float | None, lean_sign: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pinch-site pose for a lean grasp on the handle, `along` metres from the head
        centre (None: as near the centre of mass as the pads fit beside the head).
        lean_sign +1 leans the approach toward the handle's free end, so the wrist sits
        over the head; -1 leans it toward the head, the wrist over the free end."""
        u = self.stem_direction()
        head = self.data.geom_xpos[self.head_geom]
        handle = self.data.geom_xpos[self.handle_geom]
        com = self.data.subtree_com[self.wiper_body]
        head_half_width = float(np.min(self.model.geom_size[self.head_geom][:2]))
        lo = head_half_width + PAD_HALF_LENGTH + 0.006
        hi = float(self.model.geom_size[self.handle_geom][0] * 2) - PAD_HALF_LENGTH
        s = float(np.clip((com - head) @ u if along is None else along, lo, hi))
        point = head + s * u
        point[2] = handle[2]
        closing = np.array([-u[1], u[0], 0.0])
        rotation = Frames.leaned(
            closing=closing, lean_toward=lean_sign * u, lean_deg=WIPER_LEAN_DEG
        )
        return point - WIPER_STANDOFF * rotation[:, 2], rotation

    def pick_wiper(self) -> None:
        """On the shelf only the lean toward the free end clears the board above
        (kinematic probe), so this grasp sits on the handle's middle."""
        grasp, rotation = self.wiper_grasp(along=PICK_ALONG, lean_sign=1.0)
        # The wrist sits just under the board above at the grasp, so the hand comes in
        # level, pads 2 cm above the handle, and only then drops onto it.
        pre = grasp + np.array([0.0, 0.0, SLOT_ENTRY_RISE])
        porch = np.array([self.front_x - 0.12, pre[1], pre[2]])
        base = self.choose_base(
            targets=[
                (porch, rotation),
                ((porch + pre) / 2, rotation),
                (pre, rotation),
                (grasp, rotation),
            ],
            ys=[float(grasp[1] + dy) for dy in (0.0, 0.1, -0.1, 0.2)],
            holding=False,
        )
        self.grip = 0.0
        self.move_base(target=base, label="drive_to_cupboard")
        self.capture(label="at cupboard", keyframe=True)
        self.path(position=porch, rotation=rotation, label="to_slot_mouth", count=8)
        self.path(position=pre, rotation=rotation, label="into_wiper_slot", count=10)
        self.path(position=grasp, rotation=rotation, label="lower_onto_handle", count=3)
        self.capture(label="pads around handle", keyframe=True)
        self.grip = 1.0
        self.hold(count=25, label="close_on_handle")
        self.in_hand = np.linalg.inv(self.site_pose()) @ self.wiper_pose()
        self.slip = 0.0
        lift = grasp + np.array([0.0, 0.0, SLOT_ENTRY_RISE])
        self.path(position=lift, rotation=rotation, label="lift_wiper", count=4, holding=True)
        out = np.array([self.front_x - 0.15, lift[1], lift[2]])
        self.path(position=out, rotation=rotation, label="out_of_slot", count=10, holding=True)
        clear = out + np.array([-0.05, 0.0, 0.15])
        self.path(position=clear, rotation=rotation, label="lift_clear", count=4, holding=True)
        self.capture(label=f"lifted, wiper +{self.wiper_lift():.2f} m", keyframe=True)
        if self.wiper_lift() < 0.05:
            raise RuntimeError(f"Wiper not lifted (z change {self.wiper_lift():.3f} m)")

    def carry_wiper(self) -> None:
        here = self.data.qpos[self.ba].copy()
        for target in (
            here + [-0.30, 0.0, 0.0],
            here + [-0.30, 0.0, 0.5],
            here + [-0.30, 0.0, 0.0],
            here,
        ):
            self.move_base(target=target, label="carry_wiper")
        self.hold(count=20, label="hold_after_carry")
        self.capture(label=f"after carry, slip {self.slip or 0:.3f} m", keyframe=True)

    def regrasp_wiper(self) -> None:
        """The shelf grasp puts the wrist over the head, which is exactly where it
        cannot be when the head is pulled from the back of the cube slot. Set the
        wiper down on the open floor beside the box and take it again leaning the
        other way. (The cupboard top was tried first: a leaned grasp there is out of
        the arm's reach from any base pose that clears the box.)"""
        assert self.in_hand is not None
        half = float(self.model.geom_size[self.head_geom][2])
        head = np.array([*REGRASP_HEAD_XY, half + 0.004])
        stem = self.stem_start
        target = self.wiper_target(head=head, stem=stem) @ np.linalg.inv(self.in_hand)
        lifted = Frames.pose(rotation=target[:3, :3], position=target[:3, 3] + [0, 0, 0.15])
        base = self.choose_base(
            targets=[(lifted[:3, 3], target[:3, :3]), (target[:3, 3], target[:3, :3])],
            xs=REGRASP_BASE_XS,
            ys=[float(head[1] + dy) for dy in (0.0, -0.15, 0.15)],
            holding=True,
        )
        self.move_base(target=base, label="to_set_down_spot")
        for z in (0.30, 0.15, 0.05, 0.015):
            self.move_wiper(head=head + [0.0, 0.0, z], label="lower_to_floor", stem=stem)
        self.move_wiper(head=head, label="set_down_on_floor", stem=stem)
        self.notes.append(
            f"shelf grasp held with max slip {self.slip or 0:.3f} m / {self.slip_deg:.1f} deg"
        )
        self.grip = 0.0
        self.in_hand = None
        self.hold(count=20, label="release_on_floor")
        approach = self.site_pose()[:3, 2]
        self.path(
            position=self.data.site_xpos[self.site] - 0.06 * approach + [0.0, 0.0, 0.06],
            rotation=self.site_pose()[:3, :3],
            label="back_off",
            count=3,
        )
        self.capture(label="wiper set down on the floor", keyframe=True)
        grasp, rotation = self.wiper_grasp(along=REGRASP_ALONG, lean_sign=-1.0)
        pre = grasp + np.array([0.0, 0.0, 0.04])
        base = self.choose_base(
            targets=[(pre, rotation), (grasp, rotation), (grasp + [0.0, 0.0, 0.25], rotation)],
            xs=REGRASP_BASE_XS,
            ys=[float(grasp[1] + dy) for dy in (0.0, -0.15, 0.15)],
            holding=False,
        )
        self.move_base(target=base, label="to_regrasp")
        self.path(position=pre, rotation=rotation, label="regrasp_approach", count=6)
        self.path(position=grasp, rotation=rotation, label="regrasp_lower", count=4)
        self.grip = 1.0
        self.hold(count=25, label="regrasp_close")
        self.in_hand = np.linalg.inv(self.site_pose()) @ self.wiper_pose()
        self.slip, self.slip_deg = 0.0, 0.0
        self.path(
            position=grasp + [0.0, 0.0, 0.25],
            rotation=rotation,
            label="regrasp_lift",
            count=5,
            holding=True,
        )
        self.capture(label="regrasped", keyframe=True)
        if self.data.xpos[self.wiper_body][2] < 0.15:
            raise RuntimeError("Regrasp did not lift the wiper off the floor")

    def wiper_target(self, *, head: Any, stem: Any = SWEEP_STEM) -> np.ndarray:
        """Wiper body pose: flat as it started, handle pointing along `stem` (by
        default at the robot, -x), head centre at `head`."""
        u = self.stem_start
        yaw = np.arctan2(stem[1], stem[0]) - np.arctan2(u[1], u[0])
        turn = Frames.pose(rotation=Rotation.from_euler("z", yaw).as_matrix(), position=head)
        return turn @ Frames.pose(rotation=np.eye(3), position=-self.head_start) @ self.wiper_start

    def move_wiper(self, *, head: Any, label: str, stem: Any = SWEEP_STEM) -> None:
        """Open loop on the last measured grasp: the hand goes where the wiper would
        be at `head` if it did not move in the grasp."""
        assert self.in_hand is not None
        desired = self.wiper_target(head=head, stem=stem)
        target = desired @ np.linalg.inv(self.in_hand)
        distance = float(np.linalg.norm(target[:3, 3] - self.data.site_xpos[self.site]))
        try:
            self.path(
                position=target[:3, 3],
                rotation=target[:3, :3],
                label=label,
                count=max(1, int(np.ceil(distance / 0.03))),
                holding=True,
                position_tolerance=0.02,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc} | head wanted {np.round(head, 3)}, head at "
                f"{np.round(self.data.geom_xpos[self.head_geom], 3)}, handle points "
                f"{np.round(self.stem_direction(), 2)}, base {np.round(self.data.qpos[self.ba], 2)}"
            ) from exc
        error = float(np.linalg.norm(desired[:3, 3] - self.wiper_pose()[:3, 3]))
        self.worst_wiper_error = max(self.worst_wiper_error, error)
        if error > 0.05:
            raise RuntimeError(
                f"{label}: wiper {error:.3f} m off target with the hand on target "
                f"(slipped {self.slip_deg:.0f} deg in grasp)"
            )

    def recalibrate_grasp(self, *, label: str) -> None:
        """Re-measure the hand-to-wiper transform while the wiper hangs free. The
        friction grasp lets the head droop about the closing axis under its own
        weight; aiming with the transform measured at closing puts the drooped head
        into the front edge of the board."""
        assert self.in_hand is not None
        now = np.linalg.inv(self.site_pose()) @ self.wiper_pose()
        turn = Rotation.from_matrix(now[:3, :3] @ self.in_hand[:3, :3].T).magnitude()
        self.notes.append(
            f"{label}: grasp re-measured, wiper had turned {np.rad2deg(turn):.1f} deg"
        )
        self.in_hand = now

    def shelf_cube_ys(self) -> list[float]:
        ys = []
        for name in CUBES:
            x, y, z = self.core.get_object(name).get_pose()[0]
            if x > self.front_x and z > self.board[CUBE_SHELF]:
                ys.append(float(y))
        return ys

    def wiper_to(self, *, head: Any, label: str) -> None:
        """Head to `head`, the wiper flat with its handle pointing at the robot."""
        assert self.in_hand is not None
        target = self.wiper_target(head=head) @ np.linalg.inv(self.in_hand)
        self.place_base_for(target=target, ahead_of=float(head[0]), label=label)
        self.move_wiper(head=head, label=label)

    def hand_to(self, *, target: np.ndarray, label: str, tolerance: float = 0.006) -> None:
        """In contact with the board: a bounded move of the hand itself, the shortfall
        counted rather than treated as a failure."""
        self.place_base_for(
            target=target, ahead_of=float(self.data.geom_xpos[self.head_geom][0]), label=label
        )
        try:
            self.move(
                position=target[:3, 3],
                rotation=target[:3, :3],
                label=label,
                limit=CONTACT_STEP_LIMIT,
                holding=True,
                position_tolerance=tolerance,
                orientation_tolerance=0.1,
            )
        except RuntimeError:
            self.contact_shortfalls += 1

    def place_base_for(self, *, target: np.ndarray, ahead_of: float, label: str) -> None:
        """Keep the base about HEAD_AHEAD behind the head: the leaned hand only reaches
        0.6 m or more in front of the base, and the box keeps the base from coming
        closer than BASE_X, so the base carries the stroke and the arm trims."""
        base = None
        tried = []
        for ahead in HEAD_AHEADS:
            candidate = np.array([min(ahead_of - ahead, BASE_X), target[1, 3], 0.0])
            try:
                _, collision = self.ik(
                    position=target[:3, 3], rotation=target[:3, :3], base=candidate, holding=True
                )
            except RuntimeError:
                tried.append(f"{candidate[0]:.2f}: no IK")
                continue
            # With the head on the board the pad tips are on it too.
            if collision <= CONTACT_PENETRATION:
                base = candidate
                break
            tried.append(f"{candidate[0]:.2f}: {collision:.3f} m {self.touching(d=self.ik_data)}")
        if base is None:
            raise RuntimeError(f"{label}: no base pose reaches the hand target: {'; '.join(tried)}")
        if np.max(np.abs(base - self.data.qpos[self.ba])) > 0.002:
            self.move_base(target=base, label=label)

    def head_bottom(self) -> float:
        return float(
            min(
                (
                    self.data.geom_xpos[self.head_geom]
                    + self.data.geom_xmat[self.head_geom].reshape(3, 3)
                    @ (np.array(sign) * self.model.geom_size[self.head_geom])
                )[2]
                for sign in [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
            )
        )

    def lower_head_onto_board(self, *, label: str) -> None:
        """Hand straight down in small steps until the head's lowest edge is just
        above the board (HEAD_ON_BOARD)."""
        for _ in range(LOWER_STEPS):
            if self.head_bottom() <= self.board[CUBE_SHELF] + HEAD_ON_BOARD:
                return
            target = self.site_pose()
            target[2, 3] -= LOWER_STEP
            self.hand_to(target=target, label=label, tolerance=0.0015)
        self.notes.append(
            f"{label}: head still {self.head_bottom() - self.board[CUBE_SHELF]:.3f} m above board"
        )

    def level_target(self, *, x: float, above_board: float) -> np.ndarray:
        """Hand pose that would take the wiper from where it is now to flat, head
        centre at `x`, head bottom `above_board` over the cube shelf: the measured
        wiper, not the grasp at closing, so tilt in the grasp is taken out."""
        now = self.wiper_pose()
        head = self.data.geom_xpos[self.head_geom].copy()
        stem = self.stem_direction()
        level = np.array([stem[0], stem[1], 0.0])
        level /= np.linalg.norm(level)
        axis = np.cross(stem, level)
        angle = float(np.arcsin(np.clip(np.linalg.norm(axis), 0.0, 1.0)))
        turn = (
            Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()
            if angle > 1e-6
            else np.eye(3)
        )
        desired = (
            Frames.pose(rotation=turn, position=head)
            @ Frames.pose(rotation=np.eye(3), position=-head)
            @ now
        )
        half = float(self.model.geom_size[self.head_geom][2])
        drop = head[2] - half - (self.board[CUBE_SHELF] + above_board)
        desired[:3, 3] += [x - head[0], 0.0, -float(np.clip(drop, -0.01, 0.01))]
        return desired @ np.linalg.inv(now) @ self.site_pose()

    def pull(self, *, distance: float, label: str) -> None:
        """Pull the head toward the robot in STROKE_STEP increments, closed loop on the
        measured wiper (`level_target`). The cubes' drag turns the wiper in the
        friction grasp; open loop, the head tilts up and rides over the cubes
        (measured: 10-12 deg, 0/5 in the box)."""
        start_x = float(self.data.geom_xpos[self.head_geom][0])
        steps = int(round(distance / STROKE_STEP))
        for x in np.linspace(start_x, start_x - distance, steps + 1)[1:]:
            self.hand_to(target=self.level_target(x=x, above_board=HEAD_ON_BOARD), label=label)

    def reseat_wiper(self, *, label: str) -> None:
        """Lay the wiper flat on the swept strip of the cube shelf, let go and take it
        again at the grasp the stroke was planned with: the drag of each stroke turns
        it 10-15 deg in the friction grasp, and carried like that the wrist no longer
        clears the board above on the way back in."""
        lift = self.site_pose()
        lift[2, 3] += HOVER
        self.hand_to(target=lift, label=label)
        back = self.site_pose()
        back[0, 3] += RESEAT_HEAD_X - float(self.data.geom_xpos[self.head_geom][0])
        self.hand_to(target=back, label=label)
        for _ in range(3):
            self.hand_to(target=self.level_target(x=RESEAT_HEAD_X, above_board=HOVER), label=label)
        for _ in range(LOWER_STEPS // 5):
            if self.head_bottom() <= self.board[CUBE_SHELF] + 0.002:
                break
            self.hand_to(target=self.level_target(x=RESEAT_HEAD_X, above_board=0.0), label=label)
        self.notes.append(f"{label}: set down after turning {self.slip_deg:.1f} deg in the grasp")
        self.grip = 0.0
        self.in_hand = None
        self.hold(count=20, label=f"{label}_release")
        up = self.site_pose()
        up[2, 3] += 0.02
        self.hand_to(target=up, label=f"{label}_release")
        grasp, rotation = self.wiper_grasp(along=REGRASP_ALONG, lean_sign=-1.0)
        pre = Frames.pose(rotation=rotation, position=grasp + [0.0, 0.0, 0.025])
        self.hand_to(target=pre, label=f"{label}_regrasp")
        self.hand_to(
            target=Frames.pose(rotation=rotation, position=grasp),
            label=f"{label}_regrasp",
            tolerance=0.003,
        )
        self.grip = 1.0
        self.hold(count=25, label=f"{label}_close")
        self.in_hand = np.linalg.inv(self.site_pose()) @ self.wiper_pose()
        self.slip, self.slip_deg = 0.0, 0.0
        self.hand_to(
            target=self.level_target(x=RESEAT_HEAD_X, above_board=HOVER), label=f"{label}_lift"
        )

    def sweep(self) -> None:
        flat = self.board[CUBE_SHELF] + float(self.model.geom_size[self.head_geom][2])
        hover = flat + HOVER
        mouth = self.front_x - 0.12
        for sweep_pass in range(SWEEP_PASSES):
            centres = Strokes.cover(ys=self.shelf_cube_ys(), reach=STROKE_REACH, limit=STROKE_LIMIT)
            if not centres:
                return
            for y in centres:
                name = f"pass{sweep_pass}_y{y:+.2f}"
                # Carry the wiper high while the base drives, or it swings into the box.
                carry = self.site_pose()
                carry[2, 3] = max(carry[2, 3], CARRY_Z)
                self.path(
                    position=carry[:3, 3],
                    rotation=carry[:3, :3],
                    label=f"{name}_raise",
                    count=4,
                    holding=True,
                )
                self.move_base(target=[mouth - HEAD_AHEAD, y, 0.0], label=f"{name}_base")
                self.wiper_to(head=[mouth, y, hover + 0.12], label=f"{name}_above_box")
                self.wiper_to(head=[mouth, y, hover], label=f"{name}_mouth")
                self.recalibrate_grasp(label=f"{name}_mouth")
                for x in np.arange(mouth + 0.05, STROKE_START_X + 1e-9, 0.05):
                    self.wiper_to(head=[x, y, hover], label=f"{name}_reach_in")
                self.wiper_to(head=[STROKE_START_X, y, hover], label=f"{name}_reach_in")
                self.recalibrate_grasp(label=f"{name}_behind_cubes")
                self.lower_head_onto_board(label=f"{name}_lower")
                self.capture(label=f"{name}: head behind cubes", keyframe=True)
                self.pull(distance=STROKE_START_X - STROKE_END_X, label=f"{name}_pull")
                self.capture(label=f"{name}: stroke end", keyframe=True)
                head = self.data.geom_xpos[self.head_geom]
                self.notes.append(
                    f"{name}: stroke ended with head centre at x={head[0]:.3f} "
                    f"(target {STROKE_END_X}), bottom "
                    f"{self.head_bottom() - self.board[CUBE_SHELF]:+.3f} m above board, "
                    f"hand at x={self.data.site_xpos[self.site][0]:.3f}, cubes at "
                    + ", ".join(
                        f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})"
                        for p in (self.core.get_object(c).get_pose()[0] for c in CUBES)
                    )
                )
                self.reseat_wiper(label=f"{name}_reseat")
                self.wiper_to(head=[mouth, y, hover + 0.12], label=f"{name}_out")
                self.hold(count=15, label="settle")
                self.notes.append(
                    f"{name}: {self.counts()['in_box']}/5 in box after stroke, "
                    f"{self.contact_shortfalls} contact moves short so far"
                )

    # --- cube reset ----------------------------------------------------------

    def drop_cube_into_box(self) -> None:
        cube = self.core.get_object("cube_0")
        cube.set_pose([*DROP_XY, 0.10], [1.0, 0.0, 0.0, 0.0])
        cube.set_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.hold(count=self.plan.fps, label="cube_settles_in_box")
        if not self.core.cube_in_box("cube_0"):
            raise RuntimeError("Dropped cube did not settle in the box")

    def cube_grasp(self, *, centre: Any, lean_deg: float) -> tuple[np.ndarray, np.ndarray]:
        quaternion = self.core.get_object("cube_0").get_pose()[1]
        axes = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
        closing = max((axes[:, 0], axes[:, 1]), key=lambda v: abs(v[1])).copy()
        closing[2] = 0.0
        closing = closing / np.linalg.norm(closing) * np.sign(closing[1])
        toward = np.array([closing[1], -closing[0], 0.0])
        rotation = Frames.leaned(closing=closing, lean_toward=toward, lean_deg=lean_deg)
        return np.asarray(centre) - CUBE_STANDOFF * rotation[:, 2], rotation

    def cube_out_of_box_to_shelf(self) -> None:
        position = self.core.get_object("cube_0").get_pose()[0]
        place = np.array([*PLACE_XY, self.board[CUBE_SHELF] + 0.012])
        chosen = None
        for lean in CUBE_LEANS_DEG:
            pick_pose = self.cube_grasp(centre=position, lean_deg=lean)
            place_pose = self.cube_grasp(centre=place, lean_deg=lean)
            base = self.choose_base(
                targets=[pick_pose, place_pose], ys=[0.0, -0.15, 0.15], holding=True
            )
            _, a = self.ik(position=pick_pose[0], rotation=pick_pose[1], base=base)
            _, b = self.ik(position=place_pose[0], rotation=place_pose[1], base=base, holding=True)
            if a + b == 0:
                chosen = (lean, base, pick_pose, place_pose)
                break
        if chosen is None:
            raise RuntimeError("No lean reaches both the box floor and the shelf collision-free")
        lean, base, (grasp, rotation), (drop, _) = chosen
        self.notes.append(f"cube lean {lean:.0f} deg")
        approach = rotation[:, 2]
        self.grip = 0.0
        self.move_base(target=base, label="drive_to_box")
        above = grasp - 0.12 * approach + np.array([0.0, 0.0, 0.10])
        self.path(position=above, rotation=rotation, label="above_cube", count=8)
        self.path(position=grasp, rotation=rotation, label="into_box", count=6)
        self.grip = 1.0
        self.hold(count=25, label="close_on_cube")
        self.path(position=above, rotation=rotation, label="lift_out_of_box", count=6, holding=True)
        lifted = self.core.get_object("cube_0").get_pose()[0]
        self.capture(label=f"cube lifted to z={lifted[2]:.3f}", keyframe=True)
        if lifted[2] < self.board[CUBE_SHELF] - 0.1:
            raise RuntimeError(f"Cube not lifted out of the box (z={lifted[2]:.3f})")
        mouth = np.array([self.front_x - 0.12, drop[1], drop[2] + 0.04])
        self.path(position=mouth, rotation=rotation, label="to_shelf_mouth", count=6, holding=True)
        self.path(
            position=drop + [0.0, 0.0, 0.03], rotation=rotation, label="over_shelf", holding=True
        )
        self.path(position=drop, rotation=rotation, label="lower_to_shelf", count=3, holding=True)
        self.grip = 0.0
        self.hold(count=25, label="release_on_shelf")
        self.path(position=drop - 0.05 * approach, rotation=rotation, label="back_off", count=3)
        self.path(position=mouth, rotation=rotation, label="withdraw", count=4)
        placed = self.core.get_object("cube_0").get_pose()[0]
        self.capture(
            label=f"cube at ({placed[0]:.2f},{placed[1]:.2f},{placed[2]:.3f})", keyframe=True
        )
        if not (
            placed[0] > self.front_x + 0.01
            and abs(placed[2] - self.board[CUBE_SHELF] - 0.01) < 0.004
            and not self.core.cube_in_box("cube_0")
        ):
            raise RuntimeError(f"Cube not resting on shelf {CUBE_SHELF}: {np.round(placed, 3)}")

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
        if error:
            self.banner = f"{name} FAILED"
            self.banner_until = self.ticks + self.plan.fps
        self.capture(label=f"{name} ended" + (" with error" if error else ""), keyframe=True)
        c = self.counts()
        record = Phase(name=name, start_tick=start, end_tick=self.ticks, error=error, **c)
        self.phases.append(record)
        print(json.dumps(record.model_dump()), flush=True)
        return record

    def run(self) -> EpisodeResult:
        self.capture(label="reset", keyframe=True)
        success = False
        if self.episode == "pick_wiper":
            pick = self.run_phase(name="pick_wiper", action=self.pick_wiper)
            if pick.error is None:
                carry = self.run_phase(name="carry_wiper", action=self.carry_wiper)
                success = carry.error is None and self.wiper_lift() > 0.05
        elif self.episode == "sweep_cubes_into_box":
            pick = self.run_phase(name="pick_wiper", action=self.pick_wiper)
            if pick.error is None:
                regrasp = self.run_phase(name="regrasp_wiper", action=self.regrasp_wiper)
                if regrasp.error is None:
                    self.run_phase(name="sweep", action=self.sweep)
            if self.ticks + self.plan.fps <= self.plan.tick_limit:
                self.hold(count=self.plan.fps, label="settle")
            success = self.counts()["in_box"] == len(CUBES)
        elif self.episode == "cube_out_of_box_to_shelf":
            drop = self.run_phase(name="setup_drop_cube_into_box", action=self.drop_cube_into_box)
            if drop.error is None:
                place = self.run_phase(name="cube_to_shelf", action=self.cube_out_of_box_to_shelf)
                success = place.error is None
        else:
            raise ValueError(f"unknown episode {self.episode!r}")
        self.phase = "end"
        self.capture(label="end", keyframe=True)
        self.writer.close()
        sheet = Captions.contact_sheet(
            keyframes=self.thin_keyframes(),
            path=self.plan.output_dir / f"{self.stem}_contact_sheet.png",
        )
        return EpisodeResult(
            episode=self.episode,
            seed=self.plan.seed,
            scene=self.plan.scene,
            ticks=self.ticks,
            phases=tuple(self.phases),
            success=success,
            wiper_slip_m=self.slip,
            wiper_slip_deg=self.slip_deg,
            wiper_lift_m=self.wiper_lift(),
            robot_cupboard_contact_steps=self.cupboard_contact_steps,
            arm_cupboard_contact_steps=self.arm_cupboard_contact_steps,
            robot_box_contact_steps=self.box_contact_steps,
            notes=tuple(self.notes),
            video=self.video_path,
            contact_sheet=sheet,
            encoder=self.writer.encoder,
            **self.counts(),
        )

    def thin_keyframes(self) -> list[tuple[str, np.ndarray]]:
        if len(self.keyframes) <= 8:
            return self.keyframes
        picks = np.linspace(0, len(self.keyframes) - 1, 8).round().astype(int)
        return [self.keyframes[i] for i in picks]

    def close(self) -> None:
        self.writer.close()
        self.slot_renderer.close()
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
            probe = ShelfProbe(plan=plan, episode=episode)
            try:
                results.append(probe.run())
            finally:
                probe.close()
        (plan.output_dir / f"sweep_off_shelf3d_seed{plan.seed}_validation.json").write_text(
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
