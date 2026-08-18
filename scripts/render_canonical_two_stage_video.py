"""Render canonical photorealistic videos on Seed 103:
1. Two-Panel Synchronized Video:
   - Panel 1 (Left): Elevated Close-Up Camera (tracking cube grasp & displacement badge)
   - Panel 2 (Right): Canonical Viewpoint (exact untouched env.render() with scene_bg=True)
2. Single-Panel Canonical Video:
   - Full 640x480 video of the exact canonical viewpoint from env.render()
3. Two-Stage Controller (Waypoints 1 & 2 merged into single aligned standoff):
   - Stage 1: Transit directly to Aligned Pre-Grasp Standoff (z+10cm) -> Standstill Settle
   - Stage 2: Straight Vertical Insertion along Z -> Standstill Settle
   - Stage 3: Square Center Clamp & Lift
   - Stage 4: Navigate to Toss Standoff -> Windup -> Clean Toss into Goal Bin
- Speed: 10 fps (1/2 speed slow-motion)."""

import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ["DISPLAY"] = ":0"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import kinder
import kinder.envs.dynamic3d.envs  # noqa: F401  (the MODULE, not the package)
import mujoco
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from kinder_models.dynamic3d.cube_symmetry import upright_grasp_rotations
from kinder_models.dynamic3d.tidybot_pick_controller import TidyBotPickController
from kinder_models.dynamic3d.tossing.parameterized_skills import create_lifted_controllers
from kinder_models.dynamic3d.utils import (
    _ARM_MAX_ACCELERATION,
    _ARM_MAX_VELOCITY,
    _CONTROL_TIMESTEP,
    GRASP_TRANSFORM_TO_OBJECT,
    _compute_per_joint_profile,
)
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import inverse_kinematics

kinder.register_all_environments()

WINDUP_CONF_DEG = (0, 50, 180, -110, 0, -100, 90)
FULL_TOSS_CONF_DEG = (0, 20, 180, -35, 0, 25, 90)


def unwrap_joints(*, target, ref):
    target = np.array(target, dtype=np.float64)
    ref = np.array(ref, dtype=np.float64)
    diff = (target - ref + np.pi) % (2 * np.pi) - np.pi
    return ref + diff


class CanonicalTwoStagePickCubeController(TidyBotPickController):
    STANDOFF = (0.55, 0.0)
    VERTICAL_OFFSET = 0.10

    def reset(self, x, params=None):  # noqa: PLR0917  (overrides TidyBotPickController.reset)
        cube = self.objects[1]
        rotations = upright_grasp_rotations((
            x.get(cube, "qx"),
            x.get(cube, "qy"),
            x.get(cube, "qz"),
            x.get(cube, "qw"),
        ))
        for rotation in rotations:
            upright = x.copy()
            for f, v in zip(("qx", "qy", "qz", "qw"), rotation, strict=False):
                upright.set(cube, f, v)
            try:
                super().reset(upright, np.array(self.STANDOFF))
                self._setup_two_stages(plan_x=upright, rotation=rotation)
                return
            except Exception:
                continue
        raise RuntimeError("Failed to plan canonical two-stage pick")

    def _setup_two_stages(self, *, plan_x, rotation):
        target_object = self.objects[1]
        cube_pos = (
            plan_x.get(target_object, "x"),
            plan_x.get(target_object, "y"),
            plan_x.get(target_object, "z"),
        )
        cube_pose = Pose(cube_pos, rotation)

        # 1. Grasp Pose (Centered z=0)
        grasp_pose = multiply_poses(
            cube_pose, Pose((-0.005, 0.0, 0.0), GRASP_TRANSFORM_TO_OBJECT.orientation)
        )

        # 2. Unified Pre-Grasp Pose (Standoff +10cm above, FULLY YAW-ALIGNED)
        pre_grasp_pose = Pose(
            (
                grasp_pose.position[0],
                grasp_pose.position[1],
                grasp_pose.position[2] + self.VERTICAL_OFFSET,
            ),
            grasp_pose.orientation,
        )

        q_home = np.array(self.home_joints[:7])

        # Solve pre_grasp from q_home
        self._pybullet_sim.robot.set_joints(list(q_home) + [0.0] * 6)
        q_pre_raw = inverse_kinematics(self._pybullet_sim.robot, pre_grasp_pose, set_joints=True)
        q_pre = unwrap_joints(target=q_pre_raw[:7], ref=q_home)

        # Solve grasp warm-started from q_pre_raw
        self._pybullet_sim.robot.set_joints(list(q_pre_raw))
        q_grasp_raw = inverse_kinematics(self._pybullet_sim.robot, grasp_pose, set_joints=False)
        q_grasp = unwrap_joints(target=q_grasp_raw[:7], ref=q_pre)

        # Stage 1: Transit directly to aligned pre-grasp
        s1_traj, s1_dir = _compute_per_joint_profile(
            q_home, q_pre, _ARM_MAX_VELOCITY, _ARM_MAX_ACCELERATION
        )

        # Stage 2: Pure vertical descent
        v_desc = np.full(7, 0.25)
        a_desc = np.full(7, 0.8)
        s2_traj, s2_dir = _compute_per_joint_profile(q_pre, q_grasp, v_desc, a_desc)

        # Retract
        self._retract_trajectory, self._retract_traj_dir = _compute_per_joint_profile(
            q_grasp, q_home, _ARM_MAX_VELOCITY, _ARM_MAX_ACCELERATION
        )
        self._retract_start_joints = q_grasp.copy()
        self._retract_step_idx = 0

        self._stages = [
            ("Stage 1: Transit Directly to Aligned Standoff (z+10cm)", s1_traj, s1_dir, q_pre, 15),
            ("Stage 2: Straight Vertical Insertion", s2_traj, s2_dir, q_grasp, 10),
        ]
        self._stage_idx = 0
        self._stage_step_idx = 0
        self._settling = False
        self._settle_count = 0
        self._current_stage_start = q_home.copy()

    @property
    def current_stage_name(self) -> str:
        if not self._navigated:
            return "Base Navigation to Standoff"
        if self._stage_idx < len(self._stages):
            name, _, _, _, _ = self._stages[self._stage_idx]
            if self._settling:
                return f"⏸️ STOP & SETTLE: {name}"
            return name
        if not self._closed_gripper:
            return "Stage 3: Closing Gripper"
        return "Stage 4: Vertical Lift & Retract"

    def step(self):
        if not self._navigated:
            return super().step()

        if self._stage_idx < len(self._stages):
            name, traj, traj_dir, target_conf, settle_limit = self._stages[self._stage_idx]

            if not self._settling:
                idx = min(self._stage_step_idx, len(traj) - 1)
                s = float(traj[idx])
                ds = (traj[idx] - traj[idx - 1]) / _CONTROL_TIMESTEP if idx > 0 else 0.0

                curr = np.array(self._get_current_robot_arm_conf()[:7])
                target = self._current_stage_start + traj_dir * s
                err = (target - curr + np.pi) % (2 * np.pi) - np.pi

                action = np.zeros(11, dtype=np.float32)
                kp, kv = 2.0, 2.0
                action[3:10] = kp * err + traj_dir * (ds * kv)
                action[10] = 0.0

                self._stage_step_idx += 1
                if self._stage_step_idx >= len(traj):
                    self._settling = True
                    self._settle_count = 0
                return action
            else:
                curr = np.array(self._get_current_robot_arm_conf()[:7])
                err = (target_conf - curr + np.pi) % (2 * np.pi) - np.pi
                action = np.zeros(11, dtype=np.float32)
                action[3:10] = 2.5 * err
                action[10] = 0.0
                self._settle_count += 1
                if self._settle_count >= settle_limit:
                    self._settling = False
                    self._stage_idx += 1
                    self._stage_step_idx = 0
                    self._current_stage_start = target_conf.copy()
                    if self._stage_idx >= len(self._stages):
                        self._pre_grasp = True
                return action

        return super().step()


def render_canonical_videos(*, seed=103, output_dir=Path("docs/canonical_two_stage_videos")):
    env = kinder.make("kinder/Tossing3D-o1-v0", render_mode="rgb_array", scene_bg=True)
    obs, _ = env.reset(seed=seed)
    oc = env.unwrapped._object_centric_env
    sim = oc._robot_env.sim
    rc = sim._render_context_offscreen
    wide_cam_id = oc._robot_env.camera_names.index("task_view")

    state = env.observation_space.devectorize(obs)
    cube = state.get_object_from_name("cube_0")
    bin_obj = state.get_object_from_name("bin_0")
    robot = list(state.get_objects(MujocoTidyBotRobotObjectType))[0]

    pick_ctrl = CanonicalTwoStagePickCubeController((robot, cube))
    pick_ctrl.reset(state)

    tossing_skills = create_lifted_controllers(env.action_space)
    move_ctrl = tossing_skills["move_to_target"].ground((robot, bin_obj))
    windup_ctrl = tossing_skills["move_arm_to_conf"].ground((robot,))
    toss_ctrl = tossing_skills["toss"].ground((robot,))

    skills = [
        ("1. Canonical Two-Stage Pick", pick_ctrl, None, 200, []),
        ("2. Move to Toss Standoff", move_ctrl, np.array([1.30, 0.0]), 150, ["cube_0"]),
        ("3. Arm Windup", windup_ctrl, np.deg2rad(WINDUP_CONF_DEG), 100, []),
        ("4. High-Speed Toss", toss_ctrl, np.deg2rad(FULL_TOSS_CONF_DEG), 100, []),
    ]

    dual_frames = []
    canonical_single_frames = []
    total_step = 0
    init_cube_pos = np.array([
        float(state.get(cube, "x")),
        float(state.get(cube, "y")),
        float(state.get(cube, "z")),
    ])

    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_tag = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font_large = font_med = font_small = font_tag = ImageFont.load_default()

    def capture_frame(*, phase_title):
        curr_c = np.array([
            float(state.get(cube, "x")),
            float(state.get(cube, "y")),
            float(state.get(cube, "z")),
        ])
        disp = np.linalg.norm(curr_c[:2] - init_cube_pos[:2]) * 1000

        # 1. Exact Canonical Viewpoint (Untouched env.render() from default task_view)
        rc.render(width=640, height=480, camera_id=wide_cam_id)
        w_raw = rc.read_pixels(width=640, height=480)
        canonical_single_frames.append(w_raw)

        img_w = Image.fromarray(w_raw)
        draw_w = ImageDraw.Draw(img_w)
        draw_w.rectangle([10, 10, 270, 36], fill=(0, 0, 0, 200))
        draw_w.text((16, 14), "📷 CANONICAL VIEWPOINT", fill=(255, 255, 255), font=font_tag)

        # 2. Elevated Close-Up Camera View
        rc.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        rc.cam.lookat[:] = [curr_c[0], curr_c[1], curr_c[2] + 0.02]
        rc.cam.distance = 0.32
        rc.cam.elevation = -12.0
        rc.cam.azimuth = 145.0
        rc.render(width=640, height=480, camera_id=-1)
        c_raw = rc.read_pixels(width=640, height=480)
        img_c = Image.fromarray(c_raw)
        draw_c = ImageDraw.Draw(img_c)
        draw_c.rectangle([10, 10, 240, 36], fill=(0, 0, 0, 200))
        draw_c.text((16, 14), "📷 CLOSE-UP CUBE VIEW", fill=(255, 255, 255), font=font_tag)

        # Displacement badge
        draw_c.rectangle([10, 440, 340, 470], fill=(0, 0, 0, 200))
        col = (50, 255, 80) if disp < 15.0 else (255, 80, 80)
        draw_c.text((16, 446), f"Cube Displacement: {disp:.1f} mm", fill=col, font=font_med)

        # Combine into synchronized two-panel frame
        total_w = 640 * 2
        total_h = 480 + 110
        comb = Image.new("RGB", (total_w, total_h), (14, 17, 23))
        comb.paste(img_c, (0, 65))
        comb.paste(img_w, (640, 65))

        d = ImageDraw.Draw(comb)
        d.rectangle([0, 0, total_w, 65], fill=(16, 20, 26))
        d.text(
            (20, 6),
            "TWO-STAGE PICK CONTROLLER: Aligned Standoff ──> Vertical Insertion [1/2 SPEED]",
            fill=(100, 255, 100),
            font=font_large,
        )
        d.text(
            (20, 28),
            "1. Direct transit to aligned standoff (z+10cm) & STOP -> 2. Straight vertical drop & "
            "STOP -> 3. Center Clamp & Toss",
            fill=(180, 220, 180),
            font=font_small,
        )
        d.text(
            (20, 46),
            f"Active Phase: {phase_title} | Seed: {seed}",
            fill=(80, 210, 255),
            font=font_med,
        )

        d.rectangle([0, total_h - 45, total_w, total_h], fill=(12, 16, 20))
        d.text(
            (20, total_h - 35),
            "Photorealistic MimicLabs Lab2 Shading & Textures | Kinova Gen3 on TidyBot Base | Seed "
            "103",
            fill=(255, 230, 120),
            font=font_small,
        )
        dual_frames.append(np.array(comb))

    for skill_name, ctrl, params, limit, dis_col in skills:
        if params is not None:
            if dis_col:
                ctrl.reset(state, params, disable_collision_objects=dis_col)
            else:
                ctrl.reset(state, params)

        for _s in range(limit):
            action = ctrl.step()
            obs, _, _, _, _ = env.step(action)
            state = env.observation_space.devectorize(obs)
            ctrl.observe(state)

            desc = pick_ctrl.current_stage_name if skill_name.startswith("1.") else skill_name

            capture_frame(phase_title=desc)
            total_step += 1
            if ctrl.terminated():
                break

    # Flight & Goal
    for _ in range(40):
        obs, _, _, _, _ = env.step(np.zeros(11, dtype=np.float32))
        state = env.observation_space.devectorize(obs)
        capture_frame(phase_title="Flight & Landing in Goal Bin")
        total_step += 1

    env.close()

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Synchronized Two-Panel Video
    dual_mp4 = output_dir / f"canonical_two_panel_seed{seed}.mp4"
    dual_gif = output_dir / f"canonical_two_panel_seed{seed}.gif"
    imageio.mimsave(dual_mp4, dual_frames, fps=10)
    imageio.mimsave(dual_gif, dual_frames[::2], fps=5)

    # 2. Save Pure Canonical Single-View Video
    single_mp4 = output_dir / f"canonical_single_view_seed{seed}.mp4"
    single_gif = output_dir / f"canonical_single_view_seed{seed}.gif"
    imageio.mimsave(single_mp4, canonical_single_frames, fps=10)
    imageio.mimsave(single_gif, canonical_single_frames[::2], fps=5)

    print(f"Saved Dual-Panel Canonical Video to {dual_mp4}")
    print(f"Saved Single-View Canonical Video to {single_mp4}")
    return dual_mp4, single_mp4


if __name__ == "__main__":
    render_canonical_videos(seed=103)
