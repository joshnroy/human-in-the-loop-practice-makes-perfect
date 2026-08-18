"""Render official photorealistic two-panel video on Seed 103:
- Right Panel: Original default camera view from env.render() (task_view with scene_bg=True)
- Left Panel: Elevated front-side close-up camera tracking grasp and displacement
- Controller: MultiStagePickCubeController (Stage 1: Transit to z+12cm -> Stage 2: Align
  Wrist Yaw in Air -> Stage 3: Vertical Descent -> Clamp -> Toss)
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


class OfficialMultiStagePickCubeController(TidyBotPickController):
    STANDOFF = (0.55, 0.0)

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
                self._setup_multi_stage_waypoints(plan_x=upright, rotation=rotation)
                return
            except Exception:
                continue
        raise RuntimeError("Failed to plan multi-stage pick")

    def _setup_multi_stage_waypoints(self, *, plan_x, rotation):
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

        # 2. Elevated Pre-Grasp High (Standoff +12cm above, default approach orientation)
        pre_high_pose = Pose(
            (grasp_pose.position[0], grasp_pose.position[1], grasp_pose.position[2] + 0.12),
            GRASP_TRANSFORM_TO_OBJECT.orientation,
        )

        # 3. Elevated Pre-Grasp Aligned (Standoff +12cm above, yaw-aligned with cube)
        pre_aligned_pose = Pose(
            (grasp_pose.position[0], grasp_pose.position[1], grasp_pose.position[2] + 0.12),
            grasp_pose.orientation,
        )

        # Solve IKs
        q_pre_high = inverse_kinematics(self._pybullet_sim.robot, pre_high_pose, set_joints=False)
        q_pre_aligned = inverse_kinematics(
            self._pybullet_sim.robot, pre_aligned_pose, set_joints=False
        )
        q_grasp = inverse_kinematics(self._pybullet_sim.robot, grasp_pose, set_joints=False)

        assert q_pre_high is not None and q_pre_aligned is not None and q_grasp is not None, (
            "IK failed for waypoints"
        )

        # Trajectories:
        curr = np.array(self.home_joints[:7])
        s1_traj, s1_dir = _compute_per_joint_profile(
            curr, np.array(q_pre_high[:7]), _ARM_MAX_VELOCITY, _ARM_MAX_ACCELERATION
        )
        s2_traj, s2_dir = _compute_per_joint_profile(
            np.array(q_pre_high[:7]),
            np.array(q_pre_aligned[:7]),
            _ARM_MAX_VELOCITY,
            _ARM_MAX_ACCELERATION,
        )
        s3_traj, s3_dir = _compute_per_joint_profile(
            np.array(q_pre_aligned[:7]),
            np.array(q_grasp[:7]),
            _ARM_MAX_VELOCITY,
            _ARM_MAX_ACCELERATION,
        )

        self._stages = [
            ("Stage 1: Transit Above Cube (z+12cm)", s1_traj, s1_dir, np.array(q_pre_high[:7])),
            ("Stage 2: Align Wrist Yaw in Air", s2_traj, s2_dir, np.array(q_pre_aligned[:7])),
            ("Stage 3: Vertical Descent to Center", s3_traj, s3_dir, np.array(q_grasp[:7])),
        ]
        self._stage_idx = 0
        self._stage_step_idx = 0
        self._current_stage_start = curr.copy()

    @property
    def current_stage_name(self) -> str:
        if not self._navigated:
            return "Base Navigation to Standoff"
        if self._stage_idx < len(self._stages):
            return self._stages[self._stage_idx][0]
        if not self._closed_gripper:
            return "Stage 4: Closing Gripper"
        return "Stage 5: Vertical Lift & Retract"

    def step(self):
        if not self._navigated:
            return super().step()

        if self._stage_idx < len(self._stages):
            name, traj, traj_dir, target_conf = self._stages[self._stage_idx]
            idx = min(self._stage_step_idx, len(traj) - 1)
            s = float(traj[idx])
            ds = (traj[idx] - traj[idx - 1]) / _CONTROL_TIMESTEP if idx > 0 else 0.0

            curr = np.array(self._get_current_robot_arm_conf()[:7])
            target = self._current_stage_start + traj_dir * s

            action = np.zeros(11, dtype=np.float32)
            kp, kv = 2.0, 2.0
            action[3:10] = kp * (target - curr) + traj_dir * (ds * kv)
            action[10] = 0.0

            self._stage_step_idx += 1
            if self._stage_step_idx >= len(traj):
                self._current_stage_start = target_conf.copy()
                self._stage_idx += 1
                self._stage_step_idx = 0
                if self._stage_idx >= len(self._stages):
                    self._pre_grasp = True
            return action

        return super().step()


def render_official_video(*, seed=103, output_dir=Path("docs/official_two_panel_videos")):
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

    pick_ctrl = OfficialMultiStagePickCubeController((robot, cube))
    pick_ctrl.reset(state)

    tossing_skills = create_lifted_controllers(env.action_space)
    move_ctrl = tossing_skills["move_to_target"].ground((robot, bin_obj))
    windup_ctrl = tossing_skills["move_arm_to_conf"].ground((robot,))
    toss_ctrl = tossing_skills["toss"].ground((robot,))

    skills = [
        ("1. Multi-Stage Pick Cube", pick_ctrl, None, 180, []),
        ("2. Move to Toss Standoff", move_ctrl, np.array([1.30, 0.0]), 150, ["cube_0"]),
        ("3. Arm Windup", windup_ctrl, np.deg2rad(WINDUP_CONF_DEG), 100, []),
        ("4. High-Speed Toss", toss_ctrl, np.deg2rad(FULL_TOSS_CONF_DEG), 100, []),
    ]

    frames = []
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

        # Right Panel: Official Default Camera View (task_view)
        rc.render(width=640, height=480, camera_id=wide_cam_id)
        w_raw = rc.read_pixels(width=640, height=480)
        img_w = Image.fromarray(w_raw)
        draw_w = ImageDraw.Draw(img_w)
        draw_w.rectangle([10, 10, 270, 36], fill=(0, 0, 0, 200))
        draw_w.text((16, 14), "📷 DEFAULT GLOBAL SCENE VIEW", fill=(255, 255, 255), font=font_tag)

        # Left Panel: Elevated Close-Up Camera View
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
        col = (50, 255, 80) if disp < 10.0 else (255, 80, 80)
        draw_c.text((16, 446), f"Cube Displacement: {disp:.1f} mm", fill=col, font=font_med)

        # Combine
        total_w = 640 * 2
        total_h = 480 + 110
        comb = Image.new("RGB", (total_w, total_h), (14, 17, 23))
        comb.paste(img_c, (0, 65))
        comb.paste(img_w, (640, 65))

        d = ImageDraw.Draw(comb)
        d.rectangle([0, 0, total_w, 65], fill=(16, 20, 26))
        d.text(
            (20, 6),
            "MULTI-STAGE WAYPOINT CONTROLLER (Pre-Grasp High ──> Vertical Descent) [1/2 SPEED]",
            fill=(100, 255, 100),
            font=font_large,
        )
        d.text(
            (20, 28),
            "1. Transit above cube (z+12cm) -> 2. Align wrist yaw in air -> 3. Pure vertical "
            "descent -> 4. Clamp & Toss",
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
        frames.append(np.array(comb))

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
    mp4_path = output_dir / f"official_two_panel_seed{seed}.mp4"
    gif_path = output_dir / f"official_two_panel_seed{seed}.gif"

    imageio.mimsave(mp4_path, frames, fps=10)
    imageio.mimsave(gif_path, frames[::2], fps=5)
    print(f"Saved Official Two-Panel Video to {mp4_path} ({len(frames)} frames @ 10 fps)")
    return mp4_path, gif_path


if __name__ == "__main__":
    render_official_video(seed=103)
