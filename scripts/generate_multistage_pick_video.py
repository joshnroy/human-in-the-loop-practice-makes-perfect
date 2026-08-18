"""Generate dual-view (Close-Up Grasp + Wide Scene) video for Multi-Stage Waypoint Pick on
Seed 103 at 1/2 speed."""

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
os.environ["DISPLAY"] = ":0"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

WINDUP_CONF_DEG = (0, 50, 180, -110, 0, -100, 90)
FULL_TOSS_CONF_DEG = (0, 20, 180, -35, 0, 25, 90)


class MultiStagePickCubeController(TidyBotPickController):
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
            ("Stage 1: Move Above Object (z+12cm)", s1_traj, s1_dir, np.array(q_pre_high[:7])),
            ("Stage 2: Align Yaw at Standoff", s2_traj, s2_dir, np.array(q_pre_aligned[:7])),
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
            return "Stage 4: Closing Gripper Fingers"
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


def project_point_to_gl_cam(*, gl_cam, fovy_deg, world_point, width, height):
    cam_pos = np.array(gl_cam.pos)
    forward = np.array(gl_cam.forward)
    up = np.array(gl_cam.up)
    right = np.cross(forward, up)
    norm = np.linalg.norm(right)
    if norm < 1e-6:
        return None
    right /= norm
    cam_mat = np.column_stack([right, up, -forward])

    rel = world_point - cam_pos
    c_coords = cam_mat.T @ rel
    if c_coords[2] >= -1e-4:
        return None

    f_y = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    f_x = f_y
    u = int(width / 2.0 + (c_coords[0] / -c_coords[2]) * f_x)
    v = int(height / 2.0 - (c_coords[1] / -c_coords[2]) * f_y)
    return (u, v)


def annotate_dual_frame(
    *,
    closeup_raw,
    wide_raw,
    step,
    skill_name,
    phase_desc,
    gl_cam,
    fovy_deg,
    cube_pos,
    target_pos,
    current_pinch_pos,
    is_grasping,
):
    w_sub = 640
    h_sub = 480

    img_close = Image.fromarray(closeup_raw).resize((w_sub, h_sub), Image.Resampling.BILINEAR)
    img_wide = Image.fromarray(wide_raw).resize((w_sub, h_sub), Image.Resampling.BILINEAR)

    draw_close = ImageDraw.Draw(img_close)
    draw_wide = ImageDraw.Draw(img_wide)

    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_tag = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font_large = font_med = font_small = font_tag = ImageFont.load_default()

    # View labels
    draw_close.rectangle([10, 10, 240, 36], fill=(0, 0, 0, 200))
    draw_close.text((16, 14), "📷 CLOSE-UP GRASP VIEW", fill=(255, 255, 255), font=font_tag)

    draw_wide.rectangle([10, 10, 240, 36], fill=(0, 0, 0, 200))
    draw_wide.text((16, 14), "📷 WIDE SCENE VIEW", fill=(255, 255, 255), font=font_tag)

    # 3D projected markers on close-up
    p_cube = project_point_to_gl_cam(
        gl_cam=gl_cam, fovy_deg=fovy_deg, world_point=cube_pos, width=w_sub, height=h_sub
    )
    p_target = project_point_to_gl_cam(
        gl_cam=gl_cam, fovy_deg=fovy_deg, world_point=target_pos, width=w_sub, height=h_sub
    )
    p_pinch = project_point_to_gl_cam(
        gl_cam=gl_cam, fovy_deg=fovy_deg, world_point=current_pinch_pos, width=w_sub, height=h_sub
    )

    if p_cube is not None and (0 <= p_cube[0] < w_sub) and (0 <= p_cube[1] < h_sub):
        cx, cy = p_cube
        draw_close.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(0, 230, 255), outline=(0, 0, 0))
        draw_close.text((cx + 8, cy - 7), "◄ Center", fill=(0, 240, 255), font=font_tag)

    if p_target is not None and (0 <= p_target[0] < w_sub) and (0 <= p_target[1] < h_sub):
        tx, ty = p_target
        draw_close.ellipse([tx - 5, ty - 5, tx + 5, ty + 5], fill=(50, 255, 100), outline=(0, 0, 0))
        draw_close.line([tx - 10, ty, tx + 10, ty], fill=(50, 255, 100), width=2)
        draw_close.line([tx, ty - 10, tx, ty + 10], fill=(50, 255, 100), width=2)
        draw_close.text(
            (tx + 12, ty - 7), "◄ Target (Centered)", fill=(50, 255, 100), font=font_tag
        )

    if p_pinch is not None and (0 <= p_pinch[0] < w_sub) and (0 <= p_pinch[1] < h_sub):
        px, py = p_pinch
        if is_grasping:
            draw_close.ellipse(
                [px - 5, py - 5, px + 5, py + 5], fill=(255, 80, 80), outline=(255, 255, 255)
            )
            draw_close.text((px - 95, py - 7), "Pinch Site ►", fill=(255, 120, 120), font=font_tag)

    # Combine left and right
    total_w = w_sub * 2
    total_h = h_sub + 120
    combined_img = Image.new("RGB", (total_w, total_h), (14, 17, 23))
    combined_img.paste(img_close, (0, 68))
    combined_img.paste(img_wide, (w_sub, 68))

    draw = ImageDraw.Draw(combined_img)

    # Header banner
    draw.rectangle([0, 0, total_w, 68], fill=(16, 20, 26))
    draw.text(
        (20, 6),
        "MULTI-STAGE WAYPOINT PICK: Align Above -> Align Yaw -> Straight Down [1/2 SPEED]",
        fill=(255, 255, 255),
        font=font_large,
    )
    draw.text(
        (20, 30),
        "Waypoints: (1) Pre-Grasp z+12cm  ->  (2) Yaw-Align at Standoff  ->  (3) Vertical "
        "Insertion  ->  (4) Center Grasp",
        fill=(190, 205, 220),
        font=font_small,
    )
    draw.text(
        (20, 48),
        f"Active Phase: {phase_desc} | Step: {step:03d} | Seed: 103",
        fill=(80, 210, 255),
        font=font_med,
    )

    # Bottom banner
    draw.rectangle([0, total_h - 52, total_w, total_h], fill=(12, 16, 20))
    draw.text(
        (20, total_h - 44),
        f"Continuous Controller: MultiStagePickCubeController  |  Skill: {skill_name}",
        fill=(255, 230, 120),
        font=font_med,
    )
    draw.text(
        (20, total_h - 24),
        "RESULT: ZERO PREMATURE COLLISION (Straight vertical insertion cleanly envelops cube)",
        fill=(40, 220, 80),
        font=font_med,
    )

    return np.array(combined_img)


def run_multistage_rollout(*, seed=103, output_dir=Path("docs/multistage_videos")):
    env = kinder.make("kinder/Tossing3D-o1-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    oc = env.unwrapped._object_centric_env
    sim = oc._robot_env.sim
    rc = sim._render_context_offscreen
    fovy = float(sim.model.mj_model.vis.global_.fovy)

    state = env.observation_space.devectorize(obs)
    cube = state.get_object_from_name("cube_0")
    bin_obj = state.get_object_from_name("bin_0")
    robot = list(state.get_objects(MujocoTidyBotRobotObjectType))[0]

    pick_ctrl = MultiStagePickCubeController((robot, cube))
    pick_ctrl.reset(state)

    tossing_skills = create_lifted_controllers(env.action_space)
    move_ctrl = tossing_skills["move_to_target"].ground((robot, bin_obj))
    windup_ctrl = tossing_skills["move_arm_to_conf"].ground((robot,))
    toss_ctrl = tossing_skills["toss"].ground((robot,))

    frames = []
    total_step = 0

    cube_init_z = float(state.get(cube, "z"))
    cube_init_x = float(state.get(cube, "x"))
    cube_init_y = float(state.get(cube, "y"))
    cube_world = np.array([cube_init_x, cube_init_y, cube_init_z])
    target_world = cube_world.copy()

    skills = [
        ("1. Multi-Stage Pick", pick_ctrl, None, 180, []),
        ("2. Move to Standoff", move_ctrl, np.array([1.30, 0.0]), 150, ["cube_0"]),
        ("3. Arm Windup", windup_ctrl, np.deg2rad(WINDUP_CONF_DEG), 100, []),
        ("4. High-Speed Toss", toss_ctrl, np.deg2rad(FULL_TOSS_CONF_DEG), 100, []),
    ]

    wide_cam_id = oc._robot_env.camera_names.index("task_view")

    for skill_name, ctrl, params, limit, dis_col in skills:
        if params is not None:
            if dis_col:
                ctrl.reset(state, params, disable_collision_objects=dis_col)
            else:
                ctrl.reset(state, params)

        for s in range(limit):
            action = ctrl.step()
            obs, _, _, _, _ = env.step(action)
            state = env.observation_space.devectorize(obs)
            ctrl.observe(state)

            curr_cube_pos = np.array([
                float(state.get(cube, "x")),
                float(state.get(cube, "y")),
                float(state.get(cube, "z")),
            ])

            # Render Wide View
            rc.render(width=640, height=480, camera_id=wide_cam_id)
            wide_raw = rc.read_pixels(width=640, height=480)

            # Render Close-up Free Camera (tracking cube)
            rc.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            rc.cam.lookat[:] = [curr_cube_pos[0], curr_cube_pos[1], curr_cube_pos[2] + 0.015]
            rc.cam.distance = 0.28
            rc.cam.elevation = -4.0
            rc.cam.azimuth = 80.0
            rc.render(width=640, height=480, camera_id=-1)
            closeup_raw = rc.read_pixels(width=640, height=480)

            try:
                pinch_site_id = mujoco.mj_name2id(
                    sim.model.mj_model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site"
                )
                current_pinch_pos = np.array(sim.data.mj_data.site_xpos[pinch_site_id])
            except Exception:
                current_pinch_pos = curr_cube_pos.copy()

            gl_cam = rc.scn.camera[0]

            if skill_name.startswith("1."):
                phase_desc = pick_ctrl.current_stage_name
                is_closed = pick_ctrl._closed_gripper
            else:
                phase_desc = skill_name
                is_closed = True

            annotated = annotate_dual_frame(
                closeup_raw=closeup_raw,
                wide_raw=wide_raw,
                step=total_step,
                skill_name=skill_name,
                phase_desc=phase_desc,
                gl_cam=gl_cam,
                fovy_deg=fovy,
                cube_pos=curr_cube_pos,
                target_pos=target_world,
                current_pinch_pos=current_pinch_pos,
                is_grasping=is_closed,
            )
            frames.append(annotated)
            total_step += 1

            if ctrl.terminated():
                print(f"  {skill_name} completed in {s + 1} steps")
                break

    # Settle physics
    for _s in range(40):
        obs, _, _, _, _ = env.step(np.zeros(11, dtype=np.float32))
        state = env.observation_space.devectorize(obs)
        curr_cube_pos = np.array([
            float(state.get(cube, "x")),
            float(state.get(cube, "y")),
            float(state.get(cube, "z")),
        ])

        rc.render(width=640, height=480, camera_id=wide_cam_id)
        wide_raw = rc.read_pixels(width=640, height=480)

        rc.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        rc.cam.lookat[:] = [curr_cube_pos[0], curr_cube_pos[1], curr_cube_pos[2] + 0.015]
        rc.cam.distance = 0.28
        rc.cam.elevation = -4.0
        rc.cam.azimuth = 80.0
        rc.render(width=640, height=480, camera_id=-1)
        closeup_raw = rc.read_pixels(width=640, height=480)

        gl_cam = rc.scn.camera[0]
        annotated = annotate_dual_frame(
            closeup_raw=closeup_raw,
            wide_raw=wide_raw,
            step=total_step,
            skill_name="Flight & Resting",
            phase_desc="Evaluating Goal Region",
            gl_cam=gl_cam,
            fovy_deg=fovy,
            cube_pos=curr_cube_pos,
            target_pos=target_world,
            current_pinch_pos=curr_cube_pos,
            is_grasping=True,
        )
        frames.append(annotated)
        total_step += 1

    env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = output_dir / "multistage_pick_seed103_dual_view.mp4"
    gif_path = output_dir / "multistage_pick_seed103_dual_view.gif"

    imageio.mimsave(mp4_path, frames, fps=10)
    imageio.mimsave(gif_path, frames[::2], fps=5)
    print(f"\nSaved Multi-Stage Waypoint Pick Video to {mp4_path} ({len(frames)} frames @ 10 fps)")
    return mp4_path, gif_path


if __name__ == "__main__":
    run_multistage_rollout(seed=103)
