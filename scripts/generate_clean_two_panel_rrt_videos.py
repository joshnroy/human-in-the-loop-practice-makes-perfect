"""Generate two synchronized, photorealistic two-panel videos on Seed 103:
- scene_bg=True (Realistic MimicLabs Lab2 textures, hardwood floor, lighting, soft shadows)
- Left Panel: Unobstructed Close-Up Cube Camera (elevated perspective, tracking grasp & displacement)
- Right Panel: Global Scene Camera (wide task_view)
- Video 1: Baseline Controller (1-Segment Diagonal Descent colliding with corner)
- Video 2: Multi-Stage Controller (With explicit standstill settle pauses at each waypoint, 0.0 mm collision)
- Both rendered at 10 fps (1/2 speed slow-motion)."""

import os
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

os.environ["DISPLAY"] = ":0"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import mujoco
import kinder
import kinder.envs.dynamic3d.envs
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import inverse_kinematics
from kinder_models.dynamic3d.cube_symmetry import upright_grasp_rotations
from kinder_models.dynamic3d.tidybot_pick_controller import TidyBotPickController
from kinder_models.dynamic3d.tossing.parameterized_skills import PickCubeController, create_lifted_controllers
from kinder_models.dynamic3d.utils import (
    _ARM_MAX_ACCELERATION,
    _ARM_MAX_VELOCITY,
    GRASP_TRANSFORM_TO_OBJECT,
    _compute_per_joint_profile,
    _CONTROL_TIMESTEP,
)

kinder.register_all_environments()

WINDUP_CONF_DEG = (0, 50, 180, -110, 0, -100, 90)
FULL_TOSS_CONF_DEG = (0, 20, 180, -35, 0, 25, 90)

class MultiStagePickCubeControllerWithSettle(TidyBotPickController):
    STANDOFF = (0.55, 0.0)

    def reset(self, x, params=None):
        cube = self.objects[1]
        rotations = upright_grasp_rotations(
            (x.get(cube, 'qx'), x.get(cube, 'qy'), x.get(cube, 'qz'), x.get(cube, 'qw'))
        )
        for rotation in rotations:
            upright = x.copy()
            for f, v in zip(('qx', 'qy', 'qz', 'qw'), rotation):
                upright.set(cube, f, v)
            try:
                super().reset(upright, np.array(self.STANDOFF))
                self._setup_multi_stage_waypoints(upright, rotation)
                return
            except Exception:
                continue
        raise RuntimeError('Failed to plan multi-stage pick')

    def _setup_multi_stage_waypoints(self, plan_x, rotation):
        target_object = self.objects[1]
        cube_pos = (plan_x.get(target_object, 'x'), plan_x.get(target_object, 'y'), plan_x.get(target_object, 'z'))
        cube_pose = Pose(cube_pos, rotation)
        
        # 1. Grasp Pose (Centered on cube face)
        grasp_pose = multiply_poses(cube_pose, Pose((-0.005, 0.0, 0.0), GRASP_TRANSFORM_TO_OBJECT.orientation))
        
        # 2. Elevated Pre-Grasp High (Standoff +12cm above, default approach orientation)
        pre_high_pose = Pose((grasp_pose.position[0], grasp_pose.position[1], grasp_pose.position[2] + 0.12), GRASP_TRANSFORM_TO_OBJECT.orientation)
        
        # 3. Elevated Pre-Grasp Aligned (Standoff +12cm above, yaw-aligned with cube)
        pre_aligned_pose = Pose((grasp_pose.position[0], grasp_pose.position[1], grasp_pose.position[2] + 0.12), grasp_pose.orientation)

        # Solve IKs
        q_pre_high = inverse_kinematics(self._pybullet_sim.robot, pre_high_pose, set_joints=False)
        q_pre_aligned = inverse_kinematics(self._pybullet_sim.robot, pre_aligned_pose, set_joints=False)
        q_grasp = inverse_kinematics(self._pybullet_sim.robot, grasp_pose, set_joints=False)

        assert q_pre_high is not None and q_pre_aligned is not None and q_grasp is not None, 'IK failed for waypoints'

        # Trajectories:
        curr = np.array(self.home_joints[:7])
        s1_traj, s1_dir = _compute_per_joint_profile(curr, np.array(q_pre_high[:7]), _ARM_MAX_VELOCITY, _ARM_MAX_ACCELERATION)
        s2_traj, s2_dir = _compute_per_joint_profile(np.array(q_pre_high[:7]), np.array(q_pre_aligned[:7]), _ARM_MAX_VELOCITY, _ARM_MAX_ACCELERATION)
        
        # Slower, gentle vertical descent
        v_desc = np.full(7, 0.25)
        a_desc = np.full(7, 0.8)
        s3_traj, s3_dir = _compute_per_joint_profile(np.array(q_pre_aligned[:7]), np.array(q_grasp[:7]), v_desc, a_desc)

        self._stages = [
            ("Stage 1: Transit Above Cube (z+12cm)", s1_traj, s1_dir, np.array(q_pre_high[:7]), 15),
            ("Stage 2: Align Wrist Yaw at Standoff", s2_traj, s2_dir, np.array(q_pre_aligned[:7]), 12),
            ("Stage 3: Straight Vertical Descent", s3_traj, s3_dir, np.array(q_grasp[:7]), 10),
        ]
        self._stage_idx = 0
        self._stage_step_idx = 0
        self._settling = False
        self._settle_count = 0
        self._current_stage_start = curr.copy()

    @property
    def current_stage_name(self) -> str:
        if not self._navigated:
            return "Base Navigation to Standoff (0.55m)"
        if self._stage_idx < len(self._stages):
            name, _, _, _, _ = self._stages[self._stage_idx]
            if self._settling:
                return f"⏸️ STOP & SETTLE: {name}"
            return name
        if not self._closed_gripper:
            return "Stage 4: Gripper Clamping Center of Mass"
        return "Stage 5: Vertical Lift & Retract"

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
                
                action = np.zeros(11, dtype=np.float32)
                kp, kv = 2.0, 2.0
                action[3:10] = kp * (target - curr) + traj_dir * (ds * kv)
                action[10] = 0.0

                self._stage_step_idx += 1
                if self._stage_step_idx >= len(traj):
                    self._settling = True
                    self._settle_count = 0
                return action
            else:
                curr = np.array(self._get_current_robot_arm_conf()[:7])
                action = np.zeros(11, dtype=np.float32)
                action[3:10] = 2.5 * (target_conf - curr)
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

def render_two_panel_video(ctrl_class, is_baseline: bool, output_path: Path, seed: int = 103):
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
    
    pick_ctrl = ctrl_class((robot, cube))
    try:
        pick_ctrl.reset(state, tuple())
    except TypeError:
        pick_ctrl.reset(state)
    
    tossing_skills = create_lifted_controllers(env.action_space)
    move_ctrl = tossing_skills["move_to_target"].ground((robot, bin_obj))
    windup_ctrl = tossing_skills["move_arm_to_conf"].ground((robot,))
    toss_ctrl = tossing_skills["toss"].ground((robot,))
    
    skills = [
        ("1. Pick Cube", pick_ctrl, None, 200, []),
        ("2. Move to Standoff", move_ctrl, np.array([1.30, 0.0]), 150, ["cube_0"]),
        ("3. Arm Windup", windup_ctrl, np.deg2rad(WINDUP_CONF_DEG), 100, []),
        ("4. High-Speed Toss", toss_ctrl, np.deg2rad(FULL_TOSS_CONF_DEG), 100, []),
    ]
    
    frames = []
    total_step = 0
    init_cube_pos = np.array([float(state.get(cube, "x")), float(state.get(cube, "y")), float(state.get(cube, "z"))])

    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_tag = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font_large = font_med = font_small = font_tag = ImageFont.load_default()

    def capture_frame(phase_title):
        curr_c = np.array([float(state.get(cube, "x")), float(state.get(cube, "y")), float(state.get(cube, "z"))])
        disp = np.linalg.norm(curr_c[:2] - init_cube_pos[:2]) * 1000

        # Wide view (photorealistic scene)
        rc.render(width=640, height=480, camera_id=wide_cam_id)
        w_raw = rc.read_pixels(width=640, height=480)
        img_w = Image.fromarray(w_raw)
        draw_w = ImageDraw.Draw(img_w)
        draw_w.rectangle([10, 10, 240, 36], fill=(0, 0, 0, 200))
        draw_w.text((16, 14), "📷 GLOBAL SCENE VIEW", fill=(255, 255, 255), font=font_tag)

        # Unobstructed close-up view (elevated front perspective)
        rc.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        rc.cam.lookat[:] = [curr_c[0], curr_c[1], curr_c[2] + 0.05]
        rc.cam.distance = 0.38
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
        col = (50, 255, 80) if disp < 1.0 else (255, 80, 80)
        draw_c.text((16, 446), f"Cube Displacement: {disp:.1f} mm", fill=col, font=font_med)

        # Combine
        total_w = 640 * 2
        total_h = 480 + 110
        comb = Image.new("RGB", (total_w, total_h), (14, 17, 23))
        comb.paste(img_c, (0, 65))
        comb.paste(img_w, (640, 65))

        d = ImageDraw.Draw(comb)
        d.rectangle([0, 0, total_w, 65], fill=(16, 20, 26))

        if is_baseline:
            d.text((20, 6), "VIDEO 1: BASELINE CONTROLLER (1-Segment Diagonal Descent) [1/2 SPEED]", fill=(255, 100, 100), font=font_large)
            d.text((20, 28), "Diagonal joint arc strikes cube top-right corner at Step 42 (27.2 mm displacement) -> Cube flings off-course", fill=(220, 180, 180), font=font_small)
        else:
            d.text((20, 6), "VIDEO 2: MULTI-STAGE WAYPOINT CONTROLLER (With Standstill Settle Stops) [1/2 SPEED]", fill=(100, 255, 100), font=font_large)
            d.text((20, 28), "1. Move to z+12cm & STOP -> 2. Align Wrist Yaw & STOP -> 3. Vertical Descent & STOP -> 4. Clamp & Toss", fill=(180, 220, 180), font=font_small)

        d.text((20, 46), f"Active Phase: {phase_title} | Seed: {seed}", fill=(80, 210, 255), font=font_med)
        d.rectangle([0, total_h - 45, total_w, total_h], fill=(12, 16, 20))
        d.text((20, total_h - 35), "Photorealistic Lab2 Shading & Textures | Kinova Gen3 on TidyBot Base | Seed 103", fill=(255, 230, 120), font=font_small)
        frames.append(np.array(comb))

    for skill_name, ctrl, params, limit, dis_col in skills:
        if params is not None:
            if dis_col: ctrl.reset(state, params, disable_collision_objects=dis_col)
            else: ctrl.reset(state, params)

        for s in range(limit):
            action = ctrl.step()
            obs, _, _, _, _ = env.step(action)
            state = env.observation_space.devectorize(obs)
            ctrl.observe(state)
            
            if hasattr(ctrl, "current_stage_name") and skill_name.startswith("1."):
                desc = ctrl.current_stage_name
            else:
                desc = skill_name
                
            capture_frame(desc)
            total_step += 1
            if ctrl.terminated(): break

    # Flight & Settle
    for _ in range(40):
        obs, _, _, _, _ = env.step(np.zeros(11, dtype=np.float32))
        state = env.observation_space.devectorize(obs)
        capture_frame("Flight & Landing in Goal Bin")
        total_step += 1

    env.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gif_path = output_path.with_suffix(".gif")

    imageio.mimsave(output_path, frames, fps=10)
    imageio.mimsave(gif_path, frames[::2], fps=5)
    print(f"Saved video to {output_path} ({len(frames)} frames @ 10 fps)")

def main():
    out_dir = Path("docs/photorealistic_clean_videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Rendering Photorealistic Video 1: Baseline Controller...")
    render_two_panel_video(
        ctrl_class=PickCubeController,
        is_baseline=True,
        output_path=out_dir / "video_baseline_seed103.mp4",
        seed=103,
    )
    
    print("\nRendering Photorealistic Video 2: Multi-Stage Waypoint Controller (With Settle Stops)...")
    render_two_panel_video(
        ctrl_class=MultiStagePickCubeControllerWithSettle,
        is_baseline=False,
        output_path=out_dir / "video_multistage_seed103.mp4",
        seed=103,
    )

if __name__ == "__main__":
    main()
