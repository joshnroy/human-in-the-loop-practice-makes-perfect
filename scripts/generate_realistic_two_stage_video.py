"""Generate photorealistic Two-Stage Pick & Toss video on Seed 103:
- Realistic textures, hardwood floor, lighting, and soft shadows (scene_bg=True)
- Left: Unobstructed Close-Up Cube Camera (side-front view, tracking grasp & displacement)
- Right: Global Scene Camera (wide task_view of robot, base, arm, and toss into bin)
- Waypoint 1: Standoff directly above cube (z+10cm) with yaw aligned -> FULL STOP & SETTLE
- Waypoint 2: Straight vertical elevator descent -> FULL STOP & SETTLE
- Clamps center of mass -> Lifts -> Drives to toss position -> Tosses into goal bin
- Rendered at 10 fps (1/2 speed slow-motion)."""

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
from kinder_models.dynamic3d.tossing.parameterized_skills import create_lifted_controllers
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

def unwrap_joints(target, ref):
    """Unwrap continuous target angles to be within [-pi, pi] of reference."""
    target = np.array(target, dtype=np.float64)
    ref = np.array(ref, dtype=np.float64)
    diff = (target - ref + np.pi) % (2 * np.pi) - np.pi
    return ref + diff

class PerfectTwoStagePickCubeController(TidyBotPickController):
    STANDOFF = (0.55, 0.0)
    VERTICAL_OFFSET = 0.10 # +10cm strictly above cube
    DESCENT_VELOCITY = 0.20 # Slow, controlled descent (11 deg/s)
    DESCENT_ACCELERATION = 0.6

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
                self._setup_two_stage_trajectories(upright, rotation)
                return
            except Exception:
                continue
        raise RuntimeError('Failed to plan two-stage pick')

    def _setup_two_stage_trajectories(self, plan_x, rotation):
        target_object = self.objects[1]
        cube_pos = (plan_x.get(target_object, 'x'), plan_x.get(target_object, 'y'), plan_x.get(target_object, 'z'))
        cube_pose = Pose(cube_pos, rotation)
        
        # Center grasp pose
        grasp_pose = multiply_poses(cube_pose, GRASP_TRANSFORM_TO_OBJECT)
        
        # Pre-grasp pose: EXACT same (X, Y, Yaw), strictly +10cm along World Z!
        pre_grasp_pose = Pose(
            (grasp_pose.position[0], grasp_pose.position[1], grasp_pose.position[2] + self.VERTICAL_OFFSET),
            grasp_pose.orientation
        )

        q_home = np.array(self.home_joints[:7])
        self._pybullet_sim.robot.set_joints(list(q_home) + [0.0]*6)
        q_pre_raw = inverse_kinematics(self._pybullet_sim.robot, pre_grasp_pose, set_joints=True)
        
        self._pybullet_sim.robot.set_joints(list(q_pre_raw))
        q_grasp_raw = inverse_kinematics(self._pybullet_sim.robot, grasp_pose, set_joints=False)

        q_pre = unwrap_joints(q_pre_raw[:7], q_home)
        q_grasp = unwrap_joints(q_grasp_raw[:7], q_pre)

        # Stage 1: Transit to offset
        self._s1_traj, self._s1_dir = _compute_per_joint_profile(
            q_home, q_pre, _ARM_MAX_VELOCITY, _ARM_MAX_ACCELERATION
        )
        self._s1_start = q_home.copy()

        # Stage 2: Controlled vertical descent
        v_desc = np.full(7, self.DESCENT_VELOCITY)
        a_desc = np.full(7, self.DESCENT_ACCELERATION)
        self._s2_traj, self._s2_dir = _compute_per_joint_profile(
            q_pre, q_grasp, v_desc, a_desc
        )
        self._s2_start = q_pre.copy()

        # Retract
        self._retract_trajectory, self._retract_traj_dir = _compute_per_joint_profile(
            q_grasp, q_home, _ARM_MAX_VELOCITY, _ARM_MAX_ACCELERATION
        )
        self._retract_start_joints = q_grasp.copy()
        self._retract_step_idx = 0

        self._stage = 1 # 1: Transit, 2: Pre-Grasp Settle (STOP), 3: Vertical Descent, 4: Grasp Settle (STOP), 5: Retract
        self._traj_step_idx = 0
        self._settle_count = 0

    @property
    def current_stage_name(self) -> str:
        if not self._navigated:
            return "Base Navigation to Standoff (0.55m)"
        if self._stage == 1:
            return "Stage 1: Transit to Standoff Above Cube (z+10cm)"
        if self._stage == 2:
            return "⏸️ STOP & SETTLE directly above cube"
        if self._stage == 3:
            return "Stage 2: Smooth Straight Vertical Descent"
        if self._stage == 4:
            return "⏸️ STOP & SETTLE around cube center"
        if not self._closed_gripper:
            return "Stage 3: Gripper Clamping Center of Mass"
        return "Stage 4: Vertical Lift & Retract"

    def step(self):
        if not self._navigated:
            return super().step()

        # Stage 1: Transit to Pre-Grasp
        if self._stage == 1:
            idx = min(self._traj_step_idx, len(self._s1_traj) - 1)
            s = float(self._s1_traj[idx])
            ds = (self._s1_traj[idx] - self._s1_traj[idx - 1]) / _CONTROL_TIMESTEP if idx > 0 else 0.0
            
            curr = np.array(self._get_current_robot_arm_conf()[:7])
            target = self._s1_start + self._s1_dir * s
            
            action = np.zeros(11, dtype=np.float32)
            action[3:10] = 2.5 * (target - curr) + self._s1_dir * (ds * 2.0)
            action[10] = 0.0
            
            self._traj_step_idx += 1
            if self._traj_step_idx >= len(self._s1_traj):
                self._stage = 2
                self._traj_step_idx = 0
                self._settle_count = 0
            return action

        # Stage 2: FULL STOP at Pre-Grasp (Zero velocity pause)
        if self._stage == 2:
            curr = np.array(self._get_current_robot_arm_conf()[:7])
            target = self._s2_start
            
            action = np.zeros(11, dtype=np.float32)
            action[3:10] = 2.5 * (target - curr)
            action[10] = 0.0
            
            self._settle_count += 1
            if self._settle_count >= 15: # 15 steps (0.3s standstill)
                self._stage = 3
                self._traj_step_idx = 0
                self._settle_count = 0
            return action

        # Stage 3: Smooth Straight Vertical Descent
        if self._stage == 3:
            idx = min(self._traj_step_idx, len(self._s2_traj) - 1)
            s = float(self._s2_traj[idx])
            ds = (self._s2_traj[idx] - self._s2_traj[idx - 1]) / _CONTROL_TIMESTEP if idx > 0 else 0.0
            
            curr = np.array(self._get_current_robot_arm_conf()[:7])
            target = self._s2_start + self._s2_dir * s
            
            action = np.zeros(11, dtype=np.float32)
            action[3:10] = 2.5 * (target - curr) + self._s2_dir * (ds * 1.5)
            action[10] = 0.0
            
            self._traj_step_idx += 1
            if self._traj_step_idx >= len(self._s2_traj):
                self._stage = 4
                self._traj_step_idx = 0
                self._settle_count = 0
            return action

        # Stage 4: FULL STOP at Grasp Pose (Zero velocity pause before closing)
        if self._stage == 4:
            curr = np.array(self._get_current_robot_arm_conf()[:7])
            target = self._s2_start + self._s2_dir * float(self._s2_traj[-1])
            
            action = np.zeros(11, dtype=np.float32)
            action[3:10] = 2.5 * (target - curr)
            action[10] = 0.0
            
            self._settle_count += 1
            if self._settle_count >= 10: # 10 steps (0.2s standstill)
                self._pre_grasp = True
                self._stage = 5
            return action

        # Stage 5: Close Gripper & Retract Lift
        return super().step()

def render_photorealistic_video(seed=103, output_dir=Path("docs/two_stage_videos")):
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
    
    pick_ctrl = PerfectTwoStagePickCubeController((robot, cube))
    pick_ctrl.reset(state)
    
    tossing_skills = create_lifted_controllers(env.action_space)
    move_ctrl = tossing_skills["move_to_target"].ground((robot, bin_obj))
    windup_ctrl = tossing_skills["move_arm_to_conf"].ground((robot,))
    toss_ctrl = tossing_skills["toss"].ground((robot,))
    
    skills = [
        ("1. Two-Stage Pick Cube", pick_ctrl, None, 180, []),
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

        # Unobstructed close-up view (side-front perspective)
        rc.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        rc.cam.lookat[:] = [curr_c[0], curr_c[1], curr_c[2] + 0.04]
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
        col = (50, 255, 80) if disp < 5.0 else (255, 80, 80)
        draw_c.text((16, 446), f"Cube Displacement: {disp:.1f} mm", fill=col, font=font_med)

        # Combine
        total_w = 640 * 2
        total_h = 480 + 110
        comb = Image.new("RGB", (total_w, total_h), (14, 17, 23))
        comb.paste(img_c, (0, 65))
        comb.paste(img_w, (640, 65))

        d = ImageDraw.Draw(comb)
        d.rectangle([0, 0, total_w, 65], fill=(16, 20, 26))
        d.text((20, 6), "TWO-STAGE CONTROLLER: (Standoff Settle ──> Vertical Descent Settle ──> Toss) [1/2 SPEED]", fill=(100, 255, 100), font=font_large)
        d.text((20, 28), "1. Move to z+10cm & STOP -> 2. Straight vertical descent & STOP -> 3. Center Clamp & Toss", fill=(180, 220, 180), font=font_small)
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
            
            if skill_name.startswith("1."):
                desc = pick_ctrl.current_stage_name
            else:
                desc = skill_name
                
            capture_frame(desc)
            total_step += 1
            if ctrl.terminated(): break

    # Flight
    for _ in range(40):
        obs, _, _, _, _ = env.step(np.zeros(11, dtype=np.float32))
        state = env.observation_space.devectorize(obs)
        capture_frame("Flight & Landing in Goal Bin")
        total_step += 1

    env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = output_dir / f"photorealistic_two_stage_seed{seed}.mp4"
    gif_path = output_dir / f"photorealistic_two_stage_seed{seed}.gif"

    imageio.mimsave(mp4_path, frames, fps=10)
    imageio.mimsave(gif_path, frames[::2], fps=5)
    print(f"Saved Photorealistic Two-Stage Video to {mp4_path} ({len(frames)} frames @ 10 fps)")
    return mp4_path, gif_path

if __name__ == "__main__":
    render_photorealistic_video(seed=103)
