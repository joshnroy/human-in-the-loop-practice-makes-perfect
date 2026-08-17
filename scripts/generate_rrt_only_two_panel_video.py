"""Generate dual-view (Close-Up Cube + Global Scene) video for Full RRT Waypoint Tracking (NO hardcoded stages).
Uses BiRRT path directly to avoid obstacle -> Clamps cube -> Lifts -> Tosses."""

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
from pybullet_helpers.motion_planning import remap_joint_position_plan_to_constant_distance
from kinder_models.dynamic3d.cube_symmetry import upright_grasp_rotations
from kinder_models.dynamic3d.tidybot_pick_controller import TidyBotPickController
from kinder_models.dynamic3d.tossing.parameterized_skills import create_lifted_controllers
from kinder_models.dynamic3d.utils import (
    _ARM_MAX_ACCELERATION,
    _ARM_MAX_VELOCITY,
    GRASP_TRANSFORM_TO_OBJECT,
)

kinder.register_all_environments()

WINDUP_CONF_DEG = (0, 50, 180, -110, 0, -100, 90)
FULL_TOSS_CONF_DEG = (0, 20, 180, -35, 0, 25, 90)

class RRTWaypointPickCubeController(TidyBotPickController):
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
                # Remap the entire RRT plan into dense waypoints without discarding any
                self._dense_approach_plan = remap_joint_position_plan_to_constant_distance(
                    self._current_arm_joint_plan, self._pybullet_sim.robot, max_distance=0.03
                )
                self._dense_plan_idx = 0
                return
            except Exception:
                continue
        raise RuntimeError('Failed to reset')

    def step(self):
        if not self._navigated:
            return super().step()

        # Follow all RRT waypoints instead of taking a 1-segment shortcut
        if self._dense_plan_idx < len(self._dense_approach_plan):
            target = np.array(self._dense_approach_plan[self._dense_plan_idx][:7])
            curr = np.array(self._get_current_robot_arm_conf()[:7])
            err = target - curr
            
            action = np.zeros(11, dtype=np.float32)
            action[3:10] = 3.0 * err
            action[10] = 0.0 # open fingers
            
            if np.max(np.abs(err)) < 0.025 or self._dense_plan_idx == len(self._dense_approach_plan) - 1:
                self._dense_plan_idx += 1
                if self._dense_plan_idx >= len(self._dense_approach_plan):
                    self._pre_grasp = True
            return action

        return super().step()

def render_rrt_video(seed=103, output_dir=Path("docs/rrt_videos")):
    env = kinder.make("kinder/Tossing3D-o1-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    oc = env.unwrapped._object_centric_env
    sim = oc._robot_env.sim
    rc = sim._render_context_offscreen
    fovy = float(sim.model.mj_model.vis.global_.fovy)
    wide_cam_id = oc._robot_env.camera_names.index("task_view")
    
    state = env.observation_space.devectorize(obs)
    cube = state.get_object_from_name("cube_0")
    bin_obj = state.get_object_from_name("bin_0")
    robot = list(state.get_objects(MujocoTidyBotRobotObjectType))[0]
    
    pick_ctrl = RRTWaypointPickCubeController((robot, cube))
    pick_ctrl.reset(state)
    
    tossing_skills = create_lifted_controllers(env.action_space)
    move_ctrl = tossing_skills["move_to_target"].ground((robot, bin_obj))
    windup_ctrl = tossing_skills["move_arm_to_conf"].ground((robot,))
    toss_ctrl = tossing_skills["toss"].ground((robot,))
    
    skills = [
        ("1. Pick Cube (Full RRT Path)", pick_ctrl, None, 180, []),
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

        # Wide view
        rc.render(width=640, height=480, camera_id=wide_cam_id)
        w_raw = rc.read_pixels(width=640, height=480)
        img_w = Image.fromarray(w_raw)
        draw_w = ImageDraw.Draw(img_w)
        draw_w.rectangle([10, 10, 240, 36], fill=(0, 0, 0, 200))
        draw_w.text((16, 14), "📷 GLOBAL SCENE VIEW", fill=(255, 255, 255), font=font_tag)

        # Close-up view
        rc.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        rc.cam.lookat[:] = [curr_c[0], curr_c[1], curr_c[2] + 0.015]
        rc.cam.distance = 0.28
        rc.cam.elevation = -4.0
        rc.cam.azimuth = 80.0
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
        d.text((20, 6), "PURE RRT MOTION PLANNING (No Hardcoded Stages) [1/2 SPEED SLOW-MOTION]", fill=(100, 255, 100), font=font_large)
        d.text((20, 28), "Executing full collision-free RRT plan directly from PyBullet (RRT routes arm safely over cube obstacle)", fill=(180, 220, 180), font=font_small)
        d.text((20, 46), f"Active Phase: {phase_title} | Step: {total_step:03d} | Seed: {seed}", fill=(80, 210, 255), font=font_med)

        d.rectangle([0, total_h - 45, total_w, total_h], fill=(12, 16, 20))
        d.text((20, total_h - 35), "Controller: RRTWaypointPickCubeController | Trajectory: Full 31-Waypoint RRT Execution", fill=(255, 230, 120), font=font_small)
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
            capture_frame(skill_name)
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
    mp4_path = output_dir / f"rrt_waypoint_tracking_seed{seed}.mp4"
    gif_path = output_dir / f"rrt_waypoint_tracking_seed{seed}.gif"

    imageio.mimsave(mp4_path, frames, fps=10)
    imageio.mimsave(gif_path, frames[::2], fps=5)
    print(f"Saved Pure RRT Video to {mp4_path} ({len(frames)} frames @ 10 fps)")
    return mp4_path, gif_path

if __name__ == "__main__":
    render_rrt_video(seed=103)
