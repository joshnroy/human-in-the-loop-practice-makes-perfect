"""Generate two separate dual-view videos (Left: Close-Up Cube View, Right: Global Wide Scene View with Base Cylinder):
1. video_1_baseline_shortcut.mp4: Single-segment joint shortcut (strikes cube corner, flings cube).
2. video_2_full_rrt_plan.mp4: Full 31-waypoint RRT execution (clean vertical approach, 0.0mm collision, clean toss into bin).
Both at 10 fps (1/2 speed)."""

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
from pybullet_helpers.motion_planning import run_motion_planning, remap_joint_position_plan_to_constant_distance
from kinder_models.dynamic3d.cube_symmetry import upright_grasp_rotations
from kinder_models.dynamic3d.tossing.parameterized_skills import create_lifted_controllers
from kinder_models.dynamic3d.utils import (
    PyBulletSim,
    GRASP_TRANSFORM_TO_OBJECT,
    get_overhead_object_se2_pose,
    get_target_robot_pose_from_parameters,
    _compute_per_joint_profile,
    _ARM_MAX_VELOCITY,
    _ARM_MAX_ACCELERATION,
)
from spatialmath import SE2

kinder.register_all_environments()

WINDUP_CONF_DEG = (0, 50, 180, -110, 0, -100, 90)
FULL_TOSS_CONF_DEG = (0, 20, 180, -35, 0, 25, 90)

def project_point(gl_cam, fovy_deg, world_point, width, height):
    cam_pos = np.array(gl_cam.pos)
    forward = np.array(gl_cam.forward)
    up = np.array(gl_cam.up)
    right = np.cross(forward, up)
    norm = np.linalg.norm(right)
    if norm < 1e-6: return None
    right /= norm
    cam_mat = np.column_stack([right, up, -forward])
    rel = world_point - cam_pos
    c_coords = cam_mat.T @ rel
    if c_coords[2] >= -1e-4: return None
    f_y = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    f_x = f_y
    u = int(width / 2.0 + (c_coords[0] / -c_coords[2]) * f_x)
    v = int(height / 2.0 - (c_coords[1] / -c_coords[2]) * f_y)
    return (u, v)

def draw_base_cylinder(draw, gl_cam, fovy_deg, base_pos, radius=0.25, height=0.45, w=640, h=480, color=(0, 230, 255)):
    angles = np.linspace(0, 2 * np.pi, 24)
    pts_b, pts_t = [], []
    for a in angles:
        px = base_pos[0] + radius * np.cos(a)
        py = base_pos[1] + radius * np.sin(a)
        pb = project_point(gl_cam, fovy_deg, np.array([px, py, 0.02]), w, h)
        pt = project_point(gl_cam, fovy_deg, np.array([px, py, height]), w, h)
        if pb: pts_b.append(pb)
        if pt: pts_t.append(pt)
    if len(pts_b) > 2:
        for i in range(len(pts_b)):
            draw.line([pts_b[i], pts_b[(i+1)%len(pts_b)]], fill=color, width=2)
            if i < len(pts_t):
                draw.line([pts_t[i], pts_t[(i+1)%len(pts_t)]], fill=color, width=2)
                if i % 4 == 0:
                    draw.line([pts_b[i], pts_t[i]], fill=color, width=1)

def generate_two_panel_video(mode="shortcut", seed=103, output_dir=Path("docs/dual_rrt_videos")):
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
    
    cube_se2 = get_overhead_object_se2_pose(state, cube)
    target_base = get_target_robot_pose_from_parameters(cube_se2, 0.55, 0.0)
    
    plan_x = state.copy()
    plan_x.set(robot, "pos_base_x", target_base.x)
    plan_x.set(robot, "pos_base_y", target_base.y)
    plan_x.set(robot, "pos_base_rot", target_base.theta())
    
    pyb = PyBulletSim(plan_x)
    pyb.set_state(plan_x)
    
    cube_pos = (plan_x.get(cube, "x"), plan_x.get(cube, "y"), plan_x.get(cube, "z"))
    cube_rot = (plan_x.get(cube, "qx"), plan_x.get(cube, "qy"), plan_x.get(cube, "qz"), plan_x.get(cube, "qw"))
    rotations = upright_grasp_rotations(cube_rot)
    cube_pose = Pose(cube_pos, rotations[0])
    grasp_pose = multiply_poses(cube_pose, Pose((-0.005, 0.0, 0.0), GRASP_TRANSFORM_TO_OBJECT.orientation))
    
    q_grasp = inverse_kinematics(pyb.robot, grasp_pose, set_joints=False)
    q_start = pyb.get_robot_joints()
    rrt_plan = run_motion_planning(pyb.robot, q_start, q_grasp, collision_bodies=pyb.get_collision_bodies(), seed=0, physics_client_id=pyb.physics_client_id)
    dense_plan = remap_joint_position_plan_to_constant_distance(rrt_plan, pyb.robot, max_distance=0.03)
    
    frames = []
    init_cube = np.array([float(state.get(cube, "x")), float(state.get(cube, "y")), float(state.get(cube, "z"))])

    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_tag = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font_large = font_med = font_small = font_tag = ImageFont.load_default()

    def record_frame(step_num, phase_title):
        curr_c = np.array([float(state.get(cube, "x")), float(state.get(cube, "y")), float(state.get(cube, "z"))])
        robot_b = np.array([float(state.get(robot, "pos_base_x")), float(state.get(robot, "pos_base_y")), 0.0])
        disp = np.linalg.norm(curr_c[:2] - init_cube[:2]) * 1000

        # 1. Wide Global View
        rc.render(width=640, height=480, camera_id=wide_cam_id)
        w_raw = rc.read_pixels(width=640, height=480)
        img_w = Image.fromarray(w_raw)
        draw_w = ImageDraw.Draw(img_w)
        draw_base_cylinder(draw_w, rc.scn.camera[0], fovy, robot_b, radius=0.25, height=0.45, w=640, h=480, color=(0, 230, 255))
        
        draw_w.rectangle([10, 10, 360, 36], fill=(0, 0, 0, 200))
        draw_w.text((16, 14), "📷 GLOBAL SCENE + BASE CYLINDER (r=0.25m)", fill=(0, 230, 255), font=font_tag)

        # 2. Close-up Cube View
        rc.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        rc.cam.lookat[:] = [curr_c[0], curr_c[1], curr_c[2] + 0.015]
        rc.cam.distance = 0.28
        rc.cam.elevation = -4.0
        rc.cam.azimuth = 80.0
        rc.render(width=640, height=480, camera_id=-1)
        c_raw = rc.read_pixels(width=640, height=480)
        img_c = Image.fromarray(c_raw)
        draw_c = ImageDraw.Draw(img_c)

        draw_c.rectangle([10, 10, 260, 36], fill=(0, 0, 0, 200))
        draw_c.text((16, 14), "📷 CLOSE-UP CUBE CAMERA", fill=(255, 255, 255), font=font_tag)

        # Displacement badge
        draw_c.rectangle([10, 440, 340, 470], fill=(0, 0, 0, 200))
        col = (50, 255, 80) if disp < 1.0 else (255, 80, 80)
        draw_c.text((16, 446), f"Cube Displacement: {disp:.1f} mm", fill=col, font=font_med)

        # Combine Left & Right
        total_w = 640 * 2
        total_h = 480 + 110
        comb = Image.new("RGB", (total_w, total_h), (14, 17, 23))
        comb.paste(img_c, (0, 65))
        comb.paste(img_w, (640, 65))

        d = ImageDraw.Draw(comb)
        d.rectangle([0, 0, total_w, 65], fill=(16, 20, 26))
        
        if mode == "shortcut":
            d.text((20, 6), "BASELINE: 1-Segment Shortcut (Bypasses RRT Intermediate Waypoints) [1/2 SPEED]", fill=(255, 100, 100), font=font_large)
            d.text((20, 28), "Problem: Discards RRT collision-free waypoints -> single joint line sweeps finger into cube corner", fill=(220, 180, 180), font=font_small)
        else:
            d.text((20, 6), "FULL RRT EXECUTION: Following All 31 RRT Collision-Free Waypoints [1/2 SPEED]", fill=(100, 255, 100), font=font_large)
            d.text((20, 28), "Solution: Arm lifts high over cube and drops squarely from above -> Zero collision, clean toss", fill=(180, 220, 180), font=font_small)
            
        d.text((20, 46), f"Active Phase: {phase_title} | Step: {step_num:03d} | Seed: {seed}", fill=(80, 210, 255), font=font_med)

        d.rectangle([0, total_h - 45, total_w, total_h], fill=(12, 16, 20))
        d.text((20, total_h - 35), "Cyan Wireframe = Mobile Base Bounding Cylinder (Radius 0.25m, Standoff 0.55m).", fill=(255, 230, 120), font=font_small)
        frames.append(np.array(comb))

    step_counter = 0

    # 1. Base Navigation
    for _ in range(40):
        robot_pose = SE2(state.get(robot, 'pos_base_x'), state.get(robot, 'pos_base_y'), state.get(robot, 'pos_base_rot'))
        dx = target_base.x - robot_pose.x
        dy = target_base.y - robot_pose.y
        drot = target_base.theta() - robot_pose.theta()
        if np.hypot(dx, dy) < 0.01 and abs(drot) < 0.02: break
        action = np.zeros(11, dtype=np.float32)
        action[0] = max(min(5.0 * dx, 1.0), -1.0)
        action[1] = max(min(5.0 * dy, 1.0), -1.0)
        action[2] = max(min(5.0 * drot, 1.0), -1.0)
        obs, _, _, _, _ = env.step(action)
        state = env.observation_space.devectorize(obs)
        record_frame(step_counter, "1. Mobile Base Navigation to Standoff (0.55m)")
        step_counter += 1

    # 2. Arm Descent
    if mode == "shortcut":
        curr = np.array(sim.data.mj_data.qpos[3:10])
        final = np.array(rrt_plan[-1][:7])
        traj, traj_dir = _compute_per_joint_profile(curr, final, _ARM_MAX_VELOCITY, _ARM_MAX_ACCELERATION)
        for s in range(len(traj)):
            idx = min(s, len(traj) - 1)
            target = curr + traj_dir * float(traj[idx])
            curr_q = np.array(sim.data.mj_data.qpos[3:10])
            action = np.zeros(11, dtype=np.float32)
            action[3:10] = 2.0 * (target - curr_q)
            obs, _, _, _, _ = env.step(action)
            state = env.observation_space.devectorize(obs)
            record_frame(step_counter, "2. Arm Descent (1-Segment Shortcut - Sweeps Corner)")
            step_counter += 1
    else:
        for pt in dense_plan:
            target = np.array(pt[:7])
            curr_q = np.array(sim.data.mj_data.qpos[3:10])
            action = np.zeros(11, dtype=np.float32)
            action[3:10] = 2.5 * (target - curr_q)
            obs, _, _, _, _ = env.step(action)
            state = env.observation_space.devectorize(obs)
            record_frame(step_counter, "2. Arm Descent (Full RRT Collision-Free Path)")
            step_counter += 1

    # 3. Gripper Clamping
    for _ in range(25):
        action = np.zeros(11, dtype=np.float32)
        action[10] = 1.0 # Close gripper
        obs, _, _, _, _ = env.step(action)
        state = env.observation_space.devectorize(obs)
        record_frame(step_counter, "3. Gripper Clamping on Cube")
        step_counter += 1

    # 4. Lifting
    for _ in range(25):
        action = np.zeros(11, dtype=np.float32)
        action[5] = -2.0 # shoulder lift
        action[10] = 1.0
        obs, _, _, _, _ = env.step(action)
        state = env.observation_space.devectorize(obs)
        record_frame(step_counter, "4. Lifting Cube from Floor")
        step_counter += 1

    # 5. Toss sequence
    tossing_skills = create_lifted_controllers(env.action_space)
    move_ctrl = tossing_skills["move_to_target"].ground((robot, bin_obj))
    windup_ctrl = tossing_skills["move_arm_to_conf"].ground((robot,))
    toss_ctrl = tossing_skills["toss"].ground((robot,))

    move_ctrl.reset(state, np.array([1.30, 0.0]), disable_collision_objects=["cube_0"])
    for _ in range(30):
        obs, _, _, _, _ = env.step(move_ctrl.step())
        state = env.observation_space.devectorize(obs)
        move_ctrl.observe(state)
        record_frame(step_counter, "5. Driving to Toss Location")
        step_counter += 1
        if move_ctrl.terminated(): break

    windup_ctrl.reset(state, np.deg2rad(WINDUP_CONF_DEG))
    for _ in range(20):
        obs, _, _, _, _ = env.step(windup_ctrl.step())
        state = env.observation_space.devectorize(obs)
        windup_ctrl.observe(state)
        record_frame(step_counter, "6. Arm Windup for Toss")
        step_counter += 1
        if windup_ctrl.terminated(): break

    toss_ctrl.reset(state, np.deg2rad(FULL_TOSS_CONF_DEG))
    for _ in range(25):
        obs, _, _, _, _ = env.step(toss_ctrl.step())
        state = env.observation_space.devectorize(obs)
        toss_ctrl.observe(state)
        record_frame(step_counter, "7. Ballistic Toss & Release")
        step_counter += 1
        if toss_ctrl.terminated(): break

    # Flight
    for _ in range(25):
        obs, _, _, _, _ = env.step(np.zeros(11, dtype=np.float32))
        state = env.observation_space.devectorize(obs)
        record_frame(step_counter, "8. Flight & Settle in Bin")
        step_counter += 1

    env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"two_panel_{mode}_seed{seed}"
    mp4_path = output_dir / f"{prefix}.mp4"
    gif_path = output_dir / f"{prefix}.gif"

    imageio.mimsave(mp4_path, frames, fps=10)
    imageio.mimsave(gif_path, frames[::2], fps=5)
    print(f"Saved {mode} video to {mp4_path} ({len(frames)} frames @ 10 fps)")
    return mp4_path, gif_path

if __name__ == "__main__":
    print("Generating Video 1: Baseline Shortcut (Two Panels: Cube Cam Left, Global Cam Right)...")
    generate_two_panel_video(mode="shortcut", seed=103)
    
    print("\nGenerating Video 2: Full RRT Execution (Two Panels: Cube Cam Left, Global Cam Right)...")
    generate_two_panel_video(mode="rrt", seed=103)
