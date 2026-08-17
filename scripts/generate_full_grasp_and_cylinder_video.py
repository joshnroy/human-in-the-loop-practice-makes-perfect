"""Generate comprehensive dual-view video comparing:
1. Baseline Shortcut (bypasses RRT intermediate waypoints -> strikes cube corner -> flings cube)
2. Full RRT / Multi-Stage Pick (clean vertical descent -> perfect center clamp -> clean toss into bin)
With 3D base cylinder overlay and full skill execution (Drive -> Pick -> Grip -> Lift -> Toss)."""

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

def simulate_full_episode(seed=103, mode="rrt"):
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

    def capture_frame(phase_name):
        curr_c = np.array([float(state.get(cube, "x")), float(state.get(cube, "y")), float(state.get(cube, "z"))])
        robot_b = np.array([float(state.get(robot, "pos_base_x")), float(state.get(robot, "pos_base_y")), 0.0])
        
        # Wide view
        rc.render(width=640, height=480, camera_id=wide_cam_id)
        w_raw = rc.read_pixels(width=640, height=480)
        
        # Close-up view
        rc.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        rc.cam.lookat[:] = [curr_c[0], curr_c[1], curr_c[2] + 0.015]
        rc.cam.distance = 0.28
        rc.cam.elevation = -4.0
        rc.cam.azimuth = 80.0
        rc.render(width=640, height=480, camera_id=-1)
        c_raw = rc.read_pixels(width=640, height=480)
        
        disp = np.linalg.norm(curr_c[:2] - init_cube[:2]) * 1000
        frames.append((c_raw, w_raw, disp, phase_name, rc.scn.camera[0], curr_c, robot_b))

    # Phase 1: Base Navigation
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
        capture_frame("1. Base Navigation to Standoff")

    # Phase 2: Arm Approach
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
            capture_frame("2. Arm Approach (1-Segment Shortcut)")
    else:
        for pt in dense_plan:
            target = np.array(pt[:7])
            curr_q = np.array(sim.data.mj_data.qpos[3:10])
            action = np.zeros(11, dtype=np.float32)
            action[3:10] = 2.5 * (target - curr_q)
            obs, _, _, _, _ = env.step(action)
            state = env.observation_space.devectorize(obs)
            capture_frame("2. Arm Approach (Full RRT Path)")

    # Phase 3: Grip Close
    for _ in range(25):
        action = np.zeros(11, dtype=np.float32)
        action[10] = 1.0 # Close gripper
        obs, _, _, _, _ = env.step(action)
        state = env.observation_space.devectorize(obs)
        capture_frame("3. Gripper Clamping")

    # Phase 4: Lift Arm
    for _ in range(25):
        action = np.zeros(11, dtype=np.float32)
        action[5] = -2.0 # shoulder lift
        action[10] = 1.0
        obs, _, _, _, _ = env.step(action)
        state = env.observation_space.devectorize(obs)
        capture_frame("4. Lifting Cube")

    # Phase 5: Move to Toss & Throw
    tossing_skills = create_lifted_controllers(env.action_space)
    move_ctrl = tossing_skills["move_to_target"].ground((robot, bin_obj))
    windup_ctrl = tossing_skills["move_arm_to_conf"].ground((robot,))
    toss_ctrl = tossing_skills["toss"].ground((robot,))

    move_ctrl.reset(state, np.array([1.30, 0.0]), disable_collision_objects=["cube_0"])
    for _ in range(30):
        obs, _, _, _, _ = env.step(move_ctrl.step())
        state = env.observation_space.devectorize(obs)
        move_ctrl.observe(state)
        capture_frame("5. Driving to Toss Standoff")
        if move_ctrl.terminated(): break

    windup_ctrl.reset(state, np.deg2rad(WINDUP_CONF_DEG))
    for _ in range(20):
        obs, _, _, _, _ = env.step(windup_ctrl.step())
        state = env.observation_space.devectorize(obs)
        windup_ctrl.observe(state)
        capture_frame("6. Arm Windup")
        if windup_ctrl.terminated(): break

    toss_ctrl.reset(state, np.deg2rad(FULL_TOSS_CONF_DEG))
    for _ in range(25):
        obs, _, _, _, _ = env.step(toss_ctrl.step())
        state = env.observation_space.devectorize(obs)
        toss_ctrl.observe(state)
        capture_frame("7. Ballistic Toss & Release")
        if toss_ctrl.terminated(): break

    # Flight & Settle
    for _ in range(25):
        obs, _, _, _, _ = env.step(np.zeros(11, dtype=np.float32))
        state = env.observation_space.devectorize(obs)
        capture_frame("8. Flight into Bin")

    env.close()
    return frames

def render_combined_video(seed=103, output_dir=Path("docs/rrt_analysis")):
    print("Simulating Mode A (Shortcut)...")
    frames_a = simulate_full_episode(seed=seed, mode="shortcut")
    print("Simulating Mode B (Full RRT)...")
    frames_b = simulate_full_episode(seed=seed, mode="rrt")
    
    total_frames = max(len(frames_a), len(frames_b))
    combined = []
    
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font_large = font_med = font_small = ImageFont.load_default()
        
    for i in range(total_frames):
        c_a, w_a, disp_a, ph_a, cam_a, cube_a, base_a = frames_a[min(i, len(frames_a)-1)]
        c_b, w_b, disp_b, ph_b, cam_b, cube_b, base_b = frames_b[min(i, len(frames_b)-1)]
        
        img_ca = Image.fromarray(c_a).resize((480, 360))
        img_cb = Image.fromarray(c_b).resize((480, 360))
        img_wb = Image.fromarray(w_b).resize((480, 360))
        
        # Draw cylinder on wide view
        draw_wb = ImageDraw.Draw(img_wb)
        draw_base_cylinder(draw_wb, cam_b, 45.0, base_b, radius=0.25, height=0.45, w=480, h=360, color=(0, 230, 255))
        draw_wb.text((10, 10), "📷 WIDE SCENE + BASE CYLINDER (r=0.25m)", fill=(0, 230, 255), font=font_small)
        
        draw_ca = ImageDraw.Draw(img_ca)
        draw_ca.rectangle([5, 5, 320, 30], fill=(0, 0, 0, 200))
        draw_ca.text((10, 10), f"❌ SHORTCUT (Disp: {disp_a:.1f}mm)", fill=(255, 80, 80), font=font_med)
        
        draw_cb = ImageDraw.Draw(img_cb)
        draw_cb.rectangle([5, 5, 320, 30], fill=(0, 0, 0, 200))
        draw_cb.text((10, 10), f"✅ FULL RRT (Disp: {disp_b:.1f}mm)", fill=(50, 255, 80), font=font_med)
        
        # Layout: 3 panels (Shortcut Close-Up | Full RRT Close-Up | Wide Scene + Base Cylinder)
        row_img = Image.new("RGB", (480 * 3, 360 + 80), (14, 17, 23))
        row_img.paste(img_ca, (0, 50))
        row_img.paste(img_cb, (480, 50))
        row_img.paste(img_wb, (960, 50))
        
        d = ImageDraw.Draw(row_img)
        d.rectangle([0, 0, 480*3, 50], fill=(16, 20, 26))
        d.text((20, 8), f"FULL PICK & TOSS ROLLOUT: (Left: Discarded Shortcut vs Center: Full RRT Execution) | Step: {i:03d}", fill=(255, 255, 255), font=font_large)
        d.text((20, 28), f"Current Phase: {ph_b} | Seed 103", fill=(80, 210, 255), font=font_small)
        
        d.rectangle([0, 360+50, 480*3, 360+80], fill=(12, 16, 20))
        d.text((20, 360+58), "Cyan Wireframe = Mobile Base Cylinder (Radius 0.25m, Standoff 0.55m). Full RRT routes arm squarely around cube.", fill=(255, 230, 120), font=font_small)
        
        combined.append(np.array(row_img))
        
    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = output_dir / "full_grasp_and_cylinder_analysis_seed103.mp4"
    gif_path = output_dir / "full_grasp_and_cylinder_analysis_seed103.gif"
    
    imageio.mimsave(mp4_path, combined, fps=10)
    imageio.mimsave(gif_path, combined[::2], fps=5)
    print(f"\nSaved Full Grasp & Cylinder Analysis Video to {mp4_path} ({len(combined)} frames @ 10 fps)")
    return mp4_path, gif_path

if __name__ == "__main__":
    render_combined_video(seed=103)
