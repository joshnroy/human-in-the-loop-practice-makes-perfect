"""Generate zoomed-in close-up side profile videos for grasp centering and approach settling."""

import os
import sys
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

# Set rendering environment before importing MuJoCo
os.environ["DISPLAY"] = ":0"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import mujoco
import kinder
import kinder.envs.dynamic3d.envs  # registers dynamic3d envs
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from pybullet_helpers.geometry import Pose
from kinder_models.dynamic3d.utils import GRASP_TRANSFORM_TO_OBJECT
from kinder_models.dynamic3d.tidybot_pick_controller import TidyBotPickController

kinder.register_all_environments()
os.environ["DISPLAY"] = ":0"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Center grasp transform (z=0 instead of z=+10mm)
GRASP_CENTER = Pose(
    (
        GRASP_TRANSFORM_TO_OBJECT.position[0],
        GRASP_TRANSFORM_TO_OBJECT.position[1],
        0.0,
    ),
    GRASP_TRANSFORM_TO_OBJECT.orientation,
)
GRASP_HIGH = GRASP_TRANSFORM_TO_OBJECT

# Test configurations
CONFIGS = [
    {
        "id": "1_baseline",
        "title": "[1] BASELINE (Pre-Fix)",
        "subtitle": "Target: +10mm High  |  Settling: 0 Steps (Arm Droops & Closes Early)",
        "grasp_transform": GRASP_HIGH,
        "settle_steps": 0,
        "ff_gain": 0.0,
        "ff_delay": 0,
        "target_desc": "Target = +10mm Above Center (Top 1/3 of Cube)",
        "expected_outcome": "RESULT: EDGE GRASP (Closing before arrival; cube dangling/slips)",
        "outcome_color": (240, 70, 70),
    },
    {
        "id": "2_centered_only",
        "title": "[2] CENTERED TARGET ONLY",
        "subtitle": "Target: Cube Center  |  Settling: 0 Steps (Arm Droops & Closes Early)",
        "grasp_transform": GRASP_CENTER,
        "settle_steps": 0,
        "ff_gain": 0.0,
        "ff_delay": 0,
        "target_desc": "Target = Geometric Center (z=0)",
        "expected_outcome": "RESULT: UNDERSHOOTS CENTER (Tracking lag leaves arm trailing)",
        "outcome_color": (240, 160, 40),
    },
    {
        "id": "3_settled_only",
        "title": "[3] CONTROLLER SETTLING ONLY",
        "subtitle": "Target: +10mm High  |  Settling: 20 Settle Steps + Feedforward",
        "grasp_transform": GRASP_HIGH,
        "settle_steps": 20,
        "ff_gain": 2.0,
        "ff_delay": 5,
        "target_desc": "Target = +10mm Above Center (Top 1/3 of Cube)",
        "expected_outcome": "RESULT: ACCURATE ARRIVAL, but Target is Near Top Edge",
        "outcome_color": (240, 160, 40),
    },
    {
        "id": "4_both_fixed",
        "title": "[4] BOTH FIXED (Final PR #130 + #113)",
        "subtitle": "Target: Cube Center  |  Settling: 20 Settle Steps + Feedforward",
        "grasp_transform": GRASP_CENTER,
        "settle_steps": 20,
        "ff_gain": 2.0,
        "ff_delay": 5,
        "target_desc": "Target = Geometric Center (z=0)",
        "expected_outcome": "RESULT: SOLID CENTER GRASP (Pads square on cube faces, stable lift)",
        "outcome_color": (40, 220, 80),
    },
]

def create_controller(cfg, robot, cube):
    class CustomPickController(TidyBotPickController):
        GRASP_TRANSFORM = cfg["grasp_transform"]
        APPROACH_SETTLE_STEPS = cfg["settle_steps"]
        APPROACH_FEEDFORWARD_GAIN = cfg["ff_gain"]
        APPROACH_FEEDFORWARD_DELAY = cfg["ff_delay"]

    controller = CustomPickController((robot, cube))
    return controller

def project_point_to_gl_cam(gl_cam, fovy_deg, world_point, width, height):
    """Project 3D world coordinate to 2D pixel coordinate for the active free camera."""
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

def annotate_frame(raw_frame, cfg, step, phase, gl_cam, fovy_deg, cube_pos, target_pos, current_pinch_pos, width, height, is_grasping):
    img = Image.fromarray(raw_frame)
    draw = ImageDraw.Draw(img)
    
    # Load fonts
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_tag = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font_large = font_med = font_small = font_tag = ImageFont.load_default()

    # Draw header overlay banner
    draw.rectangle([0, 0, width, 72], fill=(16, 20, 26, 240))
    draw.text((15, 6), cfg["title"], fill=(255, 255, 255), font=font_large)
    draw.text((15, 30), cfg["subtitle"], fill=(190, 205, 220), font=font_small)
    draw.text((15, 48), f"Phase: {phase}  |  Step: {step:03d}", fill=(80, 210, 255), font=font_med)

    # 3D projected markers
    p_cube = project_point_to_gl_cam(gl_cam, fovy_deg, cube_pos, width, height)
    p_target = project_point_to_gl_cam(gl_cam, fovy_deg, target_pos, width, height)
    p_pinch = project_point_to_gl_cam(gl_cam, fovy_deg, current_pinch_pos, width, height)

    if p_cube is not None and (0 <= p_cube[0] < width) and (0 <= p_cube[1] < height):
        cx, cy = p_cube
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(0, 230, 255), outline=(0, 0, 0))
        draw.text((cx + 10, cy - 7), "◄ Cube Center (z=0)", fill=(0, 240, 255), font=font_tag)

    if p_target is not None and (0 <= p_target[0] < width) and (0 <= p_target[1] < height):
        tx, ty = p_target
        t_color = (255, 220, 0) if cfg["grasp_transform"] == GRASP_HIGH else (50, 255, 100)
        draw.ellipse([tx - 5, ty - 5, tx + 5, ty + 5], fill=t_color, outline=(0, 0, 0))
        draw.line([tx - 12, ty, tx + 12, ty], fill=t_color, width=2)
        draw.line([tx, ty - 12, tx, ty + 12], fill=t_color, width=2)
        
        target_label = "Target (+10mm high)" if cfg["grasp_transform"] == GRASP_HIGH else "Target (Center)"
        draw.text((tx + 15, ty - 7), f"◄ {target_label}", fill=t_color, font=font_tag)

    if p_pinch is not None and (0 <= p_pinch[0] < width) and (0 <= p_pinch[1] < height):
        px, py = p_pinch
        if is_grasping:
            draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(255, 80, 80), outline=(255, 255, 255))
            draw.text((px - 95, py - 7), "Pinch Site ►", fill=(255, 120, 120), font=font_tag)

    # Draw bottom outcome banner
    draw.rectangle([0, height - 56, width, height], fill=(12, 16, 20, 245))
    draw.text((15, height - 48), f"Grasp Spec: {cfg['target_desc']}", fill=(255, 230, 120), font=font_med)
    draw.text((15, height - 26), cfg["expected_outcome"], fill=cfg["outcome_color"], font=font_med)

    return np.array(img)

def run_rollout(cfg, seed=125, output_dir=Path("docs/grasp_videos")):
    env = kinder.make("kinder/Tossing3D-o1-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    oc = env.unwrapped._object_centric_env
    sim = oc._robot_env.sim

    state = env.observation_space.devectorize(obs)
    cube = state.get_object_from_name("cube_0")
    robot = list(state.get_objects(MujocoTidyBotRobotObjectType))[0]

    controller = create_controller(cfg, robot, cube)
    # Standoff parameter: (distance, rot)
    controller.reset(state, np.array([0.55, 0.0]))

    width = 800
    height = 600
    frames = []

    cube_init_z = float(state.get(cube, "z"))
    cube_init_x = float(state.get(cube, "x"))
    cube_init_y = float(state.get(cube, "y"))
    cube_world = np.array([cube_init_x, cube_init_y, cube_init_z])

    target_offset_z = 0.01 if cfg["grasp_transform"] == GRASP_HIGH else 0.0
    target_world = cube_world + np.array([0.0, 0.0, target_offset_z])

    print(f"\n--- Running {cfg['id']} (Zoomed-in Side Profile) ---")
    
    # Configure close-up free camera
    rc = sim._render_context_offscreen
    rc.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    rc.cam.lookat[:] = [cube_world[0], cube_world[1], cube_world[2] + 0.015]
    rc.cam.distance = 0.28
    rc.cam.elevation = -4.0
    rc.cam.azimuth = 80.0
    fovy = float(sim.model.mj_model.vis.global_.fovy)

    # Run pick controller
    for step in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        state = env.observation_space.devectorize(obs)
        controller.observe(state)
        
        # Only record once robot is navigated near the cube
        if not controller._navigated:
            continue

        if not controller._pre_grasp and not controller._closed_gripper:
            overrun = controller._approach_step_idx - len(controller._approach_trajectory)
            if overrun > 0:
                phase = f"2b. Approach Settling (Step +{overrun}/{cfg['settle_steps']})"
            else:
                phase = "2a. Arm Approach Trajectory"
        elif not controller._closed_gripper:
            phase = "3. Closing Gripper Fingers"
        else:
            phase = "4. Retracting / Lifting Cube"

        # Update camera lookat to track cube smoothly
        current_cube_pos = np.array([float(state.get(cube, "x")), float(state.get(cube, "y")), float(state.get(cube, "z"))])
        rc.cam.lookat[:] = [current_cube_pos[0], current_cube_pos[1], current_cube_pos[2] + 0.015]

        # Render zoomed-in frame
        rc.render(width=width, height=height, camera_id=-1)
        raw_frame = rc.read_pixels(width=width, height=height)

        # Read current pinch site from mujoco data
        try:
            pinch_site_id = sim.model.site("pinch_site").id
            current_pinch_pos = np.array(sim.data.mj_data.site_xpos[pinch_site_id])
        except Exception:
            current_pinch_pos = current_cube_pos.copy()

        gl_cam = rc.scn.camera[0]
        annotated = annotate_frame(
            raw_frame, cfg, step, phase, gl_cam, fovy,
            current_cube_pos, target_world, current_pinch_pos, width, height, controller._closed_gripper
        )
        frames.append(annotated)

        if controller.terminated():
            print(f"  {cfg['id']}: terminated at step {step}")
            break

    env.close()

    # Save clips
    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = output_dir / f"grasp_{cfg['id']}.mp4"
    gif_path = output_dir / f"grasp_{cfg['id']}.gif"
    
    imageio.mimsave(mp4_path, frames, fps=20)
    imageio.mimsave(gif_path, frames[::2], fps=10)
    print(f"  wrote {mp4_path} and {gif_path} ({len(frames)} frames)")
    return frames, mp4_path, gif_path

def main():
    output_dir = Path("docs/grasp_videos")
    all_runs = []
    
    for cfg in CONFIGS:
        frames, mp4, gif = run_rollout(cfg, seed=125, output_dir=output_dir)
        all_runs.append((cfg, frames, mp4, gif))

    # Build 2x2 grid comparison video
    max_len = max(len(r[1]) for r in all_runs)
    grid_frames = []
    
    for i in range(max_len):
        f1 = all_runs[0][1][min(i, len(all_runs[0][1]) - 1)]
        f2 = all_runs[1][1][min(i, len(all_runs[1][1]) - 1)]
        f3 = all_runs[2][1][min(i, len(all_runs[2][1]) - 1)]
        f4 = all_runs[3][1][min(i, len(all_runs[3][1]) - 1)]
        
        top_row = np.hstack([f1, f2])
        bot_row = np.hstack([f3, f4])
        grid_frame = np.vstack([top_row, bot_row])
        # Downscale 2x2 grid slightly so it plays smoothly (1600x1200 -> 1280x960)
        img_resized = Image.fromarray(grid_frame).resize((1280, 960), Image.Resampling.BILINEAR)
        grid_frames.append(np.array(img_resized))

    grid_mp4 = output_dir / "grasp_4way_factorial_comparison.mp4"
    grid_gif = output_dir / "grasp_4way_factorial_comparison.gif"
    imageio.mimsave(grid_mp4, grid_frames, fps=20)
    imageio.mimsave(grid_gif, grid_frames[::2], fps=10)
    print(f"\n=======================================================")
    print(f"Saved 2x2 Factorial Comparison to {grid_mp4} and {grid_gif}")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
