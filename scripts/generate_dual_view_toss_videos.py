"""Generate dual-view (Close-Up Grasp + Wide Scene) videos for Seed 103 toss rollouts at 1/2 speed."""

import os
import sys
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
from pybullet_helpers.geometry import Pose
from kinder_models.dynamic3d.utils import GRASP_TRANSFORM_TO_OBJECT
from kinder_models.dynamic3d.tidybot_pick_controller import TidyBotPickController
from kinder_models.dynamic3d.tossing.parameterized_skills import create_lifted_controllers

kinder.register_all_environments()
os.environ["DISPLAY"] = ":0"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

GRASP_CENTER = Pose(
    (
        GRASP_TRANSFORM_TO_OBJECT.position[0],
        GRASP_TRANSFORM_TO_OBJECT.position[1],
        0.0,
    ),
    GRASP_TRANSFORM_TO_OBJECT.orientation,
)
GRASP_HIGH = GRASP_TRANSFORM_TO_OBJECT

WINDUP_CONF_DEG = (0, 50, 180, -110, 0, -100, 90)
FULL_TOSS_CONF_DEG = (0, 20, 180, -35, 0, 25, 90)

CONFIGS = [
    {
        "id": "1_baseline",
        "title": "[1] BASELINE (Pre-Fix: High Grasp + No Settling)",
        "subtitle": "Edge Grasp -> Pops Out of Fingertips During Backswing (Drops at x=0.25m)",
        "gt": GRASP_HIGH,
        "settle": 0,
        "ff": 0.0,
        "ffd": 0,
        "target_desc": "Target = +10mm High (Top 1/3 of Cube)",
        "outcome": "FAIL: Cube Torques Out of Fingers During Toss Swing (x=0.25m)",
        "color": (240, 70, 70),
    },
    {
        "id": "2_centered_only",
        "title": "[2] CENTERED TARGET ONLY (Center Grasp + No Settling)",
        "subtitle": "Undershot Grasp -> Insecure Hold Slips Mid-Throw (Collides at x=0.81m)",
        "gt": GRASP_CENTER,
        "settle": 0,
        "ff": 0.0,
        "ffd": 0,
        "target_desc": "Target = Geometric Center (z=0)",
        "outcome": "FAIL: Insecure Grip Slips During Forward Throw (x=0.81m)",
        "color": (240, 150, 40),
    },
    {
        "id": "3_settled_only",
        "title": "[3] CONTROLLER SETTLING ONLY (High Target + 20 Settle Steps)",
        "subtitle": "Settled High Grasp -> Survives on Seed 103 (Fails 7 Other Seeds)",
        "gt": GRASP_HIGH,
        "settle": 20,
        "ff": 2.0,
        "ffd": 5,
        "target_desc": "Target = +10mm High (Top 1/3 of Cube)",
        "outcome": "PARTIAL: Holds on Seed 103, but Fails 7 Other Seeds",
        "color": (240, 150, 40),
    },
    {
        "id": "4_both_fixed",
        "title": "[4] BOTH FIXED (Final PR #130 + #113: Center + Settled)",
        "subtitle": "Centered Square Grasp -> Clean Ballistic Arc Directly into Bin (x=2.01m)",
        "gt": GRASP_CENTER,
        "settle": 20,
        "ff": 2.0,
        "ffd": 5,
        "target_desc": "Target = Geometric Center (z=0)",
        "outcome": "SUCCESS: Cube Flung Cleanly into Goal Bin (x=2.01m)",
        "color": (40, 220, 80),
    },
]

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

def annotate_dual_frame(closeup_raw, wide_raw, cfg, step, skill_name, phase_desc, gl_cam, fovy_deg, cube_pos, target_pos, current_pinch_pos, is_grasping):
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
    p_cube = project_point_to_gl_cam(gl_cam, fovy_deg, cube_pos, w_sub, h_sub)
    p_target = project_point_to_gl_cam(gl_cam, fovy_deg, target_pos, w_sub, h_sub)
    p_pinch = project_point_to_gl_cam(gl_cam, fovy_deg, current_pinch_pos, w_sub, h_sub)

    if p_cube is not None and (0 <= p_cube[0] < w_sub) and (0 <= p_cube[1] < h_sub):
        cx, cy = p_cube
        draw_close.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(0, 230, 255), outline=(0, 0, 0))
        draw_close.text((cx + 8, cy - 7), "◄ Center", fill=(0, 240, 255), font=font_tag)

    if p_target is not None and (0 <= p_target[0] < w_sub) and (0 <= p_target[1] < h_sub):
        tx, ty = p_target
        t_color = (255, 220, 0) if cfg["gt"] == GRASP_HIGH else (50, 255, 100)
        draw_close.ellipse([tx - 5, ty - 5, tx + 5, ty + 5], fill=t_color, outline=(0, 0, 0))
        draw_close.line([tx - 10, ty, tx + 10, ty], fill=t_color, width=2)
        draw_close.line([tx, ty - 10, tx, ty + 10], fill=t_color, width=2)
        target_label = "Target (+10mm)" if cfg["gt"] == GRASP_HIGH else "Target (Center)"
        draw_close.text((tx + 12, ty - 7), f"◄ {target_label}", fill=t_color, font=font_tag)

    if p_pinch is not None and (0 <= p_pinch[0] < w_sub) and (0 <= p_pinch[1] < h_sub):
        px, py = p_pinch
        if is_grasping:
            draw_close.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(255, 80, 80), outline=(255, 255, 255))
            draw_close.text((px - 95, py - 7), "Pinch Site ►", fill=(255, 120, 120), font=font_tag)

    # Combine left and right
    combined_img = Image.new("RGB", (w_sub * 2, h_sub + 120), (14, 17, 23))
    combined_img.paste(img_close, (0, 68))
    combined_img.paste(img_wide, (w_sub, 68))

    draw = ImageDraw.Draw(combined_img)
    total_w = w_sub * 2
    total_h = h_sub + 120

    # Header banner
    draw.rectangle([0, 0, total_w, 68], fill=(16, 20, 26))
    draw.text((20, 6), f"{cfg['title']}  [1/2 SPEED SLOW-MOTION]", fill=(255, 255, 255), font=font_large)
    draw.text((20, 30), cfg["subtitle"], fill=(190, 205, 220), font=font_small)
    draw.text((20, 48), f"Current Phase: {skill_name} | Step: {step:03d} | Seed: 103", fill=(80, 210, 255), font=font_med)

    # Bottom banner
    draw.rectangle([0, total_h - 52, total_w, total_h], fill=(12, 16, 20))
    draw.text((20, total_h - 44), f"Grasp Spec: {cfg['target_desc']}  |  {phase_desc}", fill=(255, 230, 120), font=font_med)
    draw.text((20, total_h - 24), cfg["outcome"], fill=cfg["color"], font=font_med)

    return np.array(combined_img)

def run_dual_toss(cfg, seed=103, standoff=1.35, output_dir=Path("docs/toss_videos")):
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

    tossing_skills = create_lifted_controllers(env.action_space)

    class CustomPick(TidyBotPickController):
        GRASP_TRANSFORM = cfg["gt"]
        APPROACH_SETTLE_STEPS = cfg["settle"]
        APPROACH_FEEDFORWARD_GAIN = cfg["ff"]
        APPROACH_FEEDFORWARD_DELAY = cfg["ffd"]

    pick_ctrl = CustomPick((robot, cube))
    pick_ctrl.reset(state, np.array([0.55, 0.0]))

    move_ctrl = tossing_skills["move_to_target"].ground((robot, bin_obj))
    windup_ctrl = tossing_skills["move_arm_to_conf"].ground((robot,))
    toss_ctrl = tossing_skills["toss"].ground((robot,))

    frames = []
    total_step = 0

    cube_init_z = float(state.get(cube, "z"))
    cube_init_x = float(state.get(cube, "x"))
    cube_init_y = float(state.get(cube, "y"))
    cube_world = np.array([cube_init_x, cube_init_y, cube_init_z])
    target_offset_z = 0.01 if cfg["gt"] == GRASP_HIGH else 0.0
    target_world = cube_world + np.array([0.0, 0.0, target_offset_z])

    skills = [
        ("1. Pick Cube", pick_ctrl, None, 180, []),
        ("2. Move to Toss Standoff", move_ctrl, np.array([standoff, 0.0]), 150, ["cube_0"]),
        ("3. Arm Windup", windup_ctrl, np.deg2rad(WINDUP_CONF_DEG), 100, []),
        ("4. High-Speed Toss Swing", toss_ctrl, np.deg2rad(FULL_TOSS_CONF_DEG), 100, []),
    ]

    print(f"\n=================================================")
    print(f"Running Dual-View Toss: {cfg['id']} (Seed {seed})")
    print(f"=================================================")

    # Wide camera id
    wide_cam_id = oc._robot_env.camera_names.index("task_view")

    for skill_name, ctrl, params, limit, dis_col in skills:
        if params is not None:
            if dis_col: ctrl.reset(state, params, disable_collision_objects=dis_col)
            else: ctrl.reset(state, params)

        for s in range(limit):
            action = ctrl.step()
            obs, _, _, _, _ = env.step(action)
            state = env.observation_space.devectorize(obs)
            ctrl.observe(state)
            
            curr_cube_pos = np.array([float(state.get(cube, "x")), float(state.get(cube, "y")), float(state.get(cube, "z"))])

            # Render Wide View (task_view)
            rc.render(width=640, height=480, camera_id=wide_cam_id)
            wide_raw = rc.read_pixels(width=640, height=480)

            # Render Close-up Side View (tracking cube)
            rc.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            rc.cam.lookat[:] = [curr_cube_pos[0], curr_cube_pos[1], curr_cube_pos[2] + 0.015]
            rc.cam.distance = 0.28
            rc.cam.elevation = -4.0
            rc.cam.azimuth = 80.0
            rc.render(width=640, height=480, camera_id=-1)
            closeup_raw = rc.read_pixels(width=640, height=480)

            try:
                pinch_site_id = sim.model.site("pinch_site").id
                current_pinch_pos = np.array(sim.data.mj_data.site_xpos[pinch_site_id])
            except Exception:
                current_pinch_pos = curr_cube_pos.copy()

            gl_cam = rc.scn.camera[0]
            is_closed = getattr(pick_ctrl, "_closed_gripper", True) if skill_name == "1. Pick Cube" else True

            annotated = annotate_dual_frame(
                closeup_raw, wide_raw, cfg, total_step, skill_name, f"Executing {skill_name}",
                gl_cam, fovy, curr_cube_pos, target_world, current_pinch_pos, is_closed
            )
            frames.append(annotated)
            total_step += 1

            if ctrl.terminated():
                print(f"  {skill_name} completed in {s+1} steps")
                break

    # Settle physics 40 steps
    for s in range(40):
        obs, _, _, _, _ = env.step(np.zeros(11, dtype=np.float32))
        state = env.observation_space.devectorize(obs)
        curr_cube_pos = np.array([float(state.get(cube, "x")), float(state.get(cube, "y")), float(state.get(cube, "z"))])

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
            closeup_raw, wide_raw, cfg, total_step, "Flight & Resting", "Evaluating Goal",
            gl_cam, fovy, curr_cube_pos, target_world, curr_cube_pos, True
        )
        frames.append(annotated)
        total_step += 1

    env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = output_dir / f"dual_toss_seed103_{cfg['id']}.mp4"
    gif_path = output_dir / f"dual_toss_seed103_{cfg['id']}.gif"
    
    # 1/2 speed playback (fps = 10 instead of standard 20)
    imageio.mimsave(mp4_path, frames, fps=10)
    imageio.mimsave(gif_path, frames[::2], fps=5)
    print(f"Saved {mp4_path} ({len(frames)} frames @ 10 fps = 1/2 speed)")
    return frames, mp4_path, gif_path

def main():
    output_dir = Path("docs/toss_videos")
    all_runs = []
    
    for cfg in CONFIGS:
        frames, mp4, gif = run_dual_toss(cfg, seed=103, standoff=1.35, output_dir=output_dir)
        all_runs.append((cfg, frames, mp4, gif))

    # 2x2 comparison grid of dual-view runs
    max_len = max(len(r[1]) for r in all_runs)
    grid_frames = []
    for i in range(max_len):
        f1 = all_runs[0][1][min(i, len(all_runs[0][1]) - 1)]
        f2 = all_runs[1][1][min(i, len(all_runs[1][1]) - 1)]
        f3 = all_runs[2][1][min(i, len(all_runs[2][1]) - 1)]
        f4 = all_runs[3][1][min(i, len(all_runs[3][1]) - 1)]
        grid_frame = np.vstack([np.hstack([f1, f2]), np.hstack([f3, f4])])
        img_resized = Image.fromarray(grid_frame).resize((1280, 600), Image.Resampling.BILINEAR)
        grid_frames.append(np.array(img_resized))

    grid_mp4 = output_dir / "dual_toss_seed103_4way_comparison.mp4"
    grid_gif = output_dir / "dual_toss_seed103_4way_comparison.gif"
    imageio.mimsave(grid_mp4, grid_frames, fps=10)
    imageio.mimsave(grid_gif, grid_frames[::2], fps=5)
    print(f"\nSaved 2x2 Dual Comparison to {grid_mp4}")

if __name__ == "__main__":
    main()
