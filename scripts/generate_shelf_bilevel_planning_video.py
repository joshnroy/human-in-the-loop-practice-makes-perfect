"""Generate dual-view video of Bilevel Planning executing in the Shelf3D environment."""

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
from kinder_bilevel_planning.agent import BilevelPlanningAgent
from kinder_bilevel_planning.env_models.dynamic3d.tidybot3d_shelf3D import (
    create_bilevel_planning_models,
)

kinder.register_all_environments()
os.environ["DISPLAY"] = ":0"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"


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


def annotate_shelf_frame(
    *,
    closeup_raw,
    wide_raw,
    step,
    total_steps,
    current_op,
    current_skill,
    gl_cam,
    fovy_deg,
    obj_pos,
    is_holding,
    goal_achieved,
):
    w_sub = 640
    h_sub = 480

    img_close = Image.fromarray(closeup_raw).resize((w_sub, h_sub), Image.Resampling.BILINEAR)
    img_wide = Image.fromarray(wide_raw).resize((w_sub, h_sub), Image.Resampling.BILINEAR)

    draw_close = ImageDraw.Draw(img_close)
    draw_wide = ImageDraw.Draw(img_wide)

    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 19)
        font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_tag = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font_large = font_med = font_small = font_tag = ImageFont.load_default()

    # View tags
    draw_close.rectangle([10, 10, 240, 36], fill=(0, 0, 0, 200))
    draw_close.text((16, 14), "📷 CLOSE-UP GRASP VIEW", fill=(255, 255, 255), font=font_tag)

    draw_wide.rectangle([10, 10, 240, 36], fill=(0, 0, 0, 200))
    draw_wide.text((16, 14), "📷 WIDE SCENE VIEW (Shelf3D)", fill=(255, 255, 255), font=font_tag)

    # 3D projected marker on object in close-up
    p_obj = project_point_to_gl_cam(
        gl_cam=gl_cam, fovy_deg=fovy_deg, world_point=obj_pos, width=w_sub, height=h_sub
    )
    if p_obj is not None and (0 <= p_obj[0] < w_sub) and (0 <= p_obj[1] < h_sub):
        ox, oy = p_obj
        draw_close.ellipse([ox - 4, oy - 4, ox + 4, oy + 4], fill=(0, 240, 255), outline=(0, 0, 0))
        draw_close.text((ox + 8, oy - 7), "◄ Target (cube1)", fill=(0, 240, 255), font=font_tag)

    # Combine side-by-side with HUD
    total_w = w_sub * 2
    total_h = h_sub + 120
    combined_img = Image.new("RGB", (total_w, total_h), (14, 17, 23))
    combined_img.paste(img_close, (0, 68))
    combined_img.paste(img_wide, (w_sub, 68))

    draw = ImageDraw.Draw(combined_img)

    # Header
    draw.rectangle([0, 0, total_w, 68], fill=(16, 20, 26))
    draw.text(
        (20, 6),
        "BILEVEL PLANNING IN SHELF3D: Pick & Place Execution",
        fill=(255, 255, 255),
        font=font_large,
    )
    draw.text(
        (20, 30),
        "Abstract Plan: [1] PickTargetOperator(robot, cube1)  ->  [2] PlaceTargetOperator(robot, "
        "cube1, cupboard_1)",
        fill=(190, 205, 220),
        font=font_small,
    )
    draw.text(
        (20, 48),
        f"Active Operator: {current_op}  |  Step: {step:03d}/{total_steps:03d}",
        fill=(80, 210, 255),
        font=font_med,
    )

    # Bottom Banner
    status_text = (
        "GOAL ACHIEVED: cube1 placed safely on cupboard shelf!"
        if goal_achieved
        else (
            "Holding cube1 | Carrying to cupboard"
            if is_holding
            else "Approaching & Grasping cube1 from floor"
        )
    )
    status_color = (
        (40, 220, 80) if goal_achieved else ((255, 200, 60) if is_holding else (100, 200, 255))
    )

    draw.rectangle([0, total_h - 52, total_w, total_h], fill=(12, 16, 20))
    draw.text(
        (20, total_h - 44),
        f"Continuous Controller: {current_skill}",
        fill=(255, 230, 120),
        font=font_med,
    )
    draw.text((20, total_h - 24), f"Status: {status_text}", fill=status_color, font=font_med)

    return np.array(combined_img)


def run_shelf_bilevel_rollout(*, seed=42, output_dir=Path("docs/shelf_videos")):
    env = kinder.make("kinder/Shelf3D-o1-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=seed)
    oc = env.unwrapped._object_centric_env
    sim = oc._robot_env.sim
    rc = sim._render_context_offscreen
    fovy = float(sim.model.mj_model.vis.global_.fovy)

    print(f"Creating Bilevel Planning Models for Shelf3D (seed={seed})...")
    models = create_bilevel_planning_models(env.observation_space, env.action_space, num_objects=1)
    agent = BilevelPlanningAgent(models, seed=seed)
    agent.reset(obs, info)

    planned_actions = list(agent._planned_actions)
    total_steps = len(planned_actions)
    print(f"Bilevel Planner found plan with {total_steps} continuous action steps.")

    wide_cam_id = oc._robot_env.camera_names.index("task_view")
    frames = []

    state = env.observation_space.devectorize(obs)
    cube1 = state.get_object_from_name("cube1")

    for step, action in enumerate(planned_actions):
        obs, _, _, _, _ = env.step(action)
        state = env.observation_space.devectorize(obs)

        # Get current target position
        obj_pos = np.array([
            float(state.get(cube1, "x")),
            float(state.get(cube1, "y")),
            float(state.get(cube1, "z")),
        ])

        # Render Wide View
        rc.render(width=640, height=480, camera_id=wide_cam_id)
        wide_raw = rc.read_pixels(width=640, height=480)

        # Render Close-up Free Camera (tracking target object)
        rc.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        rc.cam.lookat[:] = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.03]
        rc.cam.distance = 0.35
        rc.cam.elevation = -12.0
        rc.cam.azimuth = 75.0
        rc.render(width=640, height=480, camera_id=-1)
        close_raw = rc.read_pixels(width=640, height=480)

        # Determine phase
        # Pick is roughly first ~60 steps, Place is remaining
        is_holding = float(obj_pos[2]) > 0.15
        _is_in_cupboard = float(obj_pos[0]) > 0.8 and float(obj_pos[2]) > 0.4

        current_op = (
            "place_target(robot, cube1, cupboard_1)" if is_holding else "pick_shelf(robot, cube1)"
        )
        current_skill = (
            "TidyBotPlaceShelfController" if is_holding else "TidyBotPickShelfController"
        )

        gl_cam = rc.scn.camera[0]
        goal_done = bool(oc._check_goals())

        annotated = annotate_shelf_frame(
            closeup_raw=close_raw,
            wide_raw=wide_raw,
            step=step,
            total_steps=total_steps,
            current_op=current_op,
            current_skill=current_skill,
            gl_cam=gl_cam,
            fovy_deg=fovy,
            obj_pos=obj_pos,
            is_holding=is_holding,
            goal_achieved=goal_done,
        )
        frames.append(annotated)

    # Settle 30 steps to observe placement
    for s in range(30):
        obs, _, _, _, _ = env.step(np.zeros(11, dtype=np.float32))
        state = env.observation_space.devectorize(obs)
        obj_pos = np.array([
            float(state.get(cube1, "x")),
            float(state.get(cube1, "y")),
            float(state.get(cube1, "z")),
        ])

        rc.render(width=640, height=480, camera_id=wide_cam_id)
        wide_raw = rc.read_pixels(width=640, height=480)

        rc.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        rc.cam.lookat[:] = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.03]
        rc.cam.distance = 0.35
        rc.cam.elevation = -12.0
        rc.cam.azimuth = 75.0
        rc.render(width=640, height=480, camera_id=-1)
        close_raw = rc.read_pixels(width=640, height=480)

        gl_cam = rc.scn.camera[0]
        goal_done = bool(oc._check_goals())
        annotated = annotate_shelf_frame(
            closeup_raw=close_raw,
            wide_raw=wide_raw,
            step=total_steps + s,
            total_steps=total_steps,
            current_op="Goal Evaluation",
            current_skill="Resting in Cupboard",
            gl_cam=gl_cam,
            fovy_deg=fovy,
            obj_pos=obj_pos,
            is_holding=True,
            goal_achieved=goal_done,
        )
        frames.append(annotated)

    env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = output_dir / "shelf_bilevel_planning_dual_view.mp4"
    gif_path = output_dir / "shelf_bilevel_planning_dual_view.gif"

    # 1/2 speed playback (10 fps)
    imageio.mimsave(mp4_path, frames, fps=10)
    imageio.mimsave(gif_path, frames[::2], fps=5)
    print(f"Saved Shelf Bilevel Planning Video to {mp4_path} ({len(frames)} frames @ 10 fps)")
    return mp4_path, gif_path


if __name__ == "__main__":
    run_shelf_bilevel_rollout(seed=42)
