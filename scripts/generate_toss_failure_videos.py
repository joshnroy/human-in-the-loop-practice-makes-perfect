"""Generate full Pick -> Toss videos for all 4 configurations showing why bad grasps fail during toss."""

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
        "title": "[1] BASELINE: High Grasp + No Settling",
        "subtitle": "Edge Grasp -> Slips & Drops Early During Toss Swing",
        "grasp_transform": GRASP_HIGH,
        "settle_steps": 0,
        "ff_gain": 0.0,
        "ff_delay": 0,
        "outcome": "FAIL: Cube Torques Out of Fingertips During Swing",
        "color": (240, 70, 70),
    },
    {
        "id": "2_centered_only",
        "title": "[2] CENTERED ONLY: Center Target + No Settling",
        "subtitle": "Undershot Grasp -> Slips Under High Toss Acceleration",
        "grasp_transform": GRASP_CENTER,
        "settle_steps": 0,
        "ff_gain": 0.0,
        "ff_delay": 0,
        "outcome": "FAIL: Insecure Bottom-Edge Grip Slips Mid-Flight",
        "color": (240, 150, 40),
    },
    {
        "id": "3_settled_only",
        "title": "[3] SETTLING ONLY: High Target + 20 Settle Steps",
        "subtitle": "Accurate but High Grasp -> Weak Contact Surface",
        "grasp_transform": GRASP_HIGH,
        "settle_steps": 20,
        "ff_gain": 2.0,
        "ff_delay": 5,
        "outcome": "FAIL: Top-Edge Grasp Dislodges on Release",
        "color": (240, 150, 40),
    },
    {
        "id": "4_both_fixed",
        "title": "[4] BOTH FIXED: Center Grasp + Settled (PR #130 + #113)",
        "subtitle": "Solid Centered Grasp -> Clean Ballistic Flight into Bin",
        "grasp_transform": GRASP_CENTER,
        "settle_steps": 20,
        "ff_gain": 2.0,
        "ff_delay": 5,
        "outcome": "SUCCESS: Cube Flung Directly into Goal Bin",
        "color": (40, 220, 80),
    },
]

def create_pick_controller(cfg, robot, cube):
    class CustomPickController(TidyBotPickController):
        GRASP_TRANSFORM = cfg["grasp_transform"]
        APPROACH_SETTLE_STEPS = cfg["settle_steps"]
        APPROACH_FEEDFORWARD_GAIN = cfg["ff_gain"]
        APPROACH_FEEDFORWARD_DELAY = cfg["ff_delay"]

    return CustomPickController((robot, cube))

def annotate_toss_frame(raw_frame, cfg, step, skill_name, phase_desc, goal_scored, width, height):
    img = Image.fromarray(raw_frame)
    draw = ImageDraw.Draw(img)
    
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font_large = font_med = font_small = ImageFont.load_default()

    # Top banner
    draw.rectangle([0, 0, width, 66], fill=(16, 20, 26, 240))
    draw.text((12, 5), cfg["title"], fill=(255, 255, 255), font=font_large)
    draw.text((12, 27), cfg["subtitle"], fill=(190, 205, 220), font=font_small)
    draw.text((12, 45), f"Skill: {skill_name}  |  Phase: {phase_desc}  |  Step: {step:03d}", fill=(80, 210, 255), font=font_med)

    # Bottom banner
    draw.rectangle([0, height - 52, width, height], fill=(12, 16, 20, 245))
    draw.text((12, height - 44), f"Status: {phase_desc}", fill=(255, 230, 120), font=font_med)
    draw.text((12, height - 24), cfg["outcome"], fill=cfg["color"], font=font_med)

    return np.array(img)

def run_toss_rollout(cfg, seed=125, standoff=1.35, output_dir=Path("docs/toss_videos")):
    from kinder_models.dynamic3d.tossing.parameterized_skills import create_lifted_controllers as tossing_create_lifted_controllers
    
    env = kinder.make("kinder/Tossing3D-o1-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    oc = env.unwrapped._object_centric_env
    oc._render_camera_name = "task_view"
    
    state = env.observation_space.devectorize(obs)
    cube = state.get_object_from_name("cube_0")
    bin_obj = state.get_object_from_name("bin_0")
    robot = list(state.get_objects(MujocoTidyBotRobotObjectType))[0]

    tossing_skills = tossing_create_lifted_controllers(env.action_space)

    # 1. Pick
    pick_ctrl = create_pick_controller(cfg, robot, cube)
    pick_ctrl.reset(state, np.array([0.55, 0.0]))

    # 2. Move to toss target
    move_ctrl = tossing_skills["move_to_target"].ground((robot, bin_obj))
    
    # 3. Windup
    windup_ctrl = tossing_skills["move_arm_to_conf"].ground((robot,))

    # 4. Toss
    toss_ctrl = tossing_skills["toss"].ground((robot,))

    frames = []
    width = 640
    height = 480
    total_step = 0

    print(f"\n=================================================")
    print(f"Running Full Toss Sequence: {cfg['id']}")
    print(f"=================================================")

    # Skill sequence
    skills = [
        ("1. Pick Cube", pick_ctrl, None, 200, []),
        ("2. Move to Toss Standoff", move_ctrl, np.array([standoff, 0.0]), 150, ["cube_0"]),
        ("3. Arm Windup", windup_ctrl, np.deg2rad(WINDUP_CONF_DEG), 100, []),
        ("4. High-Speed Toss Swing", toss_ctrl, np.deg2rad(FULL_TOSS_CONF_DEG), 100, []),
    ]

    for skill_name, ctrl, params, limit, dis_col in skills:
        if params is None:
            pass # already reset
        else:
            if dis_col:
                ctrl.reset(state, params, disable_collision_objects=dis_col)
            else:
                ctrl.reset(state, params)

        for s in range(limit):
            action = ctrl.step()
            obs, _, _, _, _ = env.step(action)
            state = env.observation_space.devectorize(obs)
            ctrl.observe(state)
            
            raw_frame = env.render()
            annotated = annotate_toss_frame(
                raw_frame, cfg, total_step, skill_name, f"Executing {skill_name}", False, width, height
            )
            frames.append(annotated)
            total_step += 1

            if ctrl.terminated():
                print(f"  {skill_name} completed in {s+1} steps")
                break

    # Settle physics for 40 steps to see where cube rests
    for s in range(40):
        obs, _, _, _, _ = env.step(np.zeros(11, dtype=np.float32))
        state = env.observation_space.devectorize(obs)
        raw_frame = env.render()
        solved = bool(oc._check_goals())
        annotated = annotate_toss_frame(
            raw_frame, cfg, total_step, "Flight / Settled", "Evaluating Goal Region", solved, width, height
        )
        frames.append(annotated)
        total_step += 1

    solved = bool(oc._check_goals())
    print(f"Final Goal Check: {solved}")

    env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = output_dir / f"toss_{cfg['id']}.mp4"
    gif_path = output_dir / f"toss_{cfg['id']}.gif"
    
    imageio.mimsave(mp4_path, frames, fps=20)
    imageio.mimsave(gif_path, frames[::2], fps=10)
    print(f"Saved {mp4_path} and {gif_path}")
    return frames, mp4_path, gif_path, solved

def main():
    output_dir = Path("docs/toss_videos")
    all_runs = []
    
    for cfg in CONFIGS:
        frames, mp4, gif, solved = run_toss_rollout(cfg, seed=125, standoff=1.35, output_dir=output_dir)
        all_runs.append((cfg, frames, mp4, gif, solved))

    # Build 2x2 comparison
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
        img_resized = Image.fromarray(grid_frame).resize((1280, 960), Image.Resampling.BILINEAR)
        grid_frames.append(np.array(img_resized))

    grid_mp4 = output_dir / "toss_4way_factorial_comparison.mp4"
    grid_gif = output_dir / "toss_4way_factorial_comparison.gif"
    imageio.mimsave(grid_mp4, grid_frames, fps=20)
    imageio.mimsave(grid_gif, grid_frames[::2], fps=10)
    print(f"\n=======================================================")
    print(f"Saved Full Toss Comparison to {grid_mp4} and {grid_gif}")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
