"""Record the real bin-pouring controller from a saved in-bin planning refusal.

scripts/with_env.sh python scripts/tossing3d_pour_bin_demo.py --output-dir artifacts/pour-demo
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
from hitl_pmp.environments.tossing3d.predicates import (
    BIN_ON_GROUND,
    HAND_EMPTY,
    IN_BIN,
    ON_GROUND,
    PICK_PLANNABLE,
)

_DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests/environments/tossing3d/fixtures/pick_plannable/exp22_F.json"
)


def capture(*, fixture: Path, output_dir: Path, layout: Tossing3DLayout) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = Tossing3DEnvironment(layout=layout, scene_bg=False)
    renderer = None
    try:
        env.hard_reset()
        backend = env.kinder_backend()
        plain = json.loads(fixture.read_text())
        plain["cuboid_barrier"] = backend.snapshot_to_plain(snapshot=backend.live_state())[
            "cuboid_barrier"
        ]
        before = env.restore_plain_snapshot(plain=plain)
        if not IN_BIN.holds(before, (env.cube, env.bin)) or PICK_PLANNABLE.holds(
            before, (env.robot, env.cube)
        ):
            raise ValueError("The fixture must have an in-bin cube with no pick plan")
        original = backend.snapshot_to_plain(snapshot=backend.live_state())
        (output_dir / "before.json").write_text(json.dumps(original, indent=2) + "\n")
        import mujoco

        sim = backend._object_centric()._robot_env.sim  # noqa: SLF001
        model, data = sim.model.mj_model, sim.data.mj_data
        model.vis.global_.offwidth, model.vis.global_.offheight = 960, 600
        renderer = mujoco.Renderer(model, height=600, width=960)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [-0.60, -1.15, 0.43]
        camera.distance, camera.azimuth, camera.elevation = 2.65, 145, -29
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        title_font = ImageFont.truetype(font_path, 25)
        body_font = ImageFont.truetype(font_path, 18)
        steps = 0
        frames = []
        bin_body = model.body("bin_0").id
        with iio.get_writer(output_dir / "pour-bin-physics.mp4", fps=20) as writer:

            def record() -> None:
                nonlocal steps
                steps += 1
                renderer.update_scene(data, camera=camera)
                panel = Image.new("RGB", (960, 720), "#101827")
                panel.paste(Image.fromarray(renderer.render()), (0, 76))
                draw = ImageDraw.Draw(panel)
                draw.text(
                    (22, 13),
                    "POUR CUBE OUT OF BIN — PHYSICS ROLLOUT",
                    font=title_font,
                    fill="white",
                )
                draw.text(
                    (22, 47),
                    "Rim contact grasp  /  motor commands only  /  playback 2×",
                    font=body_font,
                    fill="#b9c8d9",
                )
                draw.text(
                    (22, 687),
                    f"Control step {steps}   |   bin height {data.xpos[bin_body, 2]:.2f} m",
                    font=body_font,
                    fill="#b9c8d9",
                )
                frame = np.asarray(panel)
                writer.append_data(frame)
                if steps % 5 == 0:
                    frames.append(panel.resize((640, 480), Image.Resampling.LANCZOS))

            backend.set_control_step_observers(before=None, after=record)
            after = env.take_action(action=np.array([env.pour_bin_id, 0, 0, 0, 0]))
            backend.set_control_step_observers(before=None, after=None)
            for _ in range(20):
                record()
        final = backend.snapshot_to_plain(snapshot=backend.live_state())
        (output_dir / "after.json").write_text(json.dumps(final, indent=2) + "\n")
        result = {
            "fixture": str(fixture),
            "layout": layout.value,
            "controller_steps": list(env.last_controller_steps()),
            "error": env.last_skill_error(),
            "cube_in_bin": IN_BIN.holds(after, (env.cube, env.bin)),
            "cube_on_floor": ON_GROUND.holds(after, (env.cube,)),
            "pick_plannable": PICK_PLANNABLE.holds(after, (env.robot, env.cube)),
            "bin_on_ground": BIN_ON_GROUND.holds(after, (env.bin,)),
            "hand_empty": HAND_EMPTY.holds(after, (env.robot,)),
            "bin_position_error_m": float(
                np.linalg.norm(np.asarray(final["bin_0"][:3]) - original["bin_0"][:3])
            ),
            "human_resets": 0,
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        frames[0].save(
            output_dir / "pour-bin-physics.gif",
            save_all=True,
            append_images=frames[1:],
            duration=250,
            loop=0,
        )
        frames[-1].save(output_dir / "final.png")
        print(json.dumps(result, indent=2), flush=True)
        if result["error"] or result["cube_in_bin"] or not result["pick_plannable"]:
            raise RuntimeError("Bin pouring failed; see result.json")
        return result
    finally:
        env.close()
        if renderer is not None:
            renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=_DEFAULT_FIXTURE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layout", choices=[v.value for v in Tossing3DLayout], default="barrier")
    args = parser.parse_args()
    capture(fixture=args.snapshot, output_dir=args.output_dir, layout=Tossing3DLayout(args.layout))


if __name__ == "__main__":
    main()
