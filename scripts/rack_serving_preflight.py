"""Bounded native rack variant demonstration; never launches learning experiments."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.skills import SKILL_IDS
from scripts.validate_rack_coverage import Coverage


def main():
    import mujoco

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--surface", choices=("wire", "concept"), default="wire")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--human-layout", action="store_true")
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=0.12)
    parser.add_argument("--scraper-tilt-degrees", type=float, default=45)
    parser.add_argument("--food-grip-command", type=float, default=0.8)
    parser.add_argument("--gripper-contact-dimension", type=int, choices=(3, 4, 6), default=4)
    parser.add_argument("--cooling-seconds", type=float, default=5.0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--scene-bg", action="store_true", help="render the full KINDER background")
    parser.add_argument("--playback-speed", type=int, choices=(1, 5), default=1)
    parser.add_argument("--recovery", action="store_true")
    parser.add_argument("--recovery-indices", nargs="+", type=int, choices=range(5))
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--individual", action="store_true")
    parser.add_argument("--reset-probe", action="store_true")
    parser.add_argument("--return-only", action="store_true")
    parser.add_argument("--sweep-only", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.recovery_indices is not None and not args.recovery:
        parser.error("--recovery-indices requires --recovery")
    if args.output.exists():
        parser.error("use a new output directory; previous evidence is preserved")
    args.output.mkdir(parents=True)
    config = RackServingConfig(
        surface_kind=args.surface,
        cooling_seconds=args.cooling_seconds,
        scraper_tilt_degrees=args.scraper_tilt_degrees,
        food_grip_command=args.food_grip_command,
        gripper_contact_dimension=args.gripper_contact_dimension,
    )
    env = RackServingEnvironment(config=config, scene_bg=args.scene_bg, skill_step_limit=3500)
    report = {
        "config": config.model_dump(),
        "source_sha256": Coverage.source_digest(),
        "scene_bg": args.scene_bg,
        "seed": args.seed,
        "layout": "human_batch_return" if args.human_layout else "fixed",
        "playback_speed": args.playback_speed,
        "actions": [],
        "complete": False,
        "thermal_model": False,
        "food_model": "rigid ellipsoids, not deformable dumplings",
        "hardware_validated": False,
        "experiment_relaunched": False,
    }
    renderer = writer = trace = None
    tick = 0
    label = "Initial scene"
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)

    def save_report():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        env.reset_to_seed(seed=args.seed)
        backend = env.backend()
        model, data = backend.model(), backend.data()
        if args.human_layout:
            env.reset_movables(destination="human_batch_return")
            report["actions"].append({
                "human": "human_batch_return",
                "phase": "initial_layout",
                "error": None,
                "cumulative_cost": backend.ledger.human_cost,
            })
            label = f"COSTED HUMAN LAYOUT: seed {args.seed}; not a robot reset"
        if args.trace:
            trace = (args.output / "physics-trace.jsonl").open("w")
        if args.reset_probe:
            tray = backend.support(name="box_0")
            backend._commit(
                placements={
                    "cube_0": (
                        np.array([
                            config.tray_center_x,
                            0,
                            tray.top_z + config.dumpling_height / 2,
                        ]),
                        np.array([1.0, 0, 0, 0]),
                    ),
                }
            )
            for _ in range(500):
                backend.sim().step()
            report["uses_privileged_test_setup"] = True
            label = "RESET PROBE: test setup placed food_0 on tray; not a robot delivery"
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, 640)
        model.vis.global_.offheight = max(model.vis.global_.offheight, 480)
        if not args.no_video:
            renderer = mujoco.Renderer(model, height=480, width=640)
            writer = imageio.get_writer(
                str(args.output / "two-views.mp4"),
                fps=10,
                codec="libx264",
                quality=7,
                ffmpeg_params=["-threads", "1"],
            )
        side = mujoco.MjvCamera()
        side.lookat[:] = [config.rack_front_x, 0, config.counter_height]
        side.distance = 1.10
        side.azimuth = 135
        side.elevation = -35
        overview = mujoco.MjvCamera()
        overview.lookat[:] = [0.95, 0, 0.65]
        overview.distance = 3.0 if args.scene_bg else 2.3
        overview.azimuth = 135
        overview.elevation = -25

        def frame(*, repeat=1):
            if writer is None:
                return
            backend.sync_clock()
            renderer.update_scene(data, camera=overview)
            wide = renderer.render().copy()
            renderer.update_scene(data, camera=side)
            close = renderer.render().copy()
            canvas = Image.new("RGB", (1280, 608), "#111827")
            canvas.paste(Image.fromarray(np.concatenate([wide, close], axis=1)), (0, 58))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (12, 5),
                f"RACK → SERVING TRAY | {args.surface.upper()} GEOMETRY | native robot physics",
                font=font,
                fill="white",
            )
            draw.text(
                (12, 30),
                "Rigid proxies; sticking / deformation NOT modeled. "
                f"{args.playback_speed}× playback.",
                font=font,
                fill="white",
            )
            locations = [backend.location(name=name) for name in backend.cube_names]
            text = (
                f"{label[:55]} | rack {locations.count('OnRack')}/5"
                f" | tray {locations.count('OnTray')}/5"
                f" | counter {locations.count('SpilledCounter')}"
                f" | floor {locations.count('SpilledFloor')}"
                f" | air/other {locations.count(None)}"
            )
            draw.text((12, 544), text[:133], font=font, fill="#fbbf24")
            draw.text(
                (12, 570),
                f"DEMO clock {backend.ledger.elapsed_seconds:.1f}s / {config.cooling_seconds:g}s"
                f"; ready={backend.ledger.ready}"
                f" | NOT temperature sensing | human cost {backend.ledger.human_cost:g}",
                font=font,
                fill="white",
            )
            pixels = np.asarray(canvas)
            for _ in range(repeat):
                writer.append_data(pixels)
            return pixels

        def record_tick():
            nonlocal tick
            tick += 1
            if trace is not None and tick % 5 == 0:
                bodies = {}
                for name in (*backend.cube_names, "rack_0", "box_0", "wiper_0"):
                    body = model.body(name).id
                    address = model.jnt_dofadr[model.body_jntadr[body]]
                    bodies[name] = {
                        "position": data.xpos[body].tolist(),
                        "velocity": data.qvel[address : address + 6].tolist(),
                    }
                trace.write(
                    json.dumps({
                        "tick": tick,
                        "time": float(data.time),
                        "skill": label,
                        "bodies": bodies,
                        "food_pad_contacts": [
                            {
                                "geoms": [
                                    model.geom(contact.geom1).name,
                                    model.geom(contact.geom2).name,
                                ],
                                "position": contact.pos.tolist(),
                                "distance": float(contact.dist),
                                "dimension": int(contact.dim),
                            }
                            for contact in data.contact[: data.ncon]
                            if {int(contact.geom1), int(contact.geom2)}
                            & env.rack_controllers().shelf.pad_geoms
                            and {int(contact.geom1), int(contact.geom2)}
                            & backend.body_geoms(name="cube_0")
                        ],
                    })
                    + "\n"
                )
                trace.flush()
            if tick % args.playback_speed == 0:
                frame()
            if tick % 100 == 0:
                backend.sync_clock()
                (args.output / "progress.json").write_text(
                    json.dumps({
                        "skill": label,
                        "ticks": tick,
                        "simulation_seconds": backend.ledger.elapsed_seconds,
                    })
                    + "\n"
                )

        first = frame(repeat=15)
        if first is not None:
            imageio.imwrite(args.output / "initial.png", first)
        backend.set_tick_observer(observer=record_tick)
        report["initial_locations"] = [backend.location(name=name) for name in backend.cube_names]
        rack, tray = backend.support(name="rack_0"), backend.support(name="box_0")
        report["measured_geometry"] = {
            "rack": rack.model_dump(),
            "tray": tray.model_dump(),
            "gap_m": rack.x_min - tray.x_max,
        }

        def action(*, name, index=0):
            nonlocal label
            label = name
            before = backend.ledger.elapsed_seconds
            env.take_action(action=np.array([SKILL_IDS[name], index, args.angle, args.speed]))
            row = {
                "skill": name,
                "food_index": index,
                "error": env.last_skill_error(),
                "controller_steps": env.last_controller_steps(),
                "elapsed_seconds": backend.ledger.elapsed_seconds,
                "duration_seconds": backend.ledger.elapsed_seconds - before,
                "locations": [backend.location(name=food) for food in backend.cube_names],
                "success": backend.task_succeeded(),
            }
            tool = env.rack_controllers().tool
            row["scraper"] = {
                "pose": tool.wiper_pose().tolist(),
                "hand_pose": tool.shelf.site_pose().tolist(),
                "blade_bottom_m": tool.capsule_bottom(),
                "table_supported": tool.table_support(),
                "handle_held": tool.pads_hold_grip(),
                "at_home": tool.at_home(),
            }
            report["actions"].append(row)
            if row["error"]:
                label = f"{name} FAILED: {row['error']}"
            frame(repeat=15)
            save_report()
            print(json.dumps(row), flush=True)

        if not args.setup_only:
            if not backend.ledger.ready:
                action(name="SweepRack")
                report["early_sweep_rejected"] = "ready" in (env.last_skill_error() or "")
            action(name="Wait")
            if args.individual or args.reset_probe:
                action(name="PickFood", index=0)
                if backend.held(name="cube_0"):
                    action(
                        name="ReturnFoodToRack" if args.reset_probe else "PlaceFoodOnTray", index=0
                    )
            else:
                action(name="PickScraper")
                if backend.held(name="wiper_0"):
                    if not args.return_only:
                        action(name="SweepRack")
                    report["sweep_metrics"] = env.rack_controllers().metrics
                    if not args.sweep_only:
                        action(name="ReturnScraper")
            if args.recovery:
                candidates = (
                    args.recovery_indices
                    if args.recovery_indices is not None
                    else [
                        index
                        for index, name in enumerate(backend.cube_names)
                        if backend.food_available(name=name)
                        and backend.location(name=name) != "OnRack"
                    ]
                )
                for index in candidates[:2]:
                    action(name="PickFood", index=index)
                    if backend.held(name=backend.cube_names[index]):
                        action(name="ReturnFoodToRack", index=index)
                for destination in ("human_batch_return", "human_tool_recovery"):
                    label = f"TRAINING HUMAN: {destination} (explicit interruption cost)"
                    try:
                        env.reset_movables(destination=destination)
                        error = None
                    except (RuntimeError, ValueError) as failure:
                        error = str(failure)
                    report["actions"].append({
                        "human": destination,
                        "error": error,
                        "cumulative_cost": backend.ledger.human_cost,
                    })
                    frame(repeat=20)
            report["final_locations"] = [backend.location(name=name) for name in backend.cube_names]
            report["final_success"] = backend.task_succeeded()
        report["complete"] = True
    except Exception as error:
        report["fatal_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["source_changed_during_run"] = report["source_sha256"] != Coverage.source_digest()
        save_report()
        env.backend().set_tick_observer(observer=None)
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        if trace is not None:
            trace.close()
        env.close()


if __name__ == "__main__":
    main()
