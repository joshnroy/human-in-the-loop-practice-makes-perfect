"""Bounded native tests and recordings of the two-parameter wiggle prototype."""

import argparse
import gc
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, ClassVar, cast

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hitl_pmp.environments.pouring3d.native_controller import NativePouringController
from hitl_pmp.environments.pouring3d.wiggle import (
    TiltWiggleController,
    TiltWigglePouringEnvironment,
)
from scripts.pouring3d_target_demo import RequestedCountDemo
from scripts.run_sweep import SweepRun, SweepRunner


class StationaryHoldController(TiltWiggleController):
    """Diagnostic matched-time control; amplitude is not a practiced parameter."""

    WIGGLE_DEGREES: ClassVar[float] = 0.0


class StationaryHoldEnvironment(TiltWigglePouringEnvironment):
    controller_class: ClassVar[type[TiltWiggleController]] = StationaryHoldController


class WiggleDemo:
    @staticmethod
    def run(
        *,
        directory: Path,
        material: str,
        angle: float,
        duration: float,
        seed: int = 0,
        mode: str = "wiggle",
        video: bool = False,
        requested_count: int = 2,
        place: bool = True,
    ) -> dict[str, Any]:
        from kinder.envs.dynamic3d.pitcher_pour3d import PitcherPour3DEnv

        directory.mkdir(parents=True, exist_ok=True)
        native = Path(inspect.getfile(PitcherPour3DEnv))
        task = native.parent / "tasks/PitcherPour3D/PitcherPour3D-o5-winco.json"
        sources = [
            native,
            task,
            Path(inspect.getfile(NativePouringController)),
            Path(inspect.getfile(TiltWiggleController)),
            Path(__file__).resolve(),
        ]
        hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        env_class = StationaryHoldEnvironment if mode == "static" else TiltWigglePouringEnvironment
        env = env_class(
            seed=seed,
            cube_material=material,
            requested_count=requested_count,
            scene_bg=True,
            task_config_path=str(task),
        )
        env.hard_reset()
        c = env.sim.controller
        trace = []
        active_skill = "initial"
        pour_success = False
        writer = (
            cast(
                Any,
                imageio.get_writer(
                    directory / "native-tilt-wiggle.mp4",
                    fps=40,
                    codec="libx264",
                    quality=7,
                ),
            )
            if video
            else None
        )
        try:
            initial = env.sim.snapshot()
            assert initial["locations"] == ["pitcher"] * 5

            def record(**kwargs: Any) -> None:
                controller = kwargs["controller"]
                locations = controller.scene.cube_locations()
                row = {
                    "time": float(controller.data.time),
                    "phase": controller.phase,
                    "skill": active_skill,
                    "locations": locations,
                    "qpos_sha256": hashlib.sha256(controller.data.qpos.tobytes()).hexdigest(),
                }
                trace.append(row)
                if writer is not None:
                    cameras = controller.scene.render_all_cameras()
                    detail = "pour_detail" if active_skill == "Pour" else "pitcher_detail"
                    speed = "1x" if active_skill == "Pour" else "4x"
                    status = (
                        ("POUR PASS" if pour_success else "POUR MISS")
                        if active_skill == "PlacePitcherAtHome"
                        else "RUNNING"
                    )
                    frame = RequestedCountDemo.frame(
                        overview=cameras["task_view_image"],
                        detail=cameras[f"{detail}_image"],
                        title=f"{material.upper()} | tilt {angle:g} deg, wiggle {duration:g} s",
                        subtitle=(
                            f"Target {requested_count} | tray {locations.count('tray')}, "
                            f"pitcher {locations.count('pitcher')} | "
                            f"{controller.phase} | {speed}"
                        ),
                        status=status,
                    )
                    for _ in range(4 if active_skill == "Pour" else 1):
                        writer.append_data(frame)

            c.callback = record
            record(controller=c)
            events = []
            for skill_id, name in ((0, "PickPitcher"), (1, "Pour"), (2, "PlacePitcherAtHome")):
                if name == "PlacePitcherAtHome" and not place:
                    break
                active_skill = name
                env.take_action(action=np.array([skill_id, 0, angle, duration]))
                event = env.last_info
                events.append(event)
                if name == "Pour":
                    pour_success = bool(event["success"])
                if event["error"] is not None or (name == "PickPitcher" and not event["success"]):
                    break
            after_pour = next(
                (event["after"] for event in events if event["skill"] == "Pour"), env.sim.snapshot()
            )
            final = env.sim.snapshot()
            eligible_count = (
                after_pour["locations"].count("tray")
                if after_pour["settled"]
                and all(location in {"pitcher", "tray"} for location in after_pour["locations"])
                and len(events) >= 2
                and events[1]["error"] is None
                else None
            )
            result = {
                "material": material,
                "seed": seed,
                "requested_count": requested_count,
                "mode": mode,
                "parameters": {"tilt_angle": angle, "wiggle_seconds": duration},
                "fixed_settings": c.fixed_settings(),
                "initial": initial,
                "after_pour": after_pour,
                "final": final,
                "events": events,
                "eligible_count": eligible_count,
                "pour_success": pour_success,
                "full_success": pour_success
                and len(events) == 3
                and events[-1]["success"]
                and final["goal"],
                "source_sha256": hashes,
                "wiggle_trace": c.wiggle_trace,
                "source_frames": len(trace),
                "video": video,
            }
            if writer is not None:
                cameras = c.scene.render_all_cameras()
                placed = next(
                    (
                        event["success"]
                        for event in events
                        if event["skill"] == "PlacePitcherAtHome"
                    ),
                    None,
                )
                placement = "not run" if placed is None else "PASS" if placed else "FAIL"
                status = (
                    "SUCCESS" if result["full_success"] else "POUR ONLY" if pour_success else "FAIL"
                )
                frame = RequestedCountDemo.frame(
                    overview=cameras["task_view_image"],
                    detail=cameras["pitcher_detail_image"],
                    title=f"{material.upper()} | target {requested_count} | final result",
                    subtitle=(
                        f"Pour {'PASS' if pour_success else 'FAIL'} "
                        f"({after_pour['locations'].count('tray')}/{requested_count}) | "
                        f"Placement {placement} | Final tray {final['locations'].count('tray')}, "
                        f"retained {final['locations'].count('pitcher')}"
                    ),
                    status=status,
                )
                for _ in range(120):
                    writer.append_data(frame)
                result["video_fps"] = 40
                result["video_result_hold_seconds"] = 3
                result["video_output_frames"] = (
                    sum(4 if row["skill"] == "Pour" else 1 for row in trace) + 120
                )
            (directory / "result.json").write_text(json.dumps(result, indent=2))
            (directory / "trace.json").write_text(json.dumps(trace, indent=2))
            print(
                json.dumps({
                    key: result[key]
                    for key in (
                        "material",
                        "seed",
                        "mode",
                        "parameters",
                        "eligible_count",
                        "pour_success",
                    )
                }),
                flush=True,
            )
            return result
        finally:
            if writer is not None:
                writer.close()
            c.callback = None
            env.sim.close()
            gc.collect()

    @staticmethod
    def discover(*, root: Path, material: str) -> None:
        angle = 97 if material == "wood" else 107
        cases = [
            ("plain", "static", angle, 0),
            ("hold", "static", angle, 2),
            ("higher-static", "static", angle + 4, 2),
            ("wiggle-2s", "wiggle", angle, 2),
            ("higher-wiggle", "wiggle", angle + 3, 2),
            ("wiggle-4s", "wiggle", angle, 4),
        ]
        results = []
        for label, mode, tilt, duration in cases:
            result = WiggleDemo.run(
                directory=root / material / "0" / label,
                material=material,
                angle=tilt,
                duration=duration,
                mode=mode,
            )
            result["label"] = label
            results.append(result)
        (root / material / "0/discovery.json").write_text(json.dumps(results, indent=2))

    @staticmethod
    def launch(*, root: Path) -> None:
        repo = Path(__file__).resolve().parents[1]
        runs = [
            SweepRun(
                method=material,
                seed=0,
                output_dir=root / material / "0",
                command=[
                    "run-heavy",
                    str(repo / "scripts/with_env.sh"),
                    sys.executable,
                    "-m",
                    "scripts.pouring3d_wiggle_demo",
                    "--root",
                    str(root),
                    "--worker",
                    material,
                ],
            )
            for material in ("wood", "foam")
        ]
        outcomes = SweepRunner.execute(runs=runs, max_workers=2)
        if not all(outcome.succeeded for outcome in outcomes):
            raise SystemExit(1)

    @staticmethod
    def plot(*, root: Path) -> None:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for axis, material in zip(axes, ("wood", "foam"), strict=True):
            rows = json.loads((root / material / "0/discovery.json").read_text())
            values = [row["after_pour"]["locations"].count("tray") for row in rows]
            axis.bar(
                range(len(rows)),
                values,
                color=["#86caa1" if row["pour_success"] else "#efaf83" for row in rows],
            )
            for index, row in enumerate(rows):
                axis.text(
                    index,
                    values[index] + 0.1,
                    str(values[index]) + ("*" if row["eligible_count"] is None else ""),
                    ha="center",
                )
            axis.set(
                xticks=range(len(rows)),
                xticklabels=[row["label"] for row in rows],
                yticks=range(6),
                ylim=(0, 5.7),
                ylabel="Tray count after Pour",
                title=f"{material.title()} | requested 2 | seed 0",
            )
            axis.tick_params(axis="x", rotation=30)
            axis.grid(axis="y", alpha=0.2)
        figure.suptitle("Two-parameter tilt → wiggle | green = exact settled two; * = rejected")
        figure.savefig(root / "wiggle-results.png", dpi=160)
        plt.close(figure)

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--worker", choices=("wood", "foam"))
        parser.add_argument("--render", choices=("wood", "foam"))
        parser.add_argument("--tilt-angle", type=float, default=108)
        parser.add_argument("--wiggle-seconds", type=float, default=2)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--plot-only", action="store_true")
        args = parser.parse_args()
        if args.worker:
            WiggleDemo.discover(root=args.root, material=args.worker)
        elif args.render:
            WiggleDemo.run(
                directory=args.root / f"video-{args.render}" / str(args.seed),
                material=args.render,
                angle=args.tilt_angle,
                duration=args.wiggle_seconds,
                seed=args.seed,
                video=True,
            )
        else:
            if not args.plot_only:
                WiggleDemo.launch(root=args.root)
            WiggleDemo.plot(root=args.root)


if __name__ == "__main__":
    WiggleDemo.main()
