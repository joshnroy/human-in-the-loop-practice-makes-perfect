"""Render table setting, human stack resets, and autonomous spill recovery.

Run: scripts/with_env.sh python -m scripts.table_setting3d_demo --output-dir <dir>
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from hitl_pmp.environments.tablesetting3d.environment import TableSetting3DEnvironment
from hitl_pmp.methods.table_setting_reset import TableSettingResetSelector


class TableSettingDemo:
    @staticmethod
    def annotate(*, frame: np.ndarray, title: str, detail: str, footer: str) -> np.ndarray:
        canvas = Image.new("RGB", (frame.shape[1], frame.shape[0] + 112), "#13242d")
        canvas.paste(Image.fromarray(frame), (0, 64))
        draw = ImageDraw.Draw(canvas)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = (
            ImageFont.truetype(font_path, 16)
            if Path(font_path).exists()
            else ImageFont.load_default()
        )
        draw.text((16, 9), title, font=font, fill="white")
        draw.text((16, 35), detail, font=font, fill="#bddadf")
        draw.text((16, frame.shape[0] + 78), footer, font=font, fill="#bddadf")
        return np.asarray(canvas)

    @staticmethod
    def record_action(
        *,
        env: TableSetting3DEnvironment,
        action: int,
        writer: "TableSettingVideo | None",
        title: str,
        footer: str,
        speedup: float,
    ) -> dict[str, Any]:
        before = env.encode_state(state=env.get_current_state()).tolist()
        first_frame = writer.frame_count if writer is not None else None
        env.take_action(action=np.array([action], dtype=np.float32))
        if writer is not None:
            trajectory = env.backend.last_trajectory
            for start, end in zip(trajectory, trajectory[1:], strict=False):
                count = max(4, int(np.ceil(end["duration"] / speedup * 12)))
                for alpha in np.linspace(0, 1, count):
                    frame = env.backend.render_snapshot(
                        robot=(1 - alpha) * start["robot"] + alpha * end["robot"],
                        plates=(1 - alpha) * start["plates"] + alpha * end["plates"],
                    )
                    writer.append_frame(
                        frame=TableSettingDemo.annotate(
                            frame=frame,
                            title=title,
                            detail=end["label"],
                            footer=footer,
                        )
                    )
        info = env.last_info
        return {
            **{key: value for key, value in info.items() if key != "action_mask"},
            "action": action,
            "state_before": before,
            "state_after": env.encode_state(state=env.get_current_state()).tolist(),
            "video_frame_start": first_frame,
            "video_frame_end": writer.frame_count if writer is not None else None,
        }

    @staticmethod
    def run(*, args: argparse.Namespace) -> dict[str, Any]:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        selector = TableSettingResetSelector()
        env = TableSetting3DEnvironment(human_competence=args.human_competence, seed=args.seed)
        writer = None
        records: list[dict[str, Any]] = []
        try:
            env.hard_reset()
            if not args.no_video:
                writer = TableSettingVideo(
                    writer=imageio.get_writer(
                        str(args.output_dir / "episode.mp4"),
                        fps=12,
                        codec="libx264",
                        macro_block_size=16,
                    )
                )
                preview = TableSettingDemo.annotate(
                    frame=env.backend.render(),
                    title="TableSetting3D | Kitchen to dining table",
                    detail="Four heavy plates; TidyBot carries one at a time",
                    footer="Skill-level simulation | scripted motion; probabilistic human spills",
                )
                imageio.imwrite(args.output_dir / "initial.png", preview)
                for _ in range(18):
                    writer.append_frame(frame=preview)
            for cycle in range(args.cycles):
                cycle_first_frame = writer.frame_count if writer is not None else None
                stage = "Calibration" if cycle < args.calibration_resets else "Adaptive choice"
                title = f"{stage} {cycle + 1} | Human competence = {args.human_competence:.2f}"
                footer = "Scripted skills | unbreakable plates | one-plate robot payload"
                events = []
                for i in range(4):
                    events.append(
                        TableSettingDemo.record_action(
                            env=env,
                            action=i,
                            writer=writer,
                            title=title,
                            footer=footer,
                            speedup=args.speedup,
                        )
                    )
                if not env.backend.goal_reached():
                    raise RuntimeError("Robot failed to set all four assigned places")
                estimate = selector.competence_estimate
                costs = env.backend.reset_costs(competence_estimate=estimate)
                choice = (
                    "human"
                    if cycle < args.calibration_resets
                    else selector.choose(
                        robot_cost=costs["robot"],
                        human_attempt_cost=env.backend.human_reset_cost(),
                        spill_recovery_cost=env.backend.robot_reset_cost(spilled=True),
                    )
                )
                footer = (
                    f"Estimated competence {estimate:.2f} | "
                    f"Human {costs['human']:.0f}s / robot {costs['robot']:.0f}s | choose {choice}"
                )
                before_reset = env.backend.elapsed_seconds
                reset_first_frame = writer.frame_count if writer is not None else None
                if writer is not None:
                    frame = TableSettingDemo.annotate(
                        frame=env.backend.render(),
                        title=title,
                        detail=f"Table set. {stage}: {choice} reset",
                        footer=footer,
                    )
                    for _ in range(24):
                        writer.append_frame(frame=frame)
                if choice == "human":
                    event = TableSettingDemo.record_action(
                        env=env,
                        action=8,
                        writer=writer,
                        title=title,
                        footer=footer,
                        speedup=args.speedup,
                    )
                    events.append(event)
                    selector.observe(success=event["human_success"])
                for i in np.flatnonzero(~env.backend.plates_at_home()):
                    events.append(
                        TableSettingDemo.record_action(
                            env=env,
                            action=int(i) + 4,
                            writer=writer,
                            title=title,
                            footer=footer,
                            speedup=args.speedup,
                        )
                    )
                if not env.backend.plates_at_home().all():
                    raise RuntimeError("Reset left a plate outside its kitchen location")
                records.append({
                    "cycle": cycle,
                    "phase": stage,
                    "choice": choice,
                    "competence_estimate_before": estimate,
                    "competence_estimate_after": selector.competence_estimate,
                    "expected_reset_seconds": costs,
                    "actual_reset_seconds": env.backend.elapsed_seconds - before_reset,
                    "video_frame_start": cycle_first_frame,
                    "video_frame_end": writer.frame_count if writer is not None else None,
                    "reset_video_frame_start": reset_first_frame,
                    "events": events,
                })
                print(
                    f"Cycle {cycle + 1}: {choice}, estimated p={estimate:.2f}, "
                    f"reset={records[-1]['actual_reset_seconds']:.1f}s",
                    flush=True,
                )
            result = {
                "format_version": 1,
                "environment": "kinder/TableSetting3D-o4-v0",
                "seed": args.seed,
                "human_competence": args.human_competence,
                "simulation_fidelity": "scripted_skill_transitions",
                "calibration_resets": args.calibration_resets,
                "cycles": records,
                "elapsed_seconds": env.backend.elapsed_seconds,
                "human_interventions": env.backend.human_interventions,
                "source_sha256": TableSettingDemo.source_fingerprints(env=env),
                "video": None,
            }
            if writer is not None:
                writer.close()
                result["video"] = {
                    "filename": "episode.mp4",
                    "fps": 12,
                    "frames": writer.frame_count,
                    "speedup": args.speedup,
                    "sha256": hashlib.sha256(
                        (args.output_dir / "episode.mp4").read_bytes()
                    ).hexdigest(),
                }
            (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            return result
        finally:
            if writer is not None:
                writer.close()
            env.close()

    @staticmethod
    def source_fingerprints(*, env: TableSetting3DEnvironment) -> dict[str, str]:
        native = Path(inspect.getfile(type(env.backend)))
        paths = {
            "demo": Path(__file__),
            "native_environment": native,
            "project_adapter": Path(inspect.getfile(TableSetting3DEnvironment)),
            "reset_selector": Path(inspect.getfile(TableSettingResetSelector)),
            "plate_geometry": native.parent / "objects" / "plate.py",
            "scene": native.parent / "tasks" / "TableSetting3D" / "TableSetting3D-o4.json",
        }
        return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--human-competence", type=float, default=0.9)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--cycles", type=int, default=4)
        parser.add_argument("--calibration-resets", type=int, default=3)
        parser.add_argument("--speedup", type=float, default=12.0)
        parser.add_argument("--no-video", action="store_true")
        args = parser.parse_args()
        if args.cycles < 1 or not 0 <= args.calibration_resets <= args.cycles:
            parser.error("Require cycles >= 1 and 0 <= calibration-resets <= cycles")
        if not np.isfinite(args.speedup) or args.speedup <= 0:
            parser.error("speedup must be positive and finite")
        TableSettingDemo.run(args=args)


class TableSettingVideo(BaseModel):
    """Stream frames to the encoder and record exact, half-open frame ranges."""

    writer: Any
    frame_count: int = 0
    closed: bool = False

    def append_frame(self, *, frame: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("Video writer is already closed")
        self.writer.append_data(frame)
        self.frame_count += 1

    def close(self) -> None:
        if not self.closed:
            self.writer.close()
            self.closed = True


if __name__ == "__main__":
    TableSettingDemo.main()
