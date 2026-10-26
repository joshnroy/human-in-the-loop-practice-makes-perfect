"""Compare requested counts using the closest settled outcomes found so far.

Each independent seed-0 trial starts with five cubes in the pitcher. Physical
placement and the standard human reset are recorded even after a count failure.
"""

import argparse
import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any, cast

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hitl_pmp.environments.pouring3d.native_controller import PourParameters
from scripts.pouring3d_native_demo import NativePouringDemo


class RequestedCountDemo:
    @staticmethod
    def render(*, directory: Path, targets: list[int]) -> None:
        for target in targets:
            # The earlier settled, spill-free metering diagnostics delivered two or five.
            # These settings minimize count error among those observed outcomes.
            parameters = PourParameters(angle=130 if target <= 3 else 135, hold_seconds=0.2)
            NativePouringDemo.run(
                directory=directory / f"target-{target}",
                seed=0,
                video=True,
                scene_bg=True,
                reset="human",
                requested_count=target,
                parameters=parameters,
                detail_video=True,
                continue_after_pour_failure=True,
            )

    @staticmethod
    def frame(
        *, overview: np.ndarray, detail: np.ndarray, title: str, subtitle: str, status: str
    ) -> np.ndarray:
        canvas = Image.new("RGB", (1280, 576), (20, 25, 33))
        canvas.paste(Image.fromarray(overview), (0, 96))
        canvas.paste(Image.fromarray(detail), (640, 96))
        draw = ImageDraw.Draw(canvas)
        font = RequestedCountDemo.font(size=28)
        small = RequestedCountDemo.font(size=20)
        draw.text((18, 9), title, font=font, fill="white")
        color = "#9ee3b6" if status == "SUCCESS" else "#ffc18c" if status == "FAIL" else "#c5d8ed"
        draw.text((18, 51), subtitle, font=small, fill=color)
        draw.text((1120, 12), status, font=small, fill=color)
        draw.rectangle((640, 96, 1279, 122), fill=(20, 25, 33))
        draw.text((650, 98), "Native close-up | transparent pitcher", font=small, fill="white")
        return np.asarray(canvas)

    @staticmethod
    def compile(*, directory: Path) -> dict[str, Any]:
        target_path = directory / "requested-targets-1-through-5.mp4"
        fps = 40
        output_frames = 0
        segments = []
        with cast(
            Any, imageio.get_writer(target_path, fps=fps, codec="libx264", quality=8)
        ) as writer:
            for target in range(1, 6):
                folder = directory / f"target-{target}"
                result = json.loads((folder / "native-results.json").read_text())
                trace = json.loads((folder / "native-frames.json").read_text())
                events = result["events"]
                assert [e["skill"] for e in events] == [
                    "PickPitcher",
                    "Pour",
                    "PlacePitcherAtHome",
                    "reset_cubes",
                ]
                assert events[0]["success"] and events[2]["success"] and events[3]["success"]
                assert result["initial"]["locations"] == ["pitcher"] * 5
                assert events[1]["before"]["locations"] == ["pitcher"] * 5
                assert result["final"]["locations"] == ["pitcher"] * 5
                assert events[3]["seconds"] == 0
                assert events[3]["before"]["qpos"] == events[2]["after"]["qpos"]
                before_reset = events[3]["before"]
                delivered = before_reset["locations"].count("tray")
                retained = before_reset["locations"].count("pitcher")
                success = before_reset["goal"]
                assert success == events[1]["success"]
                assert delivered + retained == 5 and before_reset["settled"]
                start_frame = output_frames
                seen_result = False
                with (
                    cast(Any, imageio.get_reader(folder / "native-rollout.mp4")) as overview_reader,
                    cast(Any, imageio.get_reader(folder / "native-detail.mp4")) as detail_reader,
                ):
                    assert overview_reader.count_frames() == len(trace)
                    assert detail_reader.count_frames() == len(trace)
                    for i, (overview, detail) in enumerate(
                        zip(overview_reader, detail_reader, strict=True)
                    ):
                        row = trace[i]
                        skill = row["skill"]
                        finished_pour = row["time"] >= events[1]["after"]["time"] - 1e-6
                        reset = skill == "reset_cubes"
                        status = (
                            "RESET"
                            if reset
                            else ("SUCCESS" if success else "FAIL")
                            if finished_pour
                            else "RUNNING"
                        )
                        speed = "1x" if skill == "Pour" else "4x"
                        count = row["locations"].count("tray")
                        inside = row["locations"].count("pitcher")
                        subtitle = (
                            "Human reset: five back in pitcher; tray empty "
                            "(instantaneous intervention)"
                            if reset
                            else f"Tray: {count} | Pitcher: {inside} | "
                            f"{row['phase']} | {speed} playback"
                        )
                        frame = RequestedCountDemo.frame(
                            overview=overview,
                            detail=detail,
                            title=f"TARGET {target} OF 5",
                            subtitle=subtitle,
                            status=status,
                        )
                        if i == 0:
                            card = RequestedCountDemo.frame(
                                overview=overview,
                                detail=detail,
                                title=f"TARGET {target} OF 5 - RESET START",
                                subtitle=(
                                    "Five cubes in pitcher; tray empty | "
                                    "closest settings found so far"
                                ),
                                status="READY",
                            )
                            for _ in range(2 * fps):
                                writer.append_data(card)
                                output_frames += 1
                        repeats = 4 if skill == "Pour" or reset else 1
                        for _ in range(repeats):
                            writer.append_data(frame)
                            output_frames += 1
                        if finished_pour and not reset and not seen_result:
                            seen_result = True
                            for _ in range(2 * fps):
                                writer.append_data(frame)
                                output_frames += 1
                segments.append({
                    "target": target,
                    "delivered": delivered,
                    "retained": retained,
                    "success": success,
                    "parameters": result["pour_parameters"],
                    "start_seconds": start_frame / fps,
                    "end_seconds": output_frames / fps,
                    "source_sha256": result["video_sha256"],
                    "detail_sha256": hashlib.sha256(
                        (folder / "native-detail.mp4").read_bytes()
                    ).hexdigest(),
                    "fresh_harness_start": True,
                    "human_reset_recorded": True,
                })
            summary = Image.new("RGB", (1280, 576), (20, 25, 33))
            draw = ImageDraw.Draw(summary)
            draw.text(
                (40, 25),
                "REQUESTED COUNT - BEST ATTEMPTS FOUND SO FAR",
                font=RequestedCountDemo.font(size=28),
                fill="white",
            )
            draw.text(
                (40, 80),
                "Target          Delivered          Retained          Result",
                font=RequestedCountDemo.font(size=28),
                fill="#c5d8ed",
            )
            for index, segment in enumerate(segments):
                text = (
                    f"{segment['target']}                    {segment['delivered']}"
                    f"                      {segment['retained']}                    "
                    f"{'SUCCESS' if segment['success'] else 'FAIL'}"
                )
                draw.text(
                    (40, 145 + index * 65),
                    text,
                    font=RequestedCountDemo.font(size=28),
                    fill="#9ee3b6" if segment["success"] else "#ffc18c",
                )
            draw.text(
                (40, 510),
                "Five cubes at every start. One selected attempt per target; not success rates.",
                font=RequestedCountDemo.font(size=20),
                fill="white",
            )
            for _ in range(4 * fps):
                writer.append_data(np.asarray(summary))
                output_frames += 1
        with cast(Any, imageio.get_reader(target_path)) as reader:
            actual_frames = sum(1 for _ in reader)
            assert actual_frames == output_frames
            assert reader.get_meta_data()["fps"] == fps
        manifest = {
            "video": str(target_path),
            "fps": fps,
            "frames": output_frames,
            "duration_seconds": output_frames / fps,
            "segments": segments,
            "video_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
            "selection": (
                "Closest-count settings among earlier settled, spill-free "
                "development attempts; not learned policies"
            ),
            "playback": "Pour macro at 1x; other robot motion at 4x; reset display hold at 1x",
            "pitcher_alpha": 0.2,
            "reset": (
                "Independent seed-0 starts; standard zero-time human reset "
                "recorded after physical placement in each trial"
            ),
        }
        (directory / "compilation-results.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest), flush=True)
        return manifest

    @staticmethod
    @cache
    def font(*, size: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype("DejaVuSans.ttf", size)

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--targets", type=int, nargs="+", default=[1, 2, 3, 4, 5])
        parser.add_argument("--render-only", action="store_true")
        parser.add_argument("--compile-only", action="store_true")
        args = parser.parse_args()
        if any(target not in range(1, 6) for target in args.targets):
            parser.error("targets must be in 1..5")
        if not args.compile_only:
            RequestedCountDemo.render(directory=args.output_dir, targets=args.targets)
        if not args.render_only:
            RequestedCountDemo.compile(directory=args.output_dir)


if __name__ == "__main__":
    RequestedCountDemo.main()
