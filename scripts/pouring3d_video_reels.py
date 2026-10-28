"""Native success/closest-miss replays of the frozen requested-count study."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, cast

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from scripts.pouring3d_target_demo import RequestedCountDemo
from scripts.pouring3d_wiggle_demo import WiggleDemo
from scripts.run_sweep import SweepRun, SweepRunner


class PourVideoReels:
    @staticmethod
    def choose(*, rows: list[dict[str, Any]], target: int, success: bool) -> dict[str, Any] | None:
        candidates = [
            row
            for row in rows
            if row["requested_count"] == target
            and row["eligible_count"] is not None
            and row["pour_success"] == success
        ]
        if not candidates:
            return None

        def rank(*, row: dict[str, Any]) -> tuple[Any, ...]:
            placement = next(
                (
                    event["success"]
                    for event in row["events"]
                    if event["skill"] == "PlacePitcherAtHome"
                ),
                None,
            )
            count_rank = (
                (0, 0) if success else (abs(row["eligible_count"] - target), row["eligible_count"])
            )
            return (
                *count_rank,
                not row["full_success"],
                placement is not True,
                placement is None,
                row["seed"],
                row["parameters"]["wiggle_seconds"],
                row["parameters"]["tilt_angle"],
                row["source_path"],
            )

        return min(candidates, key=lambda row: rank(row=row))

    @staticmethod
    def compare_snapshot(
        *, reference: dict[str, Any], rendered: dict[str, Any], label: str
    ) -> None:
        for key in ("qpos", "qvel", "ctrl", "time", "locations", "settled", "goal"):
            if reference[key] != rendered[key]:
                raise ValueError(f"Native replay differs at {label}: {key}")

    @staticmethod
    def launch(*, source_root: Path, root: Path) -> None:
        rows = []
        for path in sorted(source_root.glob("*/*/*/*/result.json")):
            row = json.loads(path.read_text())
            row["source_path"] = str(path.resolve())
            rows.append(row)
        selections = []
        for material in ("wood", "foam"):
            for success in (True, False):
                for target in range(1, 6):
                    row = PourVideoReels.choose(
                        rows=[row for row in rows if row["material"] == material],
                        target=target,
                        success=success,
                    )
                    if row is None:
                        continue
                    selections.append({
                        key: row[key]
                        for key in (
                            "source_path",
                            "material",
                            "seed",
                            "requested_count",
                            "parameters",
                            "eligible_count",
                            "pour_success",
                            "full_success",
                        )
                    })
        root.mkdir(parents=True, exist_ok=False)
        source = root / "source"
        source.mkdir()
        repo = Path(__file__).resolve().parents[1]
        paths = [Path(__file__).resolve(), repo / "scripts/pouring3d_wiggle_demo.py"]
        for path in paths:
            shutil.copyfile(path, source / path.name)
        (root / "protocol.json").write_text(
            json.dumps(
                {
                    "source_root": str(source_root.resolve()),
                    "selection": "One success and closest recorded count miss per material/request",
                    "success_ties": (
                        "Prefer full sequence and observed placement, then seed and duration"
                    ),
                    "failure_ties": "Count error, smaller tray count, placement, seed, duration",
                    "playback": (
                        "Pickup/placement 4x; complete Pour 1x; 2 s chapter and 3 s result cards"
                    ),
                    "selections": selections,
                    "source_sha256": {
                        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
                    },
                },
                indent=2,
            )
        )
        groups = [
            f"{material}-{outcome}"
            for material in ("wood", "foam")
            for outcome in ("success", "failure")
        ]
        outcomes = SweepRunner.execute(
            runs=[
                SweepRun(
                    method=group,
                    seed=0,
                    output_dir=root / group / "0",
                    command=[
                        "run-heavy",
                        str(repo / "scripts/with_env.sh"),
                        sys.executable,
                        "-m",
                        "scripts.pouring3d_video_reels",
                        "--root",
                        str(root),
                        "--worker",
                        group,
                    ],
                )
                for group in groups
            ],
            max_workers=2,
        )
        if not all(outcome.succeeded for outcome in outcomes):
            raise SystemExit(1)

    @staticmethod
    def worker(*, root: Path, group: str) -> None:
        material, outcome = group.split("-")
        protocol = json.loads((root / "protocol.json").read_text())
        clips = []
        for selection in protocol["selections"]:
            if selection["material"] != material or selection["pour_success"] != (
                outcome == "success"
            ):
                continue
            reference_path = Path(selection["source_path"])
            reference = json.loads(reference_path.read_text())
            target = selection["requested_count"]
            directory = root / group / "0" / f"target-{target}"
            if directory.exists():
                raise ValueError("Never overwrite an existing native replay")
            parameters = selection["parameters"]
            result = WiggleDemo.run(
                directory=directory,
                material=material,
                requested_count=target,
                seed=selection["seed"],
                angle=parameters["tilt_angle"],
                duration=parameters["wiggle_seconds"],
                video=True,
            )
            for key in (
                "material",
                "seed",
                "requested_count",
                "parameters",
                "eligible_count",
                "pour_success",
            ):
                if result[key] != reference[key]:
                    raise ValueError(f"Replay metadata differs: {key}")
            fields = ["initial", "after_pour"]
            extended_placement = len(reference["events"]) == 2
            if not extended_placement:
                fields.append("final")
                assert result["full_success"] == reference["full_success"]
            for field in fields:
                PourVideoReels.compare_snapshot(
                    reference=reference[field],
                    rendered=result[field],
                    label=field,
                )
            original_trace = json.loads(reference_path.with_name("trace.json").read_text())
            replay_trace = json.loads((directory / "trace.json").read_text())
            if replay_trace[: len(original_trace)] != original_trace:
                raise ValueError("Native motor-step trace differs from the selected attempt")
            if not extended_placement:
                assert len(replay_trace) == len(original_trace)
            for path, sha in reference["source_sha256"].items():
                if Path(path).name != "pouring3d_wiggle_demo.py":
                    assert result["source_sha256"][path] == sha, path
            audit = {
                **selection,
                "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                "exact_physical_snapshots": fields,
                "exact_original_trace_steps": len(original_trace),
                "physics_sources_unchanged": True,
                "placement_added_after_discovery": extended_placement,
                "rendered_full_success": result["full_success"],
                "placement_success": result["events"][-1]["success"]
                if result["events"][-1]["skill"] == "PlacePitcherAtHome"
                else None,
                "clip": str((directory / "native-tilt-wiggle.mp4").relative_to(root)),
            }
            (directory / "replay-audit.json").write_text(json.dumps(audit, indent=2))
            clips.append(audit)
        PourVideoReels.compile(root=root, group=group, clips=clips)

    @staticmethod
    def chapter_card(*, frame: np.ndarray, clip: dict[str, Any]) -> np.ndarray:
        canvas = Image.fromarray(frame.copy())
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, 1279, 95), fill=(20, 25, 33))
        target, actual = clip["requested_count"], clip["eligible_count"]
        title = f"{clip['material'].upper()} | target {target} | "
        title += "successful Pour" if clip["pour_success"] else f"closest miss: delivered {actual}"
        draw.text((18, 9), title, font=RequestedCountDemo.font(size=28), fill="white")
        p = clip["parameters"]
        placement = "PASS" if clip["placement_success"] else "FAIL"
        draw.text(
            (18, 51),
            f"Fresh seeded trial: five in pitcher | tilt {p['tilt_angle']:g} deg, "
            f"wiggle {p['wiggle_seconds']:g} s | placement {placement}",
            font=RequestedCountDemo.font(size=20),
            fill="#c5d8ed",
        )
        return np.asarray(canvas)

    @staticmethod
    def compile(*, root: Path, group: str, clips: list[dict[str, Any]]) -> None:
        destination = root / f"{group}.mp4"
        chapters = []
        output_frames = 0
        with cast(
            Any,
            imageio.get_writer(
                destination,
                fps=40,
                codec="libx264",
                quality=None,
                bitrate="600k",
                ffmpeg_params=["-threads", "2", "-movflags", "+faststart"],
            ),
        ) as writer:
            for clip in clips:
                path = root / clip["clip"]
                with cast(Any, imageio.get_reader(path)) as reader:
                    count = reader.count_frames()
                    result = json.loads(path.with_name("result.json").read_text())
                    assert count == result["video_output_frames"]
                    assert reader.get_meta_data()["fps"] == 40
                    card = PourVideoReels.chapter_card(frame=reader.get_data(0), clip=clip)
                    start = output_frames
                    for _ in range(80):
                        writer.append_data(card)
                    for index in range(count):
                        writer.append_data(reader.get_data(index))
                    output_frames += 80 + count
                    chapters.append({
                        **clip,
                        "start_seconds": start / 40,
                        "end_seconds": output_frames / 40,
                        "frames": count,
                        "clip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    })
        if destination.stat().st_size >= 20 * 1024 * 1024:
            raise ValueError("Reel exceeds the Notion upload size; lower the bitrate")
        with cast(Any, imageio.get_reader(destination)) as reader:
            assert reader.count_frames() == output_frames
            assert reader.get_meta_data()["size"] == (1280, 576)
        manifest = {
            "group": group,
            "video": destination.name,
            "fps": 40,
            "output_frames": output_frames,
            "duration_seconds": output_frames / 40,
            "bytes": destination.stat().st_size,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "chapters": chapters,
        }
        (root / f"{group}.json").write_text(json.dumps(manifest, indent=2))
        print(
            json.dumps({"reel": group, "clips": len(clips), "seconds": output_frames / 40}),
            flush=True,
        )

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--source-root", type=Path)
        parser.add_argument(
            "--worker", choices=("wood-success", "wood-failure", "foam-success", "foam-failure")
        )
        args = parser.parse_args()
        if args.worker:
            PourVideoReels.worker(root=args.root, group=args.worker)
        elif args.source_root:
            PourVideoReels.launch(source_root=args.source_root, root=args.root)
        else:
            parser.error("--source-root or --worker is required")


if __name__ == "__main__":
    PourVideoReels.main()
