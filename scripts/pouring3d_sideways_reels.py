"""Render and join the best frozen sideways Pour attempts for targets 1–5."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, cast

import imageio.v2 as imageio

from scripts.pouring3d_lateral_counts import LateralCounts
from scripts.pouring3d_video_reels import PourVideoReels
from scripts.run_sweep import SweepRun, SweepRunner


class SidewaysReels:
    @staticmethod
    def worker(*, root: Path, material: str) -> None:
        LateralCounts.render(root=root, material=material, include_probes=True)
        SidewaysReels.compile(root=root, material=material)

    @staticmethod
    def compile(*, root: Path, material: str) -> None:
        destination = root / f"sideways-{material}-counts.mp4"
        if destination.exists():
            raise ValueError("Never overwrite an existing native compilation")
        chapters = []
        frames = 0
        with cast(
            Any,
            imageio.get_writer(
                destination,
                fps=40,
                codec="libx264",
                quality=None,
                bitrate="350k",
                ffmpeg_params=["-threads", "2", "-movflags", "+faststart"],
            ),
        ) as writer:
            for target in range(1, 6):
                directory = root / f"video-{material}" / "0" / str(target)
                result = json.loads((directory / "result.json").read_text())
                audit = json.loads((directory / "replay-check.json").read_text())
                assert audit["exact"]
                placed = next(
                    (e["success"] for e in result["events"] if e["skill"] == "PlacePitcherAtHome"),
                    None,
                )
                clip = {
                    **{
                        k: result[k]
                        for k in (
                            "material",
                            "seed",
                            "requested_count",
                            "parameters",
                            "eligible_count",
                            "pour_success",
                            "full_success",
                        )
                    },
                    "placement_success": placed,
                    "exact_replay": audit,
                    "after_pour_locations": result["after_pour"]["locations"],
                    "after_pour_settled": result["after_pour"]["settled"],
                }
                path = directory / "native-tilt-wiggle.mp4"
                with cast(Any, imageio.get_reader(path)) as reader:
                    count = reader.count_frames()
                    assert count == result["video_output_frames"]
                    assert reader.get_meta_data()["fps"] == 40
                    card = PourVideoReels.chapter_card(frame=reader.get_data(0), clip=clip)
                    start = frames
                    for _ in range(80):
                        writer.append_data(card)
                    for index in range(count):
                        writer.append_data(reader.get_data(index))
                    frames += 80 + count
                    chapters.append({
                        **clip,
                        "start_seconds": start / 40,
                        "end_seconds": frames / 40,
                        "clip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    })
        assert destination.stat().st_size < 20 * 1024 * 1024
        with cast(Any, imageio.get_reader(destination)) as reader:
            assert reader.count_frames() == frames
            assert reader.get_meta_data()["size"] == (1280, 576)
        (root / f"sideways-{material}-counts.json").write_text(
            json.dumps(
                {
                    "material": material,
                    "video": destination.name,
                    "duration_seconds": frames / 40,
                    "bytes": destination.stat().st_size,
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "chapters": chapters,
                },
                indent=2,
            )
        )

    @staticmethod
    def launch(*, root: Path, material: str | None = None) -> None:
        repo = Path(__file__).resolve().parents[1]
        source = root / "render-source"
        paths = [
            Path(__file__).resolve(),
            repo / "scripts/pouring3d_lateral_counts.py",
            repo / "scripts/pouring3d_video_reels.py",
            repo / "scripts/pouring3d_sideways_probes.py",
        ]
        if source.exists():
            for path in paths:
                assert path.read_bytes() == (source / path.name).read_bytes()
        else:
            source.mkdir()
            for path in paths:
                shutil.copy2(path, source / path.name)
        protocol = root / "render-protocol.json"
        if not protocol.exists():
            protocol.write_text(
                json.dumps(
                    {
                        "requested_counts": list(range(1, 6)),
                        "materials": ["wood", "foam"],
                        "selection": (
                            "Best explicit-target attempt from frozen checks and extra probes"
                        ),
                        "ties": "Exact Pour, contained count error, full sequence, then seed",
                        "fresh_start_per_chapter": True,
                        "playback": "Pick/Place 4x; complete Pour 1x; chapter 2 s/result 3 s",
                        "source_sha256": {
                            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in paths
                        },
                    },
                    indent=2,
                )
            )
        materials = (material,) if material is not None else ("wood", "foam")
        outcomes = SweepRunner.execute(
            runs=[
                SweepRun(
                    method=f"video-{material}",
                    seed=0,
                    output_dir=root / f"video-{material}" / "0",
                    command=[
                        "run-heavy",
                        str(repo / "scripts/with_env.sh"),
                        sys.executable,
                        "-m",
                        "scripts.pouring3d_sideways_reels",
                        "--root",
                        str(root),
                        "--worker",
                        material,
                    ],
                )
                for material in materials
            ],
            max_workers=2,
        )
        if not all(outcome.succeeded for outcome in outcomes):
            raise SystemExit(1)

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--worker", choices=("wood", "foam"))
        parser.add_argument("--compile-only", choices=("wood", "foam"))
        parser.add_argument("--material", choices=("wood", "foam"))
        args = parser.parse_args()
        if args.worker:
            SidewaysReels.worker(root=args.root, material=args.worker)
        elif args.compile_only:
            SidewaysReels.compile(root=args.root, material=args.compile_only)
        else:
            SidewaysReels.launch(root=args.root, material=args.material)


if __name__ == "__main__":
    SidewaysReels.main()
