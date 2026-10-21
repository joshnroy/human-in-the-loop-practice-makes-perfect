"""Build a verified comparison from two recorded TableSetting3D demo runs."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio


class TableSettingComparison:
    @staticmethod
    def load_run(*, directory: Path) -> dict[str, Any]:
        report = json.loads((directory / "results.json").read_text())
        video = report.get("video")
        if report.get("format_version") != 1 or video is None:
            raise ValueError(f"{directory}: rerun the demo with video and frame metadata")
        if video.get("filename") != "episode.mp4":
            raise ValueError(f"{directory}: unsupported video filename")
        path = directory / video["filename"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != video["sha256"]:
            raise ValueError(f"{directory}: video does not match its results manifest")
        return report

    @staticmethod
    def clip_ranges(*, report: dict[str, Any], include_spill: bool) -> list[dict[str, Any]]:
        adaptive = [cycle for cycle in report["cycles"] if cycle["phase"] == "Adaptive choice"]
        if not adaptive:
            raise ValueError("Comparison needs at least one cycle after calibration")
        clips = []
        if include_spill:
            for cycle in report["cycles"]:
                failed = [event for event in cycle["events"] if event.get("human_success") is False]
                if failed:
                    clips.append({
                        "label": "Observed human spill",
                        "start": cycle["reset_video_frame_start"],
                        "end": failed[0]["video_frame_end"],
                    })
                    break
        final = adaptive[-1]
        clips.append({
            "label": f"Learned choice: {final['choice']}",
            "start": final["video_frame_start"],
            "end": final["video_frame_end"],
        })
        for clip in clips:
            if not (
                isinstance(clip["start"], int)
                and isinstance(clip["end"], int)
                and 0 <= clip["start"] < clip["end"] <= report["video"]["frames"]
            ):
                raise ValueError("Invalid recorded frame range")
        return clips

    @staticmethod
    def run(*, first: Path, second: Path, output: Path) -> dict[str, Any]:
        runs = [TableSettingComparison.load_run(directory=path) for path in (first, second)]
        if runs[0]["source_sha256"] != runs[1]["source_sha256"]:
            raise ValueError("Comparison runs must use identical simulator and demo sources")
        if runs[0]["video"]["fps"] != runs[1]["video"]["fps"]:
            raise ValueError("Comparison runs must use the same video frame rate")
        if runs[0]["video"]["speedup"] != runs[1]["video"]["speedup"]:
            raise ValueError("Comparison runs must use the same playback speed")
        output = output.resolve()
        if output.suffix != ".mp4":
            raise ValueError("Comparison output must be an .mp4 file")
        if output in {
            (first / "episode.mp4").resolve(),
            (second / "episode.mp4").resolve(),
        } or output.with_suffix(".json") in {
            (first / "results.json").resolve(),
            (second / "results.json").resolve(),
        }:
            raise ValueError("Comparison output cannot overwrite source videos or manifests")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.stem + ".partial.mp4")
        fps = runs[0]["video"]["fps"]
        manifest: dict[str, Any] = {
            "fps": fps,
            "clips": [],
            "frames": 0,
            "comparison_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        try:
            writer: Any = imageio.get_writer(str(temporary), fps=fps, codec="libx264")
            with writer:
                for index, (directory, report) in enumerate(
                    zip((first, second), runs, strict=True)
                ):
                    for clip in TableSettingComparison.clip_ranges(
                        report=report, include_spill=index == 1
                    ):
                        output_start = manifest["frames"]
                        reader: Any = imageio.get_reader(directory / "episode.mp4")
                        with reader:
                            for frame_index in range(clip["start"], clip["end"]):
                                writer.append_data(reader.get_data(frame_index))
                                manifest["frames"] += 1
                        manifest["clips"].append({
                            **clip,
                            "source": str(directory.resolve()),
                            "source_sha256": report["video"]["sha256"],
                            "output_start": output_start,
                            "output_end": manifest["frames"],
                        })
            temporary.replace(output)
            manifest["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
            output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
            return manifest
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--first", type=Path, required=True)
        parser.add_argument("--second", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        TableSettingComparison.run(first=args.first, second=args.second, output=args.output)


if __name__ == "__main__":
    TableSettingComparison.main()
