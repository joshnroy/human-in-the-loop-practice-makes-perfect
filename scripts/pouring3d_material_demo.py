"""Render provisional wood/foam physics and compare synchronized native pours."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from hitl_pmp.environments.pouring3d.native_controller import PourParameters
from scripts.pouring3d_native_demo import NativePouringDemo
from scripts.pouring3d_target_demo import RequestedCountDemo


class MaterialPouringDemo:
    @staticmethod
    def render(*, directory: Path, material: str) -> None:
        for angle in (130, 145):
            NativePouringDemo.run(
                directory=directory / material / f"angle-{angle}",
                seed=0,
                video=True,
                scene_bg=True,
                reset="human",
                requested_count=5,
                parameters=PourParameters(angle=angle, tilt_seconds=10, hold_seconds=0.2),
                detail_video=True,
                continue_after_pour_failure=True,
                cube_material=material,
                pitcher_reference="winco",
            )

    @staticmethod
    def frame(
        *,
        left: np.ndarray,
        right: np.ndarray,
        angle: int,
        time: float | None,
        counts: list[tuple[int, int]],
        label: str,
    ) -> np.ndarray:
        canvas = Image.new("RGB", (1280, 576), (20, 25, 33))
        canvas.paste(Image.fromarray(left), (0, 96))
        canvas.paste(Image.fromarray(right), (640, 96))
        draw = ImageDraw.Draw(canvas)
        title = f"{angle} degree pour | {label}"
        draw.text((16, 4), title, font=RequestedCountDemo.font(size=23), fill="white")
        for index, (material, mass, surface, self_mu) in enumerate([
            ("WOOD", "4.8 g", "0.30", "0.40"),
            ("FOAM CONTACT APPROXIMATION", "0.64 g", "0.60", "0.80"),
        ]):
            x = index * 640 + 16
            draw.text((x, 35), material, font=RequestedCountDemo.font(size=23), fill="white")
            draw.text(
                (x, 67),
                f"{mass}/cube | pitcher friction {surface} | cube-cube {self_mu}",
                font=RequestedCountDemo.font(size=17),
                fill="#c5d8ed",
            )
            draw.rectangle((index * 640, 540, (index + 1) * 640 - 1, 575), fill=(20, 25, 33))
            tray, pitcher = counts[index]
            caption = f"Tray: {tray}/5 | Pitcher: {pitcher}"
            if time is not None:
                caption += f" | t={time:.1f}s | 1x playback"
            draw.text((x, 547), caption, font=RequestedCountDemo.font(size=18), fill="white")
        return np.asarray(canvas)

    @staticmethod
    def compile(*, directory: Path) -> dict[str, Any]:
        output = directory / "wood-versus-foam-native.mp4"
        fps = 20
        frames_written = 0
        segments = []
        with cast(Any, imageio.get_writer(output, fps=fps, codec="libx264", quality=7)) as writer:
            for angle in (130, 145):
                traces, manifests, starts, ends = [], [], [], []
                for material in ("wood", "foam"):
                    folder = directory / material / f"angle-{angle}"
                    manifest = json.loads((folder / "native-results.json").read_text())
                    trace = json.loads((folder / "native-frames.json").read_text())
                    assert manifest["cube_material"] == material
                    assert manifest["pitcher_reference"] == "winco"
                    assert manifest["initial"]["locations"] == ["pitcher"] * 5
                    assert all(e["success"] for e in manifest["events"])
                    assert manifest["events"][1]["before"]["locations"] == ["pitcher"] * 5
                    starts.append(
                        next(i for i, row in enumerate(trace) if row["phase"] == "tilt pitcher")
                        - 10
                    )
                    ends.append(max(i for i, row in enumerate(trace) if row["skill"] == "Pour") + 1)
                    traces.append(trace)
                    manifests.append(manifest)
                assert manifests[0]["pour_parameters"] == manifests[1]["pour_parameters"]
                assert ends[0] - starts[0] == ends[1] - starts[1]
                start_seconds = frames_written / fps
                with (
                    cast(
                        Any,
                        imageio.get_reader(
                            directory / "wood" / f"angle-{angle}" / "native-detail.mp4"
                        ),
                    ) as wood,
                    cast(
                        Any,
                        imageio.get_reader(
                            directory / "foam" / f"angle-{angle}" / "native-detail.mp4"
                        ),
                    ) as foam,
                ):
                    setup = MaterialPouringDemo.frame(
                        left=wood.get_data(0),
                        right=foam.get_data(0),
                        angle=angle,
                        time=None,
                        counts=[(0, 5), (0, 5)],
                        label="Five-cube reset start | provisional physics",
                    )
                    for _ in range(2 * fps):
                        writer.append_data(setup)
                        frames_written += 1
                    for offset in range(ends[0] - starts[0]):
                        indices = [start + offset for start in starts]
                        rows = [traces[i][indices[i]] for i in range(2)]
                        counts = [
                            (r["locations"].count("tray"), r["locations"].count("pitcher"))
                            for r in rows
                        ]
                        frame = MaterialPouringDemo.frame(
                            left=wood.get_data(indices[0]),
                            right=foam.get_data(indices[1]),
                            angle=angle,
                            time=(offset - 10) / 10,
                            counts=counts,
                            label="Same commands | synchronized at tilt start",
                        )
                        for _ in range(2):
                            writer.append_data(frame)
                            frames_written += 1
                    for _ in range(2 * fps):
                        writer.append_data(frame)
                        frames_written += 1
                    reset = MaterialPouringDemo.frame(
                        left=wood.get_data(len(traces[0]) - 1),
                        right=foam.get_data(len(traces[1]) - 1),
                        angle=angle,
                        time=None,
                        counts=[(0, 5), (0, 5)],
                        label="Human reset after placement | five back in each pitcher",
                    )
                    for _ in range(2 * fps):
                        writer.append_data(reset)
                        frames_written += 1
                segments.append({
                    "angle": angle,
                    "start_seconds": start_seconds,
                    "end_seconds": frames_written / fps,
                    "source_start_frames": starts,
                    "source_end_frames": ends,
                    "results": [
                        {
                            "material": material,
                            "tray_count": manifests[i]["events"][1]["after"]["locations"].count(
                                "tray"
                            ),
                            "parameters": manifests[i]["material_parameters"],
                            "maximum_contact_penetration_m": max(
                                r["maximum_cube_contact_penetration_m"] for r in traces[i]
                            ),
                        }
                        for i, material in enumerate(("wood", "foam"))
                    ],
                })
        with cast(Any, imageio.get_reader(output)) as reader:
            assert sum(1 for _ in reader) == frames_written
            assert reader.get_meta_data()["fps"] == fps
        result = {
            "video": str(output),
            "fps": fps,
            "frames": frames_written,
            "duration_seconds": frames_written / fps,
            "segments": segments,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "limitations": (
                "Hypothesized friction; foam is rigid geometry with compliant contacts, "
                "not deformable foam."
            ),
            "editing": (
                "Pours shown at 1x after physical pickup; synchronized at tilt start. "
                "Reset is explicit human intervention."
            ),
        }
        (directory / "material-comparison-results.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return result

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output-dir", required=True, type=Path)
        parser.add_argument("--material", choices=["wood", "foam"])
        parser.add_argument("--compile-only", action="store_true")
        args = parser.parse_args()
        if not args.compile_only:
            for material in [args.material] if args.material else ["wood", "foam"]:
                MaterialPouringDemo.render(directory=args.output_dir, material=material)
        if args.compile_only or args.material is None:
            MaterialPouringDemo.compile(directory=args.output_dir)


if __name__ == "__main__":
    MaterialPouringDemo.main()
