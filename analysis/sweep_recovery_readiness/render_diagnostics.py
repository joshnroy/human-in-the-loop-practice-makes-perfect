"""Replay measured recovery checks, explicitly labeled constructed diagnostic fixtures."""

import json
import shutil
import subprocess
import sys
from pathlib import Path


class RecoveryVideos:
    @staticmethod
    def run():
        base = Path("scratchpad/recovery-final")
        cases = {
            "drawer-wiggle": ("test_blocked_pick_is_not_offer0", "drawer"),
            "three-cube-row": ("test_three_cube_row_tracks_all0", "floor"),
            "dropped-wiper": ("test_valid_start_dropped_wiper0", "floor"),
        }
        for name, (fixture, view) in cases.items():
            source = base / fixture / "episodes/000000-seed6/replay.jsonl"
            output = Path("scratchpad/recovery-videos") / name
            output.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, output / "replay_log.jsonl")
            steps = []
            previous = None
            last_tick = 0
            for line in source.open():
                record = json.loads(line)
                if record.get("kind") != "tick" or record["t"] == 0:
                    continue
                step = record["step"] or "constructed diagnostic fixture setup"
                if step != previous:
                    steps.append({
                        "name": step,
                        "phase": "diagnostic fixture (not experiment)",
                        "ticks": 0,
                    })
                    previous = step
                steps[-1]["ticks"] += record["t"] - last_tick
                last_tick = record["t"]
            (output / "cycle.json").write_text(json.dumps({"steps": steps}, indent=2))
            subprocess.run(
                [
                    sys.executable,
                    "scripts/render_sweep_drawer_reset.py",
                    "--run-dir",
                    str(output),
                    "--output",
                    str(output / "demo.mp4"),
                    "--fps",
                    "10",
                    "--close-up",
                    view,
                    "--ffmpeg",
                    "/home/josh/miniconda3/envs/ffnv/bin/ffmpeg",
                ],
                check=True,
            )


if __name__ == "__main__":
    RecoveryVideos.run()
