"""Six separately reported foam probes for duration sensitivity and count gaps."""

import argparse
import json
import sys
from pathlib import Path
from typing import ClassVar

from scripts.pouring3d_wiggle_demo import WiggleDemo
from scripts.run_sweep import SweepRun, SweepRunner


class SidewaysProbes:
    PLANS: ClassVar[tuple[tuple[int, float, float], ...]] = (
        (4, 106.75, 12),
        (1, 105, 4),
        (1, 105, 8),
        (1, 105, 10),
        (1, 104.75, 20),
        (4, 106.75, 20),
    )

    @staticmethod
    def worker(*, root: Path) -> None:
        for index, (target, angle, duration) in enumerate(SidewaysProbes.PLANS):
            directory = root / "foam/0/probes" / str(index)
            if directory.exists():
                raise ValueError("Never overwrite a targeted probe")
            WiggleDemo.run(
                directory=directory,
                material="foam",
                angle=angle,
                duration=duration,
                mode="lateral",
                requested_count=target,
            )

    @staticmethod
    def launch(*, root: Path) -> None:
        protocol = root / "targeted-probes-protocol.json"
        if protocol.exists():
            raise ValueError("Targeted probes already planned")
        protocol.write_text(
            json.dumps(
                {
                    "material": "foam",
                    "seed": 0,
                    "plans": SidewaysProbes.PLANS,
                    "budget": len(SidewaysProbes.PLANS),
                    "reason": "Probe missing one/four and duration sensitivity",
                    "separate_from_frozen_validation": True,
                    "changes_frozen_recipes": False,
                },
                indent=2,
            )
        )
        repo = Path(__file__).resolve().parents[1]
        outcomes = SweepRunner.execute(
            runs=[
                SweepRun(
                    method="probe-foam",
                    seed=0,
                    output_dir=root / "probe-foam/0",
                    command=[
                        "run-heavy",
                        str(repo / "scripts/with_env.sh"),
                        sys.executable,
                        "-m",
                        "scripts.pouring3d_sideways_probes",
                        "--root",
                        str(root),
                        "--worker",
                    ],
                )
            ],
            max_workers=1,
        )
        if not all(outcome.succeeded for outcome in outcomes):
            raise SystemExit(1)

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--worker", action="store_true")
        args = parser.parse_args()
        if args.worker:
            SidewaysProbes.worker(root=args.root)
        else:
            SidewaysProbes.launch(root=args.root)


if __name__ == "__main__":
    SidewaysProbes.main()
