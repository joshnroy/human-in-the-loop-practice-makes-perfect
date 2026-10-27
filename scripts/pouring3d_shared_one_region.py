"""A bounded neighboring-setting check around a shared slow one-cube recipe."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from scripts.pouring3d_wiggle_demo import WiggleDemo
from scripts.pouring3d_wood_regions import WoodRegions
from scripts.run_sweep import SweepRun, SweepRunner


class SharedOneRegion:
    @staticmethod
    def launch(*, root: Path) -> None:
        rows = WoodRegions.rows(root=root, phase="coarse")
        shared = set.intersection(*[
            {
                WoodRegions.pair(row=r)
                for r in rows
                if r["seed"] == seed and r["eligible_count"] == 1
            }
            for seed in range(3)
        ])
        center = min(shared, key=lambda p: (p[1], p[0]))
        patches = json.loads((root / "patches.json").read_text())
        assert patches["2"]["1"]["center"] == list(center)
        dest = root / "shared-one-followup"
        dest.mkdir(exist_ok=False)
        source = dest / "source"
        source.mkdir()
        active = json.loads((root / "continuation-protocol.json").read_text())
        files = dict(active["source_sha256"])
        files[str(Path(__file__).resolve())] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        for path in files:
            shutil.copy2(path, source / Path(path).name)
        plans = {}
        for seed in (0, 1):
            patch = WoodRegions.patch(
                rows=[
                    {
                        "eligible_count": 1,
                        "parameters": {"tilt_angle": center[0], "wiggle_seconds": center[1]},
                    }
                ],
                target=1,
            )
            patch["new_points"] = [p for p in patch["points"] if p != list(center)]
            plans[str(seed)] = {"1": patch}
        (dest / "patches.json").write_text(json.dumps(plans, indent=2))
        (dest / "protocol.json").write_text(
            json.dumps(
                {
                    "material": "wood",
                    "target": 1,
                    "center": center,
                    "selection": "Shared one-cube setting; shortest duration then lower angle",
                    "new_mapping_trials": 16,
                    "new_explicit_verification_trials": 2,
                    "seed_two_patch_reused_from_primary_map": True,
                    "source_sha256": files,
                },
                indent=2,
            )
        )
        repo = Path(__file__).resolve().parents[1]
        outcomes = SweepRunner.execute(
            runs=[
                SweepRun(
                    method=f"shared-one-s{seed}",
                    seed=seed,
                    output_dir=dest / f"shared-one-s{seed}" / str(seed),
                    command=[
                        "run-heavy",
                        str(repo / "scripts/with_env.sh"),
                        sys.executable,
                        "-m",
                        "scripts.pouring3d_shared_one_region",
                        "--root",
                        str(root),
                        "--worker",
                        str(seed),
                    ],
                )
                for seed in (0, 1)
            ],
            max_workers=2,
        )
        if not all(o.succeeded for o in outcomes):
            raise SystemExit(1)

    @staticmethod
    def worker(*, root: Path, seed: int) -> None:
        dest = root / "shared-one-followup"
        protocol = json.loads((dest / "protocol.json").read_text())
        WoodRegions.worker(root=dest, phase="refine", seed=seed, group=1)
        angle, duration = protocol["center"]
        result = WiggleDemo.run(
            directory=dest / f"verify-shared-s{seed}" / str(seed) / "cell-0",
            material="wood",
            angle=angle,
            duration=duration,
            mode="lateral",
            seed=seed,
            requested_count=1,
        )
        assert result["pour_success"] and result["full_success"]
        old = next(
            r
            for r in WoodRegions.rows(root=root, phase="coarse")
            if r["seed"] == seed and WoodRegions.pair(row=r) == (angle, duration)
        )
        for name in ("initial", "after_pour"):
            for field in ("qpos", "qvel", "ctrl", "time", "locations", "settled"):
                assert result[name][field] == old[name][field]

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--worker", type=int, choices=(0, 1))
        args = parser.parse_args()
        if args.worker is None:
            SharedOneRegion.launch(root=args.root)
        else:
            SharedOneRegion.worker(root=args.root, seed=args.worker)


if __name__ == "__main__":
    SharedOneRegion.main()
