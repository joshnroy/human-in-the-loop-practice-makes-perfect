"""Audit completed wood maps and export compact results and comparison figures."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.pouring3d_wood_regions import WoodRegions


class WoodRegionReport:
    @staticmethod
    def run(*, root: Path) -> dict[str, Any]:
        repo = Path(__file__).resolve().parents[1]
        summary = json.loads((root / "summary.json").read_text())
        follow = root / "shared-one-followup"
        additional = WoodRegions.rows(root=follow, phase="refine")
        extra_checks = [
            {**json.loads(p.read_text()), "source_path": str(p)}
            for p in follow.glob("verify-shared-s*/*/cell-*/result.json")
        ]
        assert len(summary["placement"]) == 9 and len(summary["verification"]) == 6
        assert len(additional) == 16 and len(extra_checks) == 2
        mapping = summary["mapping"] + additional
        all_rows = summary["placement"] + mapping + summary["verification"] + extra_checks
        native_hashes = all_rows[0]["source_sha256"]
        checked = 0
        for row in all_rows:
            assert row["source_sha256"] == native_hashes
            assert row["initial"]["locations"] == ["pitcher"] * 5
            snapshots = [row[k] for k in ("initial", "after_pour", "final")]
            for event in row["events"]:
                snapshots.extend([event["before"], event["after"]])
                if event["skill"] == "Pour":
                    snapshots.extend([event["wiggle_before"], event["wiggle_after"]])
            for snap in snapshots:
                k = row["requested_count"]
                expected = (
                    snap["settled"]
                    and snap["locations"].count("tray") == k
                    and snap["locations"].count("pitcher") == 5 - k
                )
                assert snap["goal"] == expected
                checked += 1
            assert row["pour_success"] == (
                row["after_pour"]["goal"] and row["events"][1]["error"] is None
            )
        for filename, sha in native_hashes.items():
            assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == sha
        saved = json.loads((root / "interruption-saved-cells.json").read_text())
        for filename, sha in saved["saved_files_sha256"].items():
            assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == sha
        for row in summary["placement"]:
            assert row["events"][-1]["success"]
        for row in summary["verification"] + extra_checks:
            assert row["pour_success"] and row["full_success"]
        shared_maps = []
        points = [(a, d) for a in (93.75, 94, 94.25) for d in (5.5, 6, 6.5)]
        figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        for seed, axis in enumerate(axes):
            lookup = {WoodRegions.pair(row=r): r for r in mapping if r["seed"] == seed}
            counts = [lookup[p]["eligible_count"] for p in points]
            hits = [p for p, c in zip(points, counts, strict=True) if c == 1]
            blocks = [
                [a, a + 0.25, d, d + 0.5]
                for a in (93.75, 94)
                for d in (5.5, 6)
                if all((aa, dd) in hits for aa in (a, a + 0.25) for dd in (d, d + 0.5))
            ]
            shared_maps.append({
                "seed": seed,
                "center": [94, 6],
                "points": points,
                "counts": counts,
                "successful_settings": len(hits),
                "sampled_2x2_blocks": blocks,
            })
            for (angle, duration), count in zip(points, counts, strict=True):
                axis.scatter(duration, angle, s=280, color="#CC79A7" if count == 1 else "#dddddd")
                axis.annotate(str(count), (duration, angle), ha="center", va="center")
            axis.set(
                title=f"Start {seed} | {len(hits)}/9 one-cube settings",
                xlabel="Wiggle duration (s)",
                ylabel="Held tilt (degrees)",
            )
        figure.suptitle("Shared slower one-cube recipe: 94 degrees, 6 seconds")
        figure.savefig(root / "wood-shared-one.png", dpi=150)
        plt.close(figure)
        common = {}
        for target in (1, 2):
            sets = [
                {
                    WoodRegions.pair(row=r)
                    for r in mapping
                    if r["seed"] == seed and r["eligible_count"] == target
                }
                for seed in range(3)
            ]
            common[str(target)] = sorted(set.intersection(*sets))
        shared_tested = sorted(
            set.intersection(*[
                {WoodRegions.pair(row=r) for r in mapping if r["seed"] == seed} for seed in range(3)
            ])
        )
        out = {
            "protocol": summary["protocol"],
            "shared_protocol": json.loads((follow / "protocol.json").read_text()),
            "continuation_protocol": json.loads((root / "continuation-protocol.json").read_text()),
            "primary_maps": summary["maps"],
            "shared_one_maps": shared_maps,
            "shared_successful_settings_across_three_starts": common,
            "shared_tested_settings_across_three_starts": shared_tested,
            "completed_native_episodes": len(all_rows),
            "goal_snapshots_audited": checked,
            "preserved_pre_interruption_files": len(saved["saved_files_sha256"]),
            "interrupted_attempts": [
                str(p.parent.relative_to(root)) for p in root.rglob("interruption.json")
            ],
            "native_source_sha256": native_hashes,
            "outcomes": [
                {
                    **{
                        k: r[k]
                        for k in (
                            "seed",
                            "requested_count",
                            "parameters",
                            "eligible_count",
                            "pour_success",
                            "full_success",
                            "source_path",
                        )
                    },
                    "locations": r["after_pour"]["locations"],
                    "settled": r["after_pour"]["settled"],
                    "events": [
                        {k: e[k] for k in ("skill", "success", "error")} for e in r["events"]
                    ],
                }
                for r in all_rows
            ],
        }
        (root / "audited-summary.json").write_text(json.dumps(out, indent=2) + "\n")
        for original, name in [
            ("wood-regions.png", "pouring3d-wood-regions.png"),
            ("wood-patches.png", "pouring3d-wood-patches.png"),
            ("placement-repair.png", "pouring3d-wood-placement-repair.png"),
            ("wood-shared-one.png", "pouring3d-wood-shared-one.png"),
        ]:
            shutil.copy2(root / original, repo / "docs/figures" / name)

        def portable(*, value: Any) -> Any:
            if isinstance(value, dict):
                return {portable(value=k): portable(value=v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [portable(value=v) for v in value]
            if isinstance(value, str):
                for base in (repo, root):
                    if value.startswith(str(base) + "/"):
                        return str(Path(value).relative_to(base))
            return value

        (repo / "docs/experiment-logs/data/pouring3d-wood-regions.json").write_text(
            json.dumps(portable(value=out), indent=2) + "\n"
        )
        print(
            json.dumps({
                "shared_one": [[m["seed"], m["successful_settings"]] for m in shared_maps],
                "common": common,
                "episodes": len(all_rows),
                "audited_snapshots": checked,
            })
        )
        return out

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--root", type=Path, required=True)
        WoodRegionReport.run(root=parser.parse_args().root)


if __name__ == "__main__":
    WoodRegionReport.main()
