"""Repair-check fixed placement and map wood one/two-cube parameter regions."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, ClassVar

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from scripts.pouring3d_wiggle_demo import WiggleDemo
from scripts.run_sweep import SweepRun, SweepRunner


class WoodRegions:
    ANGLES: ClassVar[tuple[float, ...]] = (94, 94.75, 95.5, 96.25, 97)
    DURATIONS: ClassVar[tuple[float, ...]] = (0.5, 2.5, 6, 12)
    PLACEMENTS: ClassVar[tuple[tuple[int, int, float, float], ...]] = (
        (0, 1, 95.5, 2.5),
        (1, 2, 95.5, 12),
        (1, 3, 97, 2),
        (0, 4, 97, 6),
        (1, 4, 97, 6),
        (2, 4, 97, 6),
        (0, 5, 97, 12),
        (1, 5, 97, 12),
        (2, 5, 97, 12),
    )

    @staticmethod
    def scores(*, row: dict[str, Any], target: int) -> bool:
        return row["eligible_count"] == target

    @staticmethod
    def pair(*, row: dict[str, Any]) -> tuple[float, float]:
        return row["parameters"]["tilt_angle"], row["parameters"]["wiggle_seconds"]

    @staticmethod
    def patch(*, rows: list[dict[str, Any]], target: int) -> dict[str, Any]:
        exact = [r for r in rows if WoodRegions.scores(row=r, target=target)]
        if exact:
            hits = {WoodRegions.pair(row=r) for r in exact}

            def rank(row: dict[str, Any]) -> tuple[Any, ...]:  # noqa: PLR0917
                a, d = WoodRegions.pair(row=row)
                i, j = WoodRegions.ANGLES.index(a), WoodRegions.DURATIONS.index(d)
                neighbors = [
                    (WoodRegions.ANGLES[ii], WoodRegions.DURATIONS[jj])
                    for ii, jj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                    if 0 <= ii < len(WoodRegions.ANGLES) and 0 <= jj < len(WoodRegions.DURATIONS)
                ]
                return -sum(p in hits for p in neighbors), d, a

            center = WoodRegions.pair(row=min(exact, key=rank))
        else:
            eligible = [r for r in rows if r["eligible_count"] is not None]
            candidates = []
            for left in eligible:
                for right in eligible:
                    if not left["eligible_count"] < target < right["eligible_count"]:
                        continue
                    a, d = WoodRegions.pair(row=left)
                    b, e = WoodRegions.pair(row=right)
                    if a != b and d != e:
                        continue
                    width = abs(a - b) / 0.75 + abs(d - e) / 2
                    candidates.append((
                        right["eligible_count"] - left["eligible_count"],
                        width,
                        round((a + b) / 2, 4),
                        round((d + e) / 2, 1),
                    ))
            if candidates:
                *_, a, d = min(candidates)
                center = a, d
            else:
                row = min(eligible, key=lambda r: abs(r["eligible_count"] - target))
                a, d = WoodRegions.pair(row=row)
                center = a + (0.25 if row["eligible_count"] < target else -0.25), d
        angle, duration = center
        points = sorted({
            (
                float(round(np.clip(angle + da, 0, 150), 4)),
                float(round(np.clip(duration + dt, 0.1, 20), 1)),
            )
            for da in (-0.25, 0, 0.25)
            for dt in (-0.5, 0, 0.5)
        })
        return {
            "center": list(center),
            "points": [list(p) for p in points],
            "coarse_exact_found": bool(exact),
        }

    @staticmethod
    def rows(*, root: Path, phase: str) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(root.glob(f"{phase}-s*-g*/*/cell-*/result.json")):
            rows.append({**json.loads(path.read_text()), "source_path": str(path)})
        return rows

    @staticmethod
    def worker(*, root: Path, phase: str, seed: int, group: int) -> None:
        protocol = json.loads((root / "protocol.json").read_text())
        continuation = root / "continuation-protocol.json"
        sources = json.loads(continuation.read_text()) if continuation.exists() else protocol
        for path, sha in sources["source_sha256"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha, path
        directory = root / f"{phase}-s{seed}-g{group}" / str(seed)
        if phase == "placement":
            _, target, angle, duration = WoodRegions.PLACEMENTS[group]
            plans = [(target, angle, duration)]
        elif phase == "coarse":
            plans = [(2, WoodRegions.ANGLES[group], d) for d in WoodRegions.DURATIONS]
        elif phase == "refine":
            plan = json.loads((root / "patches.json").read_text())[str(seed)][str(group)]
            plans = [(2, a, d) for a, d in plan["new_points"]]
        else:
            plan = json.loads((root / "verification.json").read_text())[str(seed)][str(group)]
            plans = [] if plan is None else [(group, *WoodRegions.pair(row=plan))]
        for index, (target, angle, duration) in enumerate(plans):
            dest = directory / f"cell-{index}"
            if (dest / "result.json").exists():
                saved = json.loads((dest / "result.json").read_text())
                assert saved["seed"] == seed and saved["requested_count"] == target
                assert saved["parameters"] == {"tilt_angle": angle, "wiggle_seconds": duration}
                continue
            if dest.exists():
                serial = 1
                while dest.with_name(f"{dest.name}-interrupted-{serial}").exists():
                    serial += 1
                interrupted = dest.with_name(f"{dest.name}-interrupted-{serial}")
                dest.rename(interrupted)
                (interrupted / "interruption.json").write_text(
                    json.dumps(
                        {
                            "reason": "Launcher terminated before result was saved",
                            "phase": phase,
                            "seed": seed,
                            "target": target,
                            "tilt_angle": angle,
                            "wiggle_seconds": duration,
                        },
                        indent=2,
                    )
                )
            row = WiggleDemo.run(
                directory=dest,
                material="wood",
                angle=angle,
                duration=duration,
                mode="lateral",
                seed=seed,
                requested_count=target,
                place=phase in {"placement", "verify"},
            )
            if phase in {"placement", "verify"}:
                if phase == "placement":
                    old = Path(protocol["previous_study"]) / "wood" / str(seed)
                    source = old / "validation" / str(target) / "result.json"
                else:
                    source = Path(plan["source_path"])
                previous = json.loads(source.read_text())
                for name in ("initial", "after_pour"):
                    for field in ("qpos", "qvel", "ctrl", "time", "locations", "settled"):
                        assert row[name][field] == previous[name][field], (name, field)
                assert row["events"][-1]["skill"] == "PlacePitcherAtHome"
                assert row["events"][-1]["success"], row["final"]["bodies"]["pitcher"]
                if phase == "verify":
                    assert row["pour_success"] and row["full_success"]

    @staticmethod
    def batch(*, root: Path, phase: str, groups: list[tuple[int, int]]) -> None:
        repo = Path(__file__).resolve().parents[1]
        if phase in {"refine", "verify"}:
            planfile = "patches.json" if phase == "refine" else "verification.json"
            plans = json.loads((root / planfile).read_text())
            pending = []
            for seed, group in groups:
                plan = plans[str(seed)][str(group)]
                expected = len(plan["new_points"]) if phase == "refine" else int(plan is not None)
                directory = root / f"{phase}-s{seed}-g{group}" / str(seed)
                if any(not (directory / f"cell-{i}/result.json").exists() for i in range(expected)):
                    pending.append((seed, group))
            groups = pending
        if not groups:
            return
        outcomes = SweepRunner.execute(
            runs=[
                SweepRun(
                    method=f"{phase}-s{s}-g{g}",
                    seed=s,
                    output_dir=root / f"{phase}-s{s}-g{g}" / str(s),
                    command=[
                        "run-heavy",
                        str(repo / "scripts/with_env.sh"),
                        sys.executable,
                        "-m",
                        "scripts.pouring3d_wood_regions",
                        "--root",
                        str(root),
                        "--worker",
                        phase,
                        "--seed",
                        str(s),
                        "--group",
                        str(g),
                    ],
                )
                for s, g in groups
            ],
            max_workers=2,
        )
        if not all(o.succeeded for o in outcomes):
            raise SystemExit(1)

    @staticmethod
    def launch(*, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=False)
        repo = Path(__file__).resolve().parents[1]
        paths = [
            Path(__file__).resolve(),
            repo / "scripts/pouring3d_wiggle_demo.py",
            repo / "src/hitl_pmp/environments/pouring3d/lateral_wiggle.py",
            repo / "src/hitl_pmp/environments/pouring3d/native_controller.py",
            repo / "src/hitl_pmp/environments/pouring3d/lateral_environment.py",
            repo / "reference/kindergarden/src/kinder/envs/dynamic3d/pitcher_pour3d.py",
            repo
            / "reference/kindergarden/src/kinder/envs/dynamic3d/tasks/PitcherPour3D"
            / "PitcherPour3D-o5-winco.json",
        ]
        archive = root / "source"
        archive.mkdir()
        for path in paths:
            shutil.copy2(path, archive / path.name)
        (root / "protocol.json").write_text(
            json.dumps(
                {
                    "material": "wood",
                    "seeds": [0, 1, 2],
                    "targets": [1, 2],
                    "angles": WoodRegions.ANGLES,
                    "durations": WoodRegions.DURATIONS,
                    "coarse_trials": 60,
                    "maximum_refinement_trials": 54,
                    "maximum_verification_trials": 6,
                    "placement_trials": 9,
                    "patch_delta_degrees": 0.25,
                    "patch_delta_seconds": 0.5,
                    "map_requested_count": 2,
                    "map_scoring": (
                        "Physical count scores both targets; explicit verification follows"
                    ),
                    "count_feedback_during_pour": False,
                    "previous_study": str(repo.parent / "pouring3d-sideways-counts-20261026"),
                    "source_sha256": {
                        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
                    },
                },
                indent=2,
            )
        )
        WoodRegions.batch(
            root=root,
            phase="placement",
            groups=[(p[0], i) for i, p in enumerate(WoodRegions.PLACEMENTS)],
        )
        WoodRegions.batch(
            root=root,
            phase="coarse",
            groups=[(s, a) for s in range(3) for a in range(len(WoodRegions.ANGLES))],
        )
        WoodRegions.finish(root=root)

    @staticmethod
    def resume(*, root: Path) -> None:
        assert len(WoodRegions.rows(root=root, phase="placement")) == 9
        assert len(WoodRegions.rows(root=root, phase="coarse")) == 60
        original = json.loads((root / "protocol.json").read_text())
        driver = Path(__file__).resolve()
        sources = {}
        for filename, expected in original["source_sha256"].items():
            actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
            if Path(filename) != driver:
                assert actual == expected, filename
            sources[filename] = actual
        serial = len(list(root.glob("continuation-source*"))) + 1
        archive = root / f"continuation-source-{serial}"
        archive.mkdir(exist_ok=False)
        shutil.copy2(driver, archive / driver.name)
        active = root / "continuation-protocol.json"
        if active.exists():
            shutil.copy2(active, root / f"continuation-protocol-{serial - 1}.json")
        (root / "continuation-protocol.json").write_text(
            json.dumps(
                {
                    "reason": "Resume saved cells; preserve completed and interrupted attempts",
                    "source_sha256": sources,
                    "native_sources_unchanged": True,
                },
                indent=2,
            )
        )
        WoodRegions.finish(root=root)

    @staticmethod
    def finish(*, root: Path) -> None:
        coarse = WoodRegions.rows(root=root, phase="coarse")
        patches: dict[str, Any] = {}
        for seed in range(3):
            rows = [r for r in coarse if r["seed"] == seed]
            used = {WoodRegions.pair(row=r) for r in rows}
            patches[str(seed)] = {}
            for target in (1, 2):
                patch = WoodRegions.patch(rows=rows, target=target)
                patch["new_points"] = [p for p in patch["points"] if tuple(p) not in used]
                used.update(tuple(p) for p in patch["new_points"])
                patches[str(seed)][str(target)] = patch
        if (root / "patches.json").exists():
            assert json.loads((root / "patches.json").read_text()) == patches
        else:
            (root / "patches.json").write_text(json.dumps(patches, indent=2))
        WoodRegions.batch(
            root=root, phase="refine", groups=[(s, k) for s in range(3) for k in (1, 2)]
        )
        rows = coarse + WoodRegions.rows(root=root, phase="refine")
        verification: dict[str, Any] = {}
        for seed in range(3):
            verification[str(seed)] = {}
            for target in (1, 2):
                center = patches[str(seed)][str(target)]["center"]
                hits = [
                    r
                    for r in rows
                    if r["seed"] == seed and WoodRegions.scores(row=r, target=target)
                ]
                chosen = (
                    min(
                        hits,
                        key=lambda r: (
                            abs(WoodRegions.pair(row=r)[0] - center[0]) / 0.25
                            + abs(WoodRegions.pair(row=r)[1] - center[1]) / 0.5
                        ),
                    )
                    if hits
                    else None
                )
                verification[str(seed)][str(target)] = chosen
        (root / "verification.json").write_text(json.dumps(verification, indent=2))
        WoodRegions.batch(
            root=root, phase="verify", groups=[(s, k) for s in range(3) for k in (1, 2)]
        )
        WoodRegions.report(root=root)

    @staticmethod
    def report(*, root: Path) -> None:
        coarse = WoodRegions.rows(root=root, phase="coarse")
        refined = WoodRegions.rows(root=root, phase="refine")
        patches = json.loads((root / "patches.json").read_text())
        rows = coarse + refined
        maps = []
        for seed in range(3):
            lookup = {WoodRegions.pair(row=r): r for r in rows if r["seed"] == seed}
            for target in (1, 2):
                patch = patches[str(seed)][str(target)]
                scores = [
                    WoodRegions.scores(row=lookup[tuple(p)], target=target) for p in patch["points"]
                ]
                successful = {
                    tuple(p) for p, good in zip(patch["points"], scores, strict=True) if good
                }
                angles = sorted({p[0] for p in patch["points"]})
                durations = sorted({p[1] for p in patch["points"]})
                blocks = [
                    [angles[i], angles[i + 1], durations[j], durations[j + 1]]
                    for i in range(len(angles) - 1)
                    for j in range(len(durations) - 1)
                    if all(
                        (a, d) in successful
                        for a in angles[i : i + 2]
                        for d in durations[j : j + 2]
                    )
                ]
                maps.append({
                    "seed": seed,
                    "target": target,
                    "center": patch["center"],
                    "coarse_successes": sum(
                        WoodRegions.scores(row=r, target=target)
                        for r in coarse
                        if r["seed"] == seed
                    ),
                    "coarse_trials": 20,
                    "patch_successes": sum(scores),
                    "patch_trials": len(scores),
                    "patch_points": patch["points"],
                    "patch_scores": scores,
                    "successful_sampled_2x2_blocks": blocks,
                })
        summary = {
            "protocol": json.loads((root / "protocol.json").read_text()),
            "maps": maps,
            "placement": WoodRegions.rows(root=root, phase="placement"),
            "verification": WoodRegions.rows(root=root, phase="verify"),
            "mapping": rows,
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
        fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
        for m in maps:
            ax = axes[m["seed"], m["target"] - 1]
            subset = [r for r in rows if r["seed"] == m["seed"]]
            ax.scatter(
                [WoodRegions.pair(row=r)[1] for r in subset],
                [WoodRegions.pair(row=r)[0] for r in subset],
                c=[
                    "#CC79A7" if WoodRegions.scores(row=r, target=m["target"]) else "#dddddd"
                    for r in subset
                ],
                s=65,
            )
            ax.set(
                title=(
                    f"Start {m['seed']}, target {m['target']} | "
                    f"coarse {m['coarse_successes']}/20, "
                    f"patch {m['patch_successes']}/{m['patch_trials']}"
                ),
                xlabel="Wiggle duration (s)",
                ylabel="Held tilt (degrees)",
            )
            ax.grid(alpha=0.2)
        fig.suptitle("Wood sideways Pour: purple = exact settled contained count")
        fig.savefig(root / "wood-regions.png", dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(3, 2, figsize=(10, 9), constrained_layout=True)
        for m in maps:
            ax = axes[m["seed"], m["target"] - 1]
            lookup = {WoodRegions.pair(row=r): r for r in rows if r["seed"] == m["seed"]}
            for p, success in zip(m["patch_points"], m["patch_scores"], strict=True):
                count = lookup[tuple(p)]["eligible_count"]
                ax.scatter(p[1], p[0], s=250, color="#CC79A7" if success else "#dddddd")
                ax.annotate(str(count), (p[1], p[0]), ha="center", va="center")
            ax.set(
                title=(
                    f"Start {m['seed']}, target {m['target']} | "
                    f"{m['patch_successes']}/{m['patch_trials']}"
                ),
                xlabel="Wiggle duration (s)",
                ylabel="Held tilt (degrees)",
            )
        fig.suptitle("Local neighboring settings: numbers are delivered cube counts")
        fig.savefig(root / "wood-patches.png", dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        before = []
        after = []
        labels = []
        for row in summary["placement"]:
            seed, target = row["seed"], row["requested_count"]
            previous = Path(summary["protocol"]["previous_study"])
            old = json.loads(
                (
                    previous / "wood" / str(seed) / "validation" / str(target) / "result.json"
                ).read_text()
            )
            before.append(old["final"]["bodies"]["pitcher"])
            after.append(row["final"]["bodies"]["pitcher"])
            labels.append(f"s{seed}/k{target}")
        for axis, metric, limit in zip(axes, ("rotation", "position"), (5, 25), strict=True):
            for bodies, color, label in [
                (before, "#999999", "Previous fixed return (9 trials)"),
                (after, "#CC79A7", "Repaired fixed return (9 trials)"),
            ]:
                values = []
                for body in bodies:
                    if metric == "rotation":
                        q = np.array(body["quaternion"])
                        values.append(np.rad2deg(Rotation.from_quat(q[[1, 2, 3, 0]]).magnitude()))
                    else:
                        values.append(
                            1000 * np.linalg.norm(np.array(body["position"]) - [0.6, -0.25, 0])
                        )
                axis.plot(range(len(values)), values, "o", color=color, label=label)
            axis.axhline(limit, color="#999999", linestyle=":")
            axis.set(
                xticks=range(len(labels)),
                xticklabels=labels,
                ylabel="Rotation error (degrees)"
                if metric == "rotation"
                else "Position error (mm)",
            )
            axis.tick_params(axis="x", rotation=45)
            axis.legend(fontsize=7)
        fig.suptitle("Wood: fixed placement repair, unchanged home tolerances")
        fig.savefig(root / "placement-repair.png", dpi=150)
        plt.close(fig)

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--worker", choices=("placement", "coarse", "refine", "verify"))
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--group", type=int, default=0)
        parser.add_argument("--report-only", action="store_true")
        parser.add_argument("--resume", action="store_true")
        args = parser.parse_args()
        if args.worker:
            WoodRegions.worker(root=args.root, phase=args.worker, seed=args.seed, group=args.group)
        elif args.report_only:
            WoodRegions.report(root=args.root)
        elif args.resume:
            WoodRegions.resume(root=args.root)
        else:
            WoodRegions.launch(root=args.root)


if __name__ == "__main__":
    WoodRegions.main()
