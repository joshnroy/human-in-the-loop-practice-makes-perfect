"""Prepare, never submit, the two Gurobi DP arms and their independent cost pilot.

Calibration uses reserved seeds 1000--1002, never evaluation outcomes or the paired
experiment seeds. Separate plain/always-ask episode budgets match the same 2000
available cost budget approximately; actual spending is always reported as well.
"""

import argparse
import hashlib
import json
import math
import shlex
from pathlib import Path

from hitl_pmp.cli import Cli
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.practice_makes_perfect.dp_priors import DpPriorCalibration
from scripts.sweepoffshelf3d_launch import DellaGroup, ShelfLaunch
from scripts.tossing3d_launch import SIZES, Tossing3DLaunch

REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/scratch/gpfs/TSILVER/jr2860")
CALIBRATION_SEEDS = (1000, 1001, 1002)
TARGET_COST = 2000.0


class Tossing3DDpLaunch:
    @staticmethod
    def source_hashes() -> dict[str, str]:
        files = [
            *sorted((REPO / "src/hitl_pmp/methods/practice_makes_perfect").glob("*.py")),
            REPO / "src/hitl_pmp/step_protocol.py",
            REPO / "src/hitl_pmp/core/practice_costs.py",
            REPO / "scripts/tossing3d_launch.py",
            REPO / "scripts/tossing3d_dp_priors.py",
            Path(__file__).resolve(),
        ]
        return {
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        }

    @staticmethod
    def records(
        *,
        phase: str,
        tag: str,
        code_key: str,
        report: dict | None,
        prior_calibration: Path,
    ) -> list[dict]:
        if phase not in {"calibration", "smoke", "full"}:
            raise ValueError(f"unknown DP preparation phase {phase!r}")
        if not code_key or "/" in code_key or code_key in {".", ".."}:
            raise ValueError("code key must name one existing isolated code snapshot")
        prior = DpPriorCalibration.load(path=prior_calibration)
        if (prior.environment, prior.layout, prior.variant) != ("tossing3d", "barrier", "o1"):
            raise ValueError("DP launch requires Tossing3D barrier/o1 prior calibration")
        provider = Tossing3DSkillProvider(env=Tossing3DEnvironment())
        prior.verify_robot_skills(skills=provider.skills())
        prior_sha256 = hashlib.sha256(prior_calibration.read_bytes()).hexdigest()
        if phase != "calibration":
            if report is None or report.get("target_cost") != TARGET_COST:
                raise ValueError("both experiment arms require a completed 2000-cost calibration")
            if report.get("source_sha256") != Tossing3DDpLaunch.source_hashes():
                raise ValueError("calibration is stale: algorithm source hashes differ")
            if report.get("prior_calibration_sha256") != prior_sha256:
                raise ValueError("cost calibration is stale: skill priors differ")
        size = SIZES["full" if phase == "full" else "dry"]
        seeds = (
            CALIBRATION_SEEDS if phase == "calibration" else (range(6) if phase == "full" else [0])
        )
        code = ROOT / "code" / code_key
        records = []
        for always_ask in [False, True]:
            policy = "always_ask" if always_ask else "plain"
            template = Tossing3DLaunch.arms()["ees-always-ask" if always_ask else "ees"]
            units = ["cost"] if phase == "calibration" else ["episodes", "cost"]
            for unit in units:
                schedule = "once" if unit == "episodes" else "period"
                if unit == "cost":
                    budget = TARGET_COST
                else:
                    assert report is not None
                    budget = report["arms"][policy]["episode_budget"]
                if not isinstance(budget, (float, int)) or not math.isfinite(budget) or budget <= 0:
                    raise ValueError("calibrated DP budget must be finite and positive")
                if unit == "episodes" and int(budget) != budget:
                    raise ValueError("episode budget must be integral")
                arm = f"dp-{unit}-{schedule}" + ("-always-ask" if always_ask else "")
                for seed in seeds:
                    output = ROOT / "results" / tag / arm / "dp" / str(seed)
                    argv = Tossing3DLaunch.cli_argv(
                        arm=template,
                        size=size,
                        seed=seed,
                        output_dir=output,
                        repo=code,
                    )
                    argv[argv.index("--method") + 1] = "dp"
                    argv += [
                        "--ees-score-human-skills",
                        "--dp-solver",
                        "gurobi",
                        "--dp-budget-unit",
                        unit,
                        "--dp-allocation-schedule",
                        schedule,
                        "--dp-budget",
                        str(budget),
                        "--dp-prior-calibration",
                        str(ROOT / "launch" / tag / "prior-calibration.json"),
                    ]
                    parsed = Cli.parse_args(argv=argv)
                    assert parsed.dp_solver == "gurobi" and parsed.ees_score_human_skills
                    records.append({
                        "name": f"{tag}-{arm}-s{seed}",
                        "arm": arm,
                        "seed": seed,
                        "output": str(output),
                        "argv": argv,
                    })
        return records

    @staticmethod
    def prepare(
        *,
        directory: Path,
        phase: str,
        tag: str,
        code_key: str,
        report: dict | None,
        prior_calibration: Path,
    ) -> dict:
        records = Tossing3DDpLaunch.records(
            phase=phase,
            tag=tag,
            code_key=code_key,
            report=report,
            prior_calibration=prior_calibration,
        )
        directory.mkdir(parents=True, exist_ok=True)
        prior_bytes = prior_calibration.read_bytes()
        (directory / "prior-calibration.json").write_bytes(prior_bytes)
        manifest = (
            "\n".join(
                "\t".join([
                    record["name"],
                    code_key,
                    record["arm"],
                    str(record["seed"]),
                    record["output"],
                    shlex.join(record["argv"]),
                ])
                for record in records
            )
            + "\n"
        )
        (directory / "manifest-dp.tsv").write_text(manifest)
        sbatch = ShelfLaunch.della_sbatch(
            tag=tag,
            group="dp",
            tasks=len(records),
            limits=DellaGroup(time="24:00:00" if phase == "full" else "01:02:00"),
        )
        preamble = (
            "module load gurobi/13.0.0\n"
            'export PYTHONPATH="${DP_GUROBI_PYTHONPATH:?Set the isolated Gurobi Python directory}'
            ':${PYTHONPATH:-}"\n'
        )
        assert "set -eo pipefail\n" in sbatch
        sbatch = sbatch.replace("set -eo pipefail\n", "set -eo pipefail\n" + preamble, 1)
        (directory / "array-dp.sbatch").write_text(sbatch)
        metadata = {
            "phase": phase,
            "tag": tag,
            "code_key": code_key,
            "source_sha256": Tossing3DDpLaunch.source_hashes(),
            "records": records,
            "calibration": report,
            "prior_calibration_sha256": hashlib.sha256(prior_bytes).hexdigest(),
            "submitted": False,
        }
        (directory / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata

    @staticmethod
    def calibrate(*, metadata: dict) -> dict:
        if metadata.get("phase") != "calibration":
            raise ValueError("expected reserved-seed calibration metadata")
        if metadata["source_sha256"] != Tossing3DDpLaunch.source_hashes():
            raise ValueError("pilot source hashes differ from current source")
        grouped: dict[str, list[dict]] = {"plain": [], "always_ask": []}
        for record in metadata["records"]:
            output = Path(record["output"])
            status = json.loads((output / "status.json").read_text())
            if status["phase"] != "complete" or status["practice_steps"] != 3200:
                raise ValueError(f"calibration pilot is incomplete: {output}")
            logs = [
                json.loads(line)
                for line in (output / "dp_allocations.jsonl").read_text().splitlines()
            ]
            allocations = [entry for entry in logs if entry["event"] == "allocation"]
            if not allocations or any(
                entry.get("status") != "OPTIMAL" or entry.get("solver") != "gurobi"
                for entry in allocations
            ):
                raise ValueError(f"pilot lacks certified Gurobi allocations: {output}")
            ended = [entry for entry in logs if entry["event"] == "period_end"][-1]
            if ended["episodes"] <= 0 or not math.isclose(
                ended["physical_cost"], status["physical_cost"], abs_tol=1e-7
            ):
                raise ValueError(f"pilot episode/cost accounting is incomplete: {output}")
            policy = "always_ask" if record["arm"].endswith("always-ask") else "plain"
            grouped[policy].append({
                "seed": record["seed"],
                "episodes": ended["episodes"],
                "cost": status["physical_cost"],
                "output": str(output),
            })
        arms = {}
        for policy, pilots in grouped.items():
            if sorted(pilot["seed"] for pilot in pilots) != list(CALIBRATION_SEEDS):
                raise ValueError(
                    "calibration requires exactly reserved seeds 1000--1002 per policy"
                )
            mean = sum(pilot["cost"] for pilot in pilots) / sum(
                pilot["episodes"] for pilot in pilots
            )
            arms[policy] = {
                "cost_per_episode": mean,
                "episode_budget": max(1, math.floor(TARGET_COST / mean)),
                "pilots": pilots,
            }
        return {
            "target_cost": TARGET_COST,
            "source_sha256": metadata["source_sha256"],
            "rule": "floor(2000 / pooled reserved-pilot cost per completed target rollout)",
            "arms": arms,
            "uses_evaluation_scores": False,
            "prior_calibration_sha256": metadata["prior_calibration_sha256"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--phase", choices=["calibration", "smoke", "full"], required=True)
    prepare.add_argument("--tag", required=True)
    prepare.add_argument("--code-key", required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--calibration-report", type=Path)
    prepare.add_argument("--prior-calibration", type=Path, required=True)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--launch-json", type=Path, required=True)
    calibrate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        report = (
            json.loads(args.calibration_report.read_text()) if args.calibration_report else None
        )
        metadata = Tossing3DDpLaunch.prepare(
            directory=args.out,
            phase=args.phase,
            tag=args.tag,
            code_key=args.code_key,
            report=report,
            prior_calibration=args.prior_calibration,
        )
        print(f"Prepared {len(metadata['records'])} {args.phase} tasks; submitted nothing")
    else:
        report = Tossing3DDpLaunch.calibrate(metadata=json.loads(args.launch_json.read_text()))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["arms"], indent=2))


if __name__ == "__main__":
    main()
