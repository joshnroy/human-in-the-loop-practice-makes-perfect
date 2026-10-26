"""Generate a short shared-learning validation array; never submit or run experiments.

Only the validation phase is supported. The verify command is a read-only worker
guard, run before the canonical Tossing3D worker's physics preflight and watchdog.
"""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from hitl_pmp.cli import Cli
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.practice_makes_perfect.dp_priors import DpPriorCalibration
from scripts.sweepoffshelf3d_launch import DellaGroup, ShelfLaunch
from scripts.tossing3d_dp_priors import SIMULATOR_PINS
from scripts.tossing3d_launch import DELLA, PILOT, Tossing3DLaunch

REPO = Path(__file__).resolve().parents[1]
VARIANTS = {
    "baseline": ("grounded", "observed"),
    "shared_only": ("parent", "observed"),
    "flexible_only": ("grounded", "all"),
    "shared_flexible": ("parent", "all"),
    "shared_rate": ("rate", "all"),
}
ALGORITHMS = {
    "ees": ("ees", None, None, None),
    "dp-episodes-once": ("dp", "episodes", "once", 350),
    "dp-cost-period": ("dp", "cost", "period", 2000),
}
SEEDS = (0, 1)
GROUP = "shared-learning"
ROBOT_SKILLS = {"PickCube", "MoveToTossLocationAndToss", "OpenGripper"}


class Tossing3DSharedLearningValidation:
    @staticmethod
    def validate_prior(*, path: Path) -> DpPriorCalibration:
        calibration = DpPriorCalibration.model_validate_json(path.read_text())
        if (calibration.environment, calibration.layout, calibration.variant) != (
            "tossing3d",
            "barrier",
            "o1",
        ):
            raise ValueError("prior calibration must describe Tossing3D barrier/o1")
        if calibration.simulator_pins != SIMULATOR_PINS:
            raise ValueError("prior calibration must retain the original simulator pins")
        return calibration

    @staticmethod
    def calibration(*, report: Path, prior: Path) -> dict:
        """Check historical evidence without asserting unchanged algorithm hashes."""
        evidence = json.loads(report.read_text())
        if (
            evidence.get("target_cost") != 2000
            or evidence.get("arms", {}).get("plain", {}).get("episode_budget") != 350
            or evidence.get("uses_evaluation_scores") is not False
            or not evidence.get("source_sha256")
        ):
            raise ValueError("expected the existing plain 350-episode / 2000-cost report")
        digest = hashlib.sha256(prior.read_bytes()).hexdigest()
        if evidence.get("prior_calibration_sha256") != digest:
            raise ValueError("prior calibration digest differs from the existing report")
        Tossing3DSharedLearningValidation.validate_prior(path=prior)
        return evidence

    @staticmethod
    def records(
        *,
        tag: str,
        sha: str,
        variants: tuple[str, ...] = tuple(VARIANTS),
        seeds: tuple[int, ...] = SEEDS,
        phase: str = "validation",
    ) -> list[dict]:
        if phase != "validation":
            raise ValueError("only the short validation phase is supported; no full phase")
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("sha must be a full immutable commit SHA")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", tag):
            raise ValueError("tag must contain only letters, digits, underscores and hyphens")
        if not variants or len(set(variants)) != len(variants) or set(variants) - VARIANTS.keys():
            raise ValueError("variants must be a nonempty distinct subset of the five variants")
        if not seeds or len(set(seeds)) != len(seeds) or set(seeds) - set(SEEDS):
            raise ValueError("seeds must be a nonempty distinct subset of paired seeds 0, 1")
        code = Path(DELLA) / "code" / sha
        prior = Path(DELLA) / "launch" / tag / "prior-calibration.json"
        records = []
        for variant in variants:
            sharing, groundings = VARIANTS[variant]
            for algorithm, (method, unit, schedule, budget) in ALGORITHMS.items():
                arm = f"{variant}-{algorithm}"
                for seed in seeds:
                    output = Path(DELLA) / "results" / tag / arm / method / str(seed)
                    argv = Tossing3DLaunch.cli_argv(
                        arm=Tossing3DLaunch.arms()["ees"],
                        size=PILOT,
                        seed=seed,
                        output_dir=output,
                        repo=code,
                    )
                    argv[argv.index("--method") + 1] = method
                    argv += [
                        "--ees-score-human-skills",
                        "--no-bin-pouring",
                        "--record-skill-competence",
                        "--record-learning-forecasts",
                        "--competence-sharing",
                        sharing,
                        "--practice-groundings",
                        groundings,
                    ]
                    if method == "dp":
                        argv += [
                            "--dp-solver",
                            "gurobi",
                            "--dp-budget-unit",
                            str(unit),
                            "--dp-allocation-schedule",
                            str(schedule),
                            "--dp-budget",
                            str(budget),
                            "--dp-prior-calibration",
                            str(prior),
                        ]
                    Cli.parse_args(argv=argv)
                    records.append({
                        "name": f"{tag}-{arm}-s{seed}",
                        "sha": sha,
                        "arm": arm,
                        "variant": variant,
                        "algorithm": algorithm,
                        "seed": seed,
                        "output": str(output),
                        "argv": argv,
                    })
        return records

    @staticmethod
    def prepare(
        *,
        directory: Path,
        tag: str,
        sha: str,
        calibration_report: Path,
        prior_calibration: Path,
        budget_prior_calibration: Path | None = None,
        variants: tuple[str, ...] = tuple(VARIANTS),
        seeds: tuple[int, ...] = SEEDS,
        phase: str = "validation",
    ) -> dict:
        records = Tossing3DSharedLearningValidation.records(
            tag=tag, sha=sha, variants=variants, seeds=seeds, phase=phase
        )
        budget_prior = budget_prior_calibration or prior_calibration
        evidence = Tossing3DSharedLearningValidation.calibration(
            report=calibration_report, prior=budget_prior
        )
        current = Tossing3DSharedLearningValidation.validate_prior(path=prior_calibration)
        if set(current.skills) != ROBOT_SKILLS:
            raise ValueError("current prior must cover exactly the original three robot skills")
        copies = {
            "prior-calibration.json": prior_calibration.read_bytes(),
            "budget-prior-calibration.json": budget_prior.read_bytes(),
            "calibration-report.json": calibration_report.read_bytes(),
        }
        report_digest = hashlib.sha256(copies["calibration-report.json"]).hexdigest()
        prior_digest = hashlib.sha256(copies["prior-calibration.json"]).hexdigest()
        budget_prior_digest = evidence["prior_calibration_sha256"]
        deployed = Path(DELLA) / "launch" / tag
        verify = shlex.join([
            "python",
            "-m",
            "scripts.tossing3d_shared_learning_validation",
            "verify",
            "--sha",
            sha,
            "--calibration-report",
            str(deployed / "calibration-report.json"),
            "--prior-calibration",
            str(deployed / "prior-calibration.json"),
            "--budget-prior-calibration",
            str(deployed / "budget-prior-calibration.json"),
            "--report-sha256",
            report_digest,
            "--prior-sha256",
            prior_digest,
        ])
        sbatch = ShelfLaunch.della_sbatch(
            tag=tag, group=GROUP, tasks=len(records), limits=DellaGroup(time="02:00:00")
        )
        preamble = (
            "module load gurobi/13.0.0\n"
            'export PYTHONPATH="${DP_GUROBI_PYTHONPATH:?Set the isolated Gurobi Python directory}'
            ':${PYTHONPATH:-}"\n'
        )
        sbatch = sbatch.replace("set -eo pipefail\n", "set -eo pipefail\n" + preamble, 1)
        sbatch = sbatch.replace(
            'CODE="',
            f'[ "$SHA" = "{sha}" ] || {{ echo "unexpected manifest SHA" >&2; exit 2; }}\nCODE="',
            1,
        )
        marker = 'eval "ARGS=($ARGV)"\n'
        if marker not in sbatch:
            raise RuntimeError("ShelfLaunch runtime template changed; review the worker handoff")
        sbatch = sbatch.split(marker, 1)[0] + (
            f'"$CODE/scripts/with_env.sh" {verify} '
            '--manifest "$1" --task "$SLURM_ARRAY_TASK_ID"\n'
            'exec "$CODE/scripts/with_env.sh" python -m scripts.tossing3d_launch worker '
            f'--sha {sha} --manifest "$1" --task "$SLURM_ARRAY_TASK_ID"\n'
        )
        manifest = (
            "\n".join(
                "\t".join([
                    record["name"],
                    sha,
                    record["arm"],
                    str(record["seed"]),
                    record["output"],
                    shlex.join(record["argv"]),
                ])
                for record in records
            )
            + "\n"
        )
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in copies.items():
            (directory / name).write_bytes(content)
        (directory / f"manifest-{GROUP}.tsv").write_text(manifest)
        (directory / f"array-{GROUP}.sbatch").write_text(sbatch)
        return {
            "phase": phase,
            "source_commit": sha,
            "records": records,
            "calibration_report_sha256": report_digest,
            "prior_calibration_sha256": prior_digest,
            "current_prior_calibration_sha256": prior_digest,
            "budget_reference_prior_digest": budget_prior_digest,
            "episode_budget": 350,
            "budget_interpretation": "fixed nominal historical cap; not recalibrated cost matching",
            "submitted": False,
        }

    @staticmethod
    def verify(
        *,
        sha: str,
        calibration_report: Path,
        prior_calibration: Path,
        report_sha256: str,
        prior_sha256: str,
        manifest: Path,
        task: int,
        budget_prior_calibration: Path | None = None,
    ) -> None:
        """Refuse mutable deployments, changed evidence, and stale or incomplete priors."""
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("sha must be a full immutable commit SHA")
        actual = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ],
            text=True,
        ).strip()
        if actual != sha or dirty:
            raise ValueError("deployment must be a clean checkout at the immutable source SHA")
        lines = manifest.read_text().splitlines()
        if task < 1 or task > len(lines):
            raise ValueError("expected an existing positive manifest task")
        fields = lines[task - 1].split("\t")
        if len(fields) != 6 or fields[1] != sha:
            raise ValueError("manifest task must name the immutable source SHA")
        args = Cli.parse_args(argv=shlex.split(fields[5]))
        if args.enable_bin_pouring:
            raise ValueError("validation requires --no-bin-pouring in every arm")
        if args.method == "dp" and (
            args.dp_prior_calibration is None
            or args.dp_prior_calibration.resolve() != prior_calibration.resolve()
        ):
            raise ValueError("manifest DP prior must be the digest-pinned current prior")
        for relative, package in (
            ("reference/kindergarden", "kinder"),
            ("reference/kinder-baselines", "kinder_models"),
        ):
            entry = subprocess.check_output(
                ["git", "-C", str(REPO), "ls-tree", "HEAD", relative], text=True
            ).split()
            if len(entry) < 3 or entry[2] != SIMULATOR_PINS[package]:
                raise ValueError(f"deployment changed the original simulator pin: {package}")
        for path, expected in (
            (calibration_report, report_sha256),
            (prior_calibration, prior_sha256),
        ):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"deployment evidence digest changed: {path}")
        Tossing3DSharedLearningValidation.calibration(
            report=calibration_report, prior=budget_prior_calibration or prior_calibration
        )
        Tossing3DSharedLearningValidation.validate_prior(path=prior_calibration)
        calibration = DpPriorCalibration.load(path=prior_calibration)
        calibration.verify_simulators()
        skills = Tossing3DSkillProvider(env=Tossing3DEnvironment(enable_bin_pouring=False)).skills()
        if {skill.name for skill in skills} != ROBOT_SKILLS:
            raise ValueError("validation must retain exactly the original three robot skills")
        calibration.verify_robot_skills(skills=skills)

    @staticmethod
    def parse(*, argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest="command", required=True)
        prepare = commands.add_parser("prepare", help="write launch artifacts and evidence only")
        prepare.add_argument("--phase", choices=["validation"], default="validation")
        prepare.add_argument("--tag", required=True)
        prepare.add_argument("--out", type=Path, required=True)
        prepare.add_argument(
            "--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS)
        )
        prepare.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
        verify = commands.add_parser("verify", help="read-only deployment checks")
        verify.add_argument("--report-sha256", required=True)
        verify.add_argument("--prior-sha256", required=True)
        verify.add_argument("--manifest", type=Path, required=True)
        verify.add_argument("--task", type=int, required=True)
        for command in (prepare, verify):
            command.add_argument("--sha", required=True)
            command.add_argument("--calibration-report", type=Path, required=True)
            command.add_argument("--prior-calibration", type=Path, required=True)
            command.add_argument("--budget-prior-calibration", type=Path)
        return parser.parse_args(argv)


def main(*, argv: list[str] | None = None) -> int:
    args = Tossing3DSharedLearningValidation.parse(argv=sys.argv[1:] if argv is None else argv)
    arguments = vars(args).copy()
    command = arguments.pop("command")
    if command == "prepare":
        arguments["directory"] = arguments.pop("out")
        arguments["variants"] = tuple(arguments["variants"])
        arguments["seeds"] = tuple(arguments["seeds"])
        result = Tossing3DSharedLearningValidation.prepare(**arguments)
        print(f"Prepared {len(result['records'])} validation tasks; submitted nothing")
    else:
        Tossing3DSharedLearningValidation.verify(**arguments)
        print("Verified clean source, original pins and source-bound prior calibration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
