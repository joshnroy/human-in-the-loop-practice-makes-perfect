"""The launch gate: no long practice run of a gated domain starts unless the robot
skill benchmark passed on exactly the code being launched (Josh, 2026-10-15).

"Exactly" means: a `report.json` written by `scripts/skill_benchmark.py` for this
checkout's commit, benchmarked on a clean tree, against the same two KINDER trees this
process imports, with every skill at its target. Anything else blocks with the reason,
naming each skill below target. `--skip-skill-benchmark-gate` is the explicit override;
the run's config snapshot records that it was given.

Reports live at `<root>/<domain>/<commit>/report.json`, `root` being
`$HITL_SKILL_BENCHMARK_DIR` or this checkout's `benchmark-reports/` (gitignored: a
report names its commit, so it can never be committed in it). A benchmark run on agni
writes there by default; a launch elsewhere (Della) needs the report copied alongside.

Gated: `hitl_pmp.cli` step-protocol practice runs (`--practice-step-budget`), and
`scripts/run_sweep.py` before it spawns any run.
"""

import argparse
import json
import os
from pathlib import Path

from pydantic import BaseModel

from hitl_pmp.config_snapshot import ConfigSnapshot

GATED_DOMAINS = frozenset({"sweepoffshelf3d"})
OVERRIDE_FLAG = "--skip-skill-benchmark-gate"
ROOT_VARIABLE = "HITL_SKILL_BENCHMARK_DIR"
REPORT_NAME = "report.json"


class LaunchBlocked(RuntimeError):
    """The skill benchmark gate refused a launch."""


class LaunchIdentity(BaseModel):
    """The code this process would run."""

    git_commit: str
    git_dirty: bool
    kindergarden_commit: str
    kinder_models_commit: str

    @staticmethod
    def current() -> "LaunchIdentity":
        snapshot = ConfigSnapshot.collect(args=argparse.Namespace(), fd_exec_path=None)
        return LaunchIdentity(
            git_commit=snapshot.git_commit,
            git_dirty=snapshot.git_dirty,
            kindergarden_commit=snapshot.kindergarden_commit,
            kinder_models_commit=snapshot.kinder_models_commit,
        )


class LaunchGate:
    """A static-method container, never instantiated."""

    @staticmethod
    def default_root() -> Path:
        configured = os.environ.get(ROOT_VARIABLE)
        if configured:
            return Path(configured)
        return Path(__file__).resolve().parents[2] / "benchmark-reports"

    @staticmethod
    def report_path(*, root: Path, domain: str, commit: str) -> Path:
        return root / domain / commit / REPORT_NAME

    @staticmethod
    def add_argument(*, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            OVERRIDE_FLAG,
            dest="skip_skill_benchmark_gate",
            action="store_true",
            help="Launch even without a passing skill benchmark report for this commit.",
        )

    @staticmethod
    def enforce(*, domain: str, override: bool) -> None:
        """`check` against the live checkout, printing what it decided."""
        if domain not in GATED_DOMAINS:
            return
        message = LaunchGate.check(
            domain=domain,
            root=LaunchGate.default_root(),
            identity=LaunchIdentity.current(),
            override=override,
        )
        print(message, flush=True)

    @staticmethod
    def check(*, domain: str, root: Path, identity: LaunchIdentity, override: bool) -> str:
        """Return why the launch may go ahead; raise `LaunchBlocked` if it may not."""
        if domain not in GATED_DOMAINS:
            return ""
        if override:
            return (
                f"skill benchmark gate SKIPPED for {domain} at {identity.git_commit} "
                f"({OVERRIDE_FLAG})"
            )
        if identity.git_commit in {ConfigSnapshot.UNKNOWN, ConfigSnapshot.UNSET, ""}:
            raise LaunchBlocked(
                f"the skill benchmark gate cannot tell which commit is being launched; "
                f"launch from a git checkout, or pass {OVERRIDE_FLAG}"
            )
        if identity.git_dirty:
            raise LaunchBlocked(
                f"the checkout has uncommitted changes, so no benchmark report can describe "
                f"it; commit them and benchmark that commit, or pass {OVERRIDE_FLAG}"
            )
        path = LaunchGate.report_path(root=root, domain=domain, commit=identity.git_commit)
        if not path.exists():
            raise LaunchBlocked(
                f"no skill benchmark report for {domain} at {identity.git_commit} ({path}); "
                f"run `scripts/skill_benchmark.py run` on this commit, or pass {OVERRIDE_FLAG}"
            )
        report = json.loads(path.read_text())
        problems = []
        if report.get("domain") != domain or report.get("git_commit") != identity.git_commit:
            problems.append("the report is for another domain or commit")
        if report.get("git_dirty", True):
            problems.append("the benchmark ran on a dirty tree")
        if report.get("kindergarden_commit") != identity.kindergarden_commit:
            problems.append(
                f"benchmarked on kindergarden {report.get('kindergarden_commit')}, "
                f"launching on {identity.kindergarden_commit}"
            )
        if report.get("kinder_models_commit") != identity.kinder_models_commit:
            problems.append(
                f"benchmarked on kinder-models {report.get('kinder_models_commit')}, "
                f"launching on {identity.kinder_models_commit}"
            )
        below = [
            f"{s['skill']} {s['successes']}/{s['total']}"
            for s in report.get("skills", [])
            if not s.get("passed")
        ]
        if not report.get("passed") or below:
            problems.append("skills below their benchmark target: " + ", ".join(below or ["?"]))
        if problems:
            raise LaunchBlocked(
                f"skill benchmark gate blocked {domain} at {identity.git_commit} ({path}): "
                + "; ".join(problems)
                + f". Fix them, or pass {OVERRIDE_FLAG}"
            )
        return f"skill benchmark gate passed for {domain} at {identity.git_commit}: {path}"
