"""Pull requests cancel their superseded runs and skip an unaffected Tossing3D job.

Two savings, each with a way to go wrong that no job failure would ever show:

- **Concurrency.** A burst of force-pushes queued ~15 full runs at once, all but the
  newest of them wasted. The workflow-level `concurrency` block cancels an in-progress run
  when a newer one lands on the same pull request. A push to `main` must never be
  cancelled -- each merged commit keeps its own full run -- and a group shared by two
  pushes would cancel one of them even with `cancel-in-progress` off, because GitHub
  replaces a *pending* run in a group with the newer one regardless.
- **The Tossing3D path filter.** `test / tossing3d` is the slowest job (~29 min). On a pull
  request that touches nothing its tests import or read, it is skipped. The filter is a
  hand-written list, so it is checked here against the tests' real import closure: a new
  import from a Tossing3D test into a path the filter omits fails this file instead of
  silently letting a breaking PR skip the job. A push to `main` always runs it.

The required status check is the aggregate `test` job, which therefore has to tell an
intended skip from a failure. A job whose dependency fails is *skipped*, and GitHub counts
a skipped required check as passing, so the aggregate runs under `!cancelled()` and
decides from its dependencies' results itself.
"""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
TOSSING3D_JOB = "test-tossing3d"
FILTER_NAME = "tossing3d"
# The test directories `test / tossing3d` collects, minus the real-robot one it ignores.
TOSSING3D_TEST_DIRS = ("tests/environments/tossing3d", "tests/environments/sweepoffshelf3d")
IGNORED_TEST_DIR = "tests/environments/tossing3d/real"


class Workflow:
    """A static-method container, never instantiated."""

    @staticmethod
    def load() -> dict[str, Any]:
        return yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))

    @staticmethod
    def filter_step() -> dict[str, Any]:
        steps = Workflow.load()["jobs"]["changes"]["steps"]
        (step,) = [s for s in steps if str(s.get("uses", "")).startswith("dorny/paths-filter@")]
        return step

    @staticmethod
    def filter_patterns() -> list[str]:
        return yaml.safe_load(Workflow.filter_step()["with"]["filters"])[FILTER_NAME]

    @staticmethod
    def aggregate_script() -> str:
        (step,) = Workflow.load()["jobs"]["test"]["steps"]
        body = re.search(r"<<'PY'\n(.*?)\nPY\b", step["run"], re.S)
        assert body is not None, "the aggregate `test` step must be a python3 heredoc"
        return body.group(1)


class Glob:
    """picomatch semantics as `dorny/paths-filter` applies them (`dot: true`)."""

    @staticmethod
    def matches(*, path: str, pattern: str) -> bool:
        regex = ""
        index = 0
        while index < len(pattern):
            if pattern.startswith("**/", index):
                regex += "(?:.*/)?"
                index += 3
            elif pattern.startswith("/**", index) and index + 3 == len(pattern):
                regex += "(?:/.*)?"
                index += 3
            elif pattern.startswith("**", index):
                regex += ".*"
                index += 2
            elif pattern[index] == "*":
                regex += "[^/]*"
                index += 1
            else:
                regex += re.escape(pattern[index])
                index += 1
        return re.fullmatch(regex, path) is not None

    @staticmethod
    def any_matches(*, path: str, patterns: list[str]) -> bool:
        return any(Glob.matches(path=path, pattern=pattern) for pattern in patterns)


class ImportClosure:
    """Every repo file the Tossing3D tests import, transitively, at any nesting depth.

    Static, so it over-approximates: a function-level import counts, and so does a
    `hitl_pmp.*` / `scripts.*` dotted name in a string literal (a `-m hitl_pmp.cli`
    subprocess, an `import_module` target).
    """

    ROOTS = {"hitl_pmp": REPO / "src", "scripts": REPO, "tests": REPO}
    DOTTED = re.compile(r"\b((?:hitl_pmp|scripts)(?:\.\w+)+)")

    @staticmethod
    def resolve(*, module: str) -> Path | None:
        root = ImportClosure.ROOTS.get(module.split(".")[0])
        if root is None:
            return None
        base = root.joinpath(*module.split("."))
        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def module_name(*, path: Path) -> str:
        src = REPO / "src"
        relative = path.relative_to(src if src in path.parents else REPO)
        parts = list(relative.with_suffix("").parts)
        return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)

    @staticmethod
    def imported(*, path: Path) -> set[str]:
        name = ImportClosure.module_name(path=path)
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        found: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = package.split(".")[: len(package.split(".")) - node.level + 1]
                    module = ".".join(parts + ([node.module] if node.module else []))
                else:
                    module = node.module or ""
                found |= {module} | {f"{module}.{alias.name}" for alias in node.names}
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                found |= set(ImportClosure.DOTTED.findall(node.value))
        # Importing `a.b.c` executes `a/__init__.py` and `a/b/__init__.py` too.
        return {
            ".".join(module.split(".")[:depth])
            for module in found
            for depth in range(1, module.count(".") + 2)
        }

    @staticmethod
    def files() -> set[str]:
        pending = [
            path
            for directory in TOSSING3D_TEST_DIRS
            for path in (REPO / directory).rglob("*.py")
            if not path.is_relative_to(REPO / IGNORED_TEST_DIR)
        ]
        pending += [REPO / "tests" / "conftest.py", REPO / "tests" / "__init__.py"]
        seen: set[Path] = set()
        while pending:
            path = pending.pop()
            if path in seen:
                continue
            seen.add(path)
            for module in ImportClosure.imported(path=path):
                resolved = ImportClosure.resolve(module=module)
                if resolved is not None and resolved not in seen:
                    pending.append(resolved)
        return {path.relative_to(REPO).as_posix() for path in seen}


class Aggregate:
    """Runs the aggregate `test` job's script against a synthetic `needs` context."""

    JOBS = ("changes", "test-unit", "test-tossingroom", TOSSING3D_JOB, "test-scripts")

    @staticmethod
    def passes(*, event: str, results: dict[str, str], tossing3d_flag: str) -> bool:
        needs = {
            job: {"result": results.get(job, "success"), "outputs": {}} for job in Aggregate.JOBS
        }
        needs["changes"]["outputs"] = {FILTER_NAME: tossing3d_flag}
        completed = subprocess.run(  # noqa: S603
            [sys.executable, "-c", Workflow.aggregate_script()],
            env={**os.environ, "NEEDS": json.dumps(needs), "EVENT": event},
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.returncode == 0


# --- 1. Cancel superseded runs, never a push to main -----------------------------------


def test_only_pull_request_runs_can_be_cancelled_in_progress() -> None:
    concurrency = Workflow.load()["concurrency"]
    assert concurrency["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"


def test_every_push_run_gets_a_concurrency_group_of_its_own() -> None:
    # A shared group would let a newer push replace an older pending one even without
    # cancel-in-progress, so the non-PR branch of the expression must be unique per run.
    group = Workflow.load()["concurrency"]["group"]
    expected = (
        "${{ github.event_name == 'pull_request'"
        " && format('ci-pr-{0}', github.event.pull_request.number)"
        " || format('ci-push-{0}', github.run_id) }}"
    )
    assert group == expected


def test_no_job_declares_a_concurrency_of_its_own() -> None:
    for name, job in Workflow.load()["jobs"].items():
        assert "concurrency" not in job, name


# --- 2. The Tossing3D path filter ------------------------------------------------------


def test_a_push_to_main_always_runs_the_tossing3d_job() -> None:
    jobs = Workflow.load()["jobs"]
    condition = jobs[TOSSING3D_JOB]["if"]
    assert condition.startswith("${{ !cancelled() && (github.event_name != 'pull_request' || ")
    assert f"needs.changes.outputs.{FILTER_NAME} == 'true'" in condition
    assert jobs["changes"].get("if") is None, "the changes job must run on every event"
    assert Workflow.filter_step()["if"] == "github.event_name == 'pull_request'"
    output = jobs["changes"]["outputs"][FILTER_NAME]
    assert output == (
        f"${{{{ github.event_name != 'pull_request' || steps.filter.outputs.{FILTER_NAME} }}}}"
    )


def test_the_filter_action_is_pinned_to_a_full_commit_sha() -> None:
    uses = Workflow.filter_step()["uses"]
    assert re.fullmatch(r"dorny/paths-filter@[0-9a-f]{40}", uses), uses


def test_the_filter_covers_every_file_the_tossing3d_tests_import() -> None:
    patterns = Workflow.filter_patterns()
    closure = ImportClosure.files()
    assert "src/hitl_pmp/environments/tossing3d/kinder_backend.py" in closure
    uncovered = sorted(f for f in closure if not Glob.any_matches(path=f, patterns=patterns))
    assert not uncovered, f"add these to the `{FILTER_NAME}` filter: {uncovered}"


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/ci.yml",
        "pyproject.toml",
        ".gitmodules",
        "reference/kindergarden",
        "reference/kinder-baselines",
        "conftest.py",
        "tests/conftest.py",
        "tests/methods/belief_space/conftest.py",
        "tests/environments/tossing3d/fixtures/seed3_rim.json",
        "tests/environments/sweepoffshelf3d/fixtures/neighbour_strike_states.jsonl",
        "src/hitl_pmp/environments/tossing3d/scenes/same_side.json",
        # Data the tests read by path rather than import.
        "configs/normalized_practice_costs.json",
        "benchmarks/sweepoffshelf3d/states.jsonl",
        "docs/experiment-logs/2026-08-13-tossing3d-toss-parameter-surface.json",
    ],
)
def test_the_filter_covers_dependencies_ci_config_conftests_and_data(*, path: str) -> None:
    assert Glob.any_matches(path=path, patterns=Workflow.filter_patterns())


@pytest.mark.parametrize(
    "path",
    [
        "docs/experiment-logs/2026-08-13-tossing3d-toss-parameter-surface.md",
        "README.md",
        "analysis/run_timing.py",
        "tests/analysis/test_run_timing.py",
        "tests/environments/lightswitch/test_environment.py",
        "src/hitl_pmp/full_agentic/runner.py",
    ],
)
def test_the_filter_lets_unrelated_changes_skip_the_job(*, path: str) -> None:
    assert not Glob.any_matches(path=path, patterns=Workflow.filter_patterns())


# --- The aggregate required check ------------------------------------------------------


def test_the_aggregate_check_runs_even_when_a_dependency_fails() -> None:
    job = Workflow.load()["jobs"]["test"]
    assert job["if"] == "${{ !cancelled() }}"
    assert set(job["needs"]) == set(Aggregate.JOBS)


@pytest.mark.parametrize(
    ("event", "results", "flag", "passes"),
    [
        ("push", {}, "true", True),
        ("pull_request", {}, "true", True),
        ("pull_request", {TOSSING3D_JOB: "skipped"}, "false", True),
        ("push", {TOSSING3D_JOB: "skipped"}, "true", False),
        ("pull_request", {TOSSING3D_JOB: "skipped"}, "true", False),
        ("pull_request", {"test-unit": "failure"}, "true", False),
        ("pull_request", {"test-unit": "skipped"}, "false", False),
        ("pull_request", {"changes": "failure", TOSSING3D_JOB: "skipped"}, "", False),
        ("push", {TOSSING3D_JOB: "cancelled"}, "true", False),
    ],
)
def test_the_aggregate_check_allows_only_the_intended_skip(
    *, event: str, results: dict[str, str], flag: str, passes: bool
) -> None:
    assert Aggregate.passes(event=event, results=results, tossing3d_flag=flag) is passes
