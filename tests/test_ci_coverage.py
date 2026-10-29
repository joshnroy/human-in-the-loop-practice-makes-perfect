"""Every test file is collected by some CI job.

CI's jobs list their test paths explicitly (`.github/workflows/ci.yml`), so a new test
folder runs nowhere until someone adds it. `tests/environments/sweepoffshelf3d` was
missing from every job, so none of the shelf's simulator tests ran in CI.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"


class CiPaths:
    @staticmethod
    def collected() -> set[str]:
        """Every path argument of every pytest step, across all jobs."""
        text = CI_WORKFLOW.read_text().replace("\\\n", " ")
        paths: set[str] = set()
        for line in text.splitlines():
            if "pytest" not in line:
                continue
            paths |= set(re.findall(r"(tests/[\w./*-]+)", line))
        return paths

    @staticmethod
    def covers(*, test_file: Path, paths: set[str]) -> bool:
        relative = test_file.relative_to(REPO).as_posix()
        for path in paths:
            if "*" in path and Path(relative).match(path):
                return True
            if relative == path or relative.startswith(path.rstrip("/") + "/"):
                return True
        return False


def test_every_test_file_runs_in_some_ci_job() -> None:
    paths = CiPaths.collected()
    missing = sorted(
        test_file.relative_to(REPO).as_posix()
        for test_file in (REPO / "tests").rglob("test_*.py")
        if not CiPaths.covers(test_file=test_file, paths=paths)
    )
    assert not missing, f"no CI job collects: {missing}"


class CiJobs:
    @staticmethod
    def pytest_command(*, job: str) -> str:
        """The `run:` text of the one pytest step of `job`, continuation lines joined."""
        steps = yaml.safe_load(CI_WORKFLOW.read_text())["jobs"][job]["steps"]
        runs = [step["run"] for step in steps if "pytest" in step.get("run", "")]
        assert len(runs) == 1, f"{job} has {len(runs)} pytest steps"
        return runs[0].replace("\\\n", " ")

    @staticmethod
    def collects(*, job: str, path: str) -> bool:
        arguments = CiJobs.pytest_command(job=job).split()
        selected = {argument for argument in arguments if argument.startswith("tests/")}
        ignored = {
            argument.removeprefix("--ignore=")
            for argument in arguments
            if argument.startswith("--ignore=")
        }
        return CiPaths.covers(test_file=REPO / path, paths=selected) and not CiPaths.covers(
            test_file=REPO / path, paths=ignored
        )


def test_table_environment_tests_run_only_in_the_osmesa_job() -> None:
    files = sorted((REPO / "tests/environments/sweeptable3d").glob("test_*.py"))
    assert files
    for test_file in files:
        path = test_file.relative_to(REPO).as_posix()
        assert CiJobs.collects(job="test-tossing3d", path=path), path
        for job in ("test-unit", "test-scripts", "test-tossingroom", "test-analysis"):
            assert not CiJobs.collects(job=job, path=path), (job, path)


@pytest.mark.parametrize(
    ("job", "path"),
    [
        ("test-unit", "tests/test_sweeptable3d_integration.py"),
        ("test-unit", "tests/test_step_protocol_failure.py"),
        ("test-unit", "tests/methods/belief_space/test_sweeptable3d_domain.py"),
        ("test-scripts", "tests/scripts/test_sweeptable3d_preflight.py"),
    ],
)
def test_mocked_table_learner_and_harness_checks_stay_outside_physics(*, job: str, path: str):
    assert CiJobs.collects(job=job, path=path)
    assert not CiJobs.collects(job="test-tossing3d", path=path)


def test_the_real_robot_tests_run_outside_the_osmesa_job() -> None:
    """A real-robot run trains torch samplers in-process, and torch's first optimizer
    imports triton, whose bundled LLVM segfaults against the LLVM that MuJoCo's OSMesa
    loaded earlier in the same worker. The real domain never imports MuJoCo, so its tests
    belong in a job that does not load it."""
    tossing3d = CiJobs.pytest_command(job="test-tossing3d")
    unit = CiJobs.pytest_command(job="test-unit")
    assert "--ignore=tests/environments/tossing3d/real" in tossing3d
    assert "tests/environments/tossing3d/real" in unit.split()
