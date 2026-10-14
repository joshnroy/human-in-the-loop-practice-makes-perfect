"""The `main_only` marker: slow tests that pull requests skip and `main` still runs.

CI computes one workflow-level variable from `github.event_name` and passes it to every
pytest call as `-m "$PYTEST_MARK_EXPR"`: `not main_only` on a pull request, the empty
string everywhere else. pytest treats an empty `-m` as no filter at all, so a push to
`main` runs every test. A local bare `pytest` passes no `-m`, so it runs everything too.

These tests pin the three things that can drift apart without any job failing: which
tests carry the mark, that every CI pytest call honours the variable, and that the
variable deselects on pull requests only.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"
CI_WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
MARK_EXPR_VARIABLE = "PYTEST_MARK_EXPR"

# Each entry is a test function; every parametrisation of it is marked.
EXPECTED_MAIN_ONLY = frozenset({
    "tests/environments/tossing3d/test_period_recorder.py::"
    "test_an_ordinary_cli_run_writes_one_video_per_period_and_per_sweep",
    "tests/environments/tossing3d/test_period_recorder.py::"
    "test_the_practice_video_carries_every_substep_frame_not_one_per_skill",
    "tests/environments/tossing3d/test_state_log.py::"
    "test_an_ordinary_cli_run_writes_a_state_log_with_no_manual_wiring",
    "tests/scripts/test_reproducibility.py::test_tossing3d_the_same_seed_produces_identical_results",
    "tests/scripts/test_reproducibility.py::"
    "test_tossing3d_results_do_not_depend_on_the_math_thread_count",
    "tests/environments/tossing3d/test_wide_long_range_proposal.py::"
    "test_witnesses_score_and_complete_misses_at_spanning_bin_locations",
    "tests/environments/tossing3d/test_pick_plannable.py::"
    "test_evaluating_the_predicate_does_not_change_the_trajectory",
    "tests/environments/tossing3d/test_pick_plannable.py::"
    "test_a_cube_across_the_barrier_really_is_unplannable",
})


class MainOnlyHarness:
    """A static-method container, never instantiated."""

    @staticmethod
    def collect(*, mark_expr: str, paths: list[str]) -> set[str]:
        """Node ids pytest selects under `-m <mark_expr>`, parametrisation stripped."""
        completed = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:xdist"]
            + ["-p", "no:cacheprovider", "-m", mark_expr, *paths],
            capture_output=True,
            text=True,
            cwd=REPO,
            check=False,
        )
        assert completed.returncode in {0, 5}, completed.stdout + completed.stderr
        return {line.split("[", 1)[0] for line in completed.stdout.splitlines() if "::" in line}

    @staticmethod
    def marked_files() -> list[str]:
        return sorted(
            str(path.relative_to(REPO))
            for path in TESTS.rglob("test_*.py")
            if path.name != Path(__file__).name
            and "mark.main_only" in path.read_text(encoding="utf-8")
        )

    @staticmethod
    def pytest_steps() -> list[tuple[str, str]]:
        workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
        return [
            (job_name, step["run"])
            for job_name, job in workflow["jobs"].items()
            for step in job.get("steps", [])
            if "pytest" in step.get("run", "").split()
        ]


def test_the_marker_is_registered(*, pytestconfig: pytest.Config) -> None:
    assert any(line.startswith("main_only:") for line in pytestconfig.getini("markers"))


def test_exactly_the_expected_tests_are_main_only() -> None:
    files = MainOnlyHarness.marked_files()
    assert files == sorted({node.split("::", 1)[0] for node in EXPECTED_MAIN_ONLY})
    assert MainOnlyHarness.collect(mark_expr="main_only", paths=files) == EXPECTED_MAIN_ONLY


def test_the_pull_request_expression_deselects_them_and_the_empty_one_does_not() -> None:
    files = sorted({node.split("::", 1)[0] for node in EXPECTED_MAIN_ONLY})
    pull_request = MainOnlyHarness.collect(mark_expr="not main_only", paths=files)
    push = MainOnlyHarness.collect(mark_expr="", paths=files)
    assert not pull_request & EXPECTED_MAIN_ONLY
    assert pull_request, "the pull-request path must still run the rest of these modules"
    assert push == pull_request | EXPECTED_MAIN_ONLY


def test_a_local_bare_pytest_does_not_deselect_main_only(*, pytestconfig: pytest.Config) -> None:
    assert "main_only" not in " ".join(pytestconfig.getini("addopts"))


def test_ci_computes_the_expression_from_the_event_name() -> None:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    expression = workflow["env"][MARK_EXPR_VARIABLE]
    assert expression == "${{ github.event_name == 'pull_request' && 'not main_only' || '' }}"


def test_every_ci_pytest_call_passes_the_expression() -> None:
    steps = MainOnlyHarness.pytest_steps()
    assert steps, "found no pytest step in ci.yml"
    missing = [job for job, run in steps if f'-m "${MARK_EXPR_VARIABLE}"' not in run]
    assert not missing, f'pytest steps without -m "${MARK_EXPR_VARIABLE}": {missing}'
