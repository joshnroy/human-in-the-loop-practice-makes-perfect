"""`scripts/tossing3d_ees_protocol.sh` exists because the global CLI defaults
(`--num-cycles 10 --max-steps-per-interaction 150`) are generic-runner values, not
this domain's. This domain's own protocol -- `--num-cycles 100
--max-steps-per-interaction 20` -- was hard-won across two real experiments
(`docs/experiment-logs/2026-08-08-tossing3d-100cycle-plateau.md` and
`2026-08-08-tossing3d-reset-free-remeasured.md`) and then lost again on 2026-08-19,
when an ad hoc scheduled-vs-never sweep silently fell back to the CLI defaults instead.
These tests pin the wrapper's `--dry-run` output rather than actually launching a
sweep, since this suite has no simulator available (see
`test_tossing3d_oracle_demo.py`'s own docstring for why that is the right boundary).
"""

import os
import shlex
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "scripts" / "tossing3d_ees_protocol.sh"


def test_the_wrapper_exists_and_is_executable() -> None:
    assert WRAPPER.is_file()
    assert os.access(WRAPPER, os.X_OK)


def _dry_run(*extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(WRAPPER), "--dry-run", *extra_args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_it_bakes_in_the_established_protocol_numbers() -> None:
    """The two numbers this script exists to protect, not the CLI defaults."""
    completed = _dry_run(
        "--practice-reset-policy", "scheduled", "--results-root", "/tmp/some-run"
    )
    assert completed.returncode == 0, completed.stderr
    argv = shlex.split(completed.stdout)
    method_args = argv[argv.index("--method-args") + 1]
    assert "--num-cycles 100" in method_args
    assert "--max-steps-per-interaction 20" in method_args


def test_it_drives_run_sweep_on_the_tossing3d_env_with_ees() -> None:
    completed = _dry_run(
        "--practice-reset-policy", "never", "--results-root", "/tmp/some-run"
    )
    argv = shlex.split(completed.stdout)
    assert argv[:2] == ["python", "-m"]
    assert argv[2] == "scripts.run_sweep"
    assert "--env" in argv and argv[argv.index("--env") + 1] == "tossing3d"
    assert "--methods" in argv and argv[argv.index("--methods") + 1] == "ees"


def test_it_forwards_the_practice_reset_policy_and_num_test_tasks() -> None:
    completed = _dry_run(
        "--practice-reset-policy", "scheduled", "--results-root", "/tmp/some-run"
    )
    argv = shlex.split(completed.stdout)
    shared = argv[argv.index("--shared-args") + 1]
    assert "--practice-reset-policy scheduled" in shared
    assert "--num-test-tasks 10" in shared


def test_num_seeds_defaults_to_the_papers_ten_but_is_overridable() -> None:
    default_completed = _dry_run(
        "--practice-reset-policy", "scheduled", "--results-root", "/tmp/some-run"
    )
    default_argv = shlex.split(default_completed.stdout)
    assert default_argv[default_argv.index("--num-seeds") + 1] == "10"

    overridden = _dry_run(
        "--practice-reset-policy",
        "scheduled",
        "--results-root",
        "/tmp/some-run",
        "--num-seeds",
        "3",
    )
    overridden_argv = shlex.split(overridden.stdout)
    assert overridden_argv[overridden_argv.index("--num-seeds") + 1] == "3"


def test_extra_shared_args_are_forwarded_verbatim() -> None:
    completed = _dry_run(
        "--practice-reset-policy",
        "scheduled",
        "--results-root",
        "/tmp/some-run",
        "--extra-shared-args",
        "--record-wandb",
    )
    argv = shlex.split(completed.stdout)
    shared = argv[argv.index("--shared-args") + 1]
    assert "--record-wandb" in shared


def test_it_refuses_to_run_without_a_practice_reset_policy() -> None:
    completed = _dry_run("--results-root", "/tmp/some-run")
    assert completed.returncode != 0


def test_it_refuses_to_run_without_a_results_root() -> None:
    completed = _dry_run("--practice-reset-policy", "scheduled")
    assert completed.returncode != 0
