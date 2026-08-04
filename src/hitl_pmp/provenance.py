import argparse
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, Field


class RunProvenance(BaseModel):
    """What produced a run: the resolved flags, the code, and the numerical stack.

    Written beside stats.json as provenance.json rather than folded into Metrics,
    which stays purely about measurement -- and which every archived stats.json
    already conforms to.

    This exists because of a concrete failure. Ten archived Tossing Room seeds were
    re-run from the same commit with the same flags on a second machine; four came
    back with different per-seed scores (the arm mean moved 95.0% -> 96.0%, inside
    noise, but individual seeds disagreed). Answering "did the code change, did the
    flags change, or did the environment change?" took an afternoon of bisecting
    commit dates and diffing docstrings, and every one of those questions would have
    been a single grep had the run recorded what produced it. `evaluations` says what
    happened; nothing said under what.

    Deliberately not a reproducibility *fix*: bit-identical cross-machine results
    from a floating-point training loop are not something a version pin can promise
    (see resolve_reproducibility_scope below). The goal is that a future mismatch is
    diagnosable in one command instead of reconstructed from timestamps."""

    args: dict[str, str] = Field(default_factory=dict)
    git_commit: str = "unknown"
    git_dirty: bool = False
    fast_downward_commit: str = "unknown"
    python_version: str = ""
    torch_version: str = ""
    numpy_version: str = ""
    platform_machine: str = ""
    python_executable: str = ""

    @staticmethod
    def collect(*, args: argparse.Namespace, fd_exec_path: str | None) -> "RunProvenance":
        """Snapshot everything known to vary between two runs of "the same" command.

        args is recorded *resolved* -- after argparse has applied defaults -- so a
        flag that was defaulted rather than passed is still visible. That is the
        specific gap that cost the afternoon: `--sampler-max-train-iters` had its
        default changed from 1000 to 10000 partway through the project, so an
        archived run's iteration count could not be recovered from the command line
        alone, only guessed from a directory name."""
        return RunProvenance(
            args={key: str(value) for key, value in sorted(vars(args).items())},
            git_commit=RunProvenance._git(
                repo=Path(__file__).resolve().parent, args=["rev-parse", "HEAD"]
            ),
            git_dirty=bool(
                RunProvenance._git(
                    repo=Path(__file__).resolve().parent, args=["status", "--porcelain"]
                )
            ),
            fast_downward_commit=(
                "unset"
                if fd_exec_path is None
                else RunProvenance._git(repo=Path(fd_exec_path), args=["rev-parse", "HEAD"])
            ),
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            numpy_version=np.__version__,
            platform_machine=f"{platform.system()}-{platform.machine()}",
            # Which interpreter/venv -- two envs on one machine differ only here.
            python_executable=sys.executable,
        )

    @staticmethod
    def _git(*, repo: Path, args: list[str]) -> str:
        """A git query that degrades to "unknown" rather than taking down a run.

        Provenance is diagnostic metadata; a missing .git (an sdist install, a
        tarball, a stripped container) must not be the reason an experiment fails
        after hours of compute."""
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def resolve_reproducibility_scope() -> str:
    """The honest statement of what a fixed --seed does and does not buy.

    Kept as a single string so docs, CLI help, and tests quote one source rather
    than drifting into three slightly different claims -- the codebase has already
    been bitten by self-justifying comments that were not true.

    Same machine, same environment, a fixed --seed reproduces a run exactly; that
    is what tests/scripts/test_reproducibility.py actually pins. It does NOT extend
    across machines.

    What was measured, and only this: the ten Tossing Room `results-release/ees10000`
    seeds were re-run from the same commit with the same explicit flags on a second
    machine. Six matched the archive exactly and four did not (the arm mean moved
    95.0% -> 96.0%, inside noise). Every seed agreed at the pre-training sweep and
    diverged only later -- and the pre-training sweep is the part that never calls
    torch, since an unfitted sampler draws uniformly through numpy. So the numpy
    task-sampling and Fast Downward planning paths reproduce across the two machines,
    and whatever differs sits downstream of the classifier being trained.

    Every version-level explanation was tested on one machine and refuted. Each was
    run on the differing seeds against the same commit and the same explicit flags,
    and each produced curves *bit-identical* to the control -- not merely the same
    final score:

      torch          2.5.1+cpu vs 2.13.0+cu130   identical
      Fast Downward  5ea802478 vs 6230635        identical
      numpy          1.26.4 vs 2.2.6             identical
      math threads   1 vs 24                     identical

    Four independent processes also agree bit-for-bit here, so PYTHONHASHSEED-style
    set-ordering nondeterminism is out as well. The runs are genuinely deterministic
    on a single machine; what does not carry is moving between machines.

    The surviving candidate is the platform itself -- CLAUDE.md's setup includes a
    macOS-only `brew install coreutils` step and fast_downward.py branches on
    sys.platform == "darwin", so the archived runs were probably produced on macOS
    against this Linux-x86_64 re-run, and torch's ARM and x86 kernels are different
    code producing different floats from identical inputs. That is a candidate by
    elimination, NOT a measurement: no ARM machine was available to test it, and a
    mechanism this file cannot demonstrate does not get asserted as fact here.

    The practical consequence holds regardless of which platform detail it turns out
    to be, and it is why this file records rather than pins: no version pin would
    have prevented an architecture difference. Treat per-seed values as machine-local
    and arm-level aggregates as the portable unit."""
    return (
        "A fixed --seed reproduces a run exactly on the same machine with the same "
        "installed versions. Per-seed results are NOT known to be portable across "
        "machines; arm-level aggregates are stable. See provenance.json for what "
        "produced any given run."
    )
