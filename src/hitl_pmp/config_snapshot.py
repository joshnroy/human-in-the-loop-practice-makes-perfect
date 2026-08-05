import argparse
import importlib.util
import platform
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from pydantic import BaseModel, Field


class ConfigSnapshot(BaseModel):
    """What produced a run: the flags it resolved to, the code it ran, and the
    numerical stack it ran on.

    Written beside stats.json as config_snapshot.json rather than folded into it.
    stats.json is the serialized `Metrics` and its byte-stability is what verifies
    that a change did not alter results, so a commit SHA in it would break that on
    every commit; `timing.json` is kept out for the same reason.

    This exists because of a concrete failure. Ten archived Tossing Room seeds were
    re-run from the same commit with the same flags on a second machine, and four
    came back with different per-seed scores. Answering "did the code change, did
    the flags change, or did the environment change?" took an afternoon of bisecting
    commit dates and diffing docstrings, and every one of those questions would have
    been a single grep had the run recorded the conditions it happened under.
    `evaluations` says what happened; nothing said under what.

    Deliberately a record, not a pin. Bit-identical cross-machine results out of a
    floating-point training loop are not something a version pin can promise, so the
    goal is that a future mismatch is diagnosable in one command instead of
    reconstructed from file timestamps."""

    # Two sentinels, and the difference between them is load-bearing for a reader.
    # UNSET means the thing is not part of this run at all (Fast Downward not
    # configured, KINDER not installed). UNKNOWN means it *is* part of this run but
    # git could not say which revision -- a wheel, a `git+https` install, an
    # unpacked sdist. "Absent" and "present but unidentifiable" are different facts
    # and collapsing them would hide exactly the case worth chasing.
    UNSET: ClassVar[str] = "unset"
    UNKNOWN: ClassVar[str] = "unknown"

    args: dict[str, str] = Field(default_factory=dict)
    git_commit: str = UNKNOWN
    git_dirty: bool = False
    fast_downward_commit: str = UNKNOWN
    # The two KINDER upstreams. Their distribution names and their import names
    # differ, which is a standing trap: the distribution is `kindergarden` but
    # `import kindergarden` raises ModuleNotFoundError -- the package is `kinder`.
    # Field names follow the repositories, since a repository is what a SHA
    # identifies and what a reader will `git log` against.
    kindergarden_commit: str = UNKNOWN
    kindergarden_dirty: bool = False
    kinder_models_commit: str = UNKNOWN
    kinder_models_dirty: bool = False
    python_version: str = ""
    torch_version: str = ""
    numpy_version: str = ""
    platform_machine: str = ""
    python_executable: str = ""

    @staticmethod
    def collect(*, args: argparse.Namespace, fd_exec_path: str | None) -> "ConfigSnapshot":
        """Snapshot everything known to vary between two runs of "the same" command.

        args is recorded *resolved* -- after argparse has applied defaults -- so a
        flag that was defaulted rather than passed is still visible. That is the
        specific gap that cost the afternoon: `--sampler-max-train-iters` had its
        default changed from 1000 to 10000 partway through the project, so an
        archived run's iteration count could not be recovered from the command line
        alone, only guessed from a directory name."""
        own_repo = Path(__file__).resolve().parent
        kindergarden = ConfigSnapshot._repo_state(package="kinder")
        kinder_models = ConfigSnapshot._repo_state(package="kinder_models")
        return ConfigSnapshot(
            args={key: str(value) for key, value in sorted(vars(args).items())},
            git_commit=ConfigSnapshot._git(repo=own_repo, args=["rev-parse", "HEAD"]),
            git_dirty=bool(ConfigSnapshot._git(repo=own_repo, args=["status", "--porcelain"])),
            fast_downward_commit=(
                ConfigSnapshot.UNSET
                if fd_exec_path is None
                else ConfigSnapshot._git(repo=Path(fd_exec_path), args=["rev-parse", "HEAD"])
            ),
            kindergarden_commit=kindergarden[0],
            kindergarden_dirty=kindergarden[1],
            kinder_models_commit=kinder_models[0],
            kinder_models_dirty=kinder_models[1],
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            numpy_version=np.__version__,
            platform_machine=f"{platform.system()}-{platform.machine()}",
            # Which interpreter/venv -- two envs on one machine differ only here.
            python_executable=sys.executable,
        )

    @staticmethod
    def _repo_state(*, package: str) -> tuple[str, bool]:
        """The (commit, locally-modified) pair for whichever checkout of `package`
        this interpreter would actually import.

        The dirty flag is the half a committed pin structurally cannot supply: an
        upstream tree with uncommitted edits still reports an honest SHA, and that
        SHA names code that is not what ran. Recording both is the only way a reader
        can tell "upstream at X" from "upstream at X, plus whatever was on disk".

        Three outcomes, and they stay distinguishable:
          UNSET   -- not importable here; KINDER is optional and CI never installs it
          UNKNOWN -- importable, but no git repository encloses it (a wheel, a
                     `git+https` install, an unpacked sdist)
          a SHA   -- importable out of a checkout, with its dirty flag alongside

        Not dirty when the SHA did not resolve: there is no tree to have modified."""
        package_dir = ConfigSnapshot._installed_package_dir(package=package)
        if package_dir is None:
            return (ConfigSnapshot.UNSET, False)
        commit = ConfigSnapshot._git(repo=package_dir, args=["rev-parse", "HEAD"])
        if commit == ConfigSnapshot.UNKNOWN:
            return (ConfigSnapshot.UNKNOWN, False)
        # An unreadable status after the SHA resolved counts as dirty rather than
        # clean: `_git` reports UNKNOWN, which is non-empty. Guessing "clean" would
        # be the one wrong answer here -- it asserts the tree matches the SHA.
        return (commit, bool(ConfigSnapshot._git(repo=package_dir, args=["status", "--porcelain"])))

    @staticmethod
    def _installed_package_dir(*, package: str) -> Path | None:
        """Where `package` lives, resolved through the import system rather than
        from any hardcoded location.

        A hardcoded `../kindergarden` would defeat the entire purpose: it records a
        checkout that may not be the one that ran. Going through the import system
        means a sibling clone, an editable install and a `git+https` install are all
        handled by this one path, because they are all just entries the import
        machinery already resolves.

        sys.modules first, sys.path second: if the run already imported the package,
        that module object is what actually ran and a fresh search could resolve
        somewhere else. `find_spec` is used for the fallback precisely because it
        does *not* execute the package -- KINDER pulls in pybullet and OpenGL, and a
        snapshot taken after a multi-hour run must not be the thing that imports a
        heavyweight simulator (or fails inside one)."""
        module = sys.modules.get(package)
        origin = getattr(module, "__file__", None) if module is not None else None
        if origin is None:
            try:
                spec = importlib.util.find_spec(package)
            except (ImportError, ValueError):
                # A parent package that itself fails to import, or a sys.modules
                # entry without a usable __spec__. Not importable, for our purposes.
                return None
            if spec is None:
                return None
            origin = spec.origin
            if origin is None:
                # A namespace package has no single origin file; any of its portions
                # locates the tree just as well for a `git rev-parse`.
                locations = list(spec.submodule_search_locations or [])
                return Path(locations[0]).resolve() if locations else None
        return Path(origin).resolve().parent

    @staticmethod
    def _git(*, repo: Path, args: list[str]) -> str:
        """A git query that degrades to UNKNOWN rather than taking down a run.

        `git -C <dir>` does the walk up to the enclosing repository itself, so this
        works whether `dir` is the repository root or a package nested inside it.

        This is diagnostic metadata; a missing .git (an sdist install, a tarball, a
        stripped container) must never be the reason an experiment fails after hours
        of compute."""
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ConfigSnapshot.UNKNOWN
        return completed.stdout.strip() if completed.returncode == 0 else ConfigSnapshot.UNKNOWN
