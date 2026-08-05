import argparse
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hitl_pmp.config_snapshot import ConfigSnapshot


def _args(**kwargs: object) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def _git(*, repo: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _make_repo(*, repo: Path) -> str:
    """A real git repository, so the tests exercise real `git rev-parse` output
    rather than a mock of it -- the whole point of the field is to record what a
    git command actually said about the tree that was imported."""
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo=repo, args=["init", "-q"])
    _git(repo=repo, args=["add", "-A"])
    _git(
        repo=repo,
        args=[
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=test",
            "commit",
            "-qm",
            "initial",
        ],
    )
    return _git(repo=repo, args=["rev-parse", "HEAD"])


def _install_package(*, site_dir: Path, name: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put an importable package called `name` on sys.path, the way any real
    install does, so resolution goes through the same import machinery a real
    editable/sibling/`git+https` install would."""
    package_dir = site_dir / name
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(site_dir))
    monkeypatch.delitem(sys.modules, name, raising=False)
    importlib.invalidate_caches()
    return package_dir


def _uninstall_package(*, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `name` un-importable regardless of what this machine happens to have
    installed. A None entry in sys.modules is CPython's own "this name does not
    resolve" marker, so the absent case is deterministic on a developer box that
    *does* have KINDER and in CI which never does."""
    monkeypatch.setitem(sys.modules, name, None)


# --- the KINDER upstreams: what actually ran, not what a pin claims ran --------


def test_kindergarden_commit_is_read_from_the_installed_kinder_package(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The distribution is named `kindergarden` but the import package is `kinder`,
    and the path must come from the import system -- a hardcoded ../kindergarden
    would record a checkout that never ran."""
    repo = tmp_path / "kindergarden"
    _install_package(site_dir=repo / "src", name="kinder", monkeypatch=monkeypatch)
    _uninstall_package(name="kinder_models", monkeypatch=monkeypatch)
    head = _make_repo(repo=repo)

    snapshot = ConfigSnapshot.collect(args=_args(), fd_exec_path=None)

    assert snapshot.kindergarden_commit == head
    assert snapshot.kindergarden_dirty is False


def test_kindergarden_commit_is_not_this_repository_s_own_commit(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-vacuity: a field that silently reported hitl-pmp's own SHA would pass
    every "is it a SHA" check while recording nothing about KINDER at all."""
    repo = tmp_path / "kindergarden"
    _install_package(site_dir=repo / "src", name="kinder", monkeypatch=monkeypatch)
    head = _make_repo(repo=repo)

    snapshot = ConfigSnapshot.collect(args=_args(), fd_exec_path=None)

    assert len(head) == 40
    assert snapshot.kindergarden_commit == head
    assert snapshot.kindergarden_commit != snapshot.git_commit


def test_kindergarden_dirty_when_the_checkout_has_uncommitted_changes(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case a committed pin structurally cannot see: the SHA is honest and the
    tree that ran is still not the tree that SHA names."""
    repo = tmp_path / "kindergarden"
    package_dir = _install_package(site_dir=repo / "src", name="kinder", monkeypatch=monkeypatch)
    head = _make_repo(repo=repo)
    (package_dir / "locally_edited.py").write_text("# not upstream\n")

    snapshot = ConfigSnapshot.collect(args=_args(), fd_exec_path=None)

    assert snapshot.kindergarden_commit == head
    assert snapshot.kindergarden_dirty is True


def test_kinder_models_commit_is_read_from_the_installed_kinder_models_package(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "kinder-baselines"
    _install_package(
        site_dir=repo / "kinder-models" / "src", name="kinder_models", monkeypatch=monkeypatch
    )
    _uninstall_package(name="kinder", monkeypatch=monkeypatch)
    head = _make_repo(repo=repo)

    snapshot = ConfigSnapshot.collect(args=_args(), fd_exec_path=None)

    assert snapshot.kinder_models_commit == head
    assert snapshot.kinder_models_dirty is False


def test_the_two_kinder_upstreams_are_resolved_independently(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-vacuity: the two fields must not be cross-wired to one lookup. Two
    separate repos, two different SHAs, one dirty and one clean."""
    kindergarden = tmp_path / "kindergarden"
    kinder_baselines = tmp_path / "kinder-baselines"
    _install_package(site_dir=kindergarden / "src", name="kinder", monkeypatch=monkeypatch)
    models_dir = _install_package(
        site_dir=kinder_baselines / "src", name="kinder_models", monkeypatch=monkeypatch
    )
    kindergarden_head = _make_repo(repo=kindergarden)
    kinder_models_head = _make_repo(repo=kinder_baselines)
    (models_dir / "locally_edited.py").write_text("# not upstream\n")

    snapshot = ConfigSnapshot.collect(args=_args(), fd_exec_path=None)

    assert kindergarden_head != kinder_models_head
    assert snapshot.kindergarden_commit == kindergarden_head
    assert snapshot.kinder_models_commit == kinder_models_head
    assert snapshot.kindergarden_dirty is False
    assert snapshot.kinder_models_dirty is True


def test_kinder_absent_records_a_sentinel_instead_of_raising(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KINDER is an optional dependency and CI never installs it. Absent is a fact
    to record, not an error -- and never an import at module scope."""
    _uninstall_package(name="kinder", monkeypatch=monkeypatch)
    _uninstall_package(name="kinder_models", monkeypatch=monkeypatch)

    snapshot = ConfigSnapshot.collect(args=_args(), fd_exec_path=None)

    assert snapshot.kindergarden_commit == "unset"
    assert snapshot.kinder_models_commit == "unset"
    assert snapshot.kindergarden_dirty is False
    assert snapshot.kinder_models_dirty is False


def test_kinder_installed_outside_a_git_repository_is_distinct_from_absent(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wheel or a `git+https` install leaves no .git behind. "It is here but I
    cannot tell you which commit" and "it is not here at all" are different facts
    and a reader has to be able to tell them apart."""
    _install_package(site_dir=tmp_path / "site-packages", name="kinder", monkeypatch=monkeypatch)
    _uninstall_package(name="kinder_models", monkeypatch=monkeypatch)

    snapshot = ConfigSnapshot.collect(args=_args(), fd_exec_path=None)

    assert snapshot.kindergarden_commit == "unknown"
    assert snapshot.kinder_models_commit == "unset"
    assert snapshot.kindergarden_commit != snapshot.kinder_models_commit
    assert snapshot.kindergarden_dirty is False


def test_an_already_imported_kinder_is_read_from_sys_modules(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What actually ran is what is already in sys.modules -- if the run imported
    KINDER, that module object is the authority, not a fresh sys.path search."""
    repo = tmp_path / "kindergarden"
    _install_package(site_dir=repo / "src", name="kinder", monkeypatch=monkeypatch)
    head = _make_repo(repo=repo)
    importlib.import_module("kinder")

    snapshot = ConfigSnapshot.collect(args=_args(), fd_exec_path=None)

    assert sys.modules["kinder"].__file__ is not None
    assert snapshot.kindergarden_commit == head


# --- the rest of the run's conditions -----------------------------------------


def test_collect_records_the_numerical_stack() -> None:
    """The versions that were varying between machines are exactly the ones a
    reader needs in order to rule the environment in or out."""
    snapshot = ConfigSnapshot.collect(args=_args(seed=0), fd_exec_path=None)
    assert snapshot.torch_version == torch.__version__
    assert snapshot.numpy_version == np.__version__
    assert snapshot.python_version
    assert snapshot.platform_machine


def test_collect_records_resolved_args_including_defaulted_ones() -> None:
    """The motivating gap: --sampler-max-train-iters had its default changed
    mid-project, so a run that never passed it explicitly could not have its
    iteration count recovered afterwards."""
    snapshot = ConfigSnapshot.collect(
        args=_args(seed=3, sampler_max_train_iters=10000), fd_exec_path=None
    )
    assert snapshot.args["seed"] == "3"
    assert snapshot.args["sampler_max_train_iters"] == "10000"


def test_collect_marks_fast_downward_unset_when_it_is() -> None:
    snapshot = ConfigSnapshot.collect(args=_args(), fd_exec_path=None)
    assert snapshot.fast_downward_commit == "unset"


def test_collect_still_succeeds_when_fast_downward_path_is_not_a_repository(
    *, tmp_path: Path
) -> None:
    snapshot = ConfigSnapshot.collect(args=_args(), fd_exec_path=str(tmp_path))
    assert snapshot.fast_downward_commit == "unknown"


def test_git_degrades_to_unknown_outside_a_repository(*, tmp_path: Path) -> None:
    """This is diagnostic metadata -- a missing .git must never be the reason a run
    that already burned hours of compute fails at the write step."""
    assert ConfigSnapshot._git(repo=tmp_path, args=["rev-parse", "HEAD"]) == "unknown"


def test_snapshot_round_trips_through_json() -> None:
    """It is written to disk and read back by whoever is diagnosing a mismatch."""
    original = ConfigSnapshot.collect(args=_args(seed=1), fd_exec_path=None)
    assert ConfigSnapshot.model_validate_json(original.model_dump_json()) == original
