"""`scripts/update_reference_repos.sh` keeps `reference/` in sync with the pins this
repository records, so the claims worth pinning are the ones whose failure would be
silent or destructive.

`reference/` used to be gitignored, and the script was the substitute for a pin: it
fast-forwarded each checkout to its own remote's default branch. Submodules are the
pin now, so two of the old contract's three rules are gone with it -- a submodule
tracks a *commit*, so there is no default branch to resolve (the old
`predicators`-is-on-`master` test is deliberately deleted, not ported) and nothing to
fast-forward.

The one rule that survives is the important one, and it is the reason this file exists:

* **Local work is never clobbered.** Someone may be mid-investigation inside a
  reference checkout -- an edit, a scratch branch, a commit they are bisecting. A
  submodule makes clobbering *easier* than it was, because `git submodule update` will
  happily detach a populated checkout back onto the recorded gitlink and say nothing.
  So the script only ever initialises a submodule that is **not yet there**; anything
  populated is reported, never touched.

Every test builds a synthetic superproject with local-path submodules in a tmpdir and
points the script at it with `--repo-root`. Nothing here touches the network, and
nothing here touches the real `reference/` directory.

Local-path submodules need `protocol.file.allow=always`: git has refused the `file`
transport for submodules by default since CVE-2022-39253. It is passed to the script
through `GIT_CONFIG_*` environment variables rather than baked into the script, so the
script itself carries no test-only affordance.
"""

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "update_reference_repos.sh"

# Exit codes the script contracts on, so a pre-flight check can act on them.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_SKIPPED = 2

# git refuses `file://` submodule transport by default (CVE-2022-39253). Every git
# invocation in this file -- and the script under test -- needs it re-enabled.
_ALLOW_FILE_TRANSPORT = {
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "protocol.file.allow",
    "GIT_CONFIG_VALUE_0": "always",
}

_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
}


def _env() -> dict[str, str]:
    return {**os.environ, **_GIT_IDENTITY, **_ALLOW_FILE_TRANSPORT}


def _git(*args: str, cwd: Path) -> str:
    """Run git with a fixed identity so commits work on a bare CI machine."""
    completed = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, env=_env()
    )
    return completed.stdout.strip()


def _make_origin(*, tmp_path: Path, name: str) -> Path:
    """Create a non-bare origin repo with one commit."""
    origin = tmp_path / "origins" / name
    origin.mkdir(parents=True)
    _git("init", "--initial-branch=main", cwd=origin)
    (origin / "README.md").write_text("v1\n", encoding="utf-8")
    _git("add", "README.md", cwd=origin)
    _git("commit", "-m", "v1", cwd=origin)
    return origin


def _advance_origin(*, origin: Path, text: str) -> str:
    (origin / "README.md").write_text(text, encoding="utf-8")
    _git("add", "README.md", cwd=origin)
    _git("commit", "-m", text, cwd=origin)
    return _git("rev-parse", "HEAD", cwd=origin)


def _make_superproject(*, tmp_path: Path, names: list[str]) -> tuple[Path, dict[str, Path]]:
    """A repo that records one gitlink per name under `reference/`, as this repo does.

    Returns the superproject and each submodule's origin, so a test can move the
    origin forward independently of the recorded pin.
    """
    super_root = tmp_path / "super"
    super_root.mkdir(parents=True)
    _git("init", "--initial-branch=main", cwd=super_root)
    (super_root / "README.md").write_text("superproject\n", encoding="utf-8")
    _git("add", "README.md", cwd=super_root)
    _git("commit", "-m", "init", cwd=super_root)

    origins: dict[str, Path] = {}
    for name in names:
        origin = _make_origin(tmp_path=tmp_path, name=name)
        origins[name] = origin
        _git("submodule", "add", str(origin), f"reference/{name}", cwd=super_root)
    _git("commit", "-m", "record the reference pins", cwd=super_root)
    return super_root, origins


def _deinit(*, super_root: Path, name: str) -> None:
    """Return a submodule to the state a fresh clone is in: recorded, not populated."""
    _git("submodule", "deinit", "--force", f"reference/{name}", cwd=super_root)


def _run(*, super_root: Path, extra: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), "--repo-root", str(super_root), *(extra or [])],
        capture_output=True,
        text=True,
        check=False,
        env=_env(),
    )


def test_the_script_exists_and_is_executable() -> None:
    """Documented as `scripts/update_reference_repos.sh`, not `bash scripts/...`.
    A lost mode bit is invisible in review."""
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)


def test_an_uninitialised_submodule_is_initialised(*, tmp_path: Path) -> None:
    """The fresh-clone case: `git clone` records gitlinks but leaves empty directories,
    so the whole of `reference/` is absent until something populates it."""
    super_root, _ = _make_superproject(tmp_path=tmp_path, names=["alpha"])
    _deinit(super_root=super_root, name="alpha")
    checkout = super_root / "reference" / "alpha"
    assert not (checkout / "README.md").exists()

    result = _run(super_root=super_root)

    assert result.returncode == EXIT_OK, result.stdout + result.stderr
    assert (checkout / "README.md").read_text(encoding="utf-8") == "v1\n"
    assert "initialised" in result.stdout


def test_an_already_initialised_submodule_is_a_no_op(*, tmp_path: Path) -> None:
    """Idempotence: the second run must change nothing and say so."""
    super_root, _ = _make_superproject(tmp_path=tmp_path, names=["alpha"])
    _deinit(super_root=super_root, name="alpha")

    first = _run(super_root=super_root)
    assert first.returncode == EXIT_OK, first.stdout + first.stderr
    checkout = super_root / "reference" / "alpha"
    head_after_init = _git("rev-parse", "HEAD", cwd=checkout)

    second = _run(super_root=super_root)

    assert second.returncode == EXIT_OK, second.stdout + second.stderr
    assert "already current" in second.stdout
    assert _git("rev-parse", "HEAD", cwd=checkout) == head_after_init


def test_a_drifted_submodule_is_reported_not_reset(*, tmp_path: Path) -> None:
    """The commit someone is sitting on is work. `git submodule update` would detach it
    back onto the pin without a word; this script must not."""
    super_root, origins = _make_superproject(tmp_path=tmp_path, names=["alpha"])
    checkout = super_root / "reference" / "alpha"
    pinned = _git("rev-parse", "HEAD", cwd=checkout)
    moved = _advance_origin(origin=origins["alpha"], text="v2\n")
    _git("fetch", "origin", cwd=checkout)
    _git("checkout", "--detach", moved, cwd=checkout)

    result = _run(super_root=super_root)

    assert result.returncode == EXIT_SKIPPED, result.stdout + result.stderr
    assert _git("rev-parse", "HEAD", cwd=checkout) == moved
    assert moved != pinned
    assert "drift" in result.stdout.lower()
    # Naming both ends is the point: "drifted" alone does not say from what, to what.
    assert moved[:7] in result.stdout
    assert pinned[:7] in result.stdout


def test_a_dirty_submodule_is_reported_not_clobbered(*, tmp_path: Path) -> None:
    """Someone may be mid-investigation. Their edit must survive."""
    super_root, _ = _make_superproject(tmp_path=tmp_path, names=["alpha"])
    dirty = super_root / "reference" / "alpha" / "README.md"
    dirty.write_text("local edit I care about\n", encoding="utf-8")

    result = _run(super_root=super_root)

    assert result.returncode == EXIT_SKIPPED, result.stdout + result.stderr
    assert dirty.read_text(encoding="utf-8") == "local edit I care about\n"
    assert "dirty" in result.stdout.lower()


def test_a_local_branch_at_the_pin_survives(*, tmp_path: Path) -> None:
    """A submodule at the recorded commit but on a named branch is in sync, not drifted
    -- and must still be reported without being detached back onto the gitlink."""
    super_root, _ = _make_superproject(tmp_path=tmp_path, names=["alpha"])
    checkout = super_root / "reference" / "alpha"
    _git("checkout", "-b", "my-investigation", cwd=checkout)

    result = _run(super_root=super_root)

    assert result.returncode == EXIT_OK, result.stdout + result.stderr
    assert _git("rev-parse", "--abbrev-ref", "HEAD", cwd=checkout) == "my-investigation"
    assert "already current" in result.stdout


def test_one_drifted_submodule_does_not_stop_the_others(*, tmp_path: Path) -> None:
    """A pre-flight check should still restore everything it safely can."""
    super_root, origins = _make_superproject(tmp_path=tmp_path, names=["alpha", "beta"])
    alpha = super_root / "reference" / "alpha"
    moved = _advance_origin(origin=origins["alpha"], text="v2\n")
    _git("fetch", "origin", cwd=alpha)
    _git("checkout", "--detach", moved, cwd=alpha)
    _deinit(super_root=super_root, name="beta")

    result = _run(super_root=super_root)

    assert result.returncode == EXIT_SKIPPED, result.stdout + result.stderr
    assert _git("rev-parse", "HEAD", cwd=alpha) == moved
    assert (super_root / "reference" / "beta" / "README.md").exists()


def test_check_reports_without_initialising(*, tmp_path: Path) -> None:
    """`--check` is the cheap pre-flight: a worktree must be able to ask "am I in sync?"
    without paying kindergarden's 1.08 GiB pack to find out it is not."""
    super_root, _ = _make_superproject(tmp_path=tmp_path, names=["alpha"])
    _deinit(super_root=super_root, name="alpha")
    checkout = super_root / "reference" / "alpha"

    result = _run(super_root=super_root, extra=["--check"])

    assert result.returncode == EXIT_SKIPPED, result.stdout + result.stderr
    assert not (checkout / "README.md").exists()
    assert "not initialised" in result.stdout.lower()


def test_the_summary_reports_counts_as_x_of_y(*, tmp_path: Path) -> None:
    """House rule: counts, never bare percentages."""
    super_root, _ = _make_superproject(tmp_path=tmp_path, names=["alpha", "beta"])

    result = _run(super_root=super_root)

    assert result.returncode == EXIT_OK, result.stdout + result.stderr
    assert "2/2" in result.stdout
    assert "%" not in result.stdout


def test_an_uninitialisable_submodule_exits_nonzero(*, tmp_path: Path) -> None:
    """An origin that has gone away must fail loudly, not be silently skipped.

    `git submodule deinit` leaves the clone cached under `.git/modules`, so re-init
    would otherwise succeed with no origin at all -- the cache has to go too, or this
    test passes without ever reaching the network path it claims to cover.
    """
    super_root, origins = _make_superproject(tmp_path=tmp_path, names=["alpha"])
    _deinit(super_root=super_root, name="alpha")
    shutil.rmtree(origins["alpha"])
    shutil.rmtree(super_root / ".git" / "modules" / "reference" / "alpha")

    result = _run(super_root=super_root)

    assert result.returncode == EXIT_FAILED, result.stdout + result.stderr
    assert "failed" in (result.stdout + result.stderr).lower()


def test_a_non_repository_fails_rather_than_reporting_success(*, tmp_path: Path) -> None:
    """Exit 0 from a directory that is not a git repository would make the pre-flight
    check a no-op that always passes."""
    not_a_repo = tmp_path / "elsewhere"
    not_a_repo.mkdir()

    result = _run(super_root=not_a_repo)

    assert result.returncode == EXIT_FAILED, result.stdout + result.stderr


def test_the_recorded_pins_are_the_three_reference_repos() -> None:
    """`.gitmodules` is the pin now, so the set of submodules it records is part of the
    contract the script implements. Read from the file, which needs no network and does
    not touch the real `reference/` checkouts."""
    gitmodules = (REPO_ROOT / ".gitmodules").read_text(encoding="utf-8")
    for path in ("reference/kindergarden", "reference/kinder-baselines", "reference/predicators"):
        assert f"path = {path}" in gitmodules
    # Two of the three are forks on purpose: the commits this repo depends on live on
    # unmerged branches, and a submodule hard-fails when a pinned SHA is force-pushed
    # away. A fork Josh controls cannot be force-pushed under us.
    assert "joshnroy/kindergarden" in gitmodules
    assert "joshnroy/kinder-baselines" in gitmodules


def test_kindergarden_is_initialised_blobless_by_default() -> None:
    """Its pack is 1.08 GiB; a full-blob clone is a several-minute mistake. Checked by
    reading the script, which needs no network and no clone."""
    body = SCRIPT.read_text(encoding="utf-8")
    assert "--filter=blob:none" in body
