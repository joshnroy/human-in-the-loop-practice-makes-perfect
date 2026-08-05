"""`scripts/update_reference_repos.sh` is how `reference/` is kept current, so the
claims worth pinning are the ones whose failure would be silent or destructive.

Two of them are load-bearing:

* **The default branch is per-repo, not `main`.** `reference/predicators` sits on
  `master`. A script that assumed `main` would report a spurious "skipped: not on
  the default branch" for it forever, and the skip would look like a deliberate
  choice rather than a bug.
* **Local work is never clobbered.** Someone may be mid-investigation inside a
  reference checkout -- a dirty tree, a scratch branch, a detached HEAD at some
  commit they are bisecting. The script must refuse to touch those rather than
  stash/reset/checkout over them, because the loss would be silent and
  unrecoverable.

Every test builds its remotes as local git repositories in a tmpdir and points the
script at them through `--manifest`. Nothing here touches the network, and nothing
here touches the real `reference/` directory.
"""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "update_reference_repos.sh"

# Exit codes the script contracts on, so a pre-flight check can act on them.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_SKIPPED = 2


def _git(*args: str, cwd: Path) -> str:
    """Run git with a fixed identity so commits work on a bare CI machine."""
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    completed = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, env=env
    )
    return completed.stdout.strip()


def _make_origin(*, tmp_path: Path, name: str, default_branch: str) -> Path:
    """Create a non-bare origin repo with one commit on `default_branch`.

    `git clone` reads the remote's own HEAD to decide what the default branch is,
    which is exactly the mechanism the script has to respect, so the fixture sets
    HEAD rather than relying on the ambient `init.defaultBranch`.
    """
    origin = tmp_path / "origins" / name
    origin.mkdir(parents=True)
    _git("init", f"--initial-branch={default_branch}", cwd=origin)
    (origin / "README.md").write_text("v1\n", encoding="utf-8")
    _git("add", "README.md", cwd=origin)
    _git("commit", "-m", "v1", cwd=origin)
    return origin


def _advance_origin(*, origin: Path, text: str) -> str:
    (origin / "README.md").write_text(text, encoding="utf-8")
    _git("add", "README.md", cwd=origin)
    _git("commit", "-m", text, cwd=origin)
    return _git("rev-parse", "HEAD", cwd=origin)


def _write_manifest(*, tmp_path: Path, entries: list[tuple[str, Path]]) -> Path:
    """Manifest format: one `name<TAB>url<TAB>extra-clone-flags` line per repo."""
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "".join(f"{name}\t{origin}\t\n" for name, origin in entries), encoding="utf-8"
    )
    return manifest


def _run(*, reference_dir: Path, manifest: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(SCRIPT),
            "--reference-dir",
            str(reference_dir),
            "--manifest",
            str(manifest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_script_exists_and_is_executable() -> None:
    """Documented as `scripts/update_reference_repos.sh`, not `bash scripts/...`.
    A lost mode bit is invisible in review."""
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)


def test_a_missing_repo_is_cloned(*, tmp_path: Path) -> None:
    origin = _make_origin(tmp_path=tmp_path, name="alpha", default_branch="main")
    reference_dir = tmp_path / "reference"
    manifest = _write_manifest(tmp_path=tmp_path, entries=[("alpha", origin)])
    result = _run(reference_dir=reference_dir, manifest=manifest)

    assert result.returncode == EXIT_OK, result.stderr
    assert (reference_dir / "alpha" / "README.md").read_text(encoding="utf-8") == "v1\n"
    assert "cloned" in result.stdout


def test_an_already_current_repo_is_a_no_op(*, tmp_path: Path) -> None:
    """Idempotence: the second run must change nothing and say so."""
    origin = _make_origin(tmp_path=tmp_path, name="alpha", default_branch="main")
    reference_dir = tmp_path / "reference"
    manifest = _write_manifest(tmp_path=tmp_path, entries=[("alpha", origin)])

    first = _run(reference_dir=reference_dir, manifest=manifest)
    assert first.returncode == EXIT_OK, first.stderr
    head_after_clone = _git("rev-parse", "HEAD", cwd=reference_dir / "alpha")

    second = _run(reference_dir=reference_dir, manifest=manifest)
    assert second.returncode == EXIT_OK, second.stderr
    assert "already current" in second.stdout
    assert _git("rev-parse", "HEAD", cwd=reference_dir / "alpha") == head_after_clone


def test_a_behind_repo_fast_forwards(*, tmp_path: Path) -> None:
    origin = _make_origin(tmp_path=tmp_path, name="alpha", default_branch="main")
    reference_dir = tmp_path / "reference"
    manifest = _write_manifest(tmp_path=tmp_path, entries=[("alpha", origin)])
    _run(reference_dir=reference_dir, manifest=manifest)

    new_sha = _advance_origin(origin=origin, text="v2\n")
    result = _run(reference_dir=reference_dir, manifest=manifest)

    assert result.returncode == EXIT_OK, result.stderr
    assert (reference_dir / "alpha" / "README.md").read_text(encoding="utf-8") == "v2\n"
    assert _git("rev-parse", "HEAD", cwd=reference_dir / "alpha") == new_sha
    # The report must name both ends of the move, not just say "updated".
    assert new_sha[:7] in result.stdout


def test_a_repo_whose_default_branch_is_master_is_handled(*, tmp_path: Path) -> None:
    """`reference/predicators` is on `master`. Hardcoding `main` would break it,
    and the breakage would look like a legitimate skip."""
    origin = _make_origin(tmp_path=tmp_path, name="oldstyle", default_branch="master")
    reference_dir = tmp_path / "reference"
    manifest = _write_manifest(tmp_path=tmp_path, entries=[("oldstyle", origin)])
    _run(reference_dir=reference_dir, manifest=manifest)

    new_sha = _advance_origin(origin=origin, text="v2\n")
    result = _run(reference_dir=reference_dir, manifest=manifest)

    assert result.returncode == EXIT_OK, result.stderr
    assert _git("rev-parse", "HEAD", cwd=reference_dir / "oldstyle") == new_sha
    # Nothing was skipped: a `master` default must not read as "not on main".
    assert "skipped  0/1" in result.stdout


def test_a_dirty_tree_is_skipped_not_clobbered(*, tmp_path: Path) -> None:
    """Someone may be mid-investigation. Their edit must survive."""
    origin = _make_origin(tmp_path=tmp_path, name="alpha", default_branch="main")
    reference_dir = tmp_path / "reference"
    manifest = _write_manifest(tmp_path=tmp_path, entries=[("alpha", origin)])
    _run(reference_dir=reference_dir, manifest=manifest)

    dirty = reference_dir / "alpha" / "README.md"
    dirty.write_text("local edit I care about\n", encoding="utf-8")
    _advance_origin(origin=origin, text="v2\n")

    result = _run(reference_dir=reference_dir, manifest=manifest)

    assert result.returncode == EXIT_SKIPPED
    assert dirty.read_text(encoding="utf-8") == "local edit I care about\n"
    assert "skipped" in result.stdout.lower()
    assert "dirty" in result.stdout.lower()


def test_a_non_default_branch_is_skipped_not_clobbered(*, tmp_path: Path) -> None:
    origin = _make_origin(tmp_path=tmp_path, name="alpha", default_branch="main")
    reference_dir = tmp_path / "reference"
    manifest = _write_manifest(tmp_path=tmp_path, entries=[("alpha", origin)])
    _run(reference_dir=reference_dir, manifest=manifest)

    checkout = reference_dir / "alpha"
    _git("checkout", "-b", "my-investigation", cwd=checkout)
    _advance_origin(origin=origin, text="v2\n")

    result = _run(reference_dir=reference_dir, manifest=manifest)

    assert result.returncode == EXIT_SKIPPED
    assert _git("rev-parse", "--abbrev-ref", "HEAD", cwd=checkout) == "my-investigation"
    assert (checkout / "README.md").read_text(encoding="utf-8") == "v1\n"
    assert "skipped" in result.stdout.lower()


def test_a_detached_head_is_skipped_not_clobbered(*, tmp_path: Path) -> None:
    origin = _make_origin(tmp_path=tmp_path, name="alpha", default_branch="main")
    reference_dir = tmp_path / "reference"
    manifest = _write_manifest(tmp_path=tmp_path, entries=[("alpha", origin)])
    _run(reference_dir=reference_dir, manifest=manifest)

    checkout = reference_dir / "alpha"
    pinned = _git("rev-parse", "HEAD", cwd=checkout)
    _git("checkout", "--detach", pinned, cwd=checkout)
    _advance_origin(origin=origin, text="v2\n")

    result = _run(reference_dir=reference_dir, manifest=manifest)

    assert result.returncode == EXIT_SKIPPED
    assert _git("rev-parse", "HEAD", cwd=checkout) == pinned
    assert "skipped" in result.stdout.lower()


def test_one_skipped_repo_does_not_stop_the_others(*, tmp_path: Path) -> None:
    """A pre-flight check should still update everything it safely can."""
    alpha = _make_origin(tmp_path=tmp_path, name="alpha", default_branch="main")
    beta = _make_origin(tmp_path=tmp_path, name="beta", default_branch="master")
    reference_dir = tmp_path / "reference"
    manifest = _write_manifest(tmp_path=tmp_path, entries=[("alpha", alpha), ("beta", beta)])
    _run(reference_dir=reference_dir, manifest=manifest)

    (reference_dir / "alpha" / "README.md").write_text("dirty\n", encoding="utf-8")
    beta_sha = _advance_origin(origin=beta, text="v2\n")

    result = _run(reference_dir=reference_dir, manifest=manifest)

    assert result.returncode == EXIT_SKIPPED
    assert _git("rev-parse", "HEAD", cwd=reference_dir / "beta") == beta_sha


def test_the_summary_reports_counts_as_x_of_y(*, tmp_path: Path) -> None:
    """House rule: counts, never bare percentages."""
    alpha = _make_origin(tmp_path=tmp_path, name="alpha", default_branch="main")
    beta = _make_origin(tmp_path=tmp_path, name="beta", default_branch="main")
    reference_dir = tmp_path / "reference"
    manifest = _write_manifest(tmp_path=tmp_path, entries=[("alpha", alpha), ("beta", beta)])

    result = _run(reference_dir=reference_dir, manifest=manifest)

    assert result.returncode == EXIT_OK, result.stderr
    assert "2/2" in result.stdout
    assert "%" not in result.stdout


def test_the_builtin_manifest_covers_the_three_reference_repos() -> None:
    """The default manifest is what runs when nobody passes `--manifest`, so the
    set of repos it names is part of the contract. Checked by reading the script,
    which needs no network and no clone."""
    body = SCRIPT.read_text(encoding="utf-8")
    for name in ("kindergarden", "kinder-baselines", "predicators"):
        assert name in body
    # Its pack is 1.08 GiB; a full-blob clone is a several-minute mistake.
    assert "--filter=blob:none" in body


def test_a_failing_repo_exits_nonzero(*, tmp_path: Path) -> None:
    """A URL that cannot be cloned must fail loudly, not be silently skipped."""
    reference_dir = tmp_path / "reference"
    manifest = _write_manifest(tmp_path=tmp_path, entries=[("ghost", tmp_path / "does-not-exist")])

    result = _run(reference_dir=reference_dir, manifest=manifest)

    assert result.returncode == EXIT_FAILED
    assert "failed" in result.stdout.lower() or "failed" in result.stderr.lower()
