"""`reference/` holds third-party checkouts, so the linter must not walk into it.

The five-command gate in `CLAUDE.md` is documented as runnable verbatim. It is not, in
any worktree that has run `scripts/update_reference_repos.sh`: ruff walks the populated
submodules and reports **10,824 errors across 703 files** of upstream code that this repo
neither owns nor can fix. `ruff format --check .` fails the same way, wanting to reformat
703 files.

CI never sees this, because CI never populates the submodules -- so the failure lands
only on someone doing KINDER work, which is everyone touching Tossing3D.

**Only ruff has the problem, and that is worth stating so nobody "fixes" the others.**
`mypy` is invoked as `mypy src`; `pytest` has `testpaths = ["tests"]`;
`scripts/check_doc_links.sh` scopes itself to `docs/experiment-logs/`; and
`lint-imports` works from its own `[tool.importlinter]` package list. All four are
already scoped by construction. Ruff alone is invoked as `.`.

This is a behavioural test, not a config-string assertion: it asks ruff itself which
files it would check, so it keeps passing if the exclusion is expressed some other way
and fails if `reference/` ever becomes visible again.

**`downward/` is the same problem arriving from the other direction.** Fast Downward is
a third-party checkout too, but it is found by a sibling-directory convention, so on a
workstation it sits *beside* this repo and ruff never sees it. CI caches it at
`downward/` **inside the repo root**, where `ruff check .` walks 1,444 errors of
upstream `src/translate/` code -- so this one fails only on CI, exactly inverting
`reference/`, which is populated only locally.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_REFERENCE = _REPO / "reference"
_DOWNWARD = _REPO / "downward"

# A populated submodule is the only state in which this is testable at all. A fresh clone
# and CI both leave `reference/` as empty directories, exactly as `CLAUDE.md` describes,
# and there is then nothing for ruff to walk.
_POPULATED = [p for p in (_REFERENCE.glob("*/")) if any(p.glob("**/*.py"))]
_NEEDS_SUBMODULES = pytest.mark.skipif(
    not _POPULATED, reason="reference/ submodules are not populated"
)
_NEEDS_RUFF = pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not on PATH")

# Same reasoning as `_POPULATED`, mirrored: a checkout that keeps Fast Downward as a
# sibling directory -- every local worktree -- has nothing here for ruff to walk, so the
# assertion is only meaningful where CI puts it.
_NEEDS_DOWNWARD = pytest.mark.skipif(
    not any(_DOWNWARD.glob("**/*.py")) if _DOWNWARD.is_dir() else True,
    reason="Fast Downward is not checked out inside the repo root",
)


@_NEEDS_RUFF
@_NEEDS_DOWNWARD
def test_ruff_check_does_not_walk_a_fast_downward_checkout_inside_the_repo() -> None:
    """Ask ruff for the file list it would lint, as above -- CI caches Fast Downward at
    `downward/` in the workspace, and its `src/translate/` is upstream code this repo
    neither owns nor can fix."""
    result = subprocess.run(
        ["ruff", "check", "--show-files", "."],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    walked = [
        line
        for line in result.stdout.splitlines()
        if line.strip() and Path(line.strip()).is_relative_to(_DOWNWARD)
    ]
    assert not walked, (
        f"ruff would lint {len(walked)} file(s) inside downward/, e.g. {walked[:3]}. "
        "Fast Downward must stay excluded or the documented gate cannot be run where CI "
        "checks it out inside the repo root."
    )


@_NEEDS_RUFF
@_NEEDS_SUBMODULES
def test_ruff_check_does_not_walk_the_reference_submodules() -> None:
    """Ask ruff for the file list it would lint, rather than trusting the config key."""
    result = subprocess.run(
        ["ruff", "check", "--show-files", "."],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    walked = [
        line
        for line in result.stdout.splitlines()
        if line.strip() and Path(line.strip()).is_relative_to(_REFERENCE)
    ]
    assert not walked, (
        f"ruff would lint {len(walked)} file(s) inside reference/, e.g. {walked[:3]}. "
        "Third-party checkouts must stay excluded or the documented gate cannot be run "
        "in a worktree with submodules populated."
    )


@_NEEDS_RUFF
@_NEEDS_SUBMODULES
def test_the_documented_gate_commands_pass_with_submodules_populated() -> None:
    """The actual promise: `ruff check .` and `ruff format --check .`, exactly as
    `CLAUDE.md` writes them, succeed in a worktree carrying the submodules."""
    for argv in (["ruff", "check", "."], ["ruff", "format", "--check", "."]):
        result = subprocess.run(argv, cwd=_REPO, capture_output=True, text=True, check=False)
        assert result.returncode == 0, (
            f"`{' '.join(argv)}` exited {result.returncode} with submodules populated.\n"
            f"{result.stdout[-2000:]}"
        )
