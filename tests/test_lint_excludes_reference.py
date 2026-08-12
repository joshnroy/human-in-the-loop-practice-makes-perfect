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
"""

import shutil
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_REFERENCE = _REPO / "reference"

# A populated submodule is the only state in which this is testable at all. A fresh clone
# and CI both leave `reference/` as empty directories, exactly as `CLAUDE.md` describes,
# and there is then nothing for ruff to walk.
_POPULATED = [p for p in (_REFERENCE.glob("*/")) if any(p.glob("**/*.py"))]
_NEEDS_SUBMODULES = pytest.mark.skipif(
    not _POPULATED, reason="reference/ submodules are not populated"
)
_NEEDS_RUFF = pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not on PATH")


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
