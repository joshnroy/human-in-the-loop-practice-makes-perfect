"""No prose outside `docs/experiment-logs/` may point a reader at a module #141 deleted.

The sibling to `test_cli.py`'s `RETIRED_ENVIRONMENT_NAMES`, one level up: that pins the
retired `--env` *strings* out of the registry, this pins the retired *files* out of
anything a reader would try to open or execute.

**Why the experiment-log carve-out is the whole design.** A log records what was actually
run, so a reproduction command naming `tossingroomsplit_reset_policy` is a true statement
about history and editing it would make the log false. A module docstring naming the same
file is a pointer, and a pointer to a deleted file is simply broken. Same string, opposite
treatment, decided by which directory it sits in.

`practice_diagnostics.py` is the case that motivated this: its docstring named
`scripts/tossingroomsplit_skill_traces.py` as "the previous route to these numbers" for a
full release after #141 deleted it, and nothing failed, because prose has no compiler.
"""

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

# The `analysis/` modules and `scripts/` entrypoints #141 deleted when it retired the
# three superseded Tossing Room forks. Stems rather than paths: a stale reference is just
# as broken when it spells the module dotted (`analysis.practice_makes_perfect.x`) or
# names the test file (`test_x.py`) instead of the module.
RETIRED_MODULE_STEMS = (
    "tossingroom_comparison",
    "tossingroom_goal_family_curves",
    "tossingroom_horizon_sweep",
    "tossingroom_horizon_table",
    "tossingroom_plan_traces",
    "tossingroom_reset_frequency",
    "tossingroom_reset_interval",
    "tossingroom_throw_convergence",
    "tossingroom_throw_traces",
    "tossingroomsplit_family_overlay",
    "tossingroomsplit_practice_pools",
    "tossingroomsplit_reset_policy",
    "tossingroomsplit_scaling",
    "tossingroomsplit_skill_traces",
    "tossingroomsplit_throw_rates",
    "tossingroomsplit_two_way_ledge",
)

# Where a reader goes looking for something to run. `docs/experiment-logs/` is deliberately
# absent -- see the module docstring.
_SEARCHED_DIRECTORIES = ("src", "analysis", "scripts", "tests")
_SEARCHED_SUFFIXES = frozenset({".py", ".md", ".sh", ".toml", ".yml", ".yaml"})

# This file has to name every retired stem in order to pin it.
_EXEMPT = frozenset({Path(__file__).name})


def _searched_files() -> list[Path]:
    found = [
        path
        for directory in _SEARCHED_DIRECTORIES
        for path in (_REPO / directory).rglob("*")
        if path.is_file()
        and path.suffix in _SEARCHED_SUFFIXES
        and "__pycache__" not in path.parts
        and path.name not in _EXEMPT
    ]
    found.extend(path for path in _REPO.glob("*.md") if path.name not in _EXEMPT)
    return found


@pytest.mark.parametrize("stem", RETIRED_MODULE_STEMS)
def test_no_live_prose_points_at_a_retired_module(*, stem: str) -> None:
    """A retired stem surviving here is a reader sent to a file that is not there."""
    offenders = [
        f"{path.relative_to(_REPO)}:{number}"
        for path in _searched_files()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if stem in line
    ]
    assert offenders == [], f"`{stem}` was deleted by #141 but is still named in: {offenders}"


@pytest.mark.parametrize("stem", RETIRED_MODULE_STEMS)
def test_the_retired_modules_really_are_gone(*, stem: str) -> None:
    """The guard above is vacuous if one of these stems quietly comes back as a real
    file -- then it is a live module and naming it is correct, not a defect."""
    assert list(_REPO.glob(f"**/{stem}.py")) == []
