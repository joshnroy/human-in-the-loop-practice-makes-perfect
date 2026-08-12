"""scipy is a *guaranteed* dependency here, unlike `wandb` and KINDER which are optional.

**Why this file exists.** Three optional packages in this repo are deliberately gated on
`importlib.util.find_spec` -- `wandb`, `kinder`, `kinder_models` -- so a machine without
them **skips** rather than fails. That pattern is correct for those, and wrong for scipy:
the point of adding scipy is that every environment running this gate has the *same*
statistical toolkit, so a claim computed by one agent is comparable to a claim computed by
another. An import guarded by `find_spec` would reintroduce exactly the split it removes.

**The split this closes.** Until scipy was declared, `hitl-pmp` had no scipy while the
KINDER venv had 1.14.0, so the same repository offered different statistical resolution
depending on which interpreter an agent happened to run in -- agents in the KINDER venv
reported scipy-computed statistics, agents in `hitl-pmp` hand-rolled substitutes. Two
analyses that are not directly comparable is a bad property for a research codebase.

**What this file does not claim.** It does not claim scipy raises the p-value floor of the
exact paired tests in `analysis/practice_makes_perfect/paired_tests.py`. It does not: an
exact two-sided sign-flip test over n pairs has a smallest attainable p of `2 / 2**n`
regardless of who implements it, and scipy's own `wilcoxon(method="exact")` and
`permutation_test(n_resamples=np.inf)` both return exactly `1.953125e-03` at n = 10 --
identical to the hand-rolled value. That floor is a property of the test and the sample
size, not of the implementation. scipy earns its place here by making the *toolkit*
uniform, not by making any existing number more significant.
"""

import importlib.util
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def test_scipy_is_importable() -> None:
    """The invariant itself: scipy is present wherever this suite runs, CI included.

    Deliberately not skipped when scipy is missing. A `find_spec` skip is the right
    treatment for an optional extra and the wrong one here -- it would let the very
    environment split this dependency exists to remove pass silently, which is the
    failure mode this repo has already paid for with a green gate over tests that never
    executed.
    """
    assert importlib.util.find_spec("scipy") is not None, (
        "scipy is not importable. It is a declared, non-optional dependency of this "
        'repo\'s `dev` extra -- install it with `pip install -e ".[dev]"`. It is '
        "deliberately not `find_spec`-gated the way `wandb` and KINDER are."
    )


def test_scipy_is_declared_in_the_dev_extra() -> None:
    """Importable-by-accident is not the same as declared.

    scipy arrives transitively in any environment that has installed KINDER, because
    `pybullet_helpers` requires it. So the import test above can pass on a machine where
    nothing in this repo's own metadata asks for scipy at all, and a fresh `.[dev]`
    install would then not get it. This asserts the declaration, which is what CI
    actually installs from.

    Parsed as text rather than with `tomllib`, which is 3.11+ while this project targets
    3.10, and rather than via `importlib.metadata`, which in a worktree reports the *main*
    checkout's editable install and so would answer for the wrong `pyproject.toml`.
    """
    pyproject = (_REPO / "pyproject.toml").read_text()
    dev_block = re.search(r"^dev = \[(.*?)^\]", pyproject, re.DOTALL | re.MULTILINE)
    assert dev_block is not None, "could not find the `dev` extra in pyproject.toml"
    declared = re.findall(r'"([A-Za-z0-9_.\-]+)', dev_block.group(1))
    assert "scipy" in declared, (
        f"scipy is not declared in the `dev` extra; found {declared}. CI installs "
        '`pip install -e ".[dev]"` in all three jobs, so an undeclared scipy is a scipy '
        "CI does not have."
    )
