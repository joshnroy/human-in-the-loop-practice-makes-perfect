"""`scripts/with_kinder_env.sh` used to activate a *second interpreter* -- a `kinder-venv`
virtualenv holding MuJoCo -- on the stated grounds that KINDER "cannot be merged" into
`hitl-pmp`. That justification was measured to be false (both were already Python 3.10.20;
the real constraint is `pybullet_helpers`'s `numpy<2.0` plus an exact `scipy==1.14.0`, and
`moviepy`'s `pillow<12.0`, all of which `hitl-pmp` tolerates). KINDER now installs into
`hitl-pmp` itself via the `tossing3d` extra.

So this wrapper is a thin alias for `with_env.sh`, and what is worth pinning changed with
it. Its claims now:

1. It can be invoked as written (the mode bit is committed).
2. It puts *this* checkout on the Python path -- still the load-bearing claim, since a
   worktree that does not set PYTHONPATH silently imports the main checkout's library and
   nothing errors; the run just measures the wrong thing.
3. It resolves the **same interpreter** as `with_env.sh`. This is the inversion of the old
   test, which asserted `"kinder-venv" in sys.executable`. That assertion passing again
   would mean the split had been reintroduced.
4. It adds the thread pins, which is now its *only* reason to exist as a separate file.
   Without them a bare CLI run inherits the machine default (24 here) while a swept run
   gets 1, multi-threaded float reductions reassociate, and the same seed trains to
   different weights -- so a hand-run reproduction and the sweep it reproduces are two
   different experiments. They are deliberately not in `with_env.sh`, which would impose
   them on non-simulator work.
5. It still yields an environment that can import KINDER, when KINDER is installed.

Only the last one needs KINDER present; the rest hold on a bare checkout, which is what
CI runs.
"""

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "scripts" / "with_kinder_env.sh"
PLAIN_WRAPPER = REPO_ROOT / "scripts" / "with_env.sh"

CONDA_SH_CANDIDATES = (
    "miniconda3/etc/profile.d/conda.sh",
    "anaconda3/etc/profile.d/conda.sh",
    "miniforge3/etc/profile.d/conda.sh",
)


def _conda_is_available() -> bool:
    """Same probe `with_env.sh` performs. CI installs with plain pip and has no conda, so
    every test that executes the wrapper skips there."""
    if os.environ.get("CONDA_SH"):
        return Path(os.environ["CONDA_SH"]).is_file()
    home = Path.home()
    if any((home / candidate).is_file() for candidate in CONDA_SH_CANDIDATES):
        return True
    return Path("/opt/conda/etc/profile.d/conda.sh").is_file()


needs_conda = pytest.mark.skipif(
    not _conda_is_available(), reason="no conda installation; the wrapper cannot activate an env"
)

# The *import* package name, not the distribution (`kindergarden`) -- the same gate every
# other simulator-backed test in this repo uses, so an environment without the optional
# `tossing3d` extra skips rather than fails.
needs_kinder = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None,
    reason="KINDER is an optional extra (`.[tossing3d]`) and CI never installs it",
)


def _run(*, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [str(WRAPPER), *args], capture_output=True, text=True, check=True, cwd="/"
    )


def test_the_wrapper_exists_and_is_executable() -> None:
    """Documented as `scripts/with_kinder_env.sh <command>`, not `bash scripts/...`. That
    only works if the mode bit is committed, and a lost mode bit is invisible in review."""
    assert WRAPPER.is_file()
    assert os.access(WRAPPER, os.X_OK)


@needs_conda
def test_the_wrapper_puts_this_checkout_on_the_python_path() -> None:
    """The load-bearing claim. In a worktree this must resolve to the worktree's own
    src/, never the main checkout's -- see the module docstring."""
    completed = _run(args=["python", "-c", "import hitl_pmp; print(hitl_pmp.__file__)"])
    assert Path(completed.stdout.strip()) == REPO_ROOT / "src" / "hitl_pmp" / "__init__.py"


@needs_conda
def test_the_wrapper_resolves_the_same_interpreter_as_the_plain_wrapper() -> None:
    """The environments are unified, so there is exactly one interpreter. This is the
    inversion of the old `"kinder-venv" in sys.executable` assertion: if that one could
    pass again, the split would have been reintroduced."""
    script = "import sys; print(sys.executable)"
    through_alias = _run(args=["python", "-c", script]).stdout.strip()
    through_plain = subprocess.run(  # noqa: S603
        [str(PLAIN_WRAPPER), "python", "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd="/",
    ).stdout.strip()
    assert through_alias == through_plain
    assert "kinder-venv" not in through_alias


@needs_conda
def test_the_wrapper_pins_the_math_threads_to_one() -> None:
    """The alias's only remaining reason to exist. `run_sweep.py` sets these on its
    children; pinning them here is what makes a hand-run reproduction match the sweep it
    is reproducing rather than being a different experiment."""
    script = "import os; print(os.environ['OMP_NUM_THREADS'], os.environ['MKL_NUM_THREADS'])"
    assert _run(args=["python", "-c", script]).stdout.split() == ["1", "1"]


@needs_conda
def test_the_wrapper_still_pins_the_egl_rendering_backend() -> None:
    """Inherited from `with_env.sh` now rather than set here, but the guarantee callers
    depend on is unchanged, so it stays asserted on this path too -- an alias that
    silently dropped it would reopen the silent-skip trap."""
    script = "import os; print(os.environ['MUJOCO_GL'], os.environ['PYOPENGL_PLATFORM'])"
    assert _run(args=["python", "-c", script]).stdout.split() == ["egl", "egl"]


@needs_conda
def test_with_no_command_it_reports_what_it_resolved() -> None:
    """Same affordance as `with_env.sh`: running it bare is the sanity check an agent is
    told to perform before trusting any number, so it has to print the resolved
    interpreter and the `hitl_pmp` it will actually import."""
    completed = _run(args=[])
    assert "PYTHONPATH" in completed.stdout
    assert "hitl_pmp" in completed.stdout
    assert str(REPO_ROOT) in completed.stdout


@needs_conda
@needs_kinder
def test_the_resolved_environment_can_import_kinder() -> None:
    """The point of unification: the simulator is importable from the ordinary
    environment, so the fidelity tests run in the normal gate instead of skipping. Both
    packages, because `kinder` and `kinder_models` come from two different submodules and
    a half-install is a real and previously-observed state."""
    completed = _run(args=["python", "-c", "import kinder, kinder_models; print(kinder.__file__)"])
    assert completed.stdout.strip()
