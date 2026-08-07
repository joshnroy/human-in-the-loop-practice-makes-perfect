"""`scripts/with_kinder_env.sh` is `with_env.sh`'s KINDER twin: the one command that
runs a Tossing3D job against the venv that actually has MuJoCo.

Its claims are the same two `test_with_env.py` pins, plus one this side needs and the
conda side does not. In order:

1. It can be invoked as written (the mode bit is committed).
2. It puts *this* checkout on the Python path -- the load-bearing claim, since a
   worktree that does not set PYTHONPATH silently imports the main checkout's library
   and nothing errors; the run just measures the wrong thing.
3. It uses the **KINDER venv's** interpreter, not conda's. `hitl-pmp` has no `kinder`
   at all, so a wrapper that fell back to it would fail loudly -- but the reverse
   mistake, running with KINDER's interpreter and the *main* checkout's `hitl_pmp`,
   fails silently and is the one worth pinning.
4. It exports `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`. This is not tidiness:
   `register_all_environments()` forces `osmesa` when `DISPLAY` is unset, under which
   `import mujoco` raises, and `_check_deps` swallows *every* exception -- so all
   `Dynamic3D` environments are skipped in silence and the failure surfaces much later
   as an unrelated-looking `NameNotFound`. That trap has cost an hour more than once.
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "scripts" / "with_kinder_env.sh"


def _kinder_venv_python() -> Path | None:
    """The venv the wrapper itself resolves to, or None. CI never creates it -- KINDER
    is an optional extra -- so every test that executes the wrapper skips there."""
    common = subprocess.run(  # noqa: S603
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
    )
    if common.returncode != 0:
        return None
    main_checkout = Path(common.stdout.strip()).parent
    candidate = main_checkout.parent / "kinder-venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


needs_kinder_venv = pytest.mark.skipif(
    _kinder_venv_python() is None,
    reason="no ../kinder-venv; KINDER is an optional extra and CI never installs it",
)


def _run(*, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [str(WRAPPER), *args], capture_output=True, text=True, check=True
    )


def test_the_wrapper_exists_and_is_executable() -> None:
    """Documented as `scripts/with_kinder_env.sh <command>`, not `bash scripts/...`.
    That only works if the mode bit is committed, and a lost mode bit is invisible in
    review."""
    assert WRAPPER.is_file()
    assert os.access(WRAPPER, os.X_OK)


@needs_kinder_venv
def test_the_wrapper_puts_this_checkout_on_the_python_path() -> None:
    """The load-bearing claim. In a worktree this must resolve to the worktree's own
    src/, never the main checkout's -- see the module docstring."""
    completed = _run(args=["python", "-c", "import hitl_pmp; print(hitl_pmp.__file__)"])
    assert completed.stdout.strip().startswith(str(REPO_ROOT))


@needs_kinder_venv
def test_the_wrapper_uses_the_kinder_interpreter_and_can_import_kinder() -> None:
    """`hitl-pmp` has no `kinder`; the venv does. Both halves are asserted because
    resolving the right interpreter and actually being able to import the simulator
    are separate failures."""
    completed = _run(args=["python", "-c", "import kinder, kinder_models; print(kinder.__file__)"])
    assert "kinder-venv" in _run(args=["python", "-c", "import sys; print(sys.executable)"]).stdout
    assert completed.stdout.strip()


@needs_kinder_venv
def test_the_wrapper_pins_the_egl_rendering_backend() -> None:
    """Both variables, because `register_all_environments()` falls back to `osmesa`
    when `DISPLAY` is unset and every Dynamic3D env is then skipped *in silence*."""
    script = "import os; print(os.environ['MUJOCO_GL'], os.environ['PYOPENGL_PLATFORM'])"
    completed = _run(args=["python", "-c", script])
    assert completed.stdout.split() == ["egl", "egl"]


@needs_kinder_venv
def test_with_no_command_it_reports_what_it_resolved() -> None:
    """Same affordance as `with_env.sh`: running it bare is the sanity check an agent
    is told to perform before trusting any number, so it has to print the resolved
    interpreter and the `hitl_pmp` it will actually import."""
    completed = _run(args=[])
    assert "PYTHONPATH" in completed.stdout
    assert "hitl_pmp" in completed.stdout
    assert str(REPO_ROOT) in completed.stdout
