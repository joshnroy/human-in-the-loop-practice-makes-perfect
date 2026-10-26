"""`scripts/with_env.sh` is the one command the `hitl-env` skill tells every agent
to prefix its work with, so its two central claims are worth pinning: that it can
be invoked as written, and that it puts *this* checkout on the Python path.

The second one is the load-bearing claim. A worktree that does not set PYTHONPATH
silently imports the main checkout's library -- the editable install's .pth file
holds an absolute path -- and nothing errors; the run just measures the wrong
thing. A wrapper that quietly failed to fix that would be worse than none, since
the skill instructs people to trust it.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "scripts" / "with_env.sh"


@pytest.mark.parametrize("explicit", [False, True])
def test_python_temporaries_use_disk_storage_from_any_working_directory(*, tmp_path, explicit):
    conda_sh = tmp_path / "conda.sh"
    conda_sh.write_text("conda() { return 0; }\n")
    environment = {**os.environ, "CONDA_SH": str(conda_sh)}
    environment.pop("HITL_PMP_TMPDIR", None)
    expected = REPO_ROOT / "artifacts/tmp"
    if explicit:
        expected = tmp_path / "gpfs scratch"
        environment["HITL_PMP_TMPDIR"] = str(expected)
    code = (
        "import json,os,tempfile; "
        "print(json.dumps([os.environ[k] for k in ('TMP','TMPDIR','TEMP')]"
        "+[tempfile.gettempdir()]))"
    )
    result = subprocess.run(
        [str(WRAPPER), sys.executable, "-c", code],
        cwd="/",
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == [str(expected)] * 4


# The same locations the wrapper itself probes. CI installs with plain pip and has
# no conda at all, so the tests that actually execute the wrapper skip there --
# `hitl-pmp` is a local-development environment, not a CI one.
CONDA_SH_CANDIDATES = (
    "miniconda3/etc/profile.d/conda.sh",
    "anaconda3/etc/profile.d/conda.sh",
    "miniforge3/etc/profile.d/conda.sh",
)


def _conda_is_available() -> bool:
    if os.environ.get("CONDA_SH"):
        return Path(os.environ["CONDA_SH"]).is_file()
    home = Path.home()
    if any((home / candidate).is_file() for candidate in CONDA_SH_CANDIDATES):
        return True
    return Path("/opt/conda/etc/profile.d/conda.sh").is_file()


needs_conda = pytest.mark.skipif(
    not _conda_is_available(), reason="no conda installation; the wrapper cannot activate an env"
)


def test_the_wrapper_exists_and_is_executable() -> None:
    """The skill documents `scripts/with_env.sh <command>`, not
    `bash scripts/with_env.sh <command>`. That only works if the mode bit is
    committed, and a lost mode bit is invisible in review."""
    assert WRAPPER.is_file()
    assert os.access(WRAPPER, os.X_OK)


@needs_conda
def test_the_wrapper_puts_this_checkout_on_the_python_path() -> None:
    """The whole reason the wrapper exists. `PYTHONPATH` is derived from the
    script's own location rather than `$PWD`, so this must hold no matter where
    it is invoked from -- hence running it from a directory that is not the repo
    root."""
    completed = subprocess.run(
        [str(WRAPPER), "python", "-c", "import hitl_pmp; print(hitl_pmp.__file__)"],
        capture_output=True,
        text=True,
        check=True,
        cwd="/",
    )
    resolved = Path(completed.stdout.strip())
    assert resolved == REPO_ROOT / "src" / "hitl_pmp" / "__init__.py"


@needs_conda
def test_the_wrapper_reports_its_environment_when_given_no_command() -> None:
    """The no-argument form doubles as the skill's sanity check, so it has to
    name the interpreter and the resolved import path, not just succeed."""
    completed = subprocess.run([str(WRAPPER)], capture_output=True, text=True, check=True, cwd="/")
    assert "hitl-pmp" in completed.stdout
    assert str(REPO_ROOT / "src") in completed.stdout


@needs_conda
def test_the_wrapper_pins_the_egl_rendering_backend() -> None:
    """KINDER installs into `hitl-pmp` itself now, so a plain `pytest` through this
    wrapper imports it and this pair has to be set here rather than in a second wrapper.

    Both variables, because `register_all_environments()` falls back to `osmesa` when
    `DISPLAY` is unset, under which `import mujoco` raises and `_check_deps` swallows
    *every* exception -- so all Dynamic3D environments are skipped IN SILENCE and the
    failure surfaces much later as an unrelated-looking `NameNotFound`. Asserted even on
    a machine with no KINDER, since the variables are what make the trap unreachable."""
    script = "import os; print(os.environ['MUJOCO_GL'], os.environ['PYOPENGL_PLATFORM'])"
    completed = subprocess.run(
        [str(WRAPPER), "python", "-c", script], capture_output=True, text=True, check=True, cwd="/"
    )
    assert completed.stdout.split() == ["egl", "egl"]


@needs_conda
def test_the_wrapper_propagates_the_exit_code_of_the_command_it_runs() -> None:
    """It `exec`s rather than wrapping, so a failing gate check must still look
    like a failure to whoever ran it."""
    completed = subprocess.run(
        [str(WRAPPER), sys.executable, "-c", "raise SystemExit(3)"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 3


# (import package, this checkout's submodule `src/` that must provide it). The two come
# from two different submodules, each an editable install whose .pth names the MAIN
# checkout's reference/, so each needs its own PYTHONPATH entry -- fixing one leaves a
# worktree silently testing a different copy of the other.
KINDER_PACKAGES = (
    ("kinder", REPO_ROOT / "reference" / "kindergarden" / "src"),
    ("kinder_models", REPO_ROOT / "reference" / "kinder-baselines" / "kinder-models" / "src"),
)


@needs_conda
@pytest.mark.parametrize(("package", "submodule_src"), KINDER_PACKAGES)
def test_the_wrapper_imports_kinder_from_this_checkouts_reference(
    *, package: str, submodule_src: Path
) -> None:
    """CLAUDE.md's rule is that `kinder.__file__` and `kinder_models.__file__` both
    resolve under *this* checkout's `reference/`. Without it a worktree tests whatever
    KINDER the main checkout has checked out, not the pins it declares. Skips when the
    package is not installed or this checkout's submodule is not populated -- an
    unpopulated worktree is allowed to fall back to the editable install."""
    if importlib.util.find_spec(package) is None:
        pytest.skip(f"{package} is not installed (the optional `tossing3d` extra)")
    if not submodule_src.is_dir():
        pytest.skip(f"{submodule_src} is not populated; run `git submodule update --init`")
    completed = subprocess.run(  # noqa: S603
        [str(WRAPPER), "python", "-c", f"import {package}; print({package}.__file__)"],
        capture_output=True,
        text=True,
        check=True,
        cwd="/",
    )
    assert Path(completed.stdout.strip()).is_relative_to(submodule_src)


@needs_conda
def test_the_bare_wrapper_reports_where_kinder_models_resolves() -> None:
    """The no-argument form is the sanity check; it has to name both KINDER packages,
    since they come from different submodules and can resolve to different trees."""
    completed = subprocess.run([str(WRAPPER)], capture_output=True, text=True, check=True, cwd="/")
    assert "kinder_models" in completed.stdout
