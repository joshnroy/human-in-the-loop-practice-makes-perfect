"""The headless-rendering setup every KINDER environment needs before it is imported.

Offline: these only read and write a mapping, so they run without KINDER installed.
"""

from hitl_pmp.adapters.kinder.bootstrap import FALLBACK_DISPLAY, KinderBootstrap


def test_a_real_display_wins_and_is_not_overwritten() -> None:
    environ = {"DISPLAY": ":7"}

    resolved = KinderBootstrap.configure_headless_rendering(environ=environ)

    assert resolved["DISPLAY"] == ":7"


def test_a_missing_display_gets_one_because_only_its_existence_matters() -> None:
    """`register_all_environments()` rewrites `MUJOCO_GL` to `osmesa` when `DISPLAY` is
    unset. Nothing is ever drawn to this one; it exists to stop that rewrite."""
    environ: dict[str, str] = {}

    resolved = KinderBootstrap.configure_headless_rendering(environ=environ)

    assert resolved["DISPLAY"] == FALLBACK_DISPLAY


def test_an_inherited_osmesa_is_overridden_rather_than_respected() -> None:
    """Forced, not `setdefault`, because the one inherited value known to break `import
    mujoco` on this machine is exactly the one `register_all_environments()` writes."""
    environ = {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}

    resolved = KinderBootstrap.configure_headless_rendering(environ=environ)

    assert resolved["MUJOCO_GL"] == "egl"
    assert resolved["PYOPENGL_PLATFORM"] == "egl"
