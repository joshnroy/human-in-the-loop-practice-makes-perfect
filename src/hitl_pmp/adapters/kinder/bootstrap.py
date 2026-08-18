"""Getting KINDER imported at all, which is a trap rather than a formality.

`kinder.register_all_environments()` rewrites `MUJOCO_GL` to `osmesa` whenever `DISPLAY`
is unset (`kinder/__init__.py`). Under `osmesa`, `import mujoco` raises on this machine --
and `_check_deps` swallows *every* exception, so the entire `Dynamic3D` category is
skipped in silence and the failure surfaces much later, somewhere unrelated, as

    gymnasium.error.NameNotFound: Environment `Tossing3D-o1` doesn't exist in namespace kinder.

Three things together avoid it, and all three are load-bearing:

1. `DISPLAY` exists (nothing is ever drawn to it), so the rewrite never fires.
2. `MUJOCO_GL`/`PYOPENGL_PLATFORM` are *forced* to `egl`, overriding an inherited `osmesa`.
3. A **module** inside `kinder.envs.dynamic3d` is imported -- `kinder.envs.dynamic3d.envs`,
   not the package `kinder.envs.dynamic3d`, which does not pull in `mujoco` -- so `mujoco`
   is in `sys.modules` before `register_all_environments()` runs.

`register_all_environments()` leaves `MUJOCO_GL` reading `osmesa` afterwards even when it
worked, so the environment is re-asserted on the very next line rather than left to a
caller who might forget.

This lives in `adapters/` rather than in any one domain because it is a property of KINDER,
not of Tossing3D: every Dynamic3D environment reaches the same trap.
"""

import os
from collections.abc import MutableMapping
from types import ModuleType

# A DISPLAY only has to *exist*; nothing is ever drawn to it.
FALLBACK_DISPLAY = ":0"

RENDERING_VARIABLES = ("DISPLAY", "MUJOCO_GL", "PYOPENGL_PLATFORM")


class KinderBootstrap:
    """Import-time setup for KINDER. A static-method container, never instantiated."""

    @staticmethod
    def configure_headless_rendering(
        *, environ: MutableMapping[str, str] | None = None
    ) -> dict[str, str]:
        """Point MuJoCo at EGL and make sure a `DISPLAY` exists, before KINDER is imported.

        `DISPLAY` is `setdefault`, so a real display wins; the other two are forced, for
        the reason in the module docstring. Returns what was resolved, so a caller (or a
        test) can assert on it rather than re-reading the environment.
        """
        target = os.environ if environ is None else environ
        target.setdefault("DISPLAY", FALLBACK_DISPLAY)
        target["MUJOCO_GL"] = "egl"
        target["PYOPENGL_PLATFORM"] = "egl"
        return {key: target[key] for key in RENDERING_VARIABLES}

    @staticmethod
    def register_environments() -> ModuleType:
        """Import KINDER in the order that works, register its environments, return it.

        The dynamic3d module import below is what makes registration succeed; see the
        module docstring for why deleting it fails much later and somewhere else.
        """
        KinderBootstrap.configure_headless_rendering()

        import kinder
        import kinder.envs.dynamic3d.envs  # noqa: F401  (the MODULE, not the package)

        kinder.register_all_environments()
        KinderBootstrap.configure_headless_rendering()
        return kinder
