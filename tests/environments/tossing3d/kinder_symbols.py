"""Name-tolerant lookups into kinder-models, for symbols a pin bump renames.

`kinder-baselines` #113 drops the leading underscore from four module-level constants
in `dynamic3d/utils.py` -- `_CONTROL_TIMESTEP`, `_ARM_MAX_VELOCITY`,
`_ARM_MAX_ACCELERATION`, `_TOSS_MAX_TARGET_ROTATION` -- on the grounds that a leading
underscore claimed a privacy the code did not keep, since this repo's tests import them.

That leaves the importer straddling the bump: the pinned checkout exposes the old names
and the bumped one exposes the new, so a test naming either alone is broken on one side.
Resolving by name list keeps the bump a gitlink change and nothing else, and confines the
knowledge that a rename happened to this module.

Only `tests/` needs this. `src/` imports nothing from `kinder_models.dynamic3d.utils` --
`kinder_backend.py` takes `create_lifted_controllers` and no constants -- so the rename
has no reach into the library.

Delete this once the pin is bumped and the old names are gone from every checkout this
repo supports.
"""

from types import ModuleType


class RenamedKinderSymbol:
    """Reads a module attribute that is known by more than one name."""

    @staticmethod
    def resolve(*, module: ModuleType, names: tuple[str, ...]) -> object:
        """The first of `names` the module defines, preferring the earliest.

        Order the candidates newest-first: a rename that lands as add-then-remove
        exposes both for a while, and taking the old one there defers the breakage
        to whenever it is finally deleted rather than avoiding it.

        `hasattr`, not `getattr(...) or ...` -- a constant that is legitimately zero
        would otherwise fall through to the next candidate.
        """
        for name in names:
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(
            f"{module.__name__} defines none of {names}. If kinder-models renamed "
            "this symbol again, add the new name to the front of the list."
        )
