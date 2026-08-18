"""Name- and module-tolerant lookups into kinder-models, for symbols a pin bump moves.

Two independent kinds of churn, both of which a topic-branch pin runs into repeatedly:

**A symbol is renamed.** `kinder-baselines` #113 drops the leading underscore from four
module-level constants in `dynamic3d/utils.py` -- `_CONTROL_TIMESTEP`,
`_ARM_MAX_VELOCITY`, `_ARM_MAX_ACCELERATION`, `_TOSS_MAX_TARGET_ROTATION` -- on the
grounds that a leading underscore claimed a privacy the code did not keep, since this
repo's tests import them.

**A symbol changes module.** `josh/feature/tossing-throw-controllers` splits the swing out
of `dynamic3d/tossing/parameterized_skills.py` into a sibling `toss_swing.py`, so
`toss_profile_limits` and `TOSS_SLICES_PER_CONTROL_STEP` are importable from the old
module on one line of that history and only from the new one on the other. Note the split
is partial: `parameterized_skills` re-exports several of the moved names (it imports them
for its own use) but not those two, so "which module answers" cannot be settled once for
the whole set.

Either way the importer straddles the bump -- one line of history exposes one form, the
other exposes the other, and a test naming either alone is broken on one side. Resolving
against candidate lists keeps the bump a gitlink change and nothing else, and confines the
knowledge that anything moved to this module.

Only `tests/` needs this. `src/` imports nothing from either module but
`create_lifted_controllers`, so neither the rename nor the split has any reach into the
library.

Delete this once the pin is bumped and the old forms are gone from every checkout this
repo supports.
"""

import importlib
import importlib.util
from types import ModuleType


class MovedKinderSymbol:
    """Reads an attribute whose module, as well as its name, a pin bump may have moved."""

    @staticmethod
    def resolve(*, modules: tuple[str, ...], names: tuple[str, ...]) -> object:
        """The first of `names` that the first of `modules` to define any of them holds.

        Order `modules` newest-first, for the same reason `names` is (see
        `RenamedKinderSymbol.resolve`): a split that leaves the old module re-exporting
        the name would otherwise keep reading through the compatibility shim right up
        until it was dropped.

        Modules are the outer loop and names the inner one, deliberately. A bump can
        rename *and* move in one step, and sweeping one name across every module first
        would prefer the old name in the new module -- the one pairing that is never what
        a checkout actually ships.

        Absence is decided by `find_spec`, never by catching `ModuleNotFoundError` around
        the import: a module that exists but raises because its own dependency is missing
        must fail loudly rather than silently falling through to the stale module.
        """
        tried: list[str] = []
        for module_name in modules:
            try:
                found = importlib.util.find_spec(module_name) is not None
            except ModuleNotFoundError:
                found = False
            if not found:
                tried.append(f"{module_name} (not importable)")
                continue
            module = importlib.import_module(module_name)
            try:
                return RenamedKinderSymbol.resolve(module=module, names=names)
            except AttributeError:
                tried.append(module_name)
        raise AttributeError(
            f"none of {tried} defines any of {names}. If kinder-models moved this symbol "
            "again, add the new module to the front of the list."
        )


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
