"""Tests for name- and module-tolerant lookups across a kinder-models refactor.

Deliberately uses stand-in modules rather than the installed `kinder_models`: the
point is behaviour on *both* sides of the rename, and any one checkout can only
ever exhibit one of them.
"""

import contextlib
import importlib
import importlib.util
import sys
import types
from collections.abc import Iterator

import pytest

from .kinder_symbols import MovedKinderSymbol, RenamedKinderSymbol

NEW = "CONTROL_TIMESTEP"
OLD = "_CONTROL_TIMESTEP"

# Stand-ins for the two modules `toss_profile_limits` straddles.
NEW_MODULE = "stand_in_toss_swing"
OLD_MODULE = "stand_in_parameterized_skills"
SYMBOL = "toss_profile_limits"


def _module(**attributes: object) -> types.ModuleType:
    """A stand-in for `kinder_models.dynamic3d.utils` exposing exactly these names."""
    module = types.ModuleType("stand_in_utils")
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


@contextlib.contextmanager
def _stand_in_modules(
    *, present: dict[str, dict[str, object]], absent: tuple[str, ...] = ()
) -> Iterator[None]:
    """Make `present` importable and `absent` not, for the duration of the block.

    Registered in `sys.modules` rather than written to disk, since `find_spec` and
    `import_module` both consult it first -- which is also why `__spec__` has to be set
    explicitly: `find_spec` raises `ValueError` rather than reporting absence when a
    cached module has none.

    A context manager rather than pytest's `monkeypatch` fixture because this repo bans
    positional arguments (ruff `PLR0917`, `max-positional-args = 0`) and a fixture is
    delivered positionally.
    """
    names = (*present, *absent)
    saved = {name: sys.modules.get(name) for name in names}
    try:
        for name in absent:
            sys.modules.pop(name, None)
        for name, attributes in present.items():
            module = types.ModuleType(name)
            module.__spec__ = importlib.util.spec_from_loader(name, loader=None)
            for attribute, value in attributes.items():
                setattr(module, attribute, value)
            sys.modules[name] = module
        yield
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_the_new_name_is_preferred_when_a_checkout_exposes_both() -> None:
    """A rename lands as add-then-remove often enough that both can coexist.

    Preferring the old one there would keep reading the deprecated value right up
    until it vanished, so the bump would still break -- just later, and somewhere
    less obvious.
    """
    module = _module(**{NEW: 0.1, OLD: 0.2})

    resolved = RenamedKinderSymbol.resolve(module=module, names=(NEW, OLD))

    assert resolved == 0.1


def test_the_old_name_still_resolves_before_the_pin_moves() -> None:
    """This is the side the currently-pinned checkout is on."""
    module = _module(**{OLD: 0.1})

    resolved = RenamedKinderSymbol.resolve(module=module, names=(NEW, OLD))

    assert resolved == 0.1


def test_the_new_name_resolves_after_the_pin_moves() -> None:
    """And this is the side kinder-baselines #113 puts it on."""
    module = _module(**{NEW: 0.1})

    resolved = RenamedKinderSymbol.resolve(module=module, names=(NEW, OLD))

    assert resolved == 0.1


def test_neither_name_present_names_every_candidate_it_tried() -> None:
    """A third rename would otherwise surface as a bare AttributeError deep in a
    test, with nothing pointing at the pin as the cause."""
    module = _module(SOMETHING_ELSE=0.1)

    with pytest.raises(AttributeError) as excinfo:
        RenamedKinderSymbol.resolve(module=module, names=(NEW, OLD))

    message = str(excinfo.value)
    assert NEW in message
    assert OLD in message
    assert "stand_in_utils" in message


def test_a_falsy_value_resolves_rather_than_falling_through() -> None:
    """`getattr(...) or fallback` is the obvious spelling and is wrong: a constant
    that is legitimately 0.0 would silently take the next candidate."""
    module = _module(**{NEW: 0.0, OLD: 0.2})

    resolved = RenamedKinderSymbol.resolve(module=module, names=(NEW, OLD))

    assert resolved == 0.0


def test_a_symbol_resolves_from_the_module_it_moved_to() -> None:
    """The side the bumped checkout is on: the old module is still importable but no
    longer defines the symbol, because it moved rather than being duplicated."""
    with _stand_in_modules(present={NEW_MODULE: {SYMBOL: 0.1}, OLD_MODULE: {}}):
        resolved = MovedKinderSymbol.resolve(modules=(NEW_MODULE, OLD_MODULE), names=(SYMBOL,))

    assert resolved == 0.1


def test_a_symbol_still_resolves_from_the_module_it_moved_from() -> None:
    """The side the currently-pinned checkout is on, where the new module does not exist at
    all -- so absence has to be tolerated at import time, not just at attribute lookup."""
    with _stand_in_modules(present={OLD_MODULE: {SYMBOL: 0.1}}, absent=(NEW_MODULE,)):
        resolved = MovedKinderSymbol.resolve(modules=(NEW_MODULE, OLD_MODULE), names=(SYMBOL,))

    assert resolved == 0.1


def test_the_module_it_moved_to_wins_when_both_expose_it() -> None:
    """Live at the bumped pin: `parameterized_skills` re-exports several of the names it
    now imports from `toss_swing`, so both modules answer and the order has to decide."""
    with _stand_in_modules(present={NEW_MODULE: {SYMBOL: 0.1}, OLD_MODULE: {SYMBOL: 0.2}}):
        resolved = MovedKinderSymbol.resolve(modules=(NEW_MODULE, OLD_MODULE), names=(SYMBOL,))

    assert resolved == 0.1


def test_every_name_is_tried_in_a_module_before_the_next_module() -> None:
    """A rename and a move can land in the same bump. Sweeping one name across every module
    before trying the next name would prefer the *old* name in the *new* module, which is
    the one pairing no checkout ever ships."""
    with _stand_in_modules(present={NEW_MODULE: {OLD: 0.1}, OLD_MODULE: {NEW: 0.2}}):
        resolved = MovedKinderSymbol.resolve(modules=(NEW_MODULE, OLD_MODULE), names=(NEW, OLD))

    assert resolved == 0.1


def test_a_module_that_exists_but_fails_to_import_is_not_swallowed() -> None:
    """A blanket `except ModuleNotFoundError` around the import would turn a broken KINDER
    install into a silent fall-through to the stale module -- the read-vs-run skew the pin
    exists to prevent, arriving as a wrong number rather than as an error."""

    def explode(name: str) -> types.ModuleType:  # noqa: PLR0917
        raise ModuleNotFoundError("no module named 'some_missing_dependency'")

    real_import_module = importlib.import_module
    with _stand_in_modules(present={NEW_MODULE: {SYMBOL: 0.1}, OLD_MODULE: {SYMBOL: 0.2}}):
        importlib.import_module = explode
        try:
            with pytest.raises(ModuleNotFoundError, match="some_missing_dependency"):
                MovedKinderSymbol.resolve(modules=(NEW_MODULE, OLD_MODULE), names=(SYMBOL,))
        finally:
            importlib.import_module = real_import_module


def test_no_module_defining_it_names_every_module_and_name_tried() -> None:
    """A further move would otherwise surface as a bare AttributeError naming only the last
    module, with nothing pointing at the pin as the cause."""
    with (
        _stand_in_modules(present={OLD_MODULE: {"SOMETHING_ELSE": 0.1}}, absent=(NEW_MODULE,)),
        pytest.raises(AttributeError) as excinfo,
    ):
        MovedKinderSymbol.resolve(modules=(NEW_MODULE, OLD_MODULE), names=(NEW, OLD))

    message = str(excinfo.value)
    assert NEW_MODULE in message
    assert OLD_MODULE in message
    assert NEW in message
    assert OLD in message
