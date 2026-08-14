"""Tests for name-tolerant lookups across a kinder-models rename.

Deliberately uses stand-in modules rather than the installed `kinder_models`: the
point is behaviour on *both* sides of the rename, and any one checkout can only
ever exhibit one of them.
"""

import types

import pytest

from .kinder_symbols import RenamedKinderSymbol

NEW = "CONTROL_TIMESTEP"
OLD = "_CONTROL_TIMESTEP"


def _module(**attributes: object) -> types.ModuleType:
    """A stand-in for `kinder_models.dynamic3d.utils` exposing exactly these names."""
    module = types.ModuleType("stand_in_utils")
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


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
