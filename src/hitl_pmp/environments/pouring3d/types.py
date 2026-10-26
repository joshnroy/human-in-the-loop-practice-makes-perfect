"""Relational observations derived at one instant of native physics."""

import numpy as np  # noqa: F401 - Pydantic resolves inherited array annotations here.

from hitl_pmp.core.problem.environment.types import Object, State  # noqa: F401

AbstractAtom = tuple[str, tuple[str, ...]]


class PouringState(State):
    abstract_atoms: frozenset[AbstractAtom]
