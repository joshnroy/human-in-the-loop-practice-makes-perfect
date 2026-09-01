"""Tossing3D expectimax search-state data."""

from pydantic import Field

from hitl_pmp.core.problem.tasks.types import GroundAtom

from .belief_state import Tossing3DBeliefState
from .core import EnvironmentState


class Tossing3DSearchState(EnvironmentState):
    """EES symbolic state paired with the posterior state used by the search."""

    state: Tossing3DBeliefState
    true_atoms: frozenset[GroundAtom] = Field(exclude=True)
    atoms: tuple[str, ...]
