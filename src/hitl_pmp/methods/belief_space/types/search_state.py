"""Tossing3D expectimax search-state data."""

from pydantic import BaseModel, ConfigDict, Field

from hitl_pmp.core.problem.tasks.types import GroundAtom

from .belief_state import Tossing3DBeliefState


class Tossing3DSearchState(BaseModel):
    """EES symbolic state paired with the posterior state used by the search."""

    model_config = ConfigDict(frozen=True)

    state: Tossing3DBeliefState
    true_atoms: frozenset[GroundAtom] = Field(exclude=True)
    atoms: tuple[str, ...]
