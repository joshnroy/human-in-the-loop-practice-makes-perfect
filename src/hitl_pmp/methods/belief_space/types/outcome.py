"""Tossing3D transition-outcome data."""

from pydantic import BaseModel

from hitl_pmp.core.problem.tasks.types import GroundAtom

from .belief_state import Tossing3DBeliefState


class Tossing3DOutcome(BaseModel):
    probability: float
    next_state: Tossing3DBeliefState
    next_true_atoms: frozenset[GroundAtom]
