"""The `core.State` this domain carries: flat features plus the atoms the backend's
classifier derived from the same instant of the live scene (see `backend.py`)."""

import numpy as np  # noqa: F401  (pydantic resolves State.data's annotation here)
from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.problem.environment.types import Object, State  # noqa: F401

AbstractAtom = tuple[str, tuple[str, ...]]


class SweepOffShelf3DState(State):
    abstract_atoms: frozenset[AbstractAtom] | None = None


class HumanEvent(BaseModel):
    """One human skill invocation, as the environment carried it out."""

    model_config = ConfigDict(frozen=True)

    skill: str
    moved: tuple[str, ...]
    steps_taken: int
