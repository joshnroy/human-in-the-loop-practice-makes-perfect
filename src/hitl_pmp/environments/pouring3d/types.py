"""Relational observations derived at one instant of native physics."""

import numpy as np  # noqa: F401 - Pydantic resolves inherited array annotations here.
from pydantic import BaseModel

from hitl_pmp.core.problem.environment.types import Object, State  # noqa: F401

AbstractAtom = tuple[str, tuple[str, ...]]


class PouringState(State):
    abstract_atoms: frozenset[AbstractAtom]


class PayloadPickPlan(BaseModel):
    """The verdict of a planning-only dumpling pick (`payload_pick_plan`).

    `candidate` is the grasp configuration whose planning succeeded and `goal` the
    prepared arm configuration it chose (None where the pick has no planning stage);
    `error` is the refusal dispatch raises when no candidate plans.
    """

    plannable: bool
    candidate: tuple[float, ...] | None = None
    goal: list[float] | None = None
    error: str | None = None
