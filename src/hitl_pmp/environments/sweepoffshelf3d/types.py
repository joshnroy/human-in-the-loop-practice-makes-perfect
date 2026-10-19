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


class SkillTrace(BaseModel):
    """One robot skill attempt, compact enough to log on every step: what was asked,
    how it ended, which promised effects did not hold, what changed, and every
    candidate it tried (`notes`, capped, with the overflow counted)."""

    skill: str
    objects: list[str]
    params: list[float]
    # success | plan_failure (nothing moved) | execution_failure | effects_missing
    # (no error raised, yet a promised effect does not hold) | interrupted
    outcome: str
    error: str | None
    controller_steps: int
    missing_effects: list[str]
    added: list[str]
    removed: list[str]
    notes: list["SkillNote"]
    notes_dropped: int = 0


class SkillNote(BaseModel):
    """One thing a skill tried on the way to its outcome: a candidate parameter vector
    rejected at planning time (`no_plan`), a plan refused by the MuJoCo re-check
    (`plan_blocked`), the candidate executed (`executed`) or failed in execution
    (`execution_failed`), a retract that could not run, or a scripted routine's
    retry (`scripted`)."""

    model_config = ConfigDict(frozen=True)

    stage: str
    kind: str
    reason: str
    params: list[float] | None = None


class FloorGrasp(BaseModel):
    """One candidate for the floor and box picks (`floor_pick.py`): pinch-site poses
    for the grasp, the pre-grasp it descends from and the end of the straight lift, the
    base it is reached from, and the arm at each, solved and checked in MuJoCo."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    lean_deg: float
    hand: np.ndarray
    pre: np.ndarray
    lift: np.ndarray
    base: np.ndarray
    grasp_conf: np.ndarray
    pre_conf: np.ndarray
    lift_conf: np.ndarray
