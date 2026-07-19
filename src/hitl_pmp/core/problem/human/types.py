from pydantic import BaseModel

from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Goal


class CommandStartStateDescription(BaseModel):
    # TODO: figure out what this should actually contain. The design doc notes
    # humans can't really operate on raw states -- v3 proposes natural-language
    # and/or pictorial descriptions instead of a raw State. For now this just
    # wraps State as a placeholder.
    state: State


class CommandGoalDescription(BaseModel):
    # TODO: see CommandStartStateDescription -- same open question, even though this
    # one already uses the same symbolic Goal as Task.goal rather than a raw State.
    goal: Goal


Cost = float
