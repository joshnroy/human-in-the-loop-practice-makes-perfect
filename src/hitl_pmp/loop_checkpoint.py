"""Domain-independent progress at a completed evaluation boundary."""

from pydantic import BaseModel, Field

from hitl_pmp.core.problem.tasks.types import Task


class LoopCheckpoint(BaseModel):
    """The next cycle to run and the fixed evaluation set; no live policy is retained."""

    completed_cycles: int = Field(ge=0)
    num_online_transitions: int = Field(ge=0)
    test_tasks: list[Task]
