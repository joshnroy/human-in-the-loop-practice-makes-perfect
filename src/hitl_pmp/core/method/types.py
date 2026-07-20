from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator

from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal


class LabeledAction(BaseModel):
    """A raw Action paired with a human-readable description of what produced it
    (an action-oracle's raw numbers, or a specific skill + the objects it was bound
    to). This is what lets a Renderer overlay show which action/skill was just
    taken, without Problem/Method needing a separate rendering-specific side
    channel -- Problem.run_task_episode just forwards .label to
    Renderer.render_frame's own label param."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    action: Action
    label: str


Policy = Callable[[State], LabeledAction]


class Rollout(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    states: list[State]
    actions: list[Action]

    @model_validator(mode="after")
    def _check_lengths(self) -> Rollout:
        if len(self.actions) != len(self.states) - 1:
            raise ValueError(
                f"Rollout has {len(self.states)} states but {len(self.actions)} "
                "actions; expected len(actions) == len(states) - 1."
            )
        return self


class SetupCommand(BaseModel):
    """Either the robot executes this goal itself (execute_setup_command) or it's
    handed to the human (execute_human_command) -- target says which."""

    target: SetupCommandTarget
    goal: Goal


class SetupCommandTarget(Enum):
    ROBOT = "robot"
    HUMAN = "human"


class Skill(BaseModel):
    """A lifted skill template: what a Method can select to practice/execute, before
    being bound to concrete objects or continuous parameters. Mirrors Predicate's
    shape (name + types) in problem/tasks/types.py. Per predicators/structs.py's
    NSRT / _GroundNSRT / ParameterizedOption / _Option (a 3-level lifted ->
    objects-bound -> objects+params-bound hierarchy): preconditions/effects
    (STRIPSOperator's symbolic half) are deliberately deferred -- they need a
    Variable/LiftedAtom layer (predicators: Object/Variable both subclass
    _TypedEntity, distinguished by whether the name starts with "?") that nothing
    here consumes yet."""

    model_config = ConfigDict(frozen=True)

    name: str
    types: tuple[Type, ...]
    param_dim: int


class GroundSkill(BaseModel):
    """A Skill bound to concrete objects. Params are NOT included -- continuous
    params are sampled fresh each execution (a concrete Method's job, inside
    execute_skill), so improve_skill_parameters can update the *sampler*, not one
    already-consumed param value, matching predicators' _GroundNSRT.sample_option().
    Mirrors GroundAtom's shape (predicate + objects) in problem/tasks/types.py."""

    model_config = ConfigDict(frozen=True)

    skill: Skill
    objects: tuple[Object, ...]
