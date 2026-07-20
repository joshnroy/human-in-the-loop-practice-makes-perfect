from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator

from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Predicate


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
    being bound to concrete objects or continuous parameters. Mirrors predicators'
    NSRT/STRIPSOperator: preconditions/add_effects/delete_effects are LiftedAtoms
    over this skill's own Variables (`parameters`), realizing the Variable/
    LiftedAtom layer this type deliberately deferred until a real planner (PMP's
    reproduction, planning/) needed to task-plan over skills symbolically."""

    model_config = ConfigDict(frozen=True)

    name: str
    parameters: tuple[Variable, ...]
    preconditions: frozenset[LiftedAtom]
    add_effects: frozenset[LiftedAtom]
    delete_effects: frozenset[LiftedAtom]
    param_dim: int

    @model_validator(mode="after")
    def _check_variables_are_declared_parameters(self) -> Skill:
        declared = set(self.parameters)
        referenced = {
            variable
            for atom in (*self.preconditions, *self.add_effects, *self.delete_effects)
            for variable in atom.variables
        }
        undeclared = referenced - declared
        if undeclared:
            raise ValueError(
                f"Skill {self.name!r} references variables not in its own parameters: {undeclared}"
            )
        return self


class GroundSkill(BaseModel):
    """A Skill bound to concrete objects. Params are NOT included -- continuous
    params are sampled fresh each execution (a concrete Method's job, inside
    execute_skill), so improve_skill_parameters can update the *sampler*, not one
    already-consumed param value, matching predicators' _GroundNSRT.sample_option().
    Mirrors GroundAtom's shape (predicate + objects) in problem/tasks/types.py.
    preconditions/add_effects/delete_effects ground the underlying Skill's
    LiftedAtoms by substituting objects for parameters positionally -- this is what
    lets planning/ check a candidate plan step's preconditions against the current
    state, and what lets a Method check whether execute_skill actually achieved
    add_effects (competence bookkeeping)."""

    model_config = ConfigDict(frozen=True)

    skill: Skill
    objects: tuple[Object, ...]

    @model_validator(mode="after")
    def _check_objects_match_parameters(self) -> GroundSkill:
        if len(self.objects) != len(self.skill.parameters):
            raise ValueError(
                f"GroundSkill for {self.skill.name!r} has {len(self.objects)} objects "
                f"but the skill declares {len(self.skill.parameters)} parameters."
            )
        for obj, parameter in zip(self.objects, self.skill.parameters, strict=True):
            if obj.type != parameter.type:
                raise ValueError(
                    f"GroundSkill for {self.skill.name!r}: object {obj.name!r} has type "
                    f"{obj.type.name!r}, but parameter {parameter.name!r} declares "
                    f"{parameter.type.name!r}."
                )
        return self

    @property
    def _substitution(self) -> dict[Variable, Object]:
        return dict(zip(self.skill.parameters, self.objects, strict=True))

    @property
    def preconditions(self) -> frozenset[GroundAtom]:
        return frozenset(
            atom.ground(substitution=self._substitution) for atom in self.skill.preconditions
        )

    @property
    def add_effects(self) -> frozenset[GroundAtom]:
        return frozenset(
            atom.ground(substitution=self._substitution) for atom in self.skill.add_effects
        )

    @property
    def delete_effects(self) -> frozenset[GroundAtom]:
        return frozenset(
            atom.ground(substitution=self._substitution) for atom in self.skill.delete_effects
        )


class Variable(BaseModel):
    """A typed placeholder in a lifted Skill (e.g. "?robot": robot), as opposed to
    Object, which is a concrete, named instance. Mirrors Object's shape (name +
    type) in problem/environment/types.py -- predicators' equivalent (Object and
    Variable both subclassing _TypedEntity) lives in one file since both wrap a
    Type; here Variable stays in method/types.py rather than environment/types.py
    since only Skill (Method's territory) consumes it, keeping environment/types.py
    a pure leaf."""

    model_config = ConfigDict(frozen=True)

    name: str
    type: Type


class LiftedAtom(BaseModel):
    """A Predicate applied to Variables rather than Objects -- the unground half of
    a GroundAtom (problem/tasks/types.py), used only inside a Skill's
    preconditions/add_effects/delete_effects. Mirrors Predicate.__call__
    constructing a GroundAtom: ground() here does the same, substituting concrete
    Objects for this atom's Variables."""

    model_config = ConfigDict(frozen=True)

    predicate: Predicate
    variables: tuple[Variable, ...]

    def ground(self, *, substitution: dict[Variable, Object]) -> GroundAtom:
        return GroundAtom(
            predicate=self.predicate,
            objects=tuple(substitution[variable] for variable in self.variables),
        )
