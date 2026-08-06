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


class SkillPracticeTally(BaseModel):
    """One lifted skill's practice record: how often it was executed during practice,
    how often that worked, and -- split three ways -- what chose its parameters.

    **Why this exists.** Two experiments finished on this project unable to say why
    their result happened, for the same reason: a run's `stats.json` recorded *tasks
    solved* and nothing about practice, so "the sampler was never given enough labels"
    and "the sampler has the labels and cannot use them" produced identical output.
    PR #103 answered that question once for Tossing Room, but only through a bespoke
    `scripts/` collector that subclasses `EesMethod` and imports one domain's
    `Environment` directly, so it could not be pointed at Tossing3D. This is the
    domain-agnostic half of that collector, moved into `core` where every `Method` on
    every domain reaches it.

    **The six numbers, and why six.** An attempt falls in exactly one of three pools,
    named for `SamplerChoice`'s own two flags:

    - *random* -- the epsilon-greedy branch fired. A coin flip carries no belief, so
      it is evidence about the domain, never about the sampler.
    - *informed* -- `was_informed`: the classifier's scores actually ranked the
      candidates, so these parameters reflect something it learned.
    - *fallback* -- the remainder: `LearnedSkillSampler.sample`'s uniform draw on a
      score vector that could not discriminate, plus every parameter-free skill.
      Recoverable as `num_attempts - num_random_attempts - num_informed_attempts`
      rather than stored, so the three can never disagree about the total.

    Successes are carried per pool as well as overall, because the pools are what the
    diagnosis turns on and a pooled rate silently averages them. Pooling these two
    inverted a published conclusion on this project once already (`SamplerChoice`):
    recycling's greedy draws landed 22/103 while the informed subset landed 11/56,
    which is its own epsilon-random rate.

    **How to read it.** `num_attempts == 0` (i.e. no entry at all) is "never asked".
    `num_informed_attempts == 0` with `num_attempts > 0` is "asked, but never with a
    classifier that had anything to say" -- starvation. A healthy `num_informed_attempts`
    with a low `num_informed_successes` is "asked and missed" -- inability. Those are
    different diagnoses and only this split tells them apart.

    Frozen, and every counter validated against every other, so an inconsistent tally
    cannot be constructed at all -- including by `minus`, which is how a cumulative
    counter that went backwards surfaces as an error rather than as a quietly clamped
    zero."""

    model_config = ConfigDict(frozen=True)

    num_attempts: int = 0
    num_successes: int = 0
    num_random_attempts: int = 0
    num_random_successes: int = 0
    num_informed_attempts: int = 0
    num_informed_successes: int = 0

    def num_fallback_attempts(self) -> int:
        """Attempts that were neither an epsilon-greedy coin flip nor an informed
        draw: the uniform fallback on a degenerate score vector, and every execution
        of a skill with no continuous parameters to choose. Derived rather than
        stored -- see this class's docstring."""
        return self.num_attempts - self.num_random_attempts - self.num_informed_attempts

    def num_fallback_successes(self) -> int:
        return self.num_successes - self.num_random_successes - self.num_informed_successes

    def with_attempt(
        self, *, success: bool, was_random: bool, was_informed: bool
    ) -> SkillPracticeTally:
        """This tally plus one execution, filed into exactly one pool.

        The single place an attempt is classified, so no caller can put one in two
        pools or in none. Returns a new instance (the model is frozen); the counts are
        six ints, so there is nothing to gain by mutating."""
        if was_random and was_informed:
            raise ValueError(
                "an epsilon-random draw is never informed: it ignores the classifier's "
                "scores by construction (see SamplerChoice)."
            )
        return SkillPracticeTally(
            num_attempts=self.num_attempts + 1,
            num_successes=self.num_successes + int(success),
            num_random_attempts=self.num_random_attempts + int(was_random),
            num_random_successes=self.num_random_successes + int(was_random and success),
            num_informed_attempts=self.num_informed_attempts + int(was_informed),
            num_informed_successes=self.num_informed_successes + int(was_informed and success),
        )

    def plus(self, *, other: SkillPracticeTally) -> SkillPracticeTally:
        """Field-wise sum -- how several windows are totalled over a whole run."""
        return SkillPracticeTally(
            num_attempts=self.num_attempts + other.num_attempts,
            num_successes=self.num_successes + other.num_successes,
            num_random_attempts=self.num_random_attempts + other.num_random_attempts,
            num_random_successes=self.num_random_successes + other.num_random_successes,
            num_informed_attempts=self.num_informed_attempts + other.num_informed_attempts,
            num_informed_successes=self.num_informed_successes + other.num_informed_successes,
        )

    def minus(self, *, previous: SkillPracticeTally) -> SkillPracticeTally:
        """Field-wise difference of two cumulative readings -- how method_runner.py
        turns a Method's monotonic counters into one window's record.

        A counter that went backwards produces a negative field, which the validator
        below rejects: that means two readings got out of step, which is a bug worth
        surfacing rather than clamping to zero and averaging away."""
        return SkillPracticeTally(
            num_attempts=self.num_attempts - previous.num_attempts,
            num_successes=self.num_successes - previous.num_successes,
            num_random_attempts=self.num_random_attempts - previous.num_random_attempts,
            num_random_successes=self.num_random_successes - previous.num_random_successes,
            num_informed_attempts=self.num_informed_attempts - previous.num_informed_attempts,
            num_informed_successes=(self.num_informed_successes - previous.num_informed_successes),
        )

    @model_validator(mode="after")
    def _check_counts_are_consistent(self) -> SkillPracticeTally:
        counts = {
            "num_attempts": self.num_attempts,
            "num_successes": self.num_successes,
            "num_random_attempts": self.num_random_attempts,
            "num_random_successes": self.num_random_successes,
            "num_informed_attempts": self.num_informed_attempts,
            "num_informed_successes": self.num_informed_successes,
        }
        negative = {name: value for name, value in counts.items() if value < 0}
        if negative:
            raise ValueError(f"practice counts cannot be negative: {negative}")
        if self.num_successes > self.num_attempts:
            raise ValueError(
                f"successes cannot exceed attempts: got {self.num_successes}/{self.num_attempts}"
            )
        if self.num_random_successes > self.num_random_attempts:
            raise ValueError(
                f"epsilon-random successes cannot exceed epsilon-random attempts: got "
                f"{self.num_random_successes}/{self.num_random_attempts}"
            )
        if self.num_informed_successes > self.num_informed_attempts:
            raise ValueError(
                f"informed successes cannot exceed informed attempts: got "
                f"{self.num_informed_successes}/{self.num_informed_attempts}"
            )
        # The two named pools are disjoint subsets of the attempts, so together they
        # can at most account for all of them; the slack is the uniform-fallback pool.
        # An overflow means one execution was filed into two pools.
        if self.num_random_attempts + self.num_informed_attempts > self.num_attempts:
            raise ValueError(
                f"epsilon-random ({self.num_random_attempts}) plus informed "
                f"({self.num_informed_attempts}) attempts exceed the "
                f"{self.num_attempts} attempts they are drawn from"
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
    reproduction, planning/) needed to task-plan over skills symbolically.

    `ignore_effects` is predicators' fourth effect field (`STRIPSOperator`) and is
    *not* a LiftedAtom set: it holds bare Predicates, and means "after this skill
    runs, every ground atom of these predicates becomes unknown/false, whatever its
    objects". It is what makes a non-monotone skill expressible -- Ball-Ring's
    NavigateTo* wipe *all* reachability atoms, not just the one they add, so a robot
    cannot be reachable-to two tables at once. Without it a planner happily emits a
    plan that navigates away and then picks from the table it left. In PDDL this
    becomes a universally-quantified delete (`PddlWriter._action_str`); when applying
    an operator symbolically the atoms are dropped *before* delete/add effects, so a
    predicate that is both ignored and added stays true (predicators'
    `utils.apply_operator`)."""

    model_config = ConfigDict(frozen=True)

    name: str
    parameters: tuple[Variable, ...]
    preconditions: frozenset[LiftedAtom]
    add_effects: frozenset[LiftedAtom]
    delete_effects: frozenset[LiftedAtom]
    # Deliberately absent from _check_variables_are_declared_parameters below: these
    # are Predicates, not LiftedAtoms -- they bind no variable to check.
    ignore_effects: frozenset[Predicate] = frozenset()
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
    LiftedAtoms by substituting objects for parameters positionally (ignore_effects
    is forwarded ungrounded -- it names Predicates, not atoms) -- this is what
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

    @property
    def ignore_effects(self) -> frozenset[Predicate]:
        """Passed straight through, ungrounded -- an ignore effect names a whole
        Predicate, so there is nothing to substitute objects into. Mirrors
        predicators' `_GroundNSRT.ignore_effects`, which likewise just forwards its
        parent operator's set."""
        return self.skill.ignore_effects


class Variable(BaseModel):
    """A typed placeholder in a lifted Skill (e.g. name="robot", type=robot_type), as
    opposed to Object, which is a concrete, named instance. Mirrors Object's shape
    (name + type) in problem/environment/types.py -- predicators' equivalent (Object
    and Variable both subclassing _TypedEntity) lives in one file since both wrap a
    Type; here Variable stays in method/types.py rather than environment/types.py
    since only Skill (Method's territory) consumes it, keeping environment/types.py
    a pure leaf.

    **`name` carries no leading "?".** PDDL wants one and `PddlWriter._variable_str`
    adds it at write time -- unlike predicators, whose `Variable.name` is required to
    already have it. A name written "?robot" here renders "??robot", which Fast
    Downward's translator splits into two tokens; `PddlWriter` now rejects it, because
    an earlier version of this docstring used "?robot" as its example and
    `environments/tossing3d/skills.py` followed it, silently disabling planning for
    that whole domain."""

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
