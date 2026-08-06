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
    how often that worked, and -- split four ways -- what chose its parameters.

    **Why this exists.** Two experiments finished on this project unable to say why
    their result happened, for the same reason: a run's `stats.json` recorded *tasks
    solved* and nothing about practice, so "the sampler was never given enough labels"
    and "the sampler has the labels and cannot use them" produced identical output.
    PR #103 answered that question once for Tossing Room, but only through a bespoke
    `scripts/` collector that subclasses `EesMethod` and imports one domain's
    `Environment` directly, so it could not be pointed at Tossing3D. This is the
    domain-agnostic half of that collector, moved into `core` where every `Method` on
    every domain reaches it.

    **The four pools.** An attempt falls in exactly one, named for the
    `SamplerConsultation` value that produced it (see that enum for the runtime states
    each one corresponds to):

    - *random* -- the epsilon-greedy branch fired. A coin flip carries no belief, so
      it is evidence about the domain, never about the sampler.
    - *informed* -- the classifier's scores actually ranked the candidates, so these
      parameters reflect something it learned.
    - *unparameterized* -- the skill declares `param_dim == 0`, so no sampler was ever
      constructed for it and none ever could be. Nothing was consultable.
    - *uninformative* -- the remainder: a sampler existed, was consulted, and could not
      discriminate, so `LearnedSkillSampler.sample` fell back to a uniform draw.
      Recoverable as `num_attempts` minus the three stored pools rather than stored, so
      the four can never disagree about the total.

    **Why the last two are split, which #111 did not do.** They were one pool
    ("fallback"), and their remedies are opposite. *Unparameterized* means the domain is
    decomposed wrong: the skill has no learnable parameter, so practice cannot improve
    it and the parameter has to move or the skills have to fuse. *Uninformative* means
    the sampler has a parameter and the labels do not separate it: the success predicate
    is too permissive and has to be tightened. Reported as one number they are
    indistinguishable from `stats.json`, and only reading the `Method`'s own
    skill-execution code tells them apart -- which is a code-level argument propping up a
    measurement, exactly what this instrument exists to remove. That conflation is how
    a Tossing3D design flaw survived two experiments and ~200 tasks of measurement:
    `MoveToThrowPose` (`param_dim = 1`, add effect satisfied by every standoff its
    sampler could draw) and `Toss` (`param_dim = 0`) rendered identically.

    `num_fallback_attempts` is kept, and still means exactly what it did in #111 -- the
    union of the two -- so nothing that read it changes meaning.

    Successes are carried per pool as well as overall, because the pools are what the
    diagnosis turns on and a pooled rate silently averages them. Pooling these two
    inverted a published conclusion on this project once already (`SamplerChoice`):
    recycling's greedy draws landed 22/103 while the informed subset landed 11/56,
    which is its own epsilon-random rate.

    **How to read it.** `num_attempts == 0` (i.e. no entry at all) is "never asked".
    `num_unparameterized_attempts == num_attempts` is "there is nothing here to learn"
    -- a structural fact about the domain, not an outcome. `num_uninformative_attempts`
    high with `num_informed_attempts == 0` is "asked every time, and the classifier
    never had anything to say" -- starvation or an uninformative label. A healthy
    `num_informed_attempts` with a low `num_informed_successes` is "asked and missed" --
    inability. Those are different diagnoses and only this split tells them apart.

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
    num_unparameterized_attempts: int = 0
    num_unparameterized_successes: int = 0

    def num_uninformative_attempts(self) -> int:
        """Attempts on which a sampler existed, was consulted, and could not
        discriminate -- `LearnedSkillSampler.sample`'s uniform draw on a degenerate
        score vector. The derived remainder, so the pools can never disagree about the
        total; see this class's docstring for why this one is the remainder."""
        return (
            self.num_attempts
            - self.num_random_attempts
            - self.num_informed_attempts
            - self.num_unparameterized_attempts
        )

    def num_uninformative_successes(self) -> int:
        return (
            self.num_successes
            - self.num_random_successes
            - self.num_informed_successes
            - self.num_unparameterized_successes
        )

    def num_fallback_attempts(self) -> int:
        """Attempts that were neither an epsilon-greedy coin flip nor an informed
        draw: the uniform fallback on a degenerate score vector, and every execution
        of a skill with no continuous parameters to choose.

        Unchanged in meaning and value from #111, which is why it is kept -- but it is
        the *union* of the two pools whose remedies differ, so prefer
        `num_unparameterized_attempts` and `num_uninformative_attempts` for any
        diagnosis. Derived rather than stored."""
        return self.num_attempts - self.num_random_attempts - self.num_informed_attempts

    def num_fallback_successes(self) -> int:
        return self.num_successes - self.num_random_successes - self.num_informed_successes

    def with_attempt(
        self, *, success: bool, consultation: SamplerConsultation
    ) -> SkillPracticeTally:
        """This tally plus one execution, filed into exactly one pool.

        The single place an attempt is classified, so no caller can put one in two
        pools or in none -- and taking one `SamplerConsultation` rather than a bundle of
        booleans is what makes that structural rather than validated: there is no
        "random and informed" value to reject, and no way to describe an attempt that
        belongs nowhere. Returns a new instance (the model is frozen); the counts are
        eight ints, so there is nothing to gain by mutating."""
        random = consultation is SamplerConsultation.EPSILON_RANDOM
        informed = consultation is SamplerConsultation.INFORMED
        unparameterized = consultation is SamplerConsultation.NO_SAMPLER
        return SkillPracticeTally(
            num_attempts=self.num_attempts + 1,
            num_successes=self.num_successes + int(success),
            num_random_attempts=self.num_random_attempts + int(random),
            num_random_successes=self.num_random_successes + int(random and success),
            num_informed_attempts=self.num_informed_attempts + int(informed),
            num_informed_successes=self.num_informed_successes + int(informed and success),
            num_unparameterized_attempts=self.num_unparameterized_attempts + int(unparameterized),
            num_unparameterized_successes=(
                self.num_unparameterized_successes + int(unparameterized and success)
            ),
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
            num_unparameterized_attempts=(
                self.num_unparameterized_attempts + other.num_unparameterized_attempts
            ),
            num_unparameterized_successes=(
                self.num_unparameterized_successes + other.num_unparameterized_successes
            ),
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
            num_unparameterized_attempts=(
                self.num_unparameterized_attempts - previous.num_unparameterized_attempts
            ),
            num_unparameterized_successes=(
                self.num_unparameterized_successes - previous.num_unparameterized_successes
            ),
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
            "num_unparameterized_attempts": self.num_unparameterized_attempts,
            "num_unparameterized_successes": self.num_unparameterized_successes,
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
        if self.num_unparameterized_successes > self.num_unparameterized_attempts:
            raise ValueError(
                f"unparameterized successes cannot exceed unparameterized attempts: got "
                f"{self.num_unparameterized_successes}/{self.num_unparameterized_attempts}"
            )
        # The three stored pools are disjoint subsets of the attempts, so together they
        # can at most account for all of them; the slack is the uninformative pool.
        # An overflow means one execution was filed into two pools.
        stored_attempts = (
            self.num_random_attempts
            + self.num_informed_attempts
            + self.num_unparameterized_attempts
        )
        if stored_attempts > self.num_attempts:
            raise ValueError(
                f"epsilon-random ({self.num_random_attempts}), informed "
                f"({self.num_informed_attempts}) and unparameterized "
                f"({self.num_unparameterized_attempts}) attempts sum to {stored_attempts}, "
                f"which exceeds the {self.num_attempts} attempts they are drawn from"
            )
        stored_successes = (
            self.num_random_successes
            + self.num_informed_successes
            + self.num_unparameterized_successes
        )
        if stored_successes > self.num_successes:
            raise ValueError(
                f"epsilon-random ({self.num_random_successes}), informed "
                f"({self.num_informed_successes}) and unparameterized "
                f"({self.num_unparameterized_successes}) successes sum to {stored_successes}, "
                f"which exceeds the {self.num_successes} successes they are drawn from"
            )
        # The derived pool needs its own check: none of the constraints above imply it.
        # `num_attempts=1, num_successes=1, num_random_attempts=1` satisfies every one of
        # them while leaving the remainder at 1 success out of 0 attempts -- a rate above
        # 1.0 that #111 would have serialized without complaint, and the exact class of
        # silent disagreement storing a fourth pool instead of deriving it would allow.
        if self.num_uninformative_successes() > self.num_uninformative_attempts():
            raise ValueError(
                f"uninformative successes cannot exceed uninformative attempts: got "
                f"{self.num_uninformative_successes()}/{self.num_uninformative_attempts()} "
                f"as the remainder of {self.num_successes}/{self.num_attempts}"
            )
        return self


class PracticeTargetTally(BaseModel):
    """One lifted skill's *selection* record: how often EES considered practicing it,
    how often it committed, and -- the counter this type exists for -- how often it
    refused to consider it at all because the skill already looked perfect.

    **Why this is a different quantity from `SkillPracticeTally`.** That type counts
    skill *executions*, and an execution says nothing about intent: EES plans to a
    chosen candidate's preconditions and executes the whole prefix on the way, so a
    skill can rack up hundreds of attempts without ever having been the thing EES
    wanted to practice. Tossing3D's `MoveToThrowPose` is the worked example -- 175/175
    executions recorded, and not one of them a chosen practice target, because
    `skip_perfect` had dropped every grounding of it from the candidate list. The
    execution tally could not show that, and two experiments shipped without it.

    **The four counters, and what each one licenses you to conclude.**

    - `num_scored` -- a grounding was scored, came back finite, and entered the ranked
      candidate list. This is "EES was willing to practice it".
    - `num_declined_perfect` -- a grounding scored `-math.inf` and was dropped, which
      under `skip_perfect` means and only means its *measured* success rate is exactly
      1.0. Nonzero here beside a zero `num_selected` is the failure state: the skill is
      not being practiced, and the reason is that the domain's success predicate says
      it never fails. Whether that is a genuinely solved skill or a predicate too weak
      to notice failure is not answerable from this counter -- but the counter is what
      tells you the question is live.
    - `num_selected` -- a grounding was the candidate the explorer committed to. This
      is the only counter that means "practiced on purpose".
    - `num_unreachable` -- a grounding was examined ahead of the eventual selection and
      no plan reached its preconditions. It outranked the winner and lost on
      reachability, not on score.

    **Absence is the third state, and it is deliberate.** A skill with no entry at all
    was never a candidate, because EES's candidate set is exactly the ground skills it
    has executed at least once (`competence_model` creates lazily). "Never selected
    because perfect" and "never selected because never reached" are the two readings a
    bare zero cannot separate; `num_declined_perfect` against absence separates them.

    **Keyed by the lifted skill name while scoring is keyed by the GROUND skill**, so
    every counter here sums over groundings: four perfect groundings of one skill
    contribute 4, not 1. That is the intended aggregation -- the question is about the
    sampler, and one sampler is fitted per lifted name -- but it does mean these
    numbers are grounding-executions, not decisions, and are not comparable to a count
    of `choose_practice_target` calls."""

    num_scored: int = 0
    num_declined_perfect: int = 0
    num_selected: int = 0
    num_unreachable: int = 0

    def with_scored(self) -> PracticeTargetTally:
        return self.model_copy(update={"num_scored": self.num_scored + 1})

    def with_declined_perfect(self) -> PracticeTargetTally:
        return self.model_copy(update={"num_declined_perfect": self.num_declined_perfect + 1})

    def with_selected(self) -> PracticeTargetTally:
        return self.model_copy(update={"num_selected": self.num_selected + 1})

    def with_unreachable(self) -> PracticeTargetTally:
        return self.model_copy(update={"num_unreachable": self.num_unreachable + 1})

    def plus(self, *, other: PracticeTargetTally) -> PracticeTargetTally:
        return PracticeTargetTally(
            num_scored=self.num_scored + other.num_scored,
            num_declined_perfect=self.num_declined_perfect + other.num_declined_perfect,
            num_selected=self.num_selected + other.num_selected,
            num_unreachable=self.num_unreachable + other.num_unreachable,
        )

    def minus(self, *, previous: PracticeTargetTally) -> PracticeTargetTally:
        """This window's activity from two cumulative readings. A counter that went
        backwards fails the validator rather than being clamped -- it would mean two
        Methods' readings were differenced against each other, and a silent zero there
        is worse than a crash."""
        return PracticeTargetTally(
            num_scored=self.num_scored - previous.num_scored,
            num_declined_perfect=self.num_declined_perfect - previous.num_declined_perfect,
            num_selected=self.num_selected - previous.num_selected,
            num_unreachable=self.num_unreachable - previous.num_unreachable,
        )

    @model_validator(mode="after")
    def _check_counts_are_consistent(self) -> PracticeTargetTally:
        counts = {
            "num_scored": self.num_scored,
            "num_declined_perfect": self.num_declined_perfect,
            "num_selected": self.num_selected,
            "num_unreachable": self.num_unreachable,
        }
        negative = {name: value for name, value in counts.items() if value < 0}
        if negative:
            raise ValueError(f"practice-target counts cannot be negative: {negative}")
        # Selection and the reachability rejection both happen by walking the ranked
        # list, so both are drawn from the scored pool. The walk short-circuits at the
        # first success, which is why this is an inequality and not equality: the
        # candidates below the winner were scored and never examined.
        examined = self.num_selected + self.num_unreachable
        if examined > self.num_scored:
            raise ValueError(
                f"selected ({self.num_selected}) and unreachable ({self.num_unreachable}) "
                f"groundings sum to {examined}, which exceeds the {self.num_scored} scored "
                "candidates they are drawn from"
            )
        return self


class SamplerConsultation(Enum):
    """What, if anything, a learned parameter sampler contributed to one skill
    execution -- the pool `SkillPracticeTally` files that execution into.

    An enum rather than the `(was_random, was_informed)` pair #111 threaded through,
    because the pair could express neither of the two states that actually needed
    telling apart: both rendered as `(False, False)`. Four values, each of which the
    runtime genuinely produces, and no fifth invented for symmetry:

    - `NO_SAMPLER` -- the skill declares `param_dim == 0`. No sampler was ever
      constructed for it, so none could be consulted and none ever will be. There is
      nothing here for practice to improve; if this skill is failing, the fix is to the
      domain's decomposition, not to the sampler.
    - `UNINFORMATIVE` -- a sampler exists and was consulted, and its scores did not
      discriminate among the candidates, so `LearnedSkillSampler.sample` took
      deviation 6's uniform draw. Reached by an unfitted sampler (all 0.5), by the
      single-class shortcut (all 0.0 or all 1.0), and by any fitted classifier whose
      maximum is attained by more than `uninformative_tie_fraction` of the candidates.
      If this skill is failing, the fix is to the labels -- most often a success
      predicate so permissive that every draw is a positive.
    - `INFORMED` -- the scores ranked the candidates and the argmax was taken, so the
      parameters reflect something the classifier learned.
    - `EPSILON_RANDOM` -- the epsilon-greedy branch fired. A coin flip, carrying no
      belief about the sampler.

    `NO_SAMPLER` is a property of the *skill*; the other three are properties of one
    *draw*, so a single skill's tally mixes the last three freely but is either entirely
    `NO_SAMPLER` or contains none at all.
    """

    NO_SAMPLER = "no_sampler"
    UNINFORMATIVE = "uninformative"
    INFORMED = "informed"
    EPSILON_RANDOM = "epsilon_random"


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
