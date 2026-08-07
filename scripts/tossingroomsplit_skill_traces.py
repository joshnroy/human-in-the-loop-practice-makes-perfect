"""Collect PER-SKILL practice traces for EES on Tossing Room (split throws): how many
times each lifted skill was attempted during practice, how many of those attempts
achieved the skill's own add effects, and where each skill's competence got to -- dumped
as JSON for `analysis/practice_makes_perfect/tossingroomsplit_throw_rates.py` to render.

**It serves BOTH throw representations, selected by `--env`.** `tossingroomsplit` (the
default, and every pre-existing caller's behaviour) is the CAUSAL arm: the required
force is an unobserved affine function of a bin's `throw_distance` and an item's
`weight`. `tossingroomsplitidentity` is the degenerate IDENTITY arm: the required force
IS `item.target_force`, a column every sampler already reads at index 4 of its own
classifier row. The two are otherwise the same world under the same protocol, and
`tests/environments/tossingroomsplitidentity/test_fork_equivalence.py` pins that.

One script rather than two because the shards must be the same SHAPE -- the experiment
lays the arms side by side, and the analysis keys on lifted skill names and goal
descriptions that are identical in both. `ThrowTarget` is the only place the two
representations differ here. Each shard records its own `env`, so a pooled analysis
cannot silently mix them: they are comparable side by side, never summed.

**Why this exists rather than being read off `--output-dir`, and what has since moved
there.** `stats.json` is the serialized `core.Metrics`. When this file was written it
recorded *tasks solved* and nothing about practice, so attempts, successes and the
informed/random split never left `EesMethod`'s internals and were read out here by
subclassing it.

**Half of that is now in `stats.json` for every domain.** `Metrics.
practice_outcomes_per_cycle` carries, per lifted skill and per window, exactly
`attempts` / `successes` / `random_attempts` / `random_successes` /
`informed_attempts` / `informed_successes` -- this file's whole domain-agnostic half,
available from any `--env` with no subclassing, and plotted by
`analysis/practice_makes_perfect/practice_diagnostics.py`. **Prefer that** for anything
those six numbers answer; a new domain should never grow a copy of this script.

**This file is kept, not retired, for the half that cannot move.** Everything it
records beyond those six is read from the Tossing Room DYNAMICS rather than from the
`Method`, and there is nothing domain-agnostic to promote:

- `landed` reimplements `_apply_throw`'s own condition, deliberately *not* the
  add-effect check EES scores by -- which is the point, since the audit is of whether
  those two agree. `core.Metrics` only ever sees the second.
- `prefilled` is a fact about a bin's occupancy.
- `throw_targets` / `greedy_forces` / `informed_forces` are the continuous parameter
  issued and the ground-truth force required, the latter computable only by the
  environment. Those are what separate "a sampler stuck on a confident wrong value"
  from "a sampler scattering", and no per-skill count can stand in for them.

So the overlap is subsumed and the remainder is not. `scripts/tossingroom_throw_traces.py`
sits in the same position for the same reason: there is no CLI surface for a method's
internal decisions, and adding one purely for a diagnostic would put trace plumbing in
the shipped `Method`.

**This is the same experiment as the sweep, measured a second way -- not a second
experiment.** A run is fully determined by its `--seed`, and the subclasses below
override only hooks that record; they consume no randomness and change no control flow.
So a traced run at seed *k* reproduces the sweep's seed-*k* run step for step, and
`tests/scripts/test_tossingroomsplit_skill_traces.py::test_tracing_does_not_perturb_the_run`
pins exactly that by comparing the traced run's per-sweep `(transitions, solved, total)`
triples against a stock run's. The analysis re-checks it against the real `stats.json`
files before reporting anything.

It lives in `scripts/` because it *drives* simulations, which `analysis/` may never do
(CLAUDE.md). Seeds are fixed (0..num_seeds-1), never randomly drawn, same as run_sweep.

**The committed 2026-08-05 traces are the CAPACITY-1 run.** An earlier set predating that
change to the DYNAMICS -- capacity-1 bins, a throw refused at a full bin, one emptying
button per bin, a one-item-per-bin EMPTY prefill that is now an ordering task, and an
evaluation horizon of 12 rather than 7 -- has been withdrawn and replaced rather than
re-scored. A trace taken before that change and one taken after are measurements of two
different worlds and must not be pooled or compared.
"""

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Policy
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.tossingroomsplit.environment import TossingRoomSplitEnvironment
from hitl_pmp.environments.tossingroomsplit.problem import TossingRoomSplitProblem
from hitl_pmp.environments.tossingroomsplit.skill_provider import TossingRoomSplitSkillProvider
from hitl_pmp.environments.tossingroomsplit.tasks import TossingRoomSplitTasks
from hitl_pmp.environments.tossingroomsplitidentity.environment import (
    TossingRoomSplitIdentityEnvironment,
)
from hitl_pmp.environments.tossingroomsplitidentity.problem import TossingRoomSplitIdentityProblem
from hitl_pmp.environments.tossingroomsplitidentity.skill_provider import (
    TossingRoomSplitIdentitySkillProvider,
)
from hitl_pmp.environments.tossingroomsplitidentity.tasks import TossingRoomSplitIdentityTasks
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.practice_loop import PracticeLoop, PracticeResetPolicy

# The two arms this collector serves. Both are the same world under the same protocol
# and differ only in the THROW REPRESENTATION, so they produce trace shards of identical
# shape and `analysis/practice_makes_perfect/tossingroomsplit_throw_rates.py` reads
# either without knowing which is which -- the lifted skill names and the goal
# descriptions it keys on are the same in both.
CAUSAL_ENV = "tossingroomsplit"
IDENTITY_ENV = "tossingroomsplitidentity"
ENV_CHOICES = (CAUSAL_ENV, IDENTITY_ENV)


class SkillTally(BaseModel):
    """One lifted skill's practice record inside one interaction period.

    Attempts and successes are counted from `observe_outcome`, i.e. from exactly the
    events EES itself scores a skill by -- an attempt is an execution whose outcome was
    later observed, and a success is one whose `add_effects` held. The last skill of a
    period goes unobserved by construction (`EesMethod`'s deviation 2), so a period of
    *n* skills contributes *n - 1* attempts. That is a property of the method, not of
    this file, and it applies identically to both throws.

    **`successes` is not the same thing as `landed`.** A throw's `add_effects` are
    `{<Kind>InBin(item, bin), HandEmpty(robot)}`, and `<Kind>InBin` is `count >= 1`, so
    before the capacity-1 redesign a throw made when the bin was ALREADY non-empty was
    scored a success at any force at all -- asymmetrically, since the trash bin was
    routinely in that state and the recycling bin, behind the one-way ledge with one
    throw per period, never was. `landed` and `prefilled` exist to make that difference
    measurable instead of silently inflating one skill's numbers.

    **That channel is closed on the current domain**, and these fields are kept as the
    check that it stays closed rather than as a live correction: a bin holds at most one
    item, each throw carries its bin's empty precondition, and `_apply_throw` REFUSES a
    throw at a full bin. So a throw is never issued at a non-empty bin, `prefilled` should
    be 0 for both skills, and a nonzero value is a regression rather than a datum. The
    committed 2026-08-05 run is post-redesign and reads 0/618 and 0/163."""

    attempts: int = 0
    successes: int = 0
    # Attempts the epsilon-greedy branch chose the parameters for. Kept separate because
    # a random draw says nothing about what the sampler has learned -- the two are never
    # pooled when reporting what a skill can do.
    random_attempts: int = 0
    random_successes: int = 0
    # Throws only. Read from the DYNAMICS -- the bin had room, the robot stood in that
    # bin's own room, and |force - required_force| < throw_tolerance -- not from the
    # add-effect check, so this says what the environment did rather than what EES scored.
    landed: int = 0
    landed_random: int = 0
    # Throws only. Attempts made while the target bin already held at least one item.
    # Before the capacity-1 redesign those were attempts whose scored success was
    # guaranteed before the force was chosen; now the dynamics refuse them outright and
    # the operator is inapplicable, so this should be 0 on any fresh run.
    prefilled: int = 0
    # Throws only, LEARNED-sampler draws only: the force the sampler chose and the target
    # it was aiming at, one entry each per greedy attempt. `attempts` says how often a
    # sampler was asked; these say what it answered, which is what separates "a sampler
    # stuck on a confident wrong value" from "a sampler scattering". The epsilon-random
    # draws are excluded rather than flagged, because a coin flip carries no belief and
    # pooling the two would wash the signal out.
    greedy_forces: list[float] = Field(default_factory=list)
    greedy_targets: list[float] = Field(default_factory=list)
    # Throws only, and a STRICT SUBSET of the greedy draws above: the ones whose
    # classifier scores actually ranked the candidates (`SamplerChoice.was_informed`).
    # The rest of the greedy pool is `LearnedSkillSampler.sample`'s uniform fallback on
    # a degenerate score vector -- a draw that carries no belief but is not an
    # epsilon-random one either. Kept as its own bucket rather than replacing
    # `greedy_*`, so the contaminated pool the previous log reported stays computable
    # and the two can be compared directly.
    informed_attempts: int = 0
    informed_successes: int = 0
    informed_landed: int = 0
    informed_forces: list[float] = Field(default_factory=list)
    informed_targets: list[float] = Field(default_factory=list)
    # Throws only. EVERY attempt in this period, in execution order, one entry per list.
    #
    # The `greedy_*`/`informed_*` lists above deliberately exclude the epsilon-random
    # draws, because a coin flip carries no belief and pooling the two washes out what a
    # sampler ANSWERED. But what a classifier can represent is set by its POSITIVES, and
    # `observe_outcome` feeds it every landed attempt including the random ones -- which
    # early in a run are most of them. So the greedy lists drop exactly the targets that
    # matter for the mechanism: one landing pins where the good force region sits for one
    # target, and only two landings at well-separated targets reveal the SLOPE of the
    # force/target relation. `throw_targets` restricted to `throw_landed_flags` is that
    # variable, and it is not computable from any field above.
    #
    # `throw_kinds` splits three ways rather than two, since "the epsilon branch fired"
    # and "the classifier could not rank the candidates" are different facts about a
    # non-informed draw -- the same distinction `was_random` and `was_informed` carry on
    # `SamplerChoice`.
    #
    # Every ATTEMPT and LANDING count above re-derives from these three lists -- but the
    # `successes` counts do not, and neither does `prefilled`. A scored success is EES's
    # add-effect check, which is a different event from a landing (see this class's
    # docstring), and both are kept for exactly that reason.
    throw_targets: list[float] = Field(default_factory=list)
    throw_landed_flags: list[bool] = Field(default_factory=list)
    throw_kinds: list[Literal["random", "informed", "fallback"]] = Field(default_factory=list)


class PeriodLog(BaseModel):
    """Everything observed in one interaction period, drained at the cycle boundary."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    skills: dict[str, SkillTally] = Field(default_factory=dict)

    def record(
        self,
        *,
        name: str,
        success: bool,
        was_random: bool,
        throw: "ThrowObservation | None" = None,
    ) -> None:
        tally = self.skills.setdefault(name, SkillTally())
        tally.attempts += 1
        tally.successes += int(success)
        if was_random:
            tally.random_attempts += 1
            tally.random_successes += int(success)
        if throw is not None:
            tally.landed += int(throw.landed)
            tally.prefilled += int(throw.prefilled)
            tally.throw_targets.append(throw.target)
            tally.throw_landed_flags.append(throw.landed)
            tally.throw_kinds.append(
                "random" if was_random else ("informed" if throw.informed else "fallback")
            )
            if was_random:
                tally.landed_random += int(throw.landed)
            else:
                tally.greedy_forces.append(throw.force)
                tally.greedy_targets.append(throw.target)
                if throw.informed:
                    tally.informed_attempts += 1
                    tally.informed_successes += int(success)
                    tally.informed_landed += int(throw.landed)
                    tally.informed_forces.append(throw.force)
                    tally.informed_targets.append(throw.target)

    def drain(self) -> dict:
        snapshot = {"skills": {name: tally.model_dump() for name, tally in self.skills.items()}}
        self.skills = {}
        return snapshot


class ThrowObservation(BaseModel):
    """What the DYNAMICS did with one throw, snapshotted at the state it was issued from.

    `landed` reimplements `TossingRoomSplitEnvironment._apply_throw`'s own condition --
    the bin has room under `BIN_CAPACITY`, the robot is in that bin's room, and
    `|force - required_force| < throw_tolerance` -- rather than asking whether the add
    effects held, which is the very thing being audited. The capacity term is part of
    that condition and not an optional refinement: `_apply_throw` returns without
    releasing the item when the bin is full, so a throw refused that way landed nothing
    however good the force was."""

    landed: bool
    prefilled: bool
    # The force actually issued and the force that grounding REQUIRED, the latter via
    # `ThrowTarget.of` so this file does not care which arm produced it. In the CAUSAL
    # arm the target is not a state feature at all -- it is `required_force` of the bound
    # bin's `throw_distance` and the bound item's `weight`, which only the environment
    # can compute. In the IDENTITY arm it is simply `item.target_force`, a column every
    # sampler already sees. Kept alongside the verdict rather than derived from it:
    # `landed` collapses everything about a throw into one bit, and "missed by 0.02" and
    # "missed by 0.75" are different findings.
    force: float
    target: float
    # `SamplerChoice.was_informed` for the draw that produced this throw, read off the
    # `_SkillAttempt` EesMethod already returns. It rides here rather than on
    # `observe_outcome` because that is a `core.Method` signature and this is
    # privileged instrumentation, exactly like `force`/`target` above. Defaulted so a
    # test can construct a bare observation without asserting on the sampler.
    informed: bool = False


class ThrowTarget:
    """The force one throw REQUIRED, for whichever arm's environment is in play.

    This is the ONLY place in this collector that the two arms' throw representations
    differ, which is the same claim `environments/tossingroomsplitidentity` makes about
    the domains themselves. Dispatching here rather than forking the whole script keeps
    the two arms' traces provably identical in shape -- and a shape difference is exactly
    what would make the side-by-side comparison unreadable.

    Both branches call the environment's own `required_force` rather than reimplementing
    it, so neither can drift from the dynamics that actually scored the throw.

    A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    @staticmethod
    def of(
        *,
        env: TossingRoomSplitEnvironment | TossingRoomSplitIdentityEnvironment,
        state: State,
        item: Object,
        bin_obj: Object,
    ) -> float:
        if isinstance(env, TossingRoomSplitIdentityEnvironment):
            # IDENTITY: the answer is a feature of the item, at index 4 of the throw's
            # own classifier row. Nothing privileged is being read here.
            return env.required_force(
                item_target_force=float(state.get(obj=item, feature_name="target_force"))
            )
        # CAUSAL: two observable causes combined by coefficients no sampler can see.
        return env.required_force(
            throw_distance=float(state.get(obj=bin_obj, feature_name="throw_distance")),
            item_weight=float(state.get(obj=item, feature_name="weight")),
        )


class DomainBinding:
    """Builds the `(problem, skill_provider)` pair for one arm.

    The two arms' composition roots are structurally identical -- same `Problem`, same
    `Tasks`, same `SkillProvider`, all at their defaults -- so this is a dispatch on the
    `--env` name and nothing more. Anything that had to differ beyond the throw
    representation would show up here as a divergence, which is why it is one function
    rather than two scripts.

    A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    @staticmethod
    def build(
        *, env_name: str, seed: int, num_test_tasks: int
    ) -> tuple[
        TossingRoomSplitProblem | TossingRoomSplitIdentityProblem,
        TossingRoomSplitSkillProvider | TossingRoomSplitIdentitySkillProvider,
        TossingRoomSplitEnvironment | TossingRoomSplitIdentityEnvironment,
        TossingRoomSplitProblem | TossingRoomSplitIdentityProblem,
    ]:
        """Returns (practice problem, skill provider, practice env, EVALUATION problem).

        The fourth element mirrors what `--env tossingroomsplit` does through the real
        CLI: evaluation runs on its own Environment + Tasks, so a sweep's per-episode
        `reset_to_task` cannot write into the environment practice inherits. Without it
        this script would silently reset a `never` arm `num_test_tasks` times per sweep
        and its traces would describe an arm that does not exist. Both problems are
        built from the same seed, so the evaluation Tasks yields the same test set the
        practice Tasks would have and the scheduled arm is unchanged."""
        if env_name == IDENTITY_ENV:
            identity_env = TossingRoomSplitIdentityEnvironment()
            identity_tasks = TossingRoomSplitIdentityTasks(
                env=identity_env, seed=seed, num_test_tasks=num_test_tasks
            )
            identity_eval_env = TossingRoomSplitIdentityEnvironment()
            identity_eval_tasks = TossingRoomSplitIdentityTasks(
                env=identity_eval_env, seed=seed, num_test_tasks=num_test_tasks
            )
            return (
                TossingRoomSplitIdentityProblem(env=identity_env, tasks=identity_tasks),
                TossingRoomSplitIdentitySkillProvider(env=identity_env),
                identity_env,
                TossingRoomSplitIdentityProblem(env=identity_eval_env, tasks=identity_eval_tasks),
            )
        if env_name != CAUSAL_ENV:
            raise ValueError(f"unknown --env {env_name!r}; expected one of {ENV_CHOICES}")
        env = TossingRoomSplitEnvironment()
        tasks = TossingRoomSplitTasks(env=env, seed=seed, num_test_tasks=num_test_tasks)
        eval_env = TossingRoomSplitEnvironment()
        eval_tasks = TossingRoomSplitTasks(env=eval_env, seed=seed, num_test_tasks=num_test_tasks)
        return (
            TossingRoomSplitProblem(env=env, tasks=tasks),
            TossingRoomSplitSkillProvider(env=env),
            env,
            TossingRoomSplitProblem(env=eval_env, tasks=eval_tasks),
        )


class TracingEesMethod(EesMethod):
    """Records, per interaction period, which lifted skill each observed practice outcome
    belonged to -- plus the ground skills each lifted name was ever executed with, so
    competence can be read back per name at the cycle boundary.

    `observe_outcome` is the single funnel every practice outcome passes through
    (`_EesEpisode.observe_pending` calls it, and `observe_environment_reset` routes
    through the same place), so hooking it here catches every attempt exactly once and
    cannot double-count. The override records and delegates; it changes nothing."""

    log: PeriodLog
    # Ground skills seen per lifted name, so `competence_snapshot` knows what to ask the
    # competence models about. A list rather than a set: `GroundSkill` is frozen and
    # hashable, but insertion order makes the JSON stable and the counts are tiny.
    seen: dict[str, list[GroundSkill]] = Field(default_factory=dict)
    # Practice periods only. An evaluation sweep records nothing (EES observes no
    # outcomes there), but the flag makes that explicit rather than incidental.
    practicing: bool = False
    # The in-flight throw's dynamics observation, keyed by lifted skill name, waiting for
    # the outcome to be observed. At most one per name is ever in flight: EES observes the
    # previous execution before issuing the next.
    #
    # Dropped at each cycle boundary by `drain_pending`, because EES leaves each period's
    # LAST skill unobserved by construction, so a throw issued last in a period leaves an
    # entry here that its own period will never pop. It has always been overwritten by the
    # next period's throw before anything reads it, so no count was ever wrong -- but with
    # the per-draw lists a misattribution would now be silent, since the list lengths
    # would still match `attempts` while a target and a kind rode on the wrong draw.
    pending_throws: dict[str, ThrowObservation] = Field(default_factory=dict)

    def drain_pending(self) -> None:
        """Forget any throw whose outcome the period ended before observing."""
        self.pending_throws = {}

    def get_task_policy(self, *, task: Task) -> Policy:
        self.practicing = False
        return super().get_task_policy(task=task)

    def get_practice_policy(self, *, task: Task) -> Policy:
        self.practicing = True
        return super().get_practice_policy(task=task)

    def execute_ground_skill(
        self, *, ground_skill: GroundSkill, state: State, explore: bool
    ) -> tuple[LabeledAction, object]:
        labeled, record = super().execute_ground_skill(
            ground_skill=ground_skill, state=state, explore=explore
        )
        name = ground_skill.skill.name
        if self.practicing and name in ("ThrowTrash", "ThrowRecycling"):
            # `record` is `None` exactly when the base class ran this skill with
            # `explore=False`, which during practice happens only under
            # `--reproduce-predicators-explore-target-only` (off by default, and off for
            # every run this collector has been used for). Such a draw IS an argmax and
            # may well have been informed, but the sampler's answer was not recorded, so
            # it is counted as uninformed rather than guessed at -- which under that flag
            # would over-report the uniform fallback. Asserted rather than left silent:
            # the split is the point of this collector, and a run that quietly mislabels
            # half of it is worse than one that stops.
            assert record is not None, (
                f"{name}: practiced with no sampler record, so the informed/uninformed "
                "split cannot be measured -- --reproduce-predicators-explore-target-only "
                "is not supported by this collector"
            )
            self.pending_throws[name] = self._observe_throw(
                name=name,
                state=state,
                force=float(labeled.action[2]),
                informed=record.was_informed_choice,
            )
        return labeled, record

    def _observe_throw(
        self, *, name: str, state: State, force: float, informed: bool = False
    ) -> ThrowObservation:
        env = self.env
        assert isinstance(env, TossingRoomSplitEnvironment | TossingRoomSplitIdentityEnvironment)
        trash = name == "ThrowTrash"
        item = env.trash if trash else env.recycling
        bin_obj = env.trash_bin if trash else env.recycling_bin
        bin_room = env.trash_bin_room if trash else env.recycling_bin_room
        target = ThrowTarget.of(env=env, state=state, item=item, bin_obj=bin_obj)
        robot_room = int(round(state.get(obj=env.robot, feature_name="room")))
        count = int(round(state.get(obj=bin_obj, feature_name="count")))
        refused = count >= env.BIN_CAPACITY
        return ThrowObservation(
            landed=(
                not refused and robot_room == bin_room and abs(force - target) < env.throw_tolerance
            ),
            prefilled=count >= 1,
            force=force,
            target=target,
            informed=informed,
        )

    def observe_outcome(
        self, *, ground_skill: GroundSkill, success: bool, was_random_exploration: bool = False
    ) -> None:
        name = ground_skill.skill.name
        groundings = self.seen.setdefault(name, [])
        if ground_skill not in groundings:
            groundings.append(ground_skill)
        if self.practicing:
            self.log.record(
                name=name,
                success=success,
                was_random=was_random_exploration,
                throw=self.pending_throws.pop(name, None),
            )
        super().observe_outcome(
            ground_skill=ground_skill,
            success=success,
            was_random_exploration=was_random_exploration,
        )

    def competence_snapshot(self) -> dict:
        """Each lifted skill's current competence, averaged over the ground skills it was
        actually executed with, with that count reported alongside.

        On this domain each throw has exactly one reachable grounding (the bin's room is
        pinned by `TrashBinInRoom`/`RecyclingBinInRoom`), so the mean is over one value
        -- but the count is emitted anyway, because a mean silently taken over several
        groundings would otherwise be indistinguishable from a single skill's number."""
        snapshot: dict[str, dict] = {}
        for name, groundings in self.seen.items():
            competences = [
                self.competence_model(ground_skill=grounding).get_current_competence()
                for grounding in groundings
            ]
            observations = [
                self.competence_model(ground_skill=grounding).num_observations
                for grounding in groundings
            ]
            snapshot[name] = {
                "competence": sum(competences) / len(competences),
                "num_groundings": len(groundings),
                "num_observations": sum(observations),
            }
        return snapshot


class SkillTraceCollector:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    @staticmethod
    def run_seed(
        *,
        # Defaulted to the causal arm so every pre-existing caller -- and the committed
        # 2026-08-05 reproduction command -- keeps its exact previous behaviour, with the
        # identity arm strictly opt-in. Same default as the `--env` flag, for the same
        # reason.
        env_name: str = CAUSAL_ENV,
        seed: int,
        sampler_iters: int,
        num_cycles: int,
        max_steps: int,
        num_test_tasks: int,
        # SCHEDULED keeps every pre-existing caller -- and the committed 2026-08-05
        # reproduction command -- byte-identical; the reset-free arm is strictly opt-in.
        practice_reset_policy: PracticeResetPolicy = PracticeResetPolicy.SCHEDULED,
    ) -> dict:
        """One full EES run, returning per-period skill tallies, per-cycle competence,
        and the per-sweep evaluation record (with its goal-family breakdown).

        `num_test_tasks` must be passed to the domain's `Tasks` as well as to
        `PracticeLoop.run`: the field is what the fixed goal-family composition is
        divided out of, and a disagreement silently measures a different test set."""
        log = PeriodLog()
        problem, skill_provider, env, evaluation_problem = DomainBinding.build(
            env_name=env_name, seed=seed, num_test_tasks=num_test_tasks
        )
        method = TracingEesMethod(
            env=env,
            skill_provider=skill_provider,
            seed=seed,
            sampler_max_train_iters=sampler_iters,
            log=log,
        )
        metrics = Metrics()
        periods: list[dict] = []
        competence: list[dict] = []

        def on_cycle_end() -> None:
            periods.append(log.drain())
            method.drain_pending()
            competence.append(method.competence_snapshot())

        PracticeLoop.run(
            problem=problem,
            evaluation_problem=evaluation_problem,
            method=method,
            metrics=metrics,
            num_cycles=num_cycles,
            max_steps_per_interaction=max_steps,
            num_test_tasks=num_test_tasks,
            practice_reset_policy=practice_reset_policy,
            on_cycle_end=on_cycle_end,
        )

        sweeps = [
            {
                "transitions": transitions,
                "solved": solved,
                "total": total,
                "families": SkillTraceCollector._families(metrics=metrics, index=index),
            }
            for index, (transitions, solved, total) in enumerate(metrics.evaluations)
        ]
        return {
            "seed": seed,
            "horizon": problem.max_episode_steps(),
            "sweeps": sweeps,
            "periods": periods,
            "competence": competence,
        }

    @staticmethod
    def _families(*, metrics: Metrics, index: int) -> dict[str, tuple[int, int]]:
        """`{goal description: (solved, total)}` for one evaluation sweep, read straight
        off `Metrics.breakdowns` -- the same per-task records `stats.json` carries, so
        the counts here and the counts an `analysis/` script reads out of a swept run are
        the same numbers rather than two derivations of them."""
        grouped: dict[str, tuple[int, int]] = {}
        for outcome in metrics.breakdowns[index].outcomes:
            solved, total = grouped.get(outcome.goal, (0, 0))
            grouped[outcome.goal] = (solved + int(outcome.solved), total + 1)
        return grouped

    @staticmethod
    def collect(
        *,
        label: str,
        env_name: str = CAUSAL_ENV,
        sampler_iters: int,
        seeds: list[int],
        num_cycles: int,
        max_steps: int,
        num_test_tasks: int,
        practice_reset_policy: PracticeResetPolicy = PracticeResetPolicy.SCHEDULED,
    ) -> dict:
        return {
            "label": label,
            # Recorded in the shard so a pooled analysis cannot silently mix the two
            # arms: they are the same world under two throw representations, and their
            # numbers are only comparable side by side, never summed.
            "env": env_name,
            "sampler_iters": sampler_iters,
            "num_cycles": num_cycles,
            "max_steps_per_interaction": max_steps,
            "num_test_tasks": num_test_tasks,
            # Recorded in the shard for the same reason `env` is: a scheduled-arm shard
            # and a never-arm shard describe two different practice conditions and must
            # never be pooled.
            "practice_reset_policy": practice_reset_policy.value,
            "seeds": [
                SkillTraceCollector.run_seed(
                    env_name=env_name,
                    seed=seed,
                    sampler_iters=sampler_iters,
                    num_cycles=num_cycles,
                    max_steps=max_steps,
                    num_test_tasks=num_test_tasks,
                    practice_reset_policy=practice_reset_policy,
                )
                for seed in seeds
            ],
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Name for this run set in the JSON.")
    parser.add_argument(
        "--env",
        choices=ENV_CHOICES,
        default=CAUSAL_ENV,
        help="Which throw representation to trace. Both arms are the same world under "
        "the same protocol and produce shards of identical shape, so the analysis reads "
        "either -- but shards of the two arms must never be pooled into one run set.",
    )
    parser.add_argument(
        "--sampler-max-train-iters",
        type=int,
        default=EesMethod.model_fields["sampler_max_train_iters"].default,
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Run seeds 0..N-1 -- fixed, never randomly drawn, same as run_sweep.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit seeds instead of 0..--num-seeds-1. Its only purpose is SHARDING: "
        "these runs are serial within one process and long, so a full set is collected "
        "as one process per seed and the analysis pools the shards. Still fixed values, "
        "never drawn -- a shard's seed is chosen by the caller, not by an RNG.",
    )
    parser.add_argument(
        "--practice-reset-policy",
        type=PracticeResetPolicy,
        choices=list(PracticeResetPolicy),
        default=PracticeResetPolicy.SCHEDULED,
        help="Same flag as the global CLI's. 'scheduled' (the default) is every "
        "pre-existing caller's behaviour. 'never' traces the reset-free arm -- which is "
        "the whole reason this collector reports per-period skill tallies: it is what "
        "shows WHAT practice did differently, not just how the run scored.",
    )
    parser.add_argument("--num-cycles", type=int, default=25)
    parser.add_argument("--max-steps-per-interaction", type=int, default=100)
    parser.add_argument("--num-test-tasks", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = args.seeds if args.seeds is not None else list(range(args.num_seeds))
    traces = SkillTraceCollector.collect(
        label=args.label,
        env_name=args.env,
        sampler_iters=args.sampler_max_train_iters,
        seeds=seeds,
        num_cycles=args.num_cycles,
        max_steps=args.max_steps_per_interaction,
        num_test_tasks=args.num_test_tasks,
        practice_reset_policy=args.practice_reset_policy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(traces, indent=1))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
