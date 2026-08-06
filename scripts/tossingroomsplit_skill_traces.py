"""Collect PER-SKILL practice traces for EES on Tossing Room (split throws): how many
times each lifted skill was attempted during practice, how many of those attempts
achieved the skill's own add effects, and where each skill's competence got to -- dumped
as JSON for `analysis/practice_makes_perfect/tossingroomsplit_throw_rates.py` to render.

**Why this exists rather than being read off `--output-dir`.** `stats.json` is the
serialized `core.Metrics`: it records *tasks solved*, per evaluation sweep, with a
per-task goal breakdown. That is the right record of outcomes and it is what
`scripts/run_sweep.py` produces. But the question this domain poses is about the two
throw SKILLS -- `ThrowTrash` against `ThrowRecycling` -- and how often each one was
practiced at all. Attempts, successes and competence never leave `EesMethod`'s
internals, so they are read out here by subclassing it.

That mirrors `scripts/tossingroom_throw_traces.py` exactly, for the same reason its
docstring gives: there is no CLI surface for a method's internal decisions, and adding
one purely for a diagnostic would put trace plumbing in the shipped `Method`.

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

from pydantic import BaseModel, ConfigDict, Field

from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Policy
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.tossingroomsplit.environment import TossingRoomSplitEnvironment
from hitl_pmp.environments.tossingroomsplit.problem import TossingRoomSplitProblem
from hitl_pmp.environments.tossingroomsplit.skill_provider import TossingRoomSplitSkillProvider
from hitl_pmp.environments.tossingroomsplit.tasks import TossingRoomSplitTasks
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.practice_loop import PracticeLoop


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
            if was_random:
                tally.landed_random += int(throw.landed)
            else:
                tally.greedy_forces.append(throw.force)
                tally.greedy_targets.append(throw.target)

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
    # The force actually issued and the force that grounding REQUIRED. The latter is not
    # a state feature -- it is `required_force` of the bound bin's `throw_distance` and
    # the bound item's `weight`, which only the environment can compute. Kept alongside
    # the verdict rather than derived from it: `landed` collapses everything about a
    # throw into one bit, and "missed by 0.02" and "missed by 0.75" are different
    # findings.
    force: float
    target: float


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
    pending_throws: dict[str, ThrowObservation] = Field(default_factory=dict)

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
            self.pending_throws[name] = self._observe_throw(
                name=name, state=state, force=float(labeled.action[2])
            )
        return labeled, record

    def _observe_throw(self, *, name: str, state: State, force: float) -> ThrowObservation:
        env = self.env
        assert isinstance(env, TossingRoomSplitEnvironment)
        trash = name == "ThrowTrash"
        item = env.trash if trash else env.recycling
        bin_obj = env.trash_bin if trash else env.recycling_bin
        bin_room = env.trash_bin_room if trash else env.recycling_bin_room
        target = env.required_force(
            throw_distance=float(state.get(obj=bin_obj, feature_name="throw_distance")),
            item_weight=float(state.get(obj=item, feature_name="weight")),
        )
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
        *, seed: int, sampler_iters: int, num_cycles: int, max_steps: int, num_test_tasks: int
    ) -> dict:
        """One full EES run, returning per-period skill tallies, per-cycle competence,
        and the per-sweep evaluation record (with its goal-family breakdown).

        `num_test_tasks` must be passed to `TossingRoomSplitTasks` as well as to
        `PracticeLoop.run`: the field is what the fixed goal-family composition is
        divided out of, and a disagreement silently measures a different test set."""
        log = PeriodLog()
        env = TossingRoomSplitEnvironment()
        tasks = TossingRoomSplitTasks(env=env, seed=seed, num_test_tasks=num_test_tasks)
        problem = TossingRoomSplitProblem(env=env, tasks=tasks)
        method = TracingEesMethod(
            env=env,
            skill_provider=TossingRoomSplitSkillProvider(env=env),
            seed=seed,
            sampler_max_train_iters=sampler_iters,
            log=log,
        )
        metrics = Metrics()
        periods: list[dict] = []
        competence: list[dict] = []

        def on_cycle_end() -> None:
            periods.append(log.drain())
            competence.append(method.competence_snapshot())

        PracticeLoop.run(
            problem=problem,
            method=method,
            metrics=metrics,
            num_cycles=num_cycles,
            max_steps_per_interaction=max_steps,
            num_test_tasks=num_test_tasks,
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
        sampler_iters: int,
        seeds: list[int],
        num_cycles: int,
        max_steps: int,
        num_test_tasks: int,
    ) -> dict:
        return {
            "label": label,
            "sampler_iters": sampler_iters,
            "num_cycles": num_cycles,
            "max_steps_per_interaction": max_steps,
            "num_test_tasks": num_test_tasks,
            "seeds": [
                SkillTraceCollector.run_seed(
                    seed=seed,
                    sampler_iters=sampler_iters,
                    num_cycles=num_cycles,
                    max_steps=max_steps,
                    num_test_tasks=num_test_tasks,
                )
                for seed in seeds
            ],
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Name for this run set in the JSON.")
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
        sampler_iters=args.sampler_max_train_iters,
        seeds=seeds,
        num_cycles=args.num_cycles,
        max_steps=args.max_steps_per_interaction,
        num_test_tasks=args.num_test_tasks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(traces, indent=1))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
