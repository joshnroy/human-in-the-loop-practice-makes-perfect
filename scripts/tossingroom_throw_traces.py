"""Collect Tossing Room execution traces for EES: what the robot actually does during
every evaluation sweep and every interaction period, dumped as JSON for
`analysis/practice_makes_perfect/tossingroom_throw_convergence.py` to render.

Why this exists rather than being read off `--output-dir`: `stats.json` records the
*outcome* (tasks solved), and the question this domain poses is about the one
continuous parameter EES has to learn -- `Throw`'s force, against the force each
grounding actually requires (an unobserved function of the bin's `throw_distance` and
the item's `weight`) -- which never leaves the method's internals. So this hooks the two
places the numbers live: `EesMethod.execute_ground_skill` (which force was chosen, and
whether the epsilon-greedy random branch chose it) and `Environment.take_action`
(whether the action changed the state at all, i.e. was a silent no-op -- the failure
mode that scored EES 1/10 on this domain before PR #28, so worth measuring rather than
assuming fixed).

Both hooks are ordinary subclass overrides, and the sweep/period boundary is
`Method.end_cycle` -- which `PracticeLoop` calls exactly once per cycle, after the
interaction period and before the evaluation sweep. Nothing here monkeypatches or
re-implements `PracticeLoop`, so the traces come from the same protocol a real run uses
(fixed test set included) rather than from a copy of it that could drift.

It lives in `scripts/` because it *drives* simulations, which `analysis/` may never do
(CLAUDE.md). Unlike `scripts/run_sweep.py` it does import `hitl_pmp`: run_sweep's
shell-out-only discipline exists so a sweep cannot bypass the CLI, but there is no CLI
surface for a method's internal decisions, and adding one purely for a diagnostic would
put trace plumbing in the shipped `Method`.

Seeds are fixed (0..num_seeds-1), never randomly drawn, same as run_sweep.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Policy
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.tossingroom.environment import TossingRoomEnvironment
from hitl_pmp.environments.tossingroom.problem import TossingRoomProblem
from hitl_pmp.environments.tossingroom.skill_provider import TossingRoomSkillProvider
from hitl_pmp.environments.tossingroom.skills import TossingRoomSkills
from hitl_pmp.environments.tossingroom.tasks import TossingRoomTasks
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.practice_loop import PracticeLoop


class Bucket(BaseModel):
    """Everything observed in one phase (one evaluation sweep, or one interaction
    period), reset each time it is drained."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # |chosen force - that grounding's required force|, split by whether the epsilon-greedy
    # random branch made the choice: only the greedy ones say anything about what the
    # classifier has learned, which is why the two are never pooled.
    greedy_throw_errors: list[float] = Field(default_factory=list)
    random_throw_errors: list[float] = Field(default_factory=list)
    skill_counts: dict[str, int] = Field(default_factory=dict)
    num_actions: int = 0
    num_noop_actions: int = 0

    def drain(self) -> dict:
        snapshot = {
            "greedy_throw_errors": [round(value, 6) for value in self.greedy_throw_errors],
            "random_throw_errors": [round(value, 6) for value in self.random_throw_errors],
            "skill_counts": dict(self.skill_counts),
            "num_actions": self.num_actions,
            "num_noop_actions": self.num_noop_actions,
        }
        self.greedy_throw_errors = []
        self.random_throw_errors = []
        self.skill_counts = {}
        self.num_actions = 0
        self.num_noop_actions = 0
        return snapshot


class TraceLog(BaseModel):
    """The accumulator both instrumented classes below write into, with one bucket per
    phase so an evaluation sweep is never pooled with the practice that preceded it.

    A real instance passed by construction rather than module-level state: the seed loop
    runs several times in one process, and a shared accumulator is exactly the leaking-
    ClassVar trap this repo's `core/` docstrings describe."""

    evaluation: Bucket = Field(default_factory=Bucket)
    practice: Bucket = Field(default_factory=Bucket)
    # Which phase the method is currently in, set by whichever policy getter was called.
    # "evaluation" first because PracticeLoop's very first act is an evaluation sweep.
    phase: str = "evaluation"

    @property
    def current(self) -> Bucket:
        return self.evaluation if self.phase == "evaluation" else self.practice


class TracingEnvironment(TossingRoomEnvironment):
    """Counts silent no-ops. `take_action` is total over the whole Box, so an action
    out of context leaves the state untouched rather than raising -- which is precisely
    why an over-permissive operator model can produce plans that look fine and do
    nothing."""

    log: TraceLog

    def take_action(self, *, action: Action) -> State:
        before = self.get_current_state()
        after = super().take_action(action=action)
        bucket = self.log.current
        bucket.num_actions += 1
        if all(np.allclose(before[obj], after[obj]) for obj in after.data):
            bucket.num_noop_actions += 1
        return after


class TracingEesMethod(EesMethod):
    """Records every skill executed and, for `Throw`, the error between the chosen force
    and the force that grounding actually required -- plus the phase boundaries the
    buckets need.

    The required force is not a state feature (that was the defect the throw
    representation change removed); it is `TossingRoomEnvironment.required_force` of the
    bound bin's `throw_distance` and the bound item's `weight`. Reading it here is
    privileged instrumentation, exactly as reading `target_force` used to be -- the
    method under test never sees the coefficients."""

    log: TraceLog

    def get_task_policy(self, *, task: Task) -> Policy:
        self.log.phase = "evaluation"
        return super().get_task_policy(task=task)

    def get_practice_policy(self, *, task: Task) -> Policy:
        self.log.phase = "practice"
        return super().get_practice_policy(task=task)

    def execute_ground_skill(
        self, *, ground_skill: GroundSkill, state: State, explore: bool
    ) -> tuple[LabeledAction, object]:
        labeled, record = super().execute_ground_skill(
            ground_skill=ground_skill, state=state, explore=explore
        )
        bucket = self.log.current
        name = ground_skill.skill.name
        bucket.skill_counts[name] = bucket.skill_counts.get(name, 0) + 1
        if ground_skill.skill == TossingRoomSkills.THROW:
            _robot, item, bin_obj, _room = ground_skill.objects
            assert isinstance(self.env, TossingRoomEnvironment)
            required = self.env.required_force(
                throw_distance=float(state.get(obj=bin_obj, feature_name="throw_distance")),
                item_weight=float(state.get(obj=item, feature_name="weight")),
            )
            error = abs(float(labeled.action[2]) - required)
            if record is not None and record.was_random_exploration:
                bucket.random_throw_errors.append(error)
            else:
                bucket.greedy_throw_errors.append(error)
        return labeled, record


class ThrowTraceCollector:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def run_seed(
        *, seed: int, sampler_iters: int, num_cycles: int, max_steps: int, num_test_tasks: int
    ) -> dict:
        """One full EES run, returning the per-sweep and per-period traces.

        The drain points are `end_cycle` (called between the interaction period and the
        sweep that measures it) plus one final drain after `PracticeLoop.run` returns,
        which is what makes `sweeps` line up one-to-one with `metrics.evaluations`."""
        log = TraceLog()
        env = TracingEnvironment(log=log)
        # num_test_tasks must be passed, not left at its default -- see the identical
        # note in scripts/tossingroom_horizon_sweep.py. The field is what
        # TossingRoomTasks divides between the goal families, and it has to agree with
        # the num_test_tasks handed to PracticeLoop.run below or the run is measured on
        # a 12/12/6 test set instead of the 14/14/2 every other experiment uses.
        tasks = TossingRoomTasks(env=env, seed=seed, num_test_tasks=num_test_tasks)
        problem = TossingRoomProblem(env=env, tasks=tasks)
        method = TracingEesMethod(
            env=env,
            skill_provider=TossingRoomSkillProvider(env=env),
            seed=seed,
            sampler_max_train_iters=sampler_iters,
            log=log,
        )
        metrics = Metrics()
        sweeps: list[dict] = []
        periods: list[dict] = []

        def on_cycle_end() -> None:
            sweeps.append(log.evaluation.drain())
            periods.append(log.practice.drain())

        PracticeLoop.run(
            problem=problem,
            method=method,
            metrics=metrics,
            num_cycles=num_cycles,
            max_steps_per_interaction=max_steps,
            num_test_tasks=num_test_tasks,
            on_cycle_end=on_cycle_end,
        )
        sweeps.append(log.evaluation.drain())

        for sweep, (transitions, solved, total) in zip(sweeps, metrics.evaluations, strict=True):
            sweep["transitions"] = transitions
            sweep["solved"] = solved
            sweep["total"] = total
        return {
            "seed": seed,
            "horizon": problem.max_episode_steps(),
            "sweeps": sweeps,
            "periods": periods,
        }

    @staticmethod
    def collect(
        *,
        label: str,
        sampler_iters: int,
        num_seeds: int,
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
                ThrowTraceCollector.run_seed(
                    seed=seed,
                    sampler_iters=sampler_iters,
                    num_cycles=num_cycles,
                    max_steps=max_steps,
                    num_test_tasks=num_test_tasks,
                )
                for seed in range(num_seeds)
            ],
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Name for this arm in the JSON.")
    parser.add_argument("--sampler-max-train-iters", type=int, required=True)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-cycles", type=int, default=10)
    parser.add_argument("--max-steps-per-interaction", type=int, default=150)
    parser.add_argument("--num-test-tasks", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    arm = ThrowTraceCollector.collect(
        label=args.label,
        sampler_iters=args.sampler_max_train_iters,
        num_seeds=args.num_seeds,
        num_cycles=args.num_cycles,
        max_steps=args.max_steps_per_interaction,
        num_test_tasks=args.num_test_tasks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(arm, indent=1))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
