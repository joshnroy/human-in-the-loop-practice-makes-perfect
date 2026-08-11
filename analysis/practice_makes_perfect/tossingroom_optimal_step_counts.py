"""Optimal-step-count lower bound for Tossing Room's fixed test set: for each of the
`--num-test-tasks 30`, `--seed 0` test tasks every arm in
`docs/experiment-logs/2026-08-11-human-ladder-fixed-interval-10x/` is evaluated against,
the minimum number of skill executions (policy calls) a perfect planner needs to solve it
from that task's own initial state.

**Why this exists.** Added at Josh's request alongside that PR: the "extra solves per
rescue" numbers in `human_ladder_curves.py` say how much a rescue bought relative to
`no-human`, but say nothing about how hard the underlying tasks are in absolute terms. A
per-family optimal-step-count table gives a concrete difficulty floor to read those
numbers against.

**"Steps" means skill executions (policy calls), matching every other step count in this
project** (`--max-steps-per-interaction`, `PracticeLoop`'s own step counter) -- not raw
MuJoCo/environment ticks, which do not exist on this domain anyway (Tossing Room has no
sub-skill simulation; one `take_action` call already is one whole skill).

**How it computes a lower bound without simulating anything.** `FastDownwardPlanner.plan`
runs a real optimal PDDL search (`alias="seq-opt-lmcut"`, its own default) over each
task's `(init_atoms, goal)` pair. Passing no `ground_skill_costs` gives every ground skill
`default_cost=1.0` uniformly, so the returned plan's LENGTH is exactly the minimum
skill-execution count, not a cost-weighted proxy for one. There is no environment
stepping anywhere in this module -- only a symbolic search -- so this stays within
`analysis/`'s "never drives a simulation or a `Method`" rule even though it constructs a
`Problem` directly: that construction is reused, not duplicated, from the real CLI
composition root (`Cli.parse_args` + `TossingRoomCli.build_problem`), so the 30 tasks
sampled here are byte-identical to the ones `PracticeLoop.run` draws once, up front, for
any real `--seed 0` run ("Drawn ONCE, up front" -- see that method's own docstring) --
not a second, hand-rolled `Environment`/`Tasks` construction that could silently drift
from what a real run actually evaluates against.

Reads no `--output-dir` output (unlike every other `analysis/` script here): there is
nothing to read back, since this never runs a sweep. It is closer in spirit to
`planning/`'s own tests, which shell out to a real Fast Downward rather than being
skipped.
"""

import argparse
from collections import defaultdict

from analysis.practice_makes_perfect.goal_families import GoalFamilies
from hitl_pmp.cli import Cli
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.tossingroom.cli import TossingRoomCli
from hitl_pmp.environments.tossingroom.skill_provider import TossingRoomSkillProvider
from hitl_pmp.planning.fast_downward import FastDownwardPlanner, PlanningFailure
from hitl_pmp.planning.grounding import SkillGrounder


class TossingRoomOptimalStepCounts:
    """A static-method container, never instantiated."""

    @staticmethod
    def build_test_tasks(
        *, seed: int, num_test_tasks: int
    ) -> tuple[list[Task], TossingRoomSkillProvider]:
        """The exact fixed test set a real `--seed <seed>` run evaluates against, plus
        this domain's `SkillProvider` (built off the same `Environment` instance, so its
        `objects()`/`types()` match the tasks' own initial states)."""
        args = Cli.parse_args(
            argv=[
                "--env",
                "tossingroom",
                "--method",
                "ees",
                "--seed",
                str(seed),
                "--num-test-tasks",
                str(num_test_tasks),
            ]
        )
        problem = TossingRoomCli.build_problem(args=args)
        provider = TossingRoomSkillProvider(env=problem.env)
        tasks = [problem.tasks.sample_test_task() for _ in range(num_test_tasks)]
        return tasks, provider

    @staticmethod
    def optimal_step_count(*, task: Task, provider: TossingRoomSkillProvider) -> int | None:
        """The minimum number of skill executions from this task's own initial state to
        its goal, or `None` if Fast Downward finds no plan at all (not expected on this
        domain's own test set, but reported rather than assumed)."""
        init_atoms = SkillGrounder.abstract_state(
            state=task.initial_state,
            objects=provider.objects(),
            predicates=provider.predicates(),
        )
        try:
            plan = FastDownwardPlanner.plan(
                skills=provider.skills(),
                predicates=provider.predicates(),
                types=provider.types(),
                objects=provider.objects(),
                init_atoms=init_atoms,
                goal=task.goal.atoms,
            )
        except PlanningFailure:
            return None
        return len(plan)

    @staticmethod
    def compute_all(*, seed: int, num_test_tasks: int) -> list[dict]:
        """One row per task, in the order `PracticeLoop.run` itself draws them."""
        tasks, provider = TossingRoomOptimalStepCounts.build_test_tasks(
            seed=seed, num_test_tasks=num_test_tasks
        )
        rows = []
        for index, task in enumerate(tasks):
            family = GoalFamilies.classify(goal=task.goal.describe())
            steps = TossingRoomOptimalStepCounts.optimal_step_count(task=task, provider=provider)
            rows.append({"task_index": index, "family": family, "optimal_steps": steps})
        return rows

    @staticmethod
    def summarize_by_family(*, rows: list[dict]) -> dict[str, dict]:
        """Per-family min/mean/max over tasks with a found plan, plus how many (of that
        family's total) had none -- reported rather than silently excluded, since a
        `None` there would mean this domain's own fixed test set contains an unsolvable
        task, which is itself worth knowing."""
        by_family: dict[str, list[int]] = defaultdict(list)
        unsolved: dict[str, int] = defaultdict(int)
        totals: dict[str, int] = defaultdict(int)
        for row in rows:
            totals[row["family"]] += 1
            if row["optimal_steps"] is None:
                unsolved[row["family"]] += 1
            else:
                by_family[row["family"]].append(row["optimal_steps"])
        summary = {}
        for family, values in by_family.items():
            summary[family] = {
                "n_solved": len(values),
                "n_total": totals[family],
                "n_unsolved": unsolved[family],
                "min": min(values),
                "mean": sum(values) / len(values),
                "max": max(values),
            }
        for family in unsolved:
            if family not in summary:
                summary[family] = {
                    "n_solved": 0,
                    "n_total": totals[family],
                    "n_unsolved": unsolved[family],
                    "min": None,
                    "mean": None,
                    "max": None,
                }
        return summary

    @staticmethod
    def print_report(*, rows: list[dict]) -> None:
        summary = TossingRoomOptimalStepCounts.summarize_by_family(rows=rows)
        print("optimal skill-execution count (Fast Downward, seq-opt-lmcut, unit cost)\n")
        for family in ("TRASH", "RECYCLING", "EMPTY"):
            stats = summary.get(family)
            if stats is None:
                continue
            if stats["n_solved"] == 0:
                print(f"  {family:>10}  {stats['n_unsolved']}/{stats['n_total']} unsolved")
                continue
            print(
                f"  {family:>10}  n={stats['n_solved']}/{stats['n_total']}  "
                f"min={stats['min']}  mean={stats['mean']:.2f}  max={stats['max']}"
            )
        print()
        for row in rows:
            print(
                f"  task {row['task_index']:>2}  {row['family']:>10}  steps={row['optimal_steps']}"
            )

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--num-test-tasks", type=int, default=30)
        args = parser.parse_args()
        rows = TossingRoomOptimalStepCounts.compute_all(
            seed=args.seed, num_test_tasks=args.num_test_tasks
        )
        TossingRoomOptimalStepCounts.print_report(rows=rows)


if __name__ == "__main__":
    TossingRoomOptimalStepCounts.main()
