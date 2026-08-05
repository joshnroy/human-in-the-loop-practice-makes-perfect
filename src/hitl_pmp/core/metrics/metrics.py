from pydantic import BaseModel, Field

from .types import EvaluationBreakdown, TaskOutcome


class Metrics(BaseModel):
    """Evaluation protocol -- a fully concrete, directly-usable instance now (not a
    static-method container): every method here is a genuine, reusable default, not
    a per-domain requirement, since nothing in this codebase today needs behavior
    other than what's written here -- exactly one task/goal type, and no real
    human-intervention tracking, since no Method here ever calls
    Problem.execute_human_command. This is a step further than Problem's own facade
    pattern (problem/problem.py): Problem still has one genuinely must-override
    method (run_task_episode); Metrics has none, so unlike Problem/Method/
    Environment/HumanOracle/Tasks/Renderer it isn't actually one of this project's
    abstract interfaces -- callers just construct Metrics() directly, with no
    per-domain/per-method subclass needed. A future Method that tracks real
    human-intervention cost would override num_human_interventions/
    summed_human_cost; a future multi-task environment would override
    task_training_curve_by_subtask/percentage_success_per_task_test --
    inheriting everything else unchanged either way (ordinary subclassing).

    evaluations/task_name are real instance fields now: a fresh Metrics() per
    (method, seed) run in a reproduction sweep replaces the old
    ClassVar-plus-reset() dance -- there's no shared mutable slot left to
    accidentally leak between runs or forget to reset()."""

    # Each tuple is (transitions, solved, total).
    evaluations: list[tuple[int, int, int]] = Field(default_factory=list)
    # Per-task detail behind each evaluations entry, when the caller supplies it.
    # Defaults to empty, so every stats.json written before this field existed
    # still loads, and a caller that only has aggregate counts stays valid.
    breakdowns: list[EvaluationBreakdown] = Field(default_factory=list)
    # How many times the harness put the environment back to the current practice
    # task's initial state, counted as it happened. Defaults to 0 so every
    # stats.json written before this field existed still loads.
    num_practice_resets: int = 0
    task_name: str = "default"

    def record_evaluation(
        self,
        *,
        num_online_transitions: int,
        num_solved: int,
        num_total: int,
        outcomes: tuple[TaskOutcome, ...] | None = None,
    ) -> None:
        """Records one evaluation checkpoint (e.g. after an online-learning cycle) --
        the building block task_training_curve() reports back out.

        outcomes is optional per-task detail; when given it must agree with
        num_solved/num_total, since the aggregate stays the primary record and a
        silent disagreement between the two would be undetectable downstream."""
        breakdown = (
            None
            if outcomes is None
            else EvaluationBreakdown(
                num_online_transitions=num_online_transitions, outcomes=outcomes
            )
        )
        # Validated before either list is appended to, so a rejected call leaves
        # this Metrics untouched rather than half-updated with the aggregate.
        if breakdown is not None and (
            len(breakdown.outcomes) != num_total or breakdown.num_solved() != num_solved
        ):
            raise ValueError(
                f"per-task outcomes disagree with the aggregate: got "
                f"{breakdown.num_solved()}/{len(breakdown.outcomes)} from outcomes, "
                f"{num_solved}/{num_total} from the counts"
            )
        self.evaluations.append((num_online_transitions, num_solved, num_total))
        if breakdown is not None:
            self.breakdowns.append(breakdown)

    def record_practice_reset(self) -> None:
        """Counts one free reset back to the current practice task's initial state.

        Recorded rather than rederived from the configured interval because a
        configured knob is a claim and this is the measurement: an experiment
        that varies how often the robot is rescued has to be able to show the
        resets really happened, at the rate intended, from the run's own output
        instead of from an argument about the loop's arithmetic."""
        self.num_practice_resets += 1

    def failures_by_goal(self) -> dict[str, tuple[int, int]]:
        """{goal description: (num_failed, num_total)} for the final evaluation
        sweep -- the "*which* tasks is it still failing?" view the aggregate
        curve cannot give. Empty when no per-task outcomes were recorded.

        Grouped by goal rather than listed per task because the useful unit is
        the task *family*: goal descriptions repeat across a test set, and a
        family that fails structurally (rather than by unlucky sampling) shows
        up as a whole group failing."""
        if not self.breakdowns:
            return {}
        grouped: dict[str, tuple[int, int]] = {}
        for outcome in self.breakdowns[-1].outcomes:
            failed, total = grouped.get(outcome.goal, (0, 0))
            grouped[outcome.goal] = (failed + int(not outcome.solved), total + 1)
        return grouped

    def task_training_curve(self) -> list[tuple[int, float]]:
        """(num_online_transitions, percentage_solved) pairs, in recorded order --
        e.g. Figure 4 of the "Practice Makes Perfect" paper plots exactly this,
        per approach per seed."""
        return [
            (transitions, (solved / total) if total else 0.0)
            for transitions, solved, total in self.evaluations
        ]

    def task_training_curve_by_subtask(self) -> dict[str, list[tuple[int, float]]]:
        return {self.task_name: self.task_training_curve()}

    def percentage_success_overall_test(self) -> float:
        if not self.evaluations:
            return 0.0
        _, solved, total = self.evaluations[-1]
        return solved / total if total else 0.0

    def percentage_success_per_task_test(self) -> dict[str, float]:
        return {self.task_name: self.percentage_success_overall_test()}

    def percentage_success_overall_train(self) -> float:
        """Not tracked: this reproduction only evaluates held-out test tasks
        after each online-learning cycle (matching predicators' own
        _run_testing, which only scores env.get_test_tasks())."""
        return 0.0

    def percentage_success_per_task_train(self) -> dict[str, float]:
        return {}

    def num_complete_environment_resets(self) -> int:
        return 0

    def num_human_interventions(self) -> tuple[float, int]:
        """Returns (summed cost, count); should trend down as the agent learns to
        reset itself. Trivially zero: no Method in this codebase yet ever calls
        Problem.execute_human_command."""
        return (0.0, 0)

    def summed_human_cost(self) -> float:
        return 0.0
