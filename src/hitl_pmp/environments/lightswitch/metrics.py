from typing import ClassVar

from hitl_pmp.core.metrics.metrics import Metrics


class LightSwitchMetrics(Metrics):
    """Evaluation protocol for Light Switch. `evaluations` is a shared ClassVar
    (matching LightSwitchEnvironment/LightSwitchTasks's own singleton-style
    state) -- call reset() before each (method, seed) run in a reproduction
    sweep, since a caller reading task_training_curve() right after a run
    finishes (before the next reset()) is what makes this safe to share
    across many sequential runs.

    PMP-family methods (methods/practice_makes_perfect/) never use a
    HumanOracle -- Light Switch has no irreversible action, matching
    LightSwitchProblem never setting `human` -- so num_complete_environment_resets/
    num_human_interventions/summed_human_cost are always trivially zero here.
    Light Switch also has exactly one task/goal type (LightOn), so the
    per-task-name methods just wrap the single overall curve under that name
    rather than reporting a real per-task breakdown."""

    evaluations: ClassVar[
        list[tuple[int, int, int]]
    ] = []  # (num_online_transitions, num_solved, num_total)

    @staticmethod
    def reset() -> None:
        LightSwitchMetrics.evaluations = []

    @staticmethod
    def record_evaluation(*, num_online_transitions: int, num_solved: int, num_total: int) -> None:
        LightSwitchMetrics.evaluations.append((num_online_transitions, num_solved, num_total))

    @staticmethod
    def task_training_curve() -> list[tuple[int, float]]:
        return [
            (transitions, (solved / total) if total else 0.0)
            for transitions, solved, total in LightSwitchMetrics.evaluations
        ]

    @staticmethod
    def task_training_curve_by_subtask() -> dict[str, list[tuple[int, float]]]:
        return {"light_on": LightSwitchMetrics.task_training_curve()}

    @staticmethod
    def percentage_success_overall_test() -> float:
        if not LightSwitchMetrics.evaluations:
            return 0.0
        _, solved, total = LightSwitchMetrics.evaluations[-1]
        return solved / total if total else 0.0

    @staticmethod
    def percentage_success_per_task_test() -> dict[str, float]:
        return {"light_on": LightSwitchMetrics.percentage_success_overall_test()}

    @staticmethod
    def percentage_success_overall_train() -> float:
        """Not tracked: this reproduction only evaluates held-out test tasks
        after each online-learning cycle (matching predicators' own
        _run_testing, which only scores env.get_test_tasks())."""
        return 0.0

    @staticmethod
    def percentage_success_per_task_train() -> dict[str, float]:
        return {}

    @staticmethod
    def num_complete_environment_resets() -> int:
        return 0

    @staticmethod
    def num_human_interventions() -> tuple[float, int]:
        return (0.0, 0)

    @staticmethod
    def summed_human_cost() -> float:
        return 0.0
