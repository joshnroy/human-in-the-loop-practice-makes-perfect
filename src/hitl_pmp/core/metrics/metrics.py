from typing import ClassVar


class Metrics:
    """Evaluation protocol -- a fully concrete, directly-usable static-method
    container (not an abc.ABC: every method here is a genuine, reusable default,
    not a per-domain requirement, since nothing in this codebase today needs
    behavior other than what's written here -- exactly one task/goal type, and
    no real human-intervention tracking, since no Method here ever calls
    Problem.execute_human_command). This is a step further than Problem's own
    facade pattern (problem/problem.py): Problem still has one genuinely
    must-override method (run_task_episode); Metrics has none, so unlike
    Problem/Method/Environment/HumanOracle/Tasks/Renderer it isn't actually one
    of this project's abstract interfaces -- callers just use Metrics directly,
    with no per-domain/per-method subclass needed. A future Method that tracks
    real human-intervention cost would override num_human_interventions/
    summed_human_cost; a future multi-task environment would override
    task_training_curve_by_subtask/percentage_success_per_task_test --
    inheriting everything else unchanged either way (ordinary subclassing,
    available regardless of whether the parent is abstract).

    evaluations/task_name are shared ClassVars -- call reset() before each
    (method, seed) run in a reproduction sweep, since a caller reading
    task_training_curve() right after a run finishes (before the next
    reset()) is what makes this safe to share across many sequential runs.
    This is the same single-shared-mutable-slot tradeoff Problem.env/
    Problem.tasks already make; this project only ever runs one (problem,
    method, seed) combination at a time (see e.g. analysis/ scripts
    shelling out one CLI subprocess per seed), so it's never actually
    stressed by concurrent use."""

    evaluations: ClassVar[list[tuple[int, int, int]]] = []  # (transitions, solved, total)
    task_name: ClassVar[str] = "default"

    @staticmethod
    def reset() -> None:
        Metrics.evaluations = []

    @staticmethod
    def record_evaluation(*, num_online_transitions: int, num_solved: int, num_total: int) -> None:
        """Records one evaluation checkpoint (e.g. after an online-learning cycle) --
        the building block task_training_curve() reports back out."""
        Metrics.evaluations.append((num_online_transitions, num_solved, num_total))

    @staticmethod
    def task_training_curve() -> list[tuple[int, float]]:
        """(num_online_transitions, percentage_solved) pairs, in recorded order --
        e.g. Figure 4 of the "Practice Makes Perfect" paper plots exactly this,
        per approach per seed."""
        return [
            (transitions, (solved / total) if total else 0.0)
            for transitions, solved, total in Metrics.evaluations
        ]

    @staticmethod
    def task_training_curve_by_subtask() -> dict[str, list[tuple[int, float]]]:
        return {Metrics.task_name: Metrics.task_training_curve()}

    @staticmethod
    def percentage_success_overall_test() -> float:
        if not Metrics.evaluations:
            return 0.0
        _, solved, total = Metrics.evaluations[-1]
        return solved / total if total else 0.0

    @staticmethod
    def percentage_success_per_task_test() -> dict[str, float]:
        return {Metrics.task_name: Metrics.percentage_success_overall_test()}

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
        """Returns (summed cost, count); should trend down as the agent learns to
        reset itself. Trivially zero: no Method in this codebase yet ever calls
        Problem.execute_human_command."""
        return (0.0, 0)

    @staticmethod
    def summed_human_cost() -> float:
        return 0.0
