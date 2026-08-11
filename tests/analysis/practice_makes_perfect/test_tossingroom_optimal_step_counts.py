"""`TossingRoomOptimalStepCounts` turns the fixed --seed 0 test set into a per-family
optimal-step-count table. The arithmetic (`summarize_by_family`) is pinned with hand-built
rows and needs no Fast Downward; `build_test_tasks`/`compute_all` genuinely shell out to a
real planner, matching `planning/`'s own tests rather than being skipped, so a broken
install fails loudly here too."""

from analysis.practice_makes_perfect.goal_families import GoalFamilies
from analysis.practice_makes_perfect.tossingroom_optimal_step_counts import (
    TossingRoomOptimalStepCounts,
)


def test_summarize_by_family_reports_min_mean_max() -> None:
    rows = [
        {"task_index": 0, "family": "TRASH", "optimal_steps": 5},
        {"task_index": 1, "family": "TRASH", "optimal_steps": 7},
        {"task_index": 2, "family": "RECYCLING", "optimal_steps": 4},
    ]
    summary = TossingRoomOptimalStepCounts.summarize_by_family(rows=rows)
    assert summary["TRASH"] == {
        "n_solved": 2,
        "n_total": 2,
        "n_unsolved": 0,
        "min": 5,
        "mean": 6.0,
        "max": 7,
    }
    assert summary["RECYCLING"]["min"] == 4
    assert summary["RECYCLING"]["max"] == 4


def test_summarize_by_family_reports_unsolved_tasks_rather_than_dropping_them() -> None:
    rows = [
        {"task_index": 0, "family": "EMPTY", "optimal_steps": None},
        {"task_index": 1, "family": "EMPTY", "optimal_steps": None},
    ]
    summary = TossingRoomOptimalStepCounts.summarize_by_family(rows=rows)
    assert summary["EMPTY"] == {
        "n_solved": 0,
        "n_total": 2,
        "n_unsolved": 2,
        "min": None,
        "mean": None,
        "max": None,
    }


def test_build_test_tasks_reproduces_the_fixed_composition() -> None:
    """--num-test-tasks 30 at --seed 0 is 14 TRASH / 14 RECYCLING / 2 EMPTY, the same
    composition every arm's own stats.json is checked against elsewhere in this PR."""
    tasks, provider = TossingRoomOptimalStepCounts.build_test_tasks(seed=0, num_test_tasks=30)
    assert len(tasks) == 30
    families = [GoalFamilies.classify(goal=task.goal.describe()) for task in tasks]
    assert families.count("TRASH") == 14
    assert families.count("RECYCLING") == 14
    assert families.count("EMPTY") == 2
    assert provider.env is not None


def test_optimal_step_count_finds_a_real_plan_for_a_trash_task() -> None:
    tasks, provider = TossingRoomOptimalStepCounts.build_test_tasks(seed=0, num_test_tasks=2)
    for task in tasks:
        steps = TossingRoomOptimalStepCounts.optimal_step_count(task=task, provider=provider)
        # Every task in this domain's fixed test set is solvable from its own initial
        # state (that is what makes it a fair test set), so None here is a real failure.
        assert steps is not None
        assert steps > 0


def test_compute_all_returns_one_row_per_task_with_a_recognised_family() -> None:
    rows = TossingRoomOptimalStepCounts.compute_all(seed=0, num_test_tasks=5)
    assert len(rows) == 5
    for row in rows:
        assert row["family"] in ("TRASH", "RECYCLING", "EMPTY")
        assert row["optimal_steps"] is not None and row["optimal_steps"] > 0


def test_compute_all_is_deterministic_for_a_fixed_seed() -> None:
    """Same seed, same tasks, same plans -- re-running must reproduce the same table,
    matching this project's standing seed-determinism guarantee."""
    first = TossingRoomOptimalStepCounts.compute_all(seed=0, num_test_tasks=4)
    second = TossingRoomOptimalStepCounts.compute_all(seed=0, num_test_tasks=4)
    assert first == second


def test_build_test_tasks_is_wired_to_the_seed_argument() -> None:
    """Not a strict pin on WHAT differs (only the weight seed varies per the task
    distribution's own docstring) -- just that build_test_tasks actually forwards --seed
    to Cli.parse_args rather than silently always drawing seed 0, checked on the one
    field a task's initial state actually carries per-task: the pile's weight_seed
    feature (`TossingRoomEnvironment.build_initial_state` parks the draw there)."""
    tasks_seed0, provider0 = TossingRoomOptimalStepCounts.build_test_tasks(seed=0, num_test_tasks=1)
    tasks_seed1, provider1 = TossingRoomOptimalStepCounts.build_test_tasks(seed=1, num_test_tasks=1)
    weight_seed0 = tasks_seed0[0].initial_state.get(
        obj=provider0.env.pile, feature_name="weight_seed"
    )
    weight_seed1 = tasks_seed1[0].initial_state.get(
        obj=provider1.env.pile, feature_name="weight_seed"
    )
    assert weight_seed0 != weight_seed1
