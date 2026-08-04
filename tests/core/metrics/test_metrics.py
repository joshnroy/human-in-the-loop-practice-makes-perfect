import pytest

from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.metrics.types import TaskOutcome


def test_record_evaluation_appends_a_checkpoint() -> None:
    metrics = Metrics()
    metrics.record_evaluation(num_online_transitions=0, num_solved=0, num_total=10)
    metrics.record_evaluation(num_online_transitions=150, num_solved=7, num_total=10)
    assert metrics.evaluations == [(0, 0, 10), (150, 7, 10)]


def test_task_training_curve_converts_solved_counts_to_percentages() -> None:
    metrics = Metrics()
    metrics.record_evaluation(num_online_transitions=0, num_solved=0, num_total=10)
    metrics.record_evaluation(num_online_transitions=150, num_solved=7, num_total=10)
    assert metrics.task_training_curve() == [(0, 0.0), (150, 0.7)]


def test_task_training_curve_handles_a_zero_total_without_dividing_by_zero() -> None:
    metrics = Metrics()
    metrics.record_evaluation(num_online_transitions=0, num_solved=0, num_total=0)
    assert metrics.task_training_curve() == [(0, 0.0)]


def test_task_training_curve_by_subtask_wraps_the_single_curve_under_task_name() -> None:
    metrics = Metrics(task_name="light_on")
    metrics.record_evaluation(num_online_transitions=0, num_solved=5, num_total=10)
    assert metrics.task_training_curve_by_subtask() == {"light_on": [(0, 0.5)]}


def test_percentage_success_overall_test_uses_the_most_recent_evaluation() -> None:
    metrics = Metrics()
    metrics.record_evaluation(num_online_transitions=0, num_solved=0, num_total=10)
    metrics.record_evaluation(num_online_transitions=150, num_solved=8, num_total=10)
    assert metrics.percentage_success_overall_test() == 0.8


def test_percentage_success_overall_test_is_zero_with_no_evaluations_yet() -> None:
    assert Metrics().percentage_success_overall_test() == 0.0


def test_percentage_success_per_task_test_wraps_the_overall_percentage() -> None:
    metrics = Metrics(task_name="light_on")
    metrics.record_evaluation(num_online_transitions=0, num_solved=3, num_total=10)
    assert metrics.percentage_success_per_task_test() == {"light_on": 0.3}


def test_train_metrics_are_not_tracked() -> None:
    metrics = Metrics()
    assert metrics.percentage_success_overall_train() == 0.0
    assert metrics.percentage_success_per_task_train() == {}


def test_human_and_reset_metrics_are_always_zero() -> None:
    metrics = Metrics()
    assert metrics.num_complete_environment_resets() == 0
    assert metrics.num_human_interventions() == (0.0, 0)
    assert metrics.summed_human_cost() == 0.0


def test_two_instances_do_not_share_evaluations() -> None:
    """The whole point of this refactor: no shared ClassVar/reset() dance anymore."""
    first = Metrics()
    second = Metrics()
    first.record_evaluation(num_online_transitions=0, num_solved=1, num_total=1)
    assert first.evaluations == [(0, 1, 1)]
    assert second.evaluations == []


def test_record_evaluation_without_outcomes_records_no_breakdown() -> None:
    """The per-task detail is opt-in: a caller with only aggregate counts stays
    valid, and every stats.json written before breakdowns existed still loads."""
    metrics = Metrics()
    metrics.record_evaluation(num_online_transitions=0, num_solved=3, num_total=10)
    assert metrics.breakdowns == []
    assert metrics.failures_by_goal() == {}


def test_failures_by_goal_groups_the_final_sweep_by_goal() -> None:
    metrics = Metrics()
    metrics.record_evaluation(
        num_online_transitions=100,
        num_solved=1,
        num_total=3,
        outcomes=(
            TaskOutcome(task_index=0, goal="ItemInBin(recycling, recycling_bin)", solved=False),
            TaskOutcome(task_index=1, goal="ItemInBin(recycling, recycling_bin)", solved=False),
            TaskOutcome(task_index=2, goal="ItemInBin(trash, trash_bin)", solved=True),
        ),
    )
    assert metrics.failures_by_goal() == {
        "ItemInBin(recycling, recycling_bin)": (2, 2),
        "ItemInBin(trash, trash_bin)": (0, 1),
    }


def test_failures_by_goal_reports_only_the_final_sweep() -> None:
    """The question it answers is "what is the *trained* policy still failing",
    so an earlier sweep's failures must not leak into the count."""
    metrics = Metrics()
    metrics.record_evaluation(
        num_online_transitions=0,
        num_solved=0,
        num_total=1,
        outcomes=(TaskOutcome(task_index=0, goal="g", solved=False),),
    )
    metrics.record_evaluation(
        num_online_transitions=100,
        num_solved=1,
        num_total=1,
        outcomes=(TaskOutcome(task_index=0, goal="g", solved=True),),
    )
    assert metrics.failures_by_goal() == {"g": (0, 1)}


def test_record_evaluation_rejects_outcomes_that_disagree_with_the_aggregate() -> None:
    """The aggregate stays the primary record, so a silent disagreement between
    the two would be undetectable downstream."""
    metrics = Metrics()
    with pytest.raises(ValueError, match="disagree with the aggregate"):
        metrics.record_evaluation(
            num_online_transitions=0,
            num_solved=2,
            num_total=1,
            outcomes=(TaskOutcome(task_index=0, goal="g", solved=False),),
        )
    assert metrics.evaluations == []
    assert metrics.breakdowns == []
