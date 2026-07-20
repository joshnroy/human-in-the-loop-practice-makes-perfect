from collections.abc import Iterator

import pytest

from hitl_pmp.environments.lightswitch.metrics import LightSwitchMetrics


@pytest.fixture(autouse=True)
def _reset_metrics() -> Iterator[None]:
    LightSwitchMetrics.reset()
    yield
    LightSwitchMetrics.reset()


def test_record_evaluation_appends_a_checkpoint() -> None:
    LightSwitchMetrics.record_evaluation(num_online_transitions=0, num_solved=0, num_total=10)
    LightSwitchMetrics.record_evaluation(num_online_transitions=150, num_solved=7, num_total=10)
    assert LightSwitchMetrics.evaluations == [(0, 0, 10), (150, 7, 10)]


def test_task_training_curve_converts_solved_counts_to_percentages() -> None:
    LightSwitchMetrics.record_evaluation(num_online_transitions=0, num_solved=0, num_total=10)
    LightSwitchMetrics.record_evaluation(num_online_transitions=150, num_solved=7, num_total=10)
    assert LightSwitchMetrics.task_training_curve() == [(0, 0.0), (150, 0.7)]


def test_task_training_curve_handles_a_zero_total_without_dividing_by_zero() -> None:
    LightSwitchMetrics.record_evaluation(num_online_transitions=0, num_solved=0, num_total=0)
    assert LightSwitchMetrics.task_training_curve() == [(0, 0.0)]


def test_task_training_curve_by_subtask_wraps_the_single_curve() -> None:
    LightSwitchMetrics.record_evaluation(num_online_transitions=0, num_solved=5, num_total=10)
    assert LightSwitchMetrics.task_training_curve_by_subtask() == {"light_on": [(0, 0.5)]}


def test_percentage_success_overall_test_uses_the_most_recent_evaluation() -> None:
    LightSwitchMetrics.record_evaluation(num_online_transitions=0, num_solved=0, num_total=10)
    LightSwitchMetrics.record_evaluation(num_online_transitions=150, num_solved=8, num_total=10)
    assert LightSwitchMetrics.percentage_success_overall_test() == 0.8


def test_percentage_success_overall_test_is_zero_with_no_evaluations_yet() -> None:
    assert LightSwitchMetrics.percentage_success_overall_test() == 0.0


def test_percentage_success_per_task_test_wraps_the_overall_percentage() -> None:
    LightSwitchMetrics.record_evaluation(num_online_transitions=0, num_solved=3, num_total=10)
    assert LightSwitchMetrics.percentage_success_per_task_test() == {"light_on": 0.3}


def test_train_metrics_are_not_tracked() -> None:
    assert LightSwitchMetrics.percentage_success_overall_train() == 0.0
    assert LightSwitchMetrics.percentage_success_per_task_train() == {}


def test_human_and_reset_metrics_are_always_zero() -> None:
    assert LightSwitchMetrics.num_complete_environment_resets() == 0
    assert LightSwitchMetrics.num_human_interventions() == (0.0, 0)
    assert LightSwitchMetrics.summed_human_cost() == 0.0


def test_reset_clears_recorded_evaluations() -> None:
    LightSwitchMetrics.record_evaluation(num_online_transitions=0, num_solved=1, num_total=1)
    LightSwitchMetrics.reset()
    assert LightSwitchMetrics.evaluations == []
