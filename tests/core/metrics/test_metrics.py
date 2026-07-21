from hitl_pmp.core.metrics.metrics import Metrics


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
