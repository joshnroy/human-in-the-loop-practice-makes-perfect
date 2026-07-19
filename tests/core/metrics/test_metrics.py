from hitl_pmp.core.metrics.metrics import Metrics


def test_metrics_declares_expected_abstract_methods() -> None:
    assert Metrics.__abstractmethods__ == frozenset({
        "percentage_success_per_task_test",
        "percentage_success_overall_test",
        "percentage_success_per_task_train",
        "percentage_success_overall_train",
        "num_complete_environment_resets",
        "num_human_interventions",
        "summed_human_cost",
        "task_training_curve",
        "task_training_curve_by_subtask",
    })
