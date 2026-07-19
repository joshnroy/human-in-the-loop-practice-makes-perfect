from hitl_pmp.core.problem.tasks.tasks import Tasks


def test_tasks_declares_expected_abstract_methods() -> None:
    assert Tasks.__abstractmethods__ == frozenset({"sample_train_task", "sample_test_task"})
