from hitl_pmp.core.method.method import Method


def test_method_declares_expected_abstract_methods() -> None:
    assert Method.__abstractmethods__ == frozenset({
        "reset_environment",
        "get_task_policy",
        "generate_train_task",
        "execute_setup_command",
        "execute_skill",
        "improve_skill_parameters",
    })
