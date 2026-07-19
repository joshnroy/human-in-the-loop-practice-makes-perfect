from hitl_pmp.core.problem.human.human import HumanOracle


def test_human_oracle_declares_expected_abstract_methods() -> None:
    assert HumanOracle.__abstractmethods__ == frozenset({
        "calculate_cost_for_human_command",
        "execute_human_command",
    })
