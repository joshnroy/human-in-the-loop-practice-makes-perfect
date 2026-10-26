"""Licensed Gurobi tests; enumeration independently checks the joint optimum."""

from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.methods.practice_makes_perfect.dp_allocator import DpAllocator
from hitl_pmp.methods.practice_makes_perfect.dp_gurobi import DpGurobiAllocator, DpOptimizationError
from hitl_pmp.methods.practice_makes_perfect.dp_mdp import DpMdp, DpTransition

gp = pytest.importorskip("gurobipy")


def test_paper_counterexample_uses_the_jointly_improved_route():
    mdp = DpMdp(
        initial_probabilities={0: 1},
        terminal_rewards={2: 1, 3: 4},
        transitions=[
            DpTransition(state=0, skill="easy", successor=2),
            DpTransition(state=0, skill="hard1", successor=1),
            DpTransition(state=1, skill="hard2", successor=3),
        ],
    )
    skills = ["easy", "hard1", "hard2"]
    result = DpGurobiAllocator.allocate(
        mdp=mdp,
        competences=dict.fromkeys(skills, 0.1),
        learning_rates=dict.fromkeys(skills, 0.1),
        costs=dict.fromkeys(skills, 1),
        budget=20,
    )
    assert result.allocation == {"easy": 0, "hard1": 9, "hard2": 9}
    assert result.objective == pytest.approx(4)
    assert result.status == "OPTIMAL" and result.relative_gap < 1e-7
    assert result.objective_bound == pytest.approx(result.objective)


@pytest.mark.parametrize("seed", range(16))
def test_matches_enumeration_on_branching_cyclic_multi_start_mdps(*, seed):
    rng = np.random.default_rng(seed)
    skills = ["a", "b", "c"]
    transitions = [
        DpTransition(state=state, skill=skills[int(rng.integers(3))], successor=successor)
        for state in range(3)
        for successor in range(4)
        if state != successor
    ]
    mdp = DpMdp(
        initial_probabilities={0: 0.7, 1: 0.3},
        terminal_rewards={3: 2},
        transitions=transitions,
        discount=1.0 if seed % 2 else 0.9,
    )
    probabilities = {skill: float(rng.uniform(0.1, 0.8)) for skill in skills}
    rates = {skill: float(rng.uniform(-0.05, 0.4)) for skill in skills}
    costs = {skill: float(rng.choice([0.5, 1, 2])) for skill in skills}
    budget = 3.5
    expected = DpAllocator.allocate(
        mdp=mdp,
        competences=probabilities,
        learning_rates=rates,
        costs=costs,
        budget=budget,
    )
    result = DpGurobiAllocator.allocate(
        mdp=mdp,
        competences=probabilities,
        learning_rates=rates,
        costs=costs,
        budget=budget,
    )
    expected_value = DpAllocator.predicted_success(
        mdp=mdp,
        competences=probabilities,
        learning_rates=rates,
        allocation=expected,
    )
    assert result.objective == pytest.approx(expected_value, abs=1e-7)
    assert sum(costs[skill] * result.allocation[skill] for skill in skills) == pytest.approx(
        sum(costs[skill] * expected[skill] for skill in skills)
    )


def test_unreachable_unit_probability_cycle_cannot_generate_reward():
    mdp = DpMdp(
        initial_probabilities={0: 1},
        terminal_rewards={3: 1},
        transitions=[
            DpTransition(state=1, skill="loop", successor=2),
            DpTransition(state=2, skill="loop", successor=1),
            DpTransition(state=2, skill="finish", successor=3),
        ],
    )
    result = DpGurobiAllocator.allocate(
        mdp=mdp,
        competences={"loop": 1, "finish": 0.5},
        learning_rates={"loop": 0, "finish": 0.1},
        costs={"loop": 1, "finish": 1},
        budget=5,
    )
    assert result.objective == 0
    assert result.allocation == {"loop": 0, "finish": 0}


def test_initial_terminal_reward_and_dead_end_are_handled():
    result = DpGurobiAllocator.allocate(
        mdp=DpMdp(initial_probabilities={0: 0.5, 1: 0.5}, terminal_rewards={0: 2}, transitions=[]),
        competences={},
        learning_rates={},
        costs={},
        budget=1,
    )
    assert result.objective == pytest.approx(1)
    assert result.allocation == {}


def test_time_limit_incumbent_is_not_called_optimal():
    model = SimpleNamespace(
        Status=gp.GRB.TIME_LIMIT, SolCount=1, ObjVal=0.5, ObjBound=1, MIPGap=1, Runtime=0.1
    )
    with pytest.raises(DpOptimizationError, match="no fallback"):
        DpGurobiAllocator._require_optimal(model=model, phase="test")


@pytest.mark.parametrize("task_count", [1, 3, 10])
def test_near_saturation_tie_break_preserves_a_feasible_primary_optimum(*, task_count):
    mdp = DpMdp(
        initial_probabilities={task * 3: 1 / task_count for task in range(task_count)},
        terminal_rewards={task * 3 + 2: 1 for task in range(task_count)},
        transitions=[
            transition
            for task in range(task_count)
            for transition in [
                DpTransition(state=task * 3, skill="pick", successor=task * 3 + 1),
                DpTransition(state=task * 3 + 1, skill="toss", successor=task * 3 + 2),
            ]
        ],
    )
    result = DpGurobiAllocator.allocate(
        mdp=mdp,
        competences={"pick": 0.9995657837603127, "toss": 0.15816326530612246},
        learning_rates={"pick": 9.89065490708352e-06, "toss": 0.004809736394557822},
        costs={"pick": 6.1510810810810055, "toss": 6.519999999999959},
        budget=1719.8000000000034,
    )
    assert result.allocation == {"pick": 44, "toss": 176}
    assert result.objective == pytest.approx(1, abs=1e-9)
    assert result.objective_bound <= 1 + 1e-12
    assert result.absolute_gap <= 1e-7


@pytest.mark.parametrize(
    ("failed_evaluation", "phase"),
    [(1, "primary_independent_verification"), (2, "independent_verification")],
)
def test_optimal_status_does_not_bypass_independent_value_check(
    *, monkeypatch, failed_evaluation, phase
):
    predicted_success = DpAllocator.predicted_success
    evaluations = []

    def independently_evaluate(**kwargs):
        evaluations.append(kwargs)
        actual = predicted_success(**kwargs)
        return actual - 0.1 if len(evaluations) == failed_evaluation else actual

    monkeypatch.setattr(DpAllocator, "predicted_success", staticmethod(independently_evaluate))
    with pytest.raises(DpOptimizationError) as failure:
        DpGurobiAllocator.allocate(
            mdp=DpMdp(
                initial_probabilities={0: 1},
                terminal_rewards={1: 1},
                transitions=[DpTransition(state=0, skill="toss", successor=1)],
            ),
            competences={"toss": 0.1},
            learning_rates={"toss": 0.1},
            costs={"toss": 1},
            budget=1,
        )
    assert failure.value.diagnostics["phase"] == phase
