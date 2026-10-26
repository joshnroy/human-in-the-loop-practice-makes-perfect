"""Licensed Gurobi tests; enumeration independently checks the joint optimum."""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.methods.practice_makes_perfect import dp_gurobi
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
    assert result.objective_bound >= expected_value - 1e-9
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
    [(1, "primary_independent_verification"), (2, "least_cost_tie_break")],
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


@pytest.mark.parametrize("task_count", [1, 3, 10])
@pytest.mark.parametrize("regime", ["primary_bound", "tie_break"])
def test_fixed_toss_ceiling_does_not_reject_an_independently_optimal_allocation(
    *, task_count, regime
):
    mdp = DpMdp(
        initial_probabilities={task * 3: 1 / task_count for task in range(task_count)},
        terminal_rewards={task * 3 + 2: 1 for task in range(task_count)},
        transitions=[
            transition
            for task in range(task_count)
            for transition in [
                DpTransition(state=task * 3, skill="PickCube", successor=task * 3 + 1),
                DpTransition(
                    state=task * 3 + 1,
                    skill="MoveToTossLocationAndToss",
                    successor=task * 3 + 2,
                ),
            ]
        ],
    )
    arguments = dict(
        mdp=mdp,
        competences={
            "PickCube": 0.9997349589186324,
            "MoveToTossLocationAndToss": 0.20593692022263452,
            "OpenGripper": 0.9996907854050711,
        },
        learning_rates={
            "PickCube": 5.320489724444999e-06,
            "MoveToTossLocationAndToss": -0.005847296304885589,
            "OpenGripper": 8.770520400914905e-06,
        },
        costs={
            "PickCube": 6.268644067796877,
            "MoveToTossLocationAndToss": 5.969999999999984,
            "OpenGripper": 0.01,
        },
        budget=1509.3299999999804,
    )
    expected_picks = 50
    if regime == "tie_break":
        arguments.update(
            competences={
                "PickCube": 0.9998092695021935,
                "MoveToTossLocationAndToss": 0.1054421768707483,
                "OpenGripper": 0.999780557384244,
            },
            learning_rates={
                "PickCube": 2.4192520833159016e-06,
                "MoveToTossLocationAndToss": -0.004121492346938779,
                "OpenGripper": 3.4655030917413868e-06,
            },
            costs={
                "PickCube": 6.130689655172712,
                "MoveToTossLocationAndToss": 5.0850000000001065,
                "OpenGripper": 0.01,
            },
            budget=1366.9599999999805,
        )
        expected_picks = 79
    expected = DpAllocator.allocate(**arguments)
    expected_value = DpAllocator.predicted_success(
        mdp=mdp,
        competences=arguments["competences"],
        learning_rates=arguments["learning_rates"],
        allocation=expected,
    )
    result = DpGurobiAllocator.allocate(**arguments)
    assert expected == {
        "PickCube": expected_picks,
        "MoveToTossLocationAndToss": 0,
        "OpenGripper": 0,
    }
    assert result.allocation == expected
    assert result.objective == pytest.approx(expected_value, abs=1e-12)
    assert result.objective_bound <= expected_value + 1e-12
    assert result.absolute_gap <= 1e-7


def test_bound_uses_only_affordable_relevant_training_and_keeps_unpriced_skills_fixed():
    observed = []

    class RecordingMdp(DpMdp[str]):
        def solve(self, *, competences):
            observed.append(dict(competences))
            return super().solve(competences=competences)

    competences = {
        "learning": 0.4,
        "expensive": 0.4,
        "flat": 0.4,
        "declining": 0.3,
        "unpriced": 0.2,
        "irrelevant": 0.1,
    }
    rates = {
        "learning": 0.1,
        "expensive": 0.1,
        "flat": 0,
        "declining": -0.1,
        "unpriced": 0.2,
        "irrelevant": 0.1,
    }
    skills = [skill for skill in competences if skill != "irrelevant"]
    mdp = RecordingMdp(
        initial_probabilities={0: 1},
        terminal_rewards={len(skills): 1},
        transitions=[
            DpTransition(state=index, skill=skill, successor=index + 1)
            for index, skill in enumerate(skills)
        ],
    )
    result = DpGurobiAllocator.allocate(
        mdp=mdp,
        competences=competences,
        learning_rates=rates,
        costs={"learning": 2, "expensive": 4, "flat": 1, "declining": 1, "irrelevant": 1},
        budget=3,
    )
    expected_upper = {**competences, "learning": 0.5}
    assert expected_upper in observed
    assert dict.fromkeys(competences, 1.0) not in observed
    expected_value = 0.5 * 0.4 * 0.4 * 0.3 * 0.2
    assert result.objective == pytest.approx(expected_value, abs=1e-10)
    assert result.objective_bound <= expected_value + 1e-12


def test_bound_relaxes_shared_budget_instead_of_using_a_feasible_allocation():
    observed = []

    class RecordingMdp(DpMdp[str]):
        def solve(self, *, competences):
            observed.append(dict(competences))
            return super().solve(competences=competences)

    result = DpGurobiAllocator.allocate(
        mdp=RecordingMdp(
            initial_probabilities={0: 1},
            terminal_rewards={2: 1},
            transitions=[
                DpTransition(state=0, skill="first", successor=1),
                DpTransition(state=1, skill="second", successor=2),
            ],
        ),
        competences={"first": 0.2, "second": 0.2},
        learning_rates={"first": 0.2, "second": 0.2},
        costs={"first": 1, "second": 1},
        budget=2,
    )
    assert any(
        probabilities["first"] == pytest.approx(0.6)
        and probabilities["second"] == pytest.approx(0.6)
        for probabilities in observed
    )
    assert result.objective == pytest.approx(0.16, abs=1e-7)
    assert result.objective_bound >= result.objective - 1e-12


def test_no_good_cut_excludes_only_the_exact_integer_allocation():
    for first, second in itertools.product(range(3), range(2)):
        with gp.Env(empty=True) as environment:
            environment.setParam("OutputFlag", 0)
            environment.start()
            with gp.Model(env=environment) as model:
                counts = {
                    "first": model.addVar(lb=0, ub=2, vtype=gp.GRB.INTEGER),
                    "second": model.addVar(lb=0, ub=1, vtype=gp.GRB.INTEGER),
                    "fixed": model.addVar(lb=0, ub=0, vtype=gp.GRB.INTEGER),
                }
                model.update()
                DpGurobiAllocator._exclude_allocation(
                    model=model,
                    counts=counts,
                    allocation={"first": 1, "second": 1, "fixed": 0},
                    index=1,
                )
                model.addConstr(counts["first"] == first)
                model.addConstr(counts["second"] == second)
                model.optimize()
                expected = gp.GRB.INFEASIBLE if (first, second) == (1, 1) else gp.GRB.OPTIMAL
                assert model.Status == expected


def test_tie_break_cannot_extend_the_original_solver_deadline(*, monkeypatch):
    times = iter([0, 2])
    monkeypatch.setattr(dp_gurobi, "time", SimpleNamespace(monotonic=lambda: next(times)))
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
            time_limit=1,
        )
    assert failure.value.diagnostics["reason"] == "time_limit_before_certification"


@pytest.mark.parametrize("repeat_excluded", [False, True])
def test_uncertified_tie_allocations_are_not_silently_accepted_or_overcut(
    *, monkeypatch, repeat_excluded
):
    predicted_success = DpAllocator.predicted_success
    evaluations = []
    exclusions = []

    def evaluate(**kwargs):
        evaluations.append(kwargs)
        actual = predicted_success(**kwargs)
        if repeat_excluded:
            return actual if len(evaluations) == 1 else actual - 0.1
        return actual - (9.5e-8 if len(evaluations) == 1 else 1.02e-7)

    def record_exclusion(**kwargs):
        exclusions.append(kwargs)

    monkeypatch.setattr(DpAllocator, "predicted_success", staticmethod(evaluate))
    monkeypatch.setattr(DpGurobiAllocator, "_exclude_allocation", staticmethod(record_exclusion))
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
    assert failure.value.diagnostics["phase"] == "independent_verification"
    assert len(exclusions) == int(repeat_excluded)
    assert failure.value.diagnostics["tie_break_retries"] == int(repeat_excluded)
