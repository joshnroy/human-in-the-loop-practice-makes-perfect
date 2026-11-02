"""Licensed Gurobi tests; enumeration independently checks the joint optimum."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.methods.practice_makes_perfect.dp_allocator import DpAllocator
from hitl_pmp.methods.practice_makes_perfect.dp_gurobi import DpGurobiAllocator, DpOptimizationError
from hitl_pmp.methods.practice_makes_perfect.dp_mdp import DpMdp, DpTransition

gp = pytest.importorskip("gurobipy")


@pytest.mark.parametrize("seed", [2, 3])
def test_shared_cost_period_near_saturation_matches_recorded_problem_enumeration(
    *, seed, monkeypatch
):
    """Reproduce reported diagnostics, without claiming an archived-input replay."""
    path = Path(__file__).parent / "fixtures" / "tossing3d_shared_cost_period.json"
    case = next(case for case in json.loads(path.read_text()) if case["seed"] == seed)
    failure = case["original_failure"]
    assert failure["phase"] == "primary_independent_verification"
    assert failure["objective_bound"] - failure["objective"] > 1e-7
    assert failure["solver_objective"] - failure["objective"] > 1e-7
    arguments = {**case["arguments"], "mdp": DpMdp[str].model_validate(case["arguments"]["mdp"])}
    expected = DpAllocator.allocate(**arguments)
    assert expected == case["expected_allocation"]
    expected_value = DpAllocator.predicted_success(
        **{key: value for key, value in arguments.items() if key not in {"costs", "budget"}},
        allocation=expected,
    )
    require_optimal = DpGurobiAllocator._require_optimal
    phases = []

    def check_stage(*, model, phase):
        require_optimal(model=model, phase=phase)
        phases.append(phase)
        allocation = {
            skill: int(round(model.getVarByName(f"b_{index}").X))
            for index, skill in enumerate(arguments["costs"])
        }
        actual = DpAllocator.predicted_success(
            **{key: value for key, value in arguments.items() if key not in {"costs", "budget"}},
            allocation=allocation,
        )
        assert actual == pytest.approx(expected_value, abs=1e-8, rel=0)
        assert (
            sum(arguments["costs"][skill] * count for skill, count in allocation.items())
            <= arguments["budget"]
        )
        assert phase == "task_value"
        assert model.ModelSense == gp.GRB.MAXIMIZE
        assert model.NumObj == 1
        assert model.ObjVal == pytest.approx(expected_value, abs=1e-7, rel=0)
        assert model.ObjBound == pytest.approx(expected_value, abs=1e-7, rel=0)

    monkeypatch.setattr(DpGurobiAllocator, "_require_optimal", staticmethod(check_stage))
    result = DpGurobiAllocator.allocate(**arguments)
    assert phases == ["task_value"]
    assert result.status == "OPTIMAL"
    assert result.objective == pytest.approx(expected_value, abs=1e-12, rel=0)
    assert result.objective_bound == pytest.approx(expected_value, abs=1e-9, rel=0)
    assert result.absolute_gap <= 1e-7
    assert (
        sum(arguments["costs"][skill] * count for skill, count in result.allocation.items())
        <= arguments["budget"]
    )


@pytest.mark.parametrize("declining_rate", [-0.05, -0.2, -0.25])
@pytest.mark.parametrize("budget", [0, 0.45, 0.6, 1.2])
def test_shared_declining_and_saturated_siblings_match_enumeration(*, declining_rate, budget):
    arguments = dict(
        mdp=DpMdp(
            initial_probabilities={0: 0.7, 1: 0.2, 2: 0.1},
            terminal_rewards={3: 1},
            transitions=[
                DpTransition(state=0, skill="improving", successor=3),
                DpTransition(state=1, skill="declining", successor=3),
                DpTransition(state=2, skill="saturated", successor=3),
            ],
        ),
        competences={"improving": 0.2, "declining": 0.8, "saturated": 1.0},
        learning_rates={"improving": 0.2, "declining": declining_rate, "saturated": 0.1},
        groups=dict.fromkeys(["improving", "declining", "saturated"], "shared"),
        costs={"shared": 0.15},
        budget=budget,
    )
    expected = DpAllocator.allocate(**arguments)
    expected_value = DpAllocator.predicted_success(
        **{key: value for key, value in arguments.items() if key not in {"costs", "budget"}},
        allocation=expected,
    )
    result = DpGurobiAllocator.allocate(**arguments)
    assert result.status == "OPTIMAL"
    assert result.objective == pytest.approx(expected_value, abs=1e-9, rel=0)
    assert result.objective_bound == pytest.approx(expected_value, abs=1e-9, rel=0)
    assert result.absolute_gap <= 1e-7
    assert result.allocation["shared"] * arguments["costs"]["shared"] <= budget


@pytest.mark.parametrize("seed", range(8))
def test_grouped_forecasts_match_enumeration_on_cyclic_multi_start_mdps(*, seed):
    rng = np.random.default_rng(seed)
    skills = ["left", "right", "other"]
    mdp = DpMdp(
        initial_probabilities={0: 0.7, 1: 0.3},
        terminal_rewards={3: 1},
        transitions=[
            DpTransition(state=state, skill=skills[int(rng.integers(3))], successor=successor)
            for state in range(3)
            for successor in range(4)
            if state != successor
        ],
        discount=1.0 if seed % 2 else 0.9,
    )
    arguments = dict(
        mdp=mdp,
        competences={skill: float(rng.uniform(0.1, 0.8)) for skill in skills},
        learning_rates={skill: float(rng.uniform(-0.2, 0.4)) for skill in skills},
        costs={"toss": 1.0, "other": 0.5},
        budget=3.0,
        groups={"left": "toss", "right": "toss"},
    )
    expected = DpAllocator.allocate(**arguments)
    result = DpGurobiAllocator.allocate(**arguments)
    expected_value = DpAllocator.predicted_success(
        **{key: value for key, value in arguments.items() if key not in {"costs", "budget"}},
        allocation=expected,
    )
    assert result.status == "OPTIMAL"
    assert set(result.allocation) == {"toss", "other"}
    assert result.objective == pytest.approx(expected_value, abs=1e-7, rel=0)
    assert result.objective_bound >= expected_value - 1e-9
    assert result.absolute_gap <= 1e-7
    costs = arguments["costs"]
    assert sum(costs[group] * count for group, count in result.allocation.items()) <= 3.0


def test_group_cap_and_bound_include_a_less_competent_sibling():
    result = DpGurobiAllocator.allocate(
        mdp=DpMdp(
            initial_probabilities={0: 0.5, 1: 0.5},
            terminal_rewards={2: 1},
            transitions=[
                DpTransition(state=0, skill="left", successor=2),
                DpTransition(state=1, skill="right", successor=2),
            ],
        ),
        competences={"left": 1, "right": 0.2},
        learning_rates={"left": 0.2, "right": 0.2},
        costs={"toss": 2},
        budget=20,
        groups={"left": "toss", "right": "toss"},
    )
    assert result.allocation["toss"] * 2 <= 20
    assert result.objective == pytest.approx(1, abs=1e-9, rel=0)
    assert result.objective_bound == pytest.approx(1, abs=1e-9, rel=0)


def test_grouped_allocations_keep_independent_certification(*, monkeypatch):
    original = DpAllocator.predicted_success
    evaluations = []

    def independently_evaluate(**kwargs):
        evaluations.append(kwargs)
        assert kwargs["groups"] == {"left": "toss", "right": "toss"}
        actual = original(**kwargs)
        return actual - 0.1

    monkeypatch.setattr(DpAllocator, "predicted_success", staticmethod(independently_evaluate))
    with pytest.raises(DpOptimizationError) as failure:
        DpGurobiAllocator.allocate(
            mdp=DpMdp(
                initial_probabilities={0: 1},
                terminal_rewards={2: 1},
                transitions=[
                    DpTransition(state=0, skill="left", successor=1),
                    DpTransition(state=1, skill="right", successor=2),
                ],
            ),
            competences={"left": 0.2, "right": 0.4},
            learning_rates={"left": 0.2, "right": 0.2},
            costs={"toss": 1},
            budget=1,
            groups={"left": "toss", "right": "toss"},
        )
    assert failure.value.diagnostics["phase"] == "primary_independent_verification"
    assert len(evaluations) == 1


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
    assert sum(result.allocation.values()) <= 20
    assert result.objective == pytest.approx(4, abs=1e-7, rel=0)
    assert result.status == "OPTIMAL" and result.relative_gap < 1e-7
    assert result.objective_bound == pytest.approx(result.objective, abs=1e-7, rel=0)


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
    assert result.objective == pytest.approx(expected_value, abs=1e-7, rel=0)
    assert result.objective_bound >= expected_value - 1e-9
    assert sum(costs[skill] * result.allocation[skill] for skill in skills) <= budget


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
    assert sum(result.allocation.values()) <= 5


def test_initial_terminal_reward_and_dead_end_are_handled():
    result = DpGurobiAllocator.allocate(
        mdp=DpMdp(initial_probabilities={0: 0.5, 1: 0.5}, terminal_rewards={0: 2}, transitions=[]),
        competences={},
        learning_rates={},
        costs={},
        budget=1,
    )
    assert result.objective == pytest.approx(1, abs=1e-9, rel=0)
    assert result.allocation == {}


def test_time_limit_incumbent_is_not_called_optimal():
    model = SimpleNamespace(
        Status=gp.GRB.TIME_LIMIT, SolCount=1, ObjVal=0.5, ObjBound=1, MIPGap=1, Runtime=0.1
    )
    with pytest.raises(DpOptimizationError, match="no fallback"):
        DpGurobiAllocator._require_optimal(model=model, phase="test")


@pytest.mark.parametrize("task_count", [1, 3, 10])
def test_near_saturation_certifies_a_feasible_primary_optimum(*, task_count):
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
    assert (
        result.allocation["pick"] * 6.1510810810810055
        + result.allocation["toss"] * 6.519999999999959
        <= 1719.8000000000034 + 1e-7
    )
    assert result.objective == pytest.approx(1, abs=1e-9, rel=0)
    assert result.objective_bound <= 1 + 1e-12
    assert result.absolute_gap <= 1e-7


@pytest.mark.parametrize("shortfall", [1.02e-7, 0.1])
def test_optimal_status_does_not_bypass_independent_value_check(*, monkeypatch, shortfall):
    predicted_success = DpAllocator.predicted_success
    evaluations = []

    def independently_evaluate(**kwargs):
        evaluations.append(kwargs)
        actual = predicted_success(**kwargs)
        return actual - shortfall

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
    assert failure.value.diagnostics["phase"] == "primary_independent_verification"
    assert len(evaluations) == 1


@pytest.mark.parametrize("task_count", [1, 3, 10])
@pytest.mark.parametrize("regime", ["primary_bound", "small_learning_rate"])
@pytest.mark.parametrize("toss_rate", ["negative", "constant"])
def test_fixed_toss_ceiling_does_not_reject_an_independently_optimal_allocation(
    *, task_count, regime, toss_rate, monkeypatch
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
    if regime == "small_learning_rate":
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
    if toss_rate == "constant":
        arguments["learning_rates"]["MoveToTossLocationAndToss"] = 0.0
    expected = DpAllocator.allocate(**arguments)
    expected_value = DpAllocator.predicted_success(
        mdp=mdp,
        competences=arguments["competences"],
        learning_rates=arguments["learning_rates"],
        allocation=expected,
    )
    require_optimal = DpGurobiAllocator._require_optimal

    def check_solver_certificate(*, model, phase):
        require_optimal(model=model, phase=phase)
        assert model.ObjVal == pytest.approx(expected_value, abs=1e-9, rel=0)
        assert model.ObjBound == pytest.approx(expected_value, abs=1e-9, rel=0)

    monkeypatch.setattr(
        DpGurobiAllocator, "_require_optimal", staticmethod(check_solver_certificate)
    )
    result = DpGurobiAllocator.allocate(**arguments)
    assert expected == {
        "PickCube": expected_picks,
        "MoveToTossLocationAndToss": 0,
        "OpenGripper": 0,
    }
    assert (
        sum(arguments["costs"][skill] * count for skill, count in result.allocation.items())
        <= arguments["budget"] + 1e-7
    )
    assert result.objective == pytest.approx(expected_value, abs=1e-12, rel=0)
    assert result.objective_bound == pytest.approx(expected_value, abs=1e-12, rel=0)
    assert result.absolute_gap <= 1e-7


def test_bound_uses_only_affordable_training_and_keeps_unpriced_skills_fixed():
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
    expected_upper = {**competences, "learning": 0.5, "irrelevant": 0.4}
    assert expected_upper in observed
    assert dict.fromkeys(competences, 1.0) not in observed
    expected_value = 0.5 * 0.4 * 0.4 * 0.3 * 0.2
    assert result.objective == pytest.approx(expected_value, abs=1e-10, rel=0)
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
        probabilities["first"] == pytest.approx(0.6, abs=1e-12, rel=0)
        and probabilities["second"] == pytest.approx(0.6, abs=1e-12, rel=0)
        for probabilities in observed
    )
    assert result.objective == pytest.approx(0.16, abs=1e-7, rel=0)
    assert result.objective_bound >= result.objective - 1e-12


@pytest.mark.parametrize("budget", [-1, 0, 1000])
def test_single_task_objective_and_solve_use_affordable_not_saturation_bounds(
    *, monkeypatch, budget
):
    objectives = []
    solves = []
    competences = {"learning": 0.999, "saturated": 1.0, "flat": 0.4, "declining": 0.8}
    costs = dict.fromkeys(competences, 2)

    class RecordingModel(gp.Model):
        def setObjective(self, *args, **kwargs):
            objectives.append(args[1])
            return super().setObjective(*args, **kwargs)

        def optimize(self):
            self.update()
            solves.append(self.NumObj)
            assert self.Params.OptimalityTol == 1e-9
            assert self.ModelSense == gp.GRB.MAXIMIZE
            assert self.getConstrByName("practice_budget").Sense == gp.GRB.LESS_EQUAL
            assert max(0, budget) == self.getConstrByName("practice_budget").RHS
            for index in range(len(costs)):
                count = self.getVarByName(f"b_{index}")
                assert count.VType == gp.GRB.INTEGER
                assert count.LB == 0
                assert max(0, budget // 2) == count.UB
                assert count.Obj == 0
                if budget > 0:
                    assert count.UB > 140
            return super().optimize()

    def reject_enumeration_bounds(**kwargs):
        pytest.fail("Gurobi must not use the enumeration oracle's pruned count bounds")

    monkeypatch.setattr(gp, "Model", RecordingModel)
    monkeypatch.setattr(DpAllocator, "_allocation_limits", staticmethod(reject_enumeration_bounds))
    result = DpGurobiAllocator.allocate(
        mdp=DpMdp(
            initial_probabilities={0: 1},
            terminal_rewards={1: 1},
            transitions=[DpTransition(state=0, skill="learning", successor=1)],
        ),
        competences=competences,
        learning_rates={"learning": 0.001, "saturated": 0.1, "flat": 0, "declining": -0.1},
        costs=costs,
        budget=budget,
    )
    assert objectives == [gp.GRB.MAXIMIZE]
    assert solves == [1]
    assert result.status == "OPTIMAL"
    assert result.tie_break_retries == 0
    assert result.objective == pytest.approx(1 if budget > 0 else 0.999, abs=1e-9, rel=0)
    assert sum(costs[skill] * count for skill, count in result.allocation.items()) <= max(0, budget)
