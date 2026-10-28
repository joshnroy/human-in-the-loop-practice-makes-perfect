"""Tests for DpAllocator, the exact budget allocator of the Deliberate Practice
baseline. Every expected allocation is worked out by hand or by an independent brute
force written here, never read back out of the implementation."""

import itertools
import math

import numpy as np
import pytest

from hitl_pmp.methods.practice_makes_perfect.dp_allocator import DpAllocator
from hitl_pmp.methods.practice_makes_perfect.dp_mdp import DpMdp, DpTransition


def _mdp(*, plans):
    transitions = []
    initial = {}
    rewards = {}
    state = 0
    for plan in plans:
        initial[state] = 1 / len(plans)
        for skill in plan:
            transitions.append(DpTransition(state=state, skill=skill, successor=state + 1))
            state += 1
        rewards[state] = 1
        state += 1
    return DpMdp(initial_probabilities=initial, terminal_rewards=rewards, transitions=transitions)


def _allocate(*, plans, **kwargs):
    return DpAllocator.allocate(mdp=_mdp(plans=plans), **kwargs)


def _predicted(*, plans, **kwargs):
    return DpAllocator.predicted_success(mdp=_mdp(plans=plans), **kwargs)


def _brute_force(*, plans, competences, learning_rates, costs, budget):
    """Every integer allocation that fits, scored directly: (best value, its least
    cost). No pruning, so it shares no shortcut with the allocator."""
    skills = list(costs)
    ranges = [range(int(math.floor(budget / costs[skill])) + 1) for skill in skills]
    best = (-math.inf, math.inf)
    for counts in itertools.product(*ranges):
        spent = sum(count * costs[skill] for skill, count in zip(skills, counts, strict=True))
        if spent > budget + 1e-9:
            continue
        allocation = dict(zip(skills, counts, strict=True))
        value = sum(
            math.prod(
                min(1.0, max(0.0, competences[skill] + learning_rates[skill] * allocation[skill]))
                for skill in plan
            )
            for plan in plans
        ) / len(plans)
        if value > best[0] + 1e-12 or (abs(value - best[0]) <= 1e-12 and spent < best[1]):
            best = (value, spent)
    return best


def test_predicted_success_is_the_mean_over_plans_of_each_plans_product():
    value = _predicted(
        plans=[["a", "b"], ["b"]],
        competences={"a": 0.5, "b": 0.4},
        learning_rates={"a": 0.1, "b": 0.3},
        allocation={"a": 2, "b": 1},
    )
    # a -> 0.5 + 0.2 = 0.7, b -> 0.4 + 0.3 = 0.7; plans 0.49 and 0.7.
    assert value == pytest.approx((0.49 + 0.7) / 2)


def test_predicted_success_caps_each_competence_at_one():
    value = _predicted(
        plans=[["a"]], competences={"a": 0.9}, learning_rates={"a": 0.5}, allocation={"a": 3}
    )
    assert value == pytest.approx(1.0)


def test_a_saturated_skill_gets_nothing():
    allocation = _allocate(
        plans=[["done", "learning"]],
        competences={"done": 1.0, "learning": 0.5},
        learning_rates={"done": 0.1, "learning": 0.1},
        costs={"done": 1.0, "learning": 1.0},
        budget=10.0,
    )
    # "learning" reaches 1.0 after 5 attempts; any more, or any on "done", buys
    # nothing and costs budget.
    assert allocation == {"done": 0, "learning": 5}


def test_a_skill_that_is_not_improving_gets_nothing():
    allocation = _allocate(
        plans=[["flat", "learning"]],
        competences={"flat": 0.3, "learning": 0.5},
        learning_rates={"flat": 0.0, "learning": 0.1},
        costs={"flat": 1.0, "learning": 1.0},
        budget=3.0,
    )
    assert allocation == {"flat": 0, "learning": 3}


def test_a_skill_on_no_plan_gets_nothing():
    allocation = _allocate(
        plans=[["a"]],
        competences={"a": 0.5, "elsewhere": 0.1},
        learning_rates={"a": 0.1, "elsewhere": 0.5},
        costs={"a": 1.0, "elsewhere": 1.0},
        budget=2.0,
    )
    assert allocation == {"a": 2, "elsewhere": 0}


def test_diminishing_returns_split_the_budget():
    # (0.5 + 0.1 x)(0.5 + 0.1 y) with x + y = 4: 2/2 gives 0.49, 4/0 gives 0.45.
    allocation = _allocate(
        plans=[["a", "b"]],
        competences={"a": 0.5, "b": 0.5},
        learning_rates={"a": 0.1, "b": 0.1},
        costs={"a": 1.0, "b": 1.0},
        budget=4.0,
    )
    assert allocation == {"a": 2, "b": 2}


def test_costs_are_per_attempt_and_the_budget_binds_on_their_sum():
    # A toss whose attempt carries a bundled forced reset (1 + 5) against a pick at 1:
    # one toss (0.6 * 0.6 = 0.36) loses to four picks (1.0 * 0.5 = 0.5).
    allocation = _allocate(
        plans=[["pick", "toss"]],
        competences={"pick": 0.6, "toss": 0.5},
        learning_rates={"pick": 0.1, "toss": 0.1},
        costs={"pick": 1.0, "toss": 6.0},
        budget=6.0,
    )
    assert allocation == {"pick": 4, "toss": 0}


def test_a_human_skill_is_allocated_by_the_same_rule_as_a_robot_skill():
    # The allocator has no notion of who performs a skill: relabelling the two keys
    # relabels the answer. Three attempts on the weaker skill (0.9 * 0.8 = 0.72) beat
    # saturating the stronger one and two on the weaker (1.0 * 0.7 = 0.70).
    robot_first = _allocate(
        plans=[["pick", "reset_cube_far"]],
        competences={"pick": 0.9, "reset_cube_far": 0.5},
        learning_rates={"pick": 0.1, "reset_cube_far": 0.1},
        costs={"pick": 1.0, "reset_cube_far": 1.0},
        budget=3.0,
    )
    swapped = _allocate(
        plans=[["reset_cube_far", "pick"]],
        competences={"reset_cube_far": 0.9, "pick": 0.5},
        learning_rates={"reset_cube_far": 0.1, "pick": 0.1},
        costs={"reset_cube_far": 1.0, "pick": 1.0},
        budget=3.0,
    )
    assert robot_first == {"pick": 0, "reset_cube_far": 3}
    assert swapped == {"reset_cube_far": 0, "pick": 3}


def test_no_budget_allocates_nothing():
    allocation = _allocate(
        plans=[["a"]],
        competences={"a": 0.5},
        learning_rates={"a": 0.1},
        costs={"a": 1.0},
        budget=-3.0,
    )
    assert allocation == {"a": 0}


@pytest.mark.parametrize("seed", range(40))
def test_matches_brute_force_on_small_cases(*, seed):
    rng = np.random.default_rng(seed)
    skills = [f"s{index}" for index in range(int(rng.integers(1, 4)))]
    plans = [
        [skills[int(rng.integers(len(skills)))] for _ in range(int(rng.integers(1, 4)))]
        for _ in range(int(rng.integers(1, 4)))
    ]
    competences = {skill: float(rng.uniform(0.0, 1.0)) for skill in skills}
    learning_rates = {skill: float(rng.uniform(-0.05, 0.3)) for skill in skills}
    costs = {skill: float(rng.choice([0.5, 1.0, 1.5, 2.0])) for skill in skills}
    budget = float(rng.uniform(0.0, 6.0))
    allocation = _allocate(
        plans=plans,
        competences=competences,
        learning_rates=learning_rates,
        costs=costs,
        budget=budget,
    )
    spent = sum(allocation[skill] * costs[skill] for skill in skills)
    assert spent <= budget + 1e-9
    value = _predicted(
        plans=plans, competences=competences, learning_rates=learning_rates, allocation=allocation
    )
    best_value, best_spent = _brute_force(
        plans=plans,
        competences=competences,
        learning_rates=learning_rates,
        costs=costs,
        budget=budget,
    )
    assert value == pytest.approx(best_value, abs=1e-12)
    assert spent == pytest.approx(best_spent)


def test_raises_above_the_skill_cap():
    skills = [f"s{index}" for index in range(4)]
    with pytest.raises(ValueError, match="skills"):
        _allocate(
            plans=[skills],
            competences=dict.fromkeys(skills, 0.5),
            learning_rates=dict.fromkeys(skills, 0.1),
            costs=dict.fromkeys(skills, 1.0),
            budget=10.0,
            max_skills=3,
        )


def test_raises_above_the_enumeration_cap():
    with pytest.raises(ValueError, match="enumerat"):
        _allocate(
            plans=[["a", "b"]],
            competences={"a": 0.0, "b": 0.0},
            learning_rates={"a": 0.001, "b": 0.001},
            costs={"a": 1.0, "b": 1.0},
            budget=1000.0,
            max_allocations=1000,
        )


def test_rejects_a_plan_step_with_no_competence():
    with pytest.raises(ValueError, match="competence"):
        _allocate(
            plans=[["a", "unknown"]],
            competences={"a": 0.5},
            learning_rates={"a": 0.1},
            costs={"a": 1.0},
            budget=1.0,
        )


def test_rejects_a_nonpositive_cost():
    with pytest.raises(ValueError, match="cost"):
        _allocate(
            plans=[["a"]],
            competences={"a": 0.5},
            learning_rates={"a": 0.1},
            costs={"a": 0.0},
            budget=1.0,
        )


def test_grouped_counts_improve_each_grounding_from_its_own_current_competence():
    predicted = DpAllocator.predicted_competences(
        competences={"left": 0.2, "right": 0.6, "human": 0.5},
        learning_rates={"left": 0.2, "right": 0.2, "human": 0.1},
        allocation={"toss": 2},
        groups={"left": "toss", "right": "toss"},
    )
    assert predicted == pytest.approx({"left": 0.6, "right": 1.0, "human": 0.5})


def test_shared_group_is_charged_once_and_saturates_its_last_relevant_grounding():
    arguments = dict(
        mdp=_mdp(plans=[["left"], ["right"]]),
        competences={"left": 1.0, "right": 0.2},
        learning_rates={"left": 0.2, "right": 0.2},
        costs={"toss": 2.0},
        budget=20.0,
        groups={"left": "toss", "right": "toss"},
    )
    allocation = DpAllocator.allocate(**arguments, max_skills=1, max_allocations=5)
    assert allocation == {"toss": 4}
    assert DpAllocator.predicted_success(
        **{key: value for key, value in arguments.items() if key not in {"costs", "budget"}},
        allocation=allocation,
    ) == pytest.approx(1.0)


def test_an_unpriced_group_keeps_all_its_members_fixed():
    allocation = DpAllocator.allocate(
        mdp=_mdp(plans=[["left", "right"]]),
        competences={"left": 0.2, "right": 0.8},
        learning_rates={"left": 0.2, "right": 0.2},
        costs={"unused": 1.0},
        budget=4,
        groups={"left": "toss", "right": "toss"},
    )
    assert allocation == {"unused": 0}


def test_duplicate_member_prices_are_rejected():
    with pytest.raises(ValueError, match="price allocation groups"):
        DpAllocator.allocate(
            mdp=_mdp(plans=[["left", "right"]]),
            competences={"left": 0.2, "right": 0.8},
            learning_rates={"left": 0.2, "right": 0.2},
            costs={"left": 1, "right": 1},
            budget=2,
            groups={"right": "left"},
        )


@pytest.mark.parametrize("seed", range(12))
def test_grouped_allocation_matches_independent_exhaustive_counts(*, seed):
    rng = np.random.default_rng(seed)
    plans = [["left", "right"], ["other", "right"]]
    probabilities = {skill: float(rng.uniform(0.1, 0.9)) for skill in ["left", "right", "other"]}
    rates = {skill: float(rng.uniform(-0.2, 0.4)) for skill in probabilities}
    costs = {"toss": 1.0, "other": 0.5}
    groups = {"left": "toss", "right": "toss"}
    budget = 3.0
    expected_value, expected_spent = -math.inf, math.inf
    for toss_count, other_count in itertools.product(range(4), range(7)):
        spent = toss_count + 0.5 * other_count
        if spent > budget:
            continue
        counts = {"left": toss_count, "right": toss_count, "other": other_count}
        value = sum(
            math.prod(
                min(1.0, max(0.0, probabilities[skill] + rates[skill] * counts[skill]))
                for skill in plan
            )
            for plan in plans
        ) / len(plans)
        if value > expected_value + 1e-12 or (
            value >= expected_value - 1e-12 and spent < expected_spent - 1e-12
        ):
            expected_value, expected_spent = value, spent
    allocation = DpAllocator.allocate(
        mdp=_mdp(plans=plans),
        competences=probabilities,
        learning_rates=rates,
        costs=costs,
        budget=budget,
        groups=groups,
    )
    assert DpAllocator.predicted_success(
        mdp=_mdp(plans=plans),
        competences=probabilities,
        learning_rates=rates,
        allocation=allocation,
        groups=groups,
    ) == pytest.approx(expected_value, abs=1e-12)
    assert sum(costs[group] * count for group, count in allocation.items()) == pytest.approx(
        expected_spent
    )
