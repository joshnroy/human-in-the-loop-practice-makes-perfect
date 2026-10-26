"""Task-policy optimization, independently checked against exhaustive policies."""

import itertools
import math

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import HAND_EMPTY, HOLDING, IN_BIN
from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.practice_makes_perfect.dp_allocator import DpAllocator
from hitl_pmp.methods.practice_makes_perfect.dp_mdp import DpMdp, DpTaskModel, DpTransition


def test_joint_allocation_can_abandon_the_currently_best_task_policy():
    mdp = DpMdp(
        initial_probabilities={0: 1},
        terminal_rewards={2: 1, 3: 4},
        transitions=[
            DpTransition(state=0, skill="easy", successor=2),
            DpTransition(state=0, skill="hard1", successor=1),
            DpTransition(state=1, skill="hard2", successor=3),
        ],
    )
    probabilities = dict.fromkeys(["easy", "hard1", "hard2"], 0.1)
    value, before = mdp.solve(competences=probabilities)
    assert value == pytest.approx(0.1)
    assert before[0].skill == "easy"
    allocation = DpAllocator.allocate(
        mdp=mdp,
        competences=probabilities,
        learning_rates=dict.fromkeys(probabilities, 0.1),
        costs=dict.fromkeys(probabilities, 1),
        budget=20,
    )
    assert allocation == {"easy": 0, "hard1": 9, "hard2": 9}
    assert DpAllocator.predicted_success(
        mdp=mdp,
        competences=probabilities,
        learning_rates=dict.fromkeys(probabilities, 0.1),
        allocation=allocation,
    ) == pytest.approx(4)


def test_unit_probability_cycle_does_not_replace_a_terminating_policy():
    mdp = DpMdp(
        initial_probabilities={0: 1},
        terminal_rewards={2: 1},
        transitions=[
            DpTransition(state=0, skill="loop", successor=1),
            DpTransition(state=1, skill="loop", successor=0),
            DpTransition(state=0, skill="finish", successor=2),
        ],
    )
    value, policy = mdp.solve(competences={"loop": 1, "finish": 1})
    assert value == 1
    assert policy[0].skill == "finish"
    assert mdp.curriculum(competences={"loop": 1, "finish": 1}) == ["finish"]


def test_failure_absorbs_instead_of_retrying_until_success():
    mdp = DpMdp(
        initial_probabilities={0: 1},
        terminal_rewards={1: 1},
        transitions=[DpTransition(state=0, skill="attempt", successor=1)],
    )
    assert mdp.solve(competences={"attempt": 0.3})[0] == pytest.approx(0.3)


def test_discount_is_applied_for_every_skill():
    mdp = DpMdp(
        initial_probabilities={0: 1},
        terminal_rewards={2: 1},
        transitions=[
            DpTransition(state=0, skill="a", successor=1),
            DpTransition(state=1, skill="b", successor=2),
        ],
        discount=0.9,
    )
    assert mdp.solve(competences={"a": 0.5, "b": 0.4})[0] == pytest.approx(0.9**2 * 0.5 * 0.4)


@pytest.mark.parametrize("seed", range(20))
def test_joint_solution_matches_independent_policy_and_allocation_enumeration(*, seed):
    rng = np.random.default_rng(seed)
    skills = ["a", "b", "c"]
    transitions = [
        DpTransition(state=state, skill=skills[int(rng.integers(3))], successor=successor)
        for state in range(3)
        for successor in range(state + 1, 5)
    ]
    rewards = {3: float(rng.uniform(0.5, 3)), 4: float(rng.uniform(0.5, 3))}
    mdp = DpMdp(
        initial_probabilities={0: 0.6, 1: 0.4}, terminal_rewards=rewards, transitions=transitions
    )
    probabilities = {skill: float(rng.uniform(0.1, 0.8)) for skill in skills}
    rates = {skill: float(rng.uniform(0.1, 0.3)) for skill in skills}
    costs = {skill: float(rng.choice([1, 2])) for skill in skills}
    budget = 4
    choices = [[edge for edge in transitions if edge.state == state] for state in range(3)]

    def independent_value(*, allocation):
        best = 0
        for policy in itertools.product(*choices):
            total = 0
            for initial, probability in [(0, 0.6), (1, 0.4)]:
                state = initial
                product = probability
                while state not in rewards:
                    edge = policy[state]
                    product *= min(
                        1, probabilities[edge.skill] + rates[edge.skill] * allocation[edge.skill]
                    )
                    state = edge.successor
                total += product * rewards[state]
            best = max(best, total)
        return best

    best_value, best_spent = -math.inf, math.inf
    for counts in itertools.product(range(budget + 1), repeat=3):
        allocation = dict(zip(skills, counts, strict=True))
        spent = sum(costs[skill] * allocation[skill] for skill in skills)
        if spent > budget:
            continue
        value = independent_value(allocation=allocation)
        if value > best_value + 1e-12 or abs(value - best_value) <= 1e-12 and spent < best_spent:
            best_value, best_spent = value, spent
    allocation = DpAllocator.allocate(
        mdp=mdp,
        competences=probabilities,
        learning_rates=rates,
        costs=costs,
        budget=budget,
    )
    assert independent_value(allocation=allocation) == pytest.approx(best_value)
    assert sum(costs[skill] * allocation[skill] for skill in skills) == best_spent


def _symbolic_fixture():
    start, middle, goal = [
        Predicate(name=name, types=(), holds=lambda state, objects: False)
        for name in ["Start", "Middle", "Goal"]
    ]

    def edge(*, name, before, after):
        return Skill(
            name=name,
            parameters=(),
            preconditions=frozenset({LiftedAtom(predicate=before, variables=())}),
            add_effects=frozenset({LiftedAtom(predicate=after, variables=())}),
            delete_effects=frozenset({LiftedAtom(predicate=before, variables=())}),
            param_dim=0,
        )

    skills = (
        edge(name="Easy", before=start, after=goal),
        edge(name="Hard1", before=start, after=middle),
        edge(name="Hard2", before=middle, after=goal),
    )
    tasks = [
        (
            frozenset({GroundAtom(predicate=start, objects=())}),
            frozenset({GroundAtom(predicate=goal, objects=())}),
        )
    ]
    return skills, tasks


def test_symbolic_graph_includes_alternatives_and_unobserved_downstream_skills():
    skills, tasks = _symbolic_fixture()
    mdp = DpTaskModel.build(tasks=tasks, skills=skills, objects=(), max_states=10)
    assert {edge.skill.skill.name for edge in mdp.transitions} == {"Easy", "Hard1", "Hard2"}
    probabilities = {edge.skill: 0.5 for edge in mdp.transitions}
    assert mdp.solve(competences=probabilities)[0] == pytest.approx(0.5)
    for skill in probabilities:
        probabilities[skill] = 0.1 if skill.skill.name == "Easy" else 0.9
    assert mdp.solve(competences=probabilities)[0] == pytest.approx(0.81)
    assert [skill.skill.name for skill in mdp.curriculum(competences=probabilities)] == [
        "Hard1",
        "Hard2",
    ]


def test_symbolic_graph_cap_raises_instead_of_truncating():
    skills, tasks = _symbolic_fixture()
    with pytest.raises(ValueError, match="states"):
        DpTaskModel.build(tasks=tasks, skills=skills, objects=(), max_states=1)


def test_initially_satisfied_and_unreachable_goals_have_correct_values():
    skills, tasks = _symbolic_fixture()
    initial, goal = tasks[0]
    mdp = DpTaskModel.build(
        tasks=[(goal, goal), (initial, goal)], skills=(), objects=(), max_states=2
    )
    assert mdp.solve(competences={})[0] == pytest.approx(0.5)


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan])
def test_invalid_probabilities_are_rejected(*, value):
    mdp = DpMdp(
        initial_probabilities={0: 1},
        terminal_rewards={1: 1},
        transitions=[DpTransition(state=0, skill="a", successor=1)],
    )
    with pytest.raises(ValueError, match="competence"):
        mdp.solve(competences={"a": value})


@pytest.mark.parametrize("bin_side", Tossing3DSides.objects())
@pytest.mark.parametrize("pouring", [False, True])
def test_tossing3d_empty_gripper_recovery_route_is_preserved(*, bin_side, pouring):
    env = Tossing3DEnvironment(enable_bin_pouring=pouring)
    provider = Tossing3DSkillProvider(env=env)
    objects = {
        "robot": env.robot,
        "cube": env.cube,
        "bin": env.bin,
        "barrier": env.barrier,
        "side": Tossing3DSides.robot,
    }
    grounds = {}
    for skill in provider.skills():
        bindings = dict(objects)
        if skill.name == "MoveToTossLocationAndToss":
            bindings["side"] = bin_side
        grounds[skill.name] = GroundSkill(
            skill=skill, objects=tuple(bindings[parameter.name] for parameter in skill.parameters)
        )
    recovery = {"OpenGripper", "PickCube", "MoveToTossLocationAndToss"}
    # This scene starts with a closed empty hand and the cube outside the bin.
    # Union only this route's preconditions: including pouring's InBin would
    # accidentally make the initial scene satisfy the task's goal already.
    initial = frozenset(
        atom
        for name, ground in grounds.items()
        if name in recovery
        for atom in ground.preconditions
        if atom.predicate not in {HAND_EMPTY, HOLDING}
    )
    goal = frozenset({GroundAtom(predicate=IN_BIN, objects=(env.cube, env.bin))})
    mdp = DpTaskModel.build(
        tasks=[(initial, goal)],
        skills=provider.skills(),
        objects=provider.objects(),
        max_states=20,
    )
    assert {edge.skill.skill.name for edge in mdp.transitions} == recovery
    assert not set(provider.human_cube_bin_reset_skills()) & {
        edge.skill for edge in mdp.transitions
    }
    competences = {ground: 0.5 for ground in grounds.values()}
    assert mdp.solve(competences=competences)[0] == pytest.approx(0.5**3)
    assert [ground.skill.name for ground in mdp.curriculum(competences=competences)] == [
        "OpenGripper",
        "PickCube",
        "MoveToTossLocationAndToss",
    ]


def test_tossing3d_task_mdp_can_recover_an_unpickable_in_bin_cube_by_pouring():
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env, offer_human_reset=False)
    pour = next(skill for skill in provider.skills() if skill.name == "PourCubeOutOfBin")
    grounded = GroundSkill(
        skill=pour,
        objects=(env.robot, env.cube, env.bin, env.barrier, Tossing3DSides.robot),
    )
    goal = frozenset({GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube))})
    mdp = DpTaskModel.build(
        tasks=[(grounded.preconditions, goal)],
        skills=provider.skills(),
        objects=provider.objects(),
        max_states=20,
    )
    assert mdp.solve(competences={edge.skill: 0.5 for edge in mdp.transitions})[0] == pytest.approx(
        0.25
    )
    assert [
        skill.skill.name
        for skill in mdp.curriculum(competences={edge.skill: 0.5 for edge in mdp.transitions})
    ] == ["PourCubeOutOfBin", "PickCube"]
