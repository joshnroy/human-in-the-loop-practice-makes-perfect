"""Executed deployment policies obey the objectives used to value practice."""

from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.core.method.types import LabeledAction, LiftedAtom, Skill
from hitl_pmp.core.problem.environment.types import Object, State  # noqa: F401 -- pydantic resolves State
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Predicate, Task
from hitl_pmp.methods.belief_space.tossing3d_deployment_model import evaluate_deployment_policy
from hitl_pmp.methods.deployment import DeploymentPolicy
from hitl_pmp.methods.practice_makes_perfect.competence_models import OptimisticSkillCompetenceModel


class SymbolicState(State):
    atoms: frozenset[GroundAtom]


def _fixture():
    predicates = {
        name: Predicate(name=actual, types=(), holds=lambda state, _: False)
        for name, actual in {
            "ready": "HandEmpty",
            "holding": "Holding",
            "closed": "ClosedEmpty",
            "goal": "InBin",
        }.items()
    }
    atoms = {name: GroundAtom(predicate=p, objects=()) for name, p in predicates.items()}

    def skill(*, name, before, after):
        return Skill(
            name=name,
            parameters=(),
            preconditions=frozenset({LiftedAtom(predicate=predicates[before], variables=())}),
            add_effects=frozenset({LiftedAtom(predicate=predicates[after], variables=())}),
            delete_effects=frozenset({LiftedAtom(predicate=predicates[before], variables=())}),
            param_dim=0,
        )

    skills = (
        skill(name="PickCube", before="ready", after="holding"),
        skill(name="OpenGripper", before="closed", after="ready"),
        skill(name="MoveToTossLocationAndToss", before="holding", after="goal"),
    )
    dispatched = []

    def execute(**kwargs):
        assert kwargs["explore"] is False
        name = kwargs["ground_skill"].skill.name
        dispatched.append(name)
        return LabeledAction(action=np.array([1.0]), label=name), None

    method = SimpleNamespace(
        env=SimpleNamespace(noop_action=lambda: np.array([-1.0])),
        abstract_state=lambda **kwargs: kwargs["state"].atoms,
        skills=lambda: skills,
        objects=tuple,
        execute_ground_skill=execute,
        competence_model=lambda **_: SimpleNamespace(get_current_competence=lambda: 0.5),
        _competence_models={},
        prior_alpha=1.0,
        prior_beta=1.0,
        # A domain that prices no goal: every goal state is worth 1.
        skill_provider=SimpleNamespace(goal_rewards=dict),
    )
    states = {name: SymbolicState(data={}, atoms=frozenset({atom})) for name, atom in atoms.items()}
    task = Task(initial_state=states["ready"], goal=Goal(atoms=frozenset({atoms["goal"]})))
    return method, states, task, dispatched


def test_tossing_policy_recovers_failed_pick_and_stops_after_one_toss():
    method, states, task, dispatched = _fixture()
    policy = DeploymentPolicy(kind="tossing_canonical", source_method="curtis", horizon=6).policy(
        method=method, task=task
    )
    for name in ("ready", "closed", "closed", "ready", "holding"):
        policy(states[name])
    assert dispatched == [
        "PickCube",
        "OpenGripper",
        "OpenGripper",
        "PickCube",
        "MoveToTossLocationAndToss",
    ]
    assert np.array_equal(policy(states["ready"]).action, method.env.noop_action())


@pytest.mark.parametrize("horizon", range(1, 7))
def test_executed_tossing_policy_has_the_value_used_by_practice(*, horizon):
    method, states, task, _ = _fixture()
    probabilities = {"PickCube": 0.4, "OpenGripper": 0.7, "MoveToTossLocationAndToss": 0.6}
    outcomes = {
        "PickCube": ("holding", "closed"),
        "OpenGripper": ("ready", "closed"),
        "MoveToTossLocationAndToss": ("goal", "ready"),
    }

    def value(*, history, state):
        if state == "goal":
            return 1.0
        policy = DeploymentPolicy(
            kind="tossing_canonical", source_method="expectimax", horizon=horizon
        ).policy(method=method, task=task)
        for previous in history:
            policy(states[previous])
        action = policy(states[state])
        if action.label not in outcomes:
            return 0.0
        p = probabilities[action.label]
        success, failure = outcomes[action.label]
        return p * value(history=[*history, state], state=success) + (1 - p) * value(
            history=[*history, state], state=failure
        )

    assert value(history=[], state="ready") == pytest.approx(
        evaluate_deployment_policy(
            pick_competence=0.4, open_competence=0.7, toss_competence=0.6, horizon=horizon
        )
    )


def test_dp_executes_its_task_mdp_and_failure_absorbs():
    method, states, task, dispatched = _fixture()
    policy = DeploymentPolicy(kind="dp_mdp", source_method="dp").policy(method=method, task=task)
    policy(states["ready"])
    policy(states["holding"])
    assert dispatched == ["PickCube", "MoveToTossLocationAndToss"]
    assert np.array_equal(policy(states["ready"]).action, method.env.noop_action())
    dispatched.clear()
    failed = DeploymentPolicy(kind="dp_mdp", source_method="dp").policy(method=method, task=task)
    failed(states["ready"])
    assert np.array_equal(failed(states["closed"]).action, method.env.noop_action())
    assert dispatched == ["PickCube"]


def test_canonical_holding_never_falls_back_to_opening_the_gripper():
    method, states, task, dispatched = _fixture()
    pick, opened, toss = method.skills()
    # Toss needs a missing condition; opening is nevertheless applicable while holding.
    opened = opened.model_copy(update={"preconditions": toss.preconditions})
    toss = toss.model_copy(update={"preconditions": toss.preconditions | pick.preconditions})
    method.skills = lambda: (pick, opened, toss)
    policy = DeploymentPolicy(kind="tossing_canonical").policy(method=method, task=task)
    assert np.array_equal(policy(states["holding"]).action, method.env.noop_action())
    assert dispatched == []


def test_dp_uses_cumulative_history_when_recent_window_prefers_another_route():
    from hitl_pmp.core.method.types import GroundSkill

    method, states, task, dispatched = _fixture()
    pick, _, toss = method.skills()
    direct = pick.model_copy(update={"name": "Direct", "add_effects": toss.add_effects})
    method.skills = lambda: (pick, toss, direct)
    # Old successes outweigh the last five failed periods, while EES's window drops them.
    method._competence_models = {  # noqa: SLF001
        GroundSkill(skill=pick, objects=()): OptimisticSkillCompetenceModel(
            alpha=1, beta=1, cycle_observations=[[True] * 100] + [[False] * 2 for _ in range(5)]
        ),
        GroundSkill(skill=toss, objects=()): OptimisticSkillCompetenceModel(
            alpha=1, beta=1, cycle_observations=[[True] * 100]
        ),
    }
    policy = DeploymentPolicy(kind="dp_mdp").policy(method=method, task=task)
    policy(states["ready"])
    # The direct route has p=.5; cumulative pick*toss>.8, recent pick*toss<.1.
    assert dispatched == ["PickCube"]


def test_dp_deploys_on_recent_rounds_when_the_method_estimates_competence_that_way():
    """A DP run values and practises on its estimator, so it deploys on it too: the same
    history as above, read over the last five rounds, puts pick*toss under .1."""
    from hitl_pmp.core.method.types import GroundSkill

    method, states, task, dispatched = _fixture()
    pick, _, toss = method.skills()
    direct = pick.model_copy(update={"name": "Direct", "add_effects": toss.add_effects})
    method.skills = lambda: (pick, toss, direct)
    method.dp_competence_estimator = "recent_rounds"
    method._competence_models = {  # noqa: SLF001
        GroundSkill(skill=pick, objects=()): OptimisticSkillCompetenceModel(
            alpha=1, beta=1, cycle_observations=[[True] * 100] + [[False] * 2 for _ in range(5)]
        ),
        GroundSkill(skill=toss, objects=()): OptimisticSkillCompetenceModel(
            alpha=1, beta=1, cycle_observations=[[True] * 100]
        ),
    }
    policy = DeploymentPolicy(kind="dp_mdp").policy(method=method, task=task)
    policy(states["ready"])
    assert dispatched == ["Direct"]


class _TwoRoutesDeployment:
    """A restored deployment on Two Routes, with a scripted practice history."""

    @staticmethod
    def build(*, kind: str, successes: dict[str, tuple[int, int]], priors: dict | None = None):
        from hitl_pmp.environments.tworoutes.environment import TwoRoutesEnvironment
        from hitl_pmp.environments.tworoutes.skill_provider import TwoRoutesSkillProvider
        from hitl_pmp.environments.tworoutes.tasks import TwoRoutesTasks
        from hitl_pmp.methods.deployment import SnapshotDeployment

        env = TwoRoutesEnvironment(route_a_start=1.0, route_b_start=1.0, competence_cap=1.0)
        env.hard_reset()
        provider = TwoRoutesSkillProvider(env=env, offer_human_reset=False)
        method = SnapshotDeployment(
            env=env,
            skill_provider=provider,
            deployment_policy=DeploymentPolicy(kind=kind, horizon=1, dp_priors=priors or {}),
        )
        for name, (won, lost) in successes.items():
            ground = provider.ground(name=name)
            for outcome in (*[True] * won, *[False] * lost):
                method.observe_outcome(ground_skill=ground, success=outcome)
        task = TwoRoutesTasks(env=env).sample_test_task()
        return env, method.get_task_policy(task=task), task


@pytest.mark.parametrize(
    ("history", "route"),
    [
        # 3 * 2/14 < 10/14: Route A, though Route B pays three times as much.
        ({"RouteA": (9, 1), "RouteB": (1, 9)}, "RouteA"),
        # 3 * 5/14 > 10/14: Route B, though Route A succeeds twice as often.
        ({"RouteA": (9, 1), "RouteB": (4, 6)}, "RouteB"),
    ],
)
def test_reward_greedy_takes_the_route_of_largest_reward_times_competence(*, history, route):
    from hitl_pmp.methods.practice_makes_perfect.dp_priors import DpSkillPrior

    prior = DpSkillPrior(alpha=1.0, beta=3.0, initial_learning_rate=0.0)
    env, policy, _ = _TwoRoutesDeployment.build(
        kind="reward_greedy", successes=history, priors={"RouteA": prior, "RouteB": prior}
    )
    assert policy(env.get_current_state()).label.startswith(route)


def test_reward_greedy_is_one_attempt():
    env, policy, _ = _TwoRoutesDeployment.build(kind="reward_greedy", successes={})
    first = policy(env.get_current_state())
    assert not np.array_equal(first.action, env.noop_action())
    env.set_state(state=env.build_state(after_a=True))
    assert np.array_equal(policy(env.get_current_state()).action, env.noop_action())


def test_reward_greedy_values_an_unpractised_route_at_its_prior_not_the_ees_default():
    """Untried, Route B is worth 3 * 1/4 under Beta(1, 3), less than Route A's 20/24.
    EES's Beta(10, 1) would call it 0.91 competent and deploy it."""
    from hitl_pmp.methods.practice_makes_perfect.dp_priors import DpSkillPrior

    prior = DpSkillPrior(alpha=1.0, beta=3.0, initial_learning_rate=0.0)
    env, policy, _ = _TwoRoutesDeployment.build(
        kind="reward_greedy",
        successes={"RouteA": (19, 1)},
        priors={"RouteA": prior, "RouteB": prior},
    )
    assert policy(env.get_current_state()).label.startswith("RouteA")
    env, policy, _ = _TwoRoutesDeployment.build(kind="reward_greedy", successes={"RouteA": (19, 1)})
    assert policy(env.get_current_state()).label.startswith("RouteB")


def test_dp_deployment_weighs_the_goal_reward_and_shared_ees_does_not():
    history = {"RouteA": (9, 1), "RouteB": (5, 5)}
    env, policy, _ = _TwoRoutesDeployment.build(kind="dp_mdp", successes=history)
    assert policy(env.get_current_state()).label.startswith("RouteB")
    env, policy, _ = _TwoRoutesDeployment.build(kind="shared_ees", successes=history)
    assert policy(env.get_current_state()).label.startswith("RouteA")
