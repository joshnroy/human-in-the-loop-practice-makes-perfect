"""--chained-route-b: Route B is two learned skills in a row. Route B's success leaves the
robot mid-route, Route B2 finishes, and a failure at either step ends the attempt in a
state of its own that needs the human's reset."""

from types import SimpleNamespace

import pytest

from hitl_pmp.environments.tworoutes.environment import (
    HUMAN_RESET_B,
    ROUTE_A,
    ROUTE_B,
    ROUTE_B2,
    TwoRoutesEnvironment,
)
from hitl_pmp.environments.tworoutes.skill_provider import TwoRoutesSkillProvider
from hitl_pmp.methods.belief_space.rewarded_routes_domain import RewardedRoutesPracticeDomain
from hitl_pmp.methods.belief_space.tossing3d_observation_model import make_default_tossing3d_belief
from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
    FrozenSkillEffects,
    binary_outcomes,
)


def _env(*, b: float, b2: float) -> TwoRoutesEnvironment:
    env = TwoRoutesEnvironment(
        both_resets=True,
        robot_reset=False,
        chained_route_b=True,
        route_b_start=b,
        route_b_rate=0.0,
        route_b2_start=b2,
        route_b2_rate=0.0,
        competence_cap=1.0,
    )
    env.hard_reset()
    return env


def _on(*, env: TwoRoutesEnvironment) -> set[str]:
    names = TwoRoutesEnvironment.robot_type.feature_names
    return {n for n in names if env.flag(name=n)}


def test_route_b_success_waits_mid_route_and_b2_finishes():
    env = _env(b=1.0, b2=1.0)
    env.take_action(action=env.action_for(skill=ROUTE_B))
    assert _on(env=env) == {"after", "after_b", "mid_b"}
    env.take_action(action=env.action_for(skill=ROUTE_B2))
    assert _on(env=env) == {"after", "after_b2", "via_b", "done"}
    assert env.attempts == {ROUTE_A: 0, ROUTE_B: 1, ROUTE_B2: 1}


@pytest.mark.parametrize(
    ("b", "b2", "ends"), [(0.0, 1.0, {"after", "after_b"}), (1.0, 0.0, {"after", "after_b2"})]
)
def test_a_failure_at_either_step_ends_in_its_own_state(*, b, b2, ends):
    env = _env(b=b, b2=b2)
    env.take_action(action=env.action_for(skill=ROUTE_B))
    if b:
        env.take_action(action=env.action_for(skill=ROUTE_B2))
    assert _on(env=env) == ends
    assert env.reset_movables(destination=HUMAN_RESET_B)
    assert _on(env=env) == {"ready"}


def test_route_b2_is_refused_from_ready():
    env = _env(b=1.0, b2=1.0)
    env.take_action(action=env.action_for(skill=ROUTE_B2))
    assert env.last_error is not None and _on(env=env) == {"ready"}


def test_the_model_forecasts_each_end_state():
    provider = TwoRoutesSkillProvider(
        env=TwoRoutesEnvironment(both_resets=True, chained_route_b=True),
        both_resets=True,
        robot_reset=False,
        chained_route_b=True,
    )
    domain = RewardedRoutesPracticeDomain.from_provider(provider=provider)
    belief = make_default_tossing3d_belief(domain=domain, num_particles=8)
    expected = {
        ROUTE_B: ({"MidB", "After", "AfterB"}, {"After", "AfterB"}),
        ROUTE_B2: ({"Done", "ViaB", "After", "AfterB2"}, {"After", "AfterB2"}),
    }
    for name, (success, failure) in expected.items():
        skill = provider.ground(name=name)
        start = skill.preconditions | (
            frozenset(
                a
                for a in provider.ground(name=ROUTE_B).add_effects
                if a.predicate.name in ("After", "AfterB")
            )
            if name == ROUTE_B2
            else frozenset()
        )
        effects = FrozenSkillEffects(
            table={skill: (skill.add_effects, skill.delete_effects, frozenset())}
        )
        branches = binary_outcomes(
            state=belief,
            true_atoms=start,
            ground_skill=skill,
            probability=0.5,
            cost=0.0,
            effects=effects,
            domain=domain,
        )
        got = [{a.predicate.name for a in atoms} for _, _, atoms in branches]
        assert got == [success, failure], (name, got)


def test_ours_values_route_b_as_both_its_skills():
    chained = TwoRoutesSkillProvider(
        env=TwoRoutesEnvironment(both_resets=True, chained_route_b=True),
        both_resets=True,
        robot_reset=False,
        chained_route_b=True,
    )
    domain = RewardedRoutesPracticeDomain.from_provider(provider=chained)
    assert domain.route_prerequisites == ((ROUTE_B2, (ROUTE_B,)),)
    assert ROUTE_B in dict(domain.belief_sources) and ROUTE_B not in domain.deterministic_skills

    theta = {
        ROUTE_A: SimpleNamespace(competence=0.5),
        ROUTE_B: SimpleNamespace(competence=0.5),
        ROUTE_B2: SimpleNamespace(competence=0.4),
    }
    rewards = dict(domain.route_rewards)
    assert domain.evaluate_theta(theta=theta) == pytest.approx(
        max(rewards[ROUTE_A] * 0.5, rewards[ROUTE_B2] * 0.5 * 0.4)
    )


@pytest.mark.parametrize("chained", [False, True])
def test_the_chain_walk_starts_where_a_robot_reset_leaves_off(*, chained: bool):
    # With no human reset offered, the robot's own reset is what restores the start; a
    # walk that knew only the human's reset would loop between a route and its reset.
    provider = TwoRoutesSkillProvider(
        env=TwoRoutesEnvironment(both_resets=chained, chained_route_b=chained),
        both_resets=chained,
        offer_human_reset=False,
        chained_route_b=chained,
    )
    routes = (ROUTE_A, ROUTE_B2) if chained else (ROUTE_A, ROUTE_B)
    expected = ((ROUTE_B2, (ROUTE_B,)),) if chained else ()
    assert RewardedRoutesPracticeDomain.chains(provider=provider, routes=routes) == expected


def test_a_one_skill_route_has_no_prerequisites():
    plain = TwoRoutesSkillProvider(env=TwoRoutesEnvironment(both_resets=True), both_resets=True)
    assert RewardedRoutesPracticeDomain.from_provider(provider=plain).route_prerequisites == ()


def test_a_setup_run_does_not_train_the_skill_it_runs():
    env = _env(b=1.0, b2=1.0)
    env.execution_trains = False
    env.take_action(action=env.action_for(skill=ROUTE_B))
    assert env.attempts[ROUTE_B] == 0 and env.execution_trains
    env.take_action(action=env.action_for(skill=ROUTE_B2))
    assert env.attempts == {ROUTE_A: 0, ROUTE_B: 0, ROUTE_B2: 1}


def test_on_a_chain_an_outcome_is_weighed_at_the_competence_already_bought():
    from hitl_pmp.methods.belief_space.tossing3d_observation_model import skill_belief_model
    from hitl_pmp.methods.belief_space.types.weighted_hypothesis_belief import (
        WeightedHypothesisBelief,
    )

    chained = TwoRoutesSkillProvider(
        env=TwoRoutesEnvironment(both_resets=True, chained_route_b=True),
        both_resets=True,
        robot_reset=False,
        chained_route_b=True,
    )
    domain = RewardedRoutesPracticeDomain.from_provider(provider=chained)
    assert domain.competence_advances_every_try
    # Two hypotheses, equal at the start: one learns (0.5 a try), one never does.
    belief = WeightedHypothesisBelief(
        competences=(0.0, 0.0),
        learning_rates=(0.5, 0.0),
        costs=(0.0, 0.0),
        probabilities=(0.5, 0.5),
    )
    state = make_default_tossing3d_belief(domain=domain, num_particles=8)
    state = state.model_copy(
        update={
            "skill_beliefs": {**state.skill_beliefs, ROUTE_B: belief},
            "pending_examples": {ROUTE_B: 1},
        }
    )
    after = skill_belief_model(
        ground_skill=chained.ground(name=ROUTE_B), domain=domain
    ).observe_outcome(state=state, success=True, was_random_exploration=False, resample=False)
    # A success after one try is only possible for the learner (0.5 vs 0.0).
    assert after.skill_beliefs[ROUTE_B].probabilities[0] == pytest.approx(1.0)
    assert after.pending_examples[ROUTE_B] == 1
