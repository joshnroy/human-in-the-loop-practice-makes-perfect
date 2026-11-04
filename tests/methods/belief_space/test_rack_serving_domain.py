"""Rack-specific value, belief routing, and CLI composition without native physics."""

from unittest.mock import Mock

import numpy as np
import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.racktoserving3d.cli import RackServingCli
from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.objects import NUM_FOOD
from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
from hitl_pmp.environments.racktoserving3d.predicates import (
    RackServing3DAtoms,
    RackServing3DPredicates,
)
from hitl_pmp.environments.racktoserving3d.problem import EPISODE_HORIZON
from hitl_pmp.environments.racktoserving3d.skill_provider import RackServingSkillProvider
from hitl_pmp.environments.racktoserving3d.skills import RackServing3DSkills as Skills
from hitl_pmp.methods.belief_space.competence_inference import InferenceConfig
from hitl_pmp.methods.belief_space.practice_domain import SWEEP_OFF_SHELF_DOMAIN, TOSSING3D_DOMAIN
from hitl_pmp.methods.belief_space.rack_serving_domain import (
    RACK_HUMAN_SKILLS,
    RACK_NUM_FOOD,
    RACK_ROBOT_SKILLS,
    RACK_SERVING_DOMAIN,
    RackServingDeployment,
)
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    make_default_tossing3d_belief,
    skill_belief_model,
)
from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
    make_tossing3d_search_state,
    transition_outcomes,
)
from hitl_pmp.methods.belief_space.types.practice_example_source import PracticeExampleSource
from hitl_pmp.methods.belief_space.types.skill_belief import SkillHypothesis
from hitl_pmp.methods.practice_makes_perfect.cli import Tossing3DPomdpCli
from hitl_pmp.step_protocol import StepPracticeRunner, StepProtocolEntry
from tests.practice_cost_fixtures import CostFixtures


def _value(*, horizon, **competence):
    return float(
        RackServingDeployment.values(
            competences={name: np.array([competence.get(name, 0.0)]) for name in RACK_ROBOT_SKILLS},
            horizon=horizon,
        )[0]
    )


def _belief():
    return make_default_tossing3d_belief(
        domain=RACK_SERVING_DOMAIN,
        model="local_trend",
        engine="grid",
        num_particles=32,
        inference_config=InferenceConfig(competence_bins=9, learning_rate_bins=4),
    )


def _atoms(*, tool_home=True, food_on_rack=True):
    named = {("HandEmpty", (Objects.robot.name,))}
    named.add(("ScraperHome" if tool_home else "ScraperOnFloor", (Objects.scraper.name,)))
    location = "OnRack" if food_on_rack else "OnTray"
    support = Objects.rack if food_on_rack else Objects.tray
    named |= {(location, (food.name, support.name)) for food in Objects.food}
    named = RackServing3DAtoms.with_summaries(atoms=frozenset(named))
    predicates = {predicate.name: predicate for predicate in RackServing3DPredicates.all()}
    objects = {obj.name: obj for obj in Objects.all_objects()}
    return (
        frozenset(
            GroundAtom(predicate=predicates[name], objects=tuple(objects[item] for item in names))
            for name, names in named
        )
        | RackServing3DPredicates.static_atoms()
    )


def test_domain_matches_rack_skill_names_horizon_and_observation_sources():
    assert set(RACK_ROBOT_SKILLS) == {skill.name for skill in Skills.robot_skills()}
    assert set(RACK_HUMAN_SKILLS) == {skill.name for skill in Skills.human_skills()}
    assert RACK_SERVING_DOMAIN.deployment_horizon == EPISODE_HORIZON == 2 * NUM_FOOD + 2
    assert RACK_SERVING_DOMAIN.sampler_skill == Skills.GATHER_AND_SWEEP.name
    assert RACK_SERVING_DOMAIN.human_skill_names == frozenset(RACK_HUMAN_SKILLS)
    assert RACK_SERVING_DOMAIN.failure_effect_skills == frozenset((
        *RACK_ROBOT_SKILLS,
        *RACK_HUMAN_SKILLS,
    ))
    assert not RACK_SERVING_DOMAIN.deterministic_skills
    assert not RACK_SERVING_DOMAIN.shared_beliefs
    assert not RACK_SERVING_DOMAIN.reduce_expectimax_search
    for name, source in RACK_SERVING_DOMAIN.belief_sources:
        assert RACK_SERVING_DOMAIN.belief_name(skill_name=name) == name
        assert source == (
            PracticeExampleSource.SAMPLER
            if name == "GatherAndSweep"
            else PracticeExampleSource.OUTCOME
        )


def test_a_fresh_episode_can_sweep_without_waiting():
    assert _value(horizon=2, PickScraper=1, GatherAndSweep=1) == 1
    assert _value(horizon=1, PickScraper=1, GatherAndSweep=1) == 0


def test_pick_place_is_an_independent_deployment_route_with_no_scraper_or_human():
    assert _value(horizon=10, PickFood=1, PlaceFoodOnTray=1) == 1
    assert _value(horizon=9, PickFood=1, PlaceFoodOnTray=1) == 0


def test_no_skill_or_return_to_rack_alone_can_deliver_food():
    assert _value(horizon=0) == 0
    assert _value(horizon=12, PickFood=1, ReturnFoodToRack=1) == 0
    assert _value(horizon=12, PickScraper=1, ReturnScraper=1) == 0


def test_retries_and_alternative_routes_are_valued_without_forcing_a_sweep():
    assert _value(horizon=4, PickScraper=1, GatherAndSweep=0.5) == pytest.approx(1 - 0.5**3)
    sweep = _value(horizon=12, PickScraper=0.9, GatherAndSweep=0.5)
    carry = _value(horizon=12, PickFood=0.95, PlaceFoodOnTray=0.95)
    both = _value(
        horizon=12, PickScraper=0.9, GatherAndSweep=0.5, PickFood=0.95, PlaceFoodOnTray=0.95
    )
    assert both >= max(sweep, carry) - 1e-12


def test_deployment_transitions_include_rack_recovery_but_never_human_actions():
    moves = RackServingDeployment.transitions()
    assert ("ReturnFoodToRack", (5, "empty", True)) in moves[(4, "food", True)]
    assert ("ReturnScraper", (5, "empty", True)) in moves[(5, "scraper", False)]
    assert ("PickFood", (4, "food", True)) in moves[(5, "empty", True)]
    assert ("PickFood", (4, "food", True)) in moves[(4, "empty", True)]
    assert all(successor in moves for options in moves.values() for _, successor in options)
    names = {name for options in moves.values() for name, _ in options}
    assert names == set(RACK_ROBOT_SKILLS)
    assert not names.intersection(RACK_HUMAN_SKILLS)


def test_value_interfaces_agree_and_never_read_human_beliefs():
    domain = RACK_SERVING_DOMAIN.model_copy(update={"deployment_horizon": 2})
    competence = {
        name: float(name in {"PickScraper", "GatherAndSweep"}) for name in RACK_ROBOT_SKILLS
    }
    beliefs = {
        name: Mock(sample=Mock(return_value=np.array([[value, 0.0]] * 4)))
        for name, value in competence.items()
    }
    samples = domain.sample_policy_values(beliefs=beliefs, rng=np.random.default_rng(7), count=4)
    np.testing.assert_array_equal(samples, np.ones(4))
    theta = {
        name: SkillHypothesis(competence=value, learning_rate=0.0)
        for name, value in competence.items()
    }
    assert domain.evaluate_theta(theta=theta) == 1
    draws = domain.sample_thetas(
        beliefs=beliefs,
        sample_skills=lambda belief, count: (
            [SkillHypothesis(competence=belief.sample(rng=None, count=1)[0, 0], learning_rate=0)]
            * count
        ),
        count=4,
    )
    assert all(domain.evaluate_theta(theta=draw) == 1 for draw in draws)


def test_expected_value_is_reproducible_and_ignores_human_competence():
    belief = _belief()
    robots = {name: belief.skill_beliefs[name] for name in RACK_ROBOT_SKILLS}
    first = RACK_SERVING_DOMAIN.expected_policy_value(beliefs=robots)
    assert 0 < first < 1
    assert RACK_SERVING_DOMAIN.expected_policy_value(beliefs=robots) == first
    assert RACK_SERVING_DOMAIN.expected_policy_value(beliefs=belief.skill_beliefs) == first


@pytest.mark.parametrize("index", [0, 1])
def test_human_forecasts_charge_failures_and_update_only_their_belief(*, index):
    env = RackServingEnvironment(config=RackServingConfig(human_interruption_cost=7.5))
    action = RackServingSkillProvider(env=env).human_cube_bin_reset_skills()[index]
    before = _atoms(tool_home=False, food_on_rack=False)
    assert action.preconditions <= before
    state = _belief().model_copy(update={"expected_execution_costs": {str(action): 7.5}})
    competence = state.skill_beliefs[action.skill.name].mean_competence()
    outcomes = transition_outcomes(
        environment_state=make_tossing3d_search_state(state=state, true_atoms=before),
        state=state,
        action=action,
        ground_skills=(action,),
        effects={action: (action.add_effects, action.delete_effects, action.ignore_effects)},
        exploration_epsilon=0.0,
        random_toss_competence=0.0,
        domain=RACK_SERVING_DOMAIN,
    )
    assert [weight for weight, _, _ in outcomes] == pytest.approx([competence, 1 - competence])
    for _, successor, _ in outcomes:
        assert successor.accumulated_cost == 7.5
        for name in RACK_ROBOT_SKILLS:
            assert successor.skill_beliefs[name] == state.skill_beliefs[name]
    assert outcomes[1][2] == before
    assert action.expected_add_effects(before_atoms=before) <= outcomes[0][2]
    observed = skill_belief_model(ground_skill=action, domain=RACK_SERVING_DOMAIN).observe_outcome(
        state=state, success=False, was_random_exploration=False, observed_cost=7.5
    )
    assert observed.pending_examples[action.skill.name] == 1
    assert observed.skill_beliefs[action.skill.name].mean_competence() < competence
    assert env._backend is None


@pytest.mark.parametrize("solver", ["determinized_astar", "expectimax"])
@pytest.mark.parametrize(
    "mode,offer_human", [("training", True), ("training", False), ("deployment", True)]
)
def test_real_method_builds_from_rack_skills_and_filters_humans(*, solver, mode, offer_human):
    env = RackServingEnvironment(config=RackServingConfig(mode=mode, human_interruption_cost=7.5))
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=RackServingSkillProvider(env=env, offer_human_reset=offer_human),
        pomdp_domain=RACK_SERVING_DOMAIN,
        pomdp_solver=solver,
        pomdp_search_depth=1,
        pomdp_num_particles=32,
        pomdp_inference_engine="grid",
        pomdp_grid_competence_bins=9,
        pomdp_grid_learning_rate_bins=4,
    )
    humans = set(RACK_HUMAN_SKILLS) if mode == "training" and offer_human else set()
    assert set(method.pomdp_state.skill_beliefs) == set(RACK_ROBOT_SKILLS) | humans
    assert set(method.pomdp_state.sampler_training) == {"GatherAndSweep"}
    actions = method._pomdp_model.ground_skills
    assert {action.skill.name for action in actions} == set(RACK_ROBOT_SKILLS) | humans
    assert RACK_SERVING_DOMAIN.human_skill_names == frozenset(RACK_HUMAN_SKILLS)
    assert env._backend is None


@pytest.mark.parametrize("solver", ["determinized_astar", "expectimax"])
def test_cli_builds_real_rack_method_without_running(*, tmp_path, monkeypatch, solver):
    args = Cli.parse_args(
        argv=[
            "--env",
            "racktoserving3d",
            "--method",
            "pomdp",
            "--pomdp-solver",
            solver,
            "--practice-step-budget",
            "3",
            *CostFixtures.argv(env="racktoserving3d"),
            "--practice-reset-policy",
            "never",
            "--defer-rendering",
            "--output-dir",
            str(tmp_path),
            "--pomdp-num-particles",
            "32",
            "--pomdp-inference-engine",
            "grid",
            "--pomdp-grid-competence-bins",
            "9",
            "--pomdp-grid-learning-rate-bins",
            "4",
        ]
    )
    runner = Mock()
    monkeypatch.setattr(StepPracticeRunner, "run", runner)
    from hitl_pmp.environments.racktoserving3d.step_protocol import RackServingStepProtocol

    StepProtocolEntry.run(
        args=args,
        method_cli=Tossing3DPomdpCli,
        env_cli=RackServingCli,
        environment=RackServingStepProtocol,
    )
    method = runner.call_args.kwargs["method"]
    assert method.pomdp_domain.deployment_horizon == EPISODE_HORIZON
    # Determinized A* searches no depth, so the snapshot records None for it.
    expected_depth = RACK_SERVING_DOMAIN.default_search_depth if solver == "expectimax" else None
    assert args.pomdp_search_depth == expected_depth
    assert method.env._backend is None
    assert Tossing3DPomdpCli.DOMAINS["SweepOffShelf3DCli"] is SWEEP_OFF_SHELF_DOMAIN
    assert Tossing3DPomdpCli.DOMAINS["Tossing3DCli"] is TOSSING3D_DOMAIN


def test_no_wait_in_the_planner_domain_and_a_fresh_episode_can_sweep_at_once():
    assert "Wait" not in RACK_ROBOT_SKILLS
    moves = RackServingDeployment.transitions()
    start = (RACK_NUM_FOOD, "empty", True)
    assert start in moves
    assert all(len(state) == 3 for state in moves)
    assert all(skill != "Wait" for options in moves.values() for skill, _ in options)
    holding = dict(moves[(RACK_NUM_FOOD, "scraper", False)])
    assert holding["GatherAndSweep"] == (0, "scraper", False)
