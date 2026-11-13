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
    RACK_SWEEP_SKILLS,
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
            if name in {"GatherAndSweep", "Sweep"}
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
    assert set(method.pomdp_state.sampler_training) == {"GatherAndSweep", "Sweep"}
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


def test_sweep_is_a_second_sampler_route_from_the_rack_with_the_scraper_held():
    assert RACK_ROBOT_SKILLS[-1] == "Sweep"
    assert "Sweep" in RACK_SERVING_DOMAIN.sampler_skills
    assert "Sweep" in RACK_SERVING_DOMAIN.optional_skill_names
    holding = RackServingDeployment.transitions()[(RACK_NUM_FOOD, "scraper", False)]
    assert ("Sweep", (0, "scraper", False)) in holding
    assert ("GatherAndSweep", (0, "scraper", False)) in holding
    assert _value(horizon=2, PickScraper=1, Sweep=1) == 1
    assert _value(horizon=4, PickScraper=1, Sweep=0.25, GatherAndSweep=0.5) == pytest.approx(
        1 - 0.5**3
    )


def test_sweep_appends_after_every_existing_belief_so_prior_seeds_do_not_move():
    names = [name for name, _ in RACK_SERVING_DOMAIN.belief_sources]
    assert names == [
        "PickScraper",
        "GatherAndSweep",
        "ReturnScraper",
        "PickFood",
        "PlaceFoodOnTray",
        "ReturnFoodToRack",
        "human_batch_return",
        "human_tool_recovery",
        "Sweep",
    ]


def test_a_provider_without_sweep_drops_its_belief_and_still_values_deployment():
    from hitl_pmp.methods.belief_space.practice_domain import PracticeDomainWithoutSkills

    domain = PracticeDomainWithoutSkills.keep_offered(
        domain=RACK_SERVING_DOMAIN,
        offered=frozenset(RACK_ROBOT_SKILLS[:-1]) | frozenset(RACK_HUMAN_SKILLS),
    )
    assert "Sweep" not in dict(domain.belief_sources)
    beliefs = {
        name: Mock(sample=Mock(return_value=np.array([[1.0, 0.0]] * 4)))
        for name in RACK_ROBOT_SKILLS[:-1]
    }
    values = domain.sample_policy_values(beliefs=beliefs, rng=np.random.default_rng(0), count=4)
    np.testing.assert_array_equal(values, np.ones(4))


def test_the_deployment_model_sweeps_only_a_full_rack_like_the_operators():
    # Mirrors `test_a_rack_missing_any_food_cannot_be_swept`: after a partial sweep the
    # rest is picked, not swept again.
    moves = RackServingDeployment.transitions()
    for on_rack in range(RACK_NUM_FOOD):
        options = {skill for skill, _ in moves[(on_rack, "scraper", False)]}
        assert not options & set(RACK_SWEEP_SKILLS), on_rack


@pytest.mark.parametrize("prior", [None, (4.0, 1.0)])
def test_the_competence_prior_flag_overrides_the_domains_beta_prior(
    *, tmp_path, monkeypatch, prior
):
    # Josh, 2026-11-06: launch Curtis at prior mean 0.25 (the domain's Beta(1, 3)) and 0.8.
    flag = [] if prior is None else ["--pomdp-competence-prior", *(str(v) for v in prior)]
    args = Cli.parse_args(
        argv=[
            "--env",
            "racktoserving3d",
            "--method",
            "pomdp",
            "--pomdp-solver",
            "determinized_astar",
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
            *flag,
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
    domain = runner.call_args.kwargs["method"].pomdp_domain
    expected = (1.0, 3.0) if prior is None else prior
    assert (domain.initial_competence_alpha, domain.initial_competence_beta) == expected
    assert (
        RACK_SERVING_DOMAIN.initial_competence_alpha,
        RACK_SERVING_DOMAIN.initial_competence_beta,
    ) == (1.0, 3.0)


def _one_attempt(*, horizon, **competence):
    return float(
        RackServingDeployment.solve(
            competences={name: np.array([value]) for name, value in competence.items()},
            horizon=horizon,
            step_weight=0.0,
            one_bulk_attempt=True,
        )[0][0]
    )


def test_one_bulk_attempt_gives_a_failed_sweep_no_retry_and_takes_the_better_skill():
    assert _one_attempt(horizon=12, PickScraper=1, GatherAndSweep=0.5) == pytest.approx(0.5)
    assert _one_attempt(horizon=12, PickScraper=1, GatherAndSweep=0.9, Sweep=0.3) == (
        pytest.approx(0.9)
    )
    # Picking the scraper up is not the bulk attempt: it may still be retried.
    assert _one_attempt(horizon=3, PickScraper=0.5, GatherAndSweep=1) == pytest.approx(0.75)


def test_a_provider_without_placement_is_valued_as_one_bulk_attempt():
    from hitl_pmp.methods.belief_space.practice_domain import PracticeDomainWithoutSkills

    offered = frozenset(RACK_ROBOT_SKILLS) - {"PlaceFoodOnTray"}
    domain = PracticeDomainWithoutSkills.keep_offered(
        domain=RACK_SERVING_DOMAIN, offered=offered | frozenset(RACK_HUMAN_SKILLS)
    )
    assert "PlaceFoodOnTray" not in dict(domain.belief_sources)
    theta = {
        name: Mock(competence=0.5 if name == "GatherAndSweep" else 1.0)
        for name in domain.robot_skills()
    }
    theta["Sweep"] = Mock(competence=0.3)
    assert domain.evaluate_theta(theta=theta) == pytest.approx(0.5)
    with_placement = {**theta, "PlaceFoodOnTray": Mock(competence=1.0)}
    assert RACK_SERVING_DOMAIN.evaluate_theta(theta=with_placement) == pytest.approx(1.0)


TWO_ROUTE_CONFIG = RackServingConfig(bulk_transfer_only=True, two_route_rewards=True)


def test_priced_bulk_outcomes_are_terminal_values_weighed_by_competence():
    def value(**competence):
        return float(
            RackServingDeployment.solve(
                competences={name: np.array([v]) for name, v in competence.items()},
                horizon=12,
                step_weight=0.0,
                one_bulk_attempt=True,
                bulk_rewards={"Sweep": 3.0, "GatherAndSweep": 5.0},
            )[0][0]
        )

    assert value(PickScraper=1, Sweep=0.9, GatherAndSweep=0.4) == pytest.approx(2.7)
    assert value(PickScraper=1, Sweep=0.9, GatherAndSweep=0.8) == pytest.approx(4.0)
    assert value(PickScraper=1, Sweep=1.0) == pytest.approx(3.0)


def test_the_rack_domain_takes_its_bulk_rewards_from_a_provider_that_prices_goals():
    from hitl_pmp.methods.belief_space.practice_domain import PracticeDomainWithoutSkills
    from hitl_pmp.methods.belief_space.rack_serving_domain import RackServingPracticeDomain

    plain = RackServingSkillProvider(env=RackServingEnvironment())
    assert RACK_SERVING_DOMAIN.with_goal_rewards(provider=plain) is RACK_SERVING_DOMAIN
    provider = RackServingSkillProvider(env=RackServingEnvironment(config=TWO_ROUTE_CONFIG))
    domain = RACK_SERVING_DOMAIN.with_goal_rewards(provider=provider)
    assert isinstance(domain, RackServingPracticeDomain)
    assert dict(domain.bulk_rewards) == {"Sweep": 3.0, "GatherAndSweep": 5.0}
    domain = PracticeDomainWithoutSkills.keep_offered(
        domain=domain,
        offered=frozenset(skill.name for skill in provider.skills()) | frozenset(RACK_HUMAN_SKILLS),
    )
    theta = {name: Mock(competence=1.0) for name in domain.robot_skills()}
    theta["GatherAndSweep"] = Mock(competence=0.5)
    assert domain.evaluate_theta(theta=theta) == pytest.approx(3.0)
    theta["GatherAndSweep"] = Mock(competence=0.7)
    assert domain.evaluate_theta(theta=theta) == pytest.approx(3.5)


def _plan(*, step_weight=0.0, **competence):
    return RackServingDeployment.plan(
        competences={name: np.array([v]) for name, v in competence.items()},
        horizon=12,
        step_weight=step_weight,
        one_bulk_attempt=True,
        bulk_rewards={"Sweep": 3.0, "GatherAndSweep": 5.0},
    )


def test_the_deployment_plan_is_the_choice_the_value_recursion_makes():
    start, held = (5, "empty", True), (5, "scraper", False)
    plan = _plan(PickScraper=1, Sweep=0.9, GatherAndSweep=0.4)
    assert plan[start, 12] == ("PickScraper", held)
    assert plan[held, 11][0] == "Sweep"
    assert _plan(PickScraper=1, Sweep=0.9, GatherAndSweep=0.8)[held, 11][0] == "GatherAndSweep"
    # Each step is charged as the value is: at a weight that makes every route a loss the
    # plan stops, exactly where `solve` returns 0.
    costly = _plan(step_weight=0.01, PickScraper=1, Sweep=0.9, GatherAndSweep=0.8)
    assert costly[start, 12] is None
    # With no steps left there is nothing to choose.
    assert (start, 0) not in plan


def test_the_plans_expected_value_is_the_value_models_at_one_competence_sample():
    competences = {"PickScraper": 0.7, "Sweep": 0.6, "GatherAndSweep": 0.5}
    weight = 0.0001
    plan = _plan(step_weight=weight, **competences)
    value = float(
        RackServingDeployment.solve(
            competences={name: np.array([v]) for name, v in competences.items()},
            horizon=12,
            step_weight=weight,
            one_bulk_attempt=True,
            bulk_rewards={"Sweep": 3.0, "GatherAndSweep": 5.0},
        )[0][0]
    )
    # Follow the plan exactly: PickScraper is retried while steps remain, the bulk skill once.
    from hitl_pmp.methods.belief_space.rack_serving_domain import RACK_SKILL_STEPS

    def follow(*, state, remaining):
        choice = plan.get((state, remaining))
        if choice is None:
            return 0.0
        skill, successor = choice
        c = competences[skill]
        ok, bad = RACK_SKILL_STEPS[skill]
        if skill in ("Sweep", "GatherAndSweep"):
            reward = {"Sweep": 3.0, "GatherAndSweep": 5.0}[skill]
            return c * (reward - weight * ok) - (1 - c) * weight * bad
        return c * (follow(state=successor, remaining=remaining - 1) - weight * ok) + (1 - c) * (
            follow(state=state, remaining=remaining - 1) - weight * bad
        )

    assert follow(state=(5, "empty", True), remaining=12) == pytest.approx(value)


def test_the_abstract_deployment_state_is_read_from_measured_atoms():
    atoms = {("OnRack", (f"cube_{i}", "rack_0")) for i in range(5)}
    assert RackServingDeployment.state_of(
        atoms=atoms | {("HandEmpty", ("robot",)), ("ScraperHome", ("wiper_0",))}
    ) == (5, "empty", True)
    assert RackServingDeployment.state_of(
        atoms=atoms | {("HoldingScraper", ("robot", "wiper_0")), ("ScraperAway", ("wiper_0",))}
    ) == (5, "scraper", False)


def test_the_domains_deployment_plan_is_the_recursion_at_the_beliefs_mean_competences():
    provider = RackServingSkillProvider(env=RackServingEnvironment(config=TWO_ROUTE_CONFIG))
    from hitl_pmp.methods.belief_space.practice_domain import PracticeDomainWithoutSkills

    domain = PracticeDomainWithoutSkills.keep_offered(
        domain=RACK_SERVING_DOMAIN.with_goal_rewards(provider=provider),
        offered=frozenset(skill.name for skill in provider.skills()) | frozenset(RACK_HUMAN_SKILLS),
    ).model_copy(update={"deployment_cost_lambda": 0.0015, "robot_step_price": 0.01})

    def belief(*, mean):
        return Mock(sample=lambda rng, count: np.full((count, 1), mean))

    means = {name: 0.9 for name in domain.robot_skills()} | {"Sweep": 0.8, "GatherAndSweep": 0.3}
    plan = domain.deployment_plan(beliefs={name: belief(mean=m) for name, m in means.items()})
    expected = RackServingDeployment.plan(
        competences={name: np.array([m]) for name, m in means.items()},
        horizon=domain.deployment_horizon,
        step_weight=0.0015 * 0.01,
        one_bulk_attempt=True,
        bulk_rewards={"Sweep": 3.0, "GatherAndSweep": 5.0},
    )
    assert plan == expected
    held = (5, "scraper", False)
    assert plan[(5, "empty", True), domain.deployment_horizon][0] == "PickScraper"
    # 3 * 0.8 beats 5 * 0.3, net of each skill's steps at the domain's weight.
    assert plan[held, domain.deployment_horizon - 1][0] == "Sweep"


def test_a_named_skill_can_start_from_its_own_competence_prior():
    domain = Tossing3DPomdpCli.with_prior_means(
        domain=RACK_SERVING_DOMAIN, means={"PickScraper": 0.9, "ReturnScraper": 0.9}
    )
    assert dict((name, (a, b)) for name, a, b in domain.competence_prior_overrides) == {
        "PickScraper": (pytest.approx(9.0), pytest.approx(1.0)),
        "ReturnScraper": (pytest.approx(9.0), pytest.approx(1.0)),
    }
    belief = make_default_tossing3d_belief(
        domain=domain,
        model="local_trend",
        engine="grid",
        num_particles=32,
        inference_config=InferenceConfig(competence_bins=41, learning_rate_bins=4),
    )
    means = {name: b.mean_competence() for name, b in belief.skill_beliefs.items()}
    assert means["PickScraper"] == pytest.approx(0.9, abs=0.03)
    assert means["ReturnScraper"] == pytest.approx(0.9, abs=0.03)
    plain = _belief().skill_beliefs
    assert means["Sweep"] == pytest.approx(0.25, abs=0.03)
    assert set(means) == set(plain)
    assert Tossing3DPomdpCli.with_prior_means(domain=RACK_SERVING_DOMAIN, means={}) is (
        RACK_SERVING_DOMAIN
    )
    with pytest.raises(ValueError, match="NoSuchSkill"):
        Tossing3DPomdpCli.with_prior_means(domain=RACK_SERVING_DOMAIN, means={"NoSuchSkill": 0.5})


def test_ours_starting_learning_rate_prior_scales_with_the_skills_parameter_count():
    provider = RackServingSkillProvider(env=RackServingEnvironment(config=TWO_ROUTE_CONFIG))
    domain = Tossing3DPomdpCli.with_learning_rates_by_parameters(
        domain=RACK_SERVING_DOMAIN, provider=provider, enabled=True
    )
    scales = dict(domain.learning_rate_prior_scales)
    assert scales["Sweep"] == 1.0
    assert scales["GatherAndSweep"] == pytest.approx(1 / 3)
    assert scales["PickScraper"] == 0.0
    # The person's skills are not the robot's to tune and keep the shared prior.
    assert not set(scales) & set(RACK_HUMAN_SKILLS)

    def beliefs(*, domain, engine):
        return make_default_tossing3d_belief(
            domain=domain,
            model="local_trend",
            engine=engine,
            num_particles=2000,
            inference_config=InferenceConfig(competence_bins=9, learning_rate_bins=31),
        ).skill_beliefs

    for engine in ("grid", "particle"):
        scaled = beliefs(domain=domain, engine=engine)
        plain = beliefs(domain=RACK_SERVING_DOMAIN, engine=engine)
        base = plain["Sweep"].mean_learning_rate()
        assert scaled["Sweep"].mean_learning_rate() == pytest.approx(base)
        assert scaled["GatherAndSweep"].mean_learning_rate() == pytest.approx(base / 3, rel=0.15)
        assert scaled["PickScraper"].mean_learning_rate() == pytest.approx(0.0, abs=1e-9)
        # Competence priors are untouched.
        assert scaled["PickScraper"].mean_competence() == pytest.approx(
            plain["PickScraper"].mean_competence()
        )
        assert scaled["human_batch_return"].mean_learning_rate() == pytest.approx(
            plain["human_batch_return"].mean_learning_rate()
        )
    # Off: the very same domain object, so beliefs are built exactly as today.
    assert (
        Tossing3DPomdpCli.with_learning_rates_by_parameters(
            domain=RACK_SERVING_DOMAIN, provider=provider, enabled=False
        )
        is RACK_SERVING_DOMAIN
    )


def _initial_rate_beliefs(*, base: float | None, enabled: bool = True, num_particles: int):
    provider = RackServingSkillProvider(env=RackServingEnvironment(config=TWO_ROUTE_CONFIG))
    domain = Tossing3DPomdpCli.with_learning_rates_by_parameters(
        domain=RACK_SERVING_DOMAIN, provider=provider, enabled=enabled, base=base
    )
    return make_default_tossing3d_belief(
        domain=domain,
        model="local_trend",
        engine="particle",
        num_particles=num_particles,
        seed=0,
    ).skill_beliefs


def test_ours_learning_rate_prior_base_sets_each_skills_starting_mean_to_dps():
    """With base 0.1, a robot skill's starting learning-rate belief has mean
    0.1 / param_dim, DP's own starting rate (Josh, 2026-11-13). A half-normal's sample
    mean over 20000 particles has a relative standard error of about 0.5%; the
    tolerance is 3%."""
    beliefs = _initial_rate_beliefs(base=0.1, num_particles=20000)
    plain = _initial_rate_beliefs(base=None, enabled=False, num_particles=20000)
    assert beliefs["Sweep"].mean_learning_rate() == pytest.approx(0.100, rel=0.03)
    assert beliefs["GatherAndSweep"].mean_learning_rate() == pytest.approx(0.1 / 3, rel=0.03)
    assert beliefs["PickScraper"].mean_learning_rate() == pytest.approx(0.0, abs=1e-9)
    # Still a half-normal: its standard deviation is its mean times sqrt(pi / 2 - 1).
    rates = beliefs["Sweep"].arrays()[0][:, 1]
    assert rates.std() == pytest.approx(0.1 * np.sqrt(np.pi / 2 - 1), rel=0.03)
    # The person's skills keep the shared prior, particle for particle, and no
    # competence prior moves.
    for name in RACK_HUMAN_SKILLS:
        assert beliefs[name].latent_values == plain[name].latent_values
    for name in beliefs:
        assert np.array_equal(beliefs[name].arrays()[0][:, 0], plain[name].arrays()[0][:, 0])


def test_ours_learning_rate_prior_without_a_base_is_todays_particles():
    """Digests of every initial belief's bytes, recorded from the code before the base
    existed (2026-11-13), with and without the scaling by parameters."""
    import hashlib

    def digest(*, beliefs) -> str:
        hashed = hashlib.sha256()
        for name in sorted(beliefs):
            hashed.update(name.encode())
            hashed.update(beliefs[name].latent_values)
            hashed.update(beliefs[name].state_weights)
        return hashed.hexdigest()

    assert digest(beliefs=_initial_rate_beliefs(base=None, num_particles=2000)) == (
        "4ede892340ffdb8f2f20feb43df8eb0d61df43e42a5f3c4060300fa8de69163b"
    )
    assert digest(beliefs=_initial_rate_beliefs(base=None, enabled=False, num_particles=2000)) == (
        "92528eda5564791c3be08ace8742211d5a4d5cc267c5d5451ffd24931ec1b364"
    )
