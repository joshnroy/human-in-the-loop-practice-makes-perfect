"""A known competence curve: competence after n tries is min(1, start + rate * n), and
nothing a practice outcome says moves it -- for ours' belief and for deployment."""

from types import SimpleNamespace

import pytest

from hitl_pmp.methods.belief_space.rewarded_routes_domain import RewardedRoutesPracticeDomain
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    make_default_tossing3d_belief,
)
from hitl_pmp.methods.belief_space.types.practice_example_source import PracticeExampleSource
from hitl_pmp.methods.deployment import DeploymentPolicy

DOMAIN = RewardedRoutesPracticeDomain(
    route_rewards=(("RouteA", 3.0), ("RouteB", 6.0)),
    belief_sources=(
        ("RouteA", PracticeExampleSource.OUTCOME),
        ("RouteB", PracticeExampleSource.OUTCOME),
    ),
    known_competence_curves=(("RouteA", 0.25, 0.25), ("RouteB", 0.25, 0.02)),
)


class TestKnownCompetenceCurve:
    @staticmethod
    def test_ours_belief_is_the_curve_and_outcomes_do_not_move_it() -> None:
        belief = make_default_tossing3d_belief(
            domain=DOMAIN, model="local_trend", engine="grid"
        ).skill_beliefs["RouteA"]
        assert belief.mean_competence() == pytest.approx(0.25)
        failed = belief.condition_outcome(success=False)
        assert failed.mean_competence() == pytest.approx(0.25)
        assert failed.refit(training_examples=2).mean_competence() == pytest.approx(0.75)
        assert belief.refit(training_examples=5).mean_competence() == pytest.approx(1.0)

    @staticmethod
    def test_a_known_rate_alone_leaves_competence_to_be_inferred() -> None:
        domain = DOMAIN.model_copy(
            update={"known_competence_curves": (), "known_learning_rates": (("RouteA", 0.25),)}
        )
        belief = make_default_tossing3d_belief(
            domain=domain, model="local_trend", engine="grid"
        ).skill_beliefs["RouteA"]
        assert belief.mean_competence() == pytest.approx(0.25, abs=0.01)
        assert belief.mean_learning_rate() == pytest.approx(0.25)
        assert belief.refit(training_examples=1).mean_competence() > 0.45
        succeeded = belief.condition_outcome(success=True).mean_competence()
        failed = belief.condition_outcome(success=False).mean_competence()
        assert failed < belief.mean_competence() < succeeded

    @staticmethod
    def test_a_routes_deployment_prior_is_its_own_planning_prior() -> None:
        domain = DOMAIN.model_copy(update={"competence_prior_overrides": (("RouteB", 1.0, 99.0),)})
        assert domain.route_competence_prior(name="RouteB", alpha=1.0, beta=3.0) == (1.0, 99.0)
        assert domain.route_competence_prior(name="RouteA", alpha=1.0, beta=3.0) == (1.0, 3.0)

    @staticmethod
    def test_named_skills_count_as_tries_even_without_parameters() -> None:
        from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

        route = SimpleNamespace(skill=SimpleNamespace(name="RouteA", param_dim=0))
        reset = SimpleNamespace(skill=SimpleNamespace(name="ResetA", param_dim=0))
        assert EesMethod.counts_as_try(ground_skill=route, try_skills=frozenset({"RouteA"}))
        assert not EesMethod.counts_as_try(ground_skill=reset, try_skills=frozenset({"RouteA"}))
        assert not EesMethod.counts_as_try(ground_skill=route, try_skills=None)

    @staticmethod
    def test_a_skill_prior_gives_ours_that_beta_and_that_rate_distribution() -> None:
        domain = DOMAIN.model_copy(
            update={
                "known_competence_curves": (),
                "skill_priors": (("RouteA", 0.1, 10.0, 0.1, 0.05),),
            }
        )
        belief = make_default_tossing3d_belief(
            domain=domain, model="local_trend", engine="grid"
        ).skill_beliefs["RouteA"]
        assert belief.mean_competence() == pytest.approx(0.1, abs=0.01)
        assert belief.mean_learning_rate() == pytest.approx(0.1, abs=0.01)
        assert belief.refit(training_examples=1).mean_competence() > 0.15
        assert domain.route_competence_prior(name="RouteA", alpha=1.0, beta=3.0) == (
            pytest.approx(1.0),
            pytest.approx(9.0),
        )

    @staticmethod
    def test_a_skill_prior_gives_dp_that_beta_and_that_rate_mean() -> None:
        from hitl_pmp.methods.practice_makes_perfect.dp_method import DpMethod
        from hitl_pmp.methods.practice_makes_perfect.dp_priors import DpSkillPrior

        prior = DpMethod.skill_prior(c_mean=0.1, c_strength=10.0, rate_mean=0.1)
        assert prior == DpSkillPrior(alpha=1.0, beta=9.0, initial_learning_rate=0.1)

    @staticmethod
    def test_current_competence_accounts_for_learning() -> None:
        from hitl_pmp.methods.deployment import CurrentCompetence

        # Route A of the toy: prior centred on 0.1 + 0.1/try; five tries made at true
        # competence 0.1..0.5 with one success, on the 4th try.
        outcomes = [False, False, False, True, False]
        current = CurrentCompetence.estimate(prior=(0.1, 10.0, 0.1, 0.05), outcomes=outcomes)
        lagging = (1 + 1) / (1 + 9 + 5)
        assert current > 0.4 > lagging
        assert CurrentCompetence.estimate(prior=(0.5, 10.0, 0.5, 0.25), outcomes=[]) == (
            pytest.approx(0.5, abs=0.02)
        )

    @staticmethod
    def test_deployment_uses_current_competence_when_given_a_curve_prior() -> None:
        ground = SimpleNamespace(skill=SimpleNamespace(name="RouteA"))
        history = SimpleNamespace(cycle_observations=[[False, False], [False, True, False]])
        method = SimpleNamespace(
            _competence_models=SimpleNamespace(get=lambda key: history if key is ground else None),
            prior_alpha=1.0,
            prior_beta=9.0,
        )
        policy = DeploymentPolicy(
            kind="reward_greedy", curve_priors={"RouteA": (0.1, 10.0, 0.1, 0.05)}
        )
        assert policy.recent_competence(method=method, ground=ground) > 0.4
        assert policy.dp_competence(method=method, ground=ground) > 0.4

    @staticmethod
    def test_deployment_reads_the_curve_at_the_try_count() -> None:
        ground = SimpleNamespace(skill=SimpleNamespace(name="RouteA"))
        history = SimpleNamespace(cycle_observations=[[0, 0], [1]])
        method = SimpleNamespace(
            _competence_models=SimpleNamespace(get=lambda key: history if key is ground else None),
            prior_alpha=1.0,
            prior_beta=3.0,
        )
        policy = DeploymentPolicy(kind="dp_mdp", known_curves={"RouteA": (0.25, 0.25)})
        assert policy.dp_competence(method=method, ground=ground) == pytest.approx(1.0)
        assert policy.recent_competence(method=method, ground=ground) == pytest.approx(1.0)
