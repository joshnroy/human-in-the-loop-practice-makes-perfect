"""`RewardedRoutesPracticeDomain.expected_policy_value` is what the deployed robot earns:
it runs the route its belief rates best, so the value is that route's expected reward
(the product of its skills' mean competences, times its reward). It must not credit an
uncertain route with the upside of knowing the true competences (Josh, 2026-11-16: an
uninformed prior made the uncertain chained route look better than it was). A belief
the search has already valued is answered from a memo."""

import pytest

from hitl_pmp.methods.belief_space.rewarded_routes_domain import RewardedRoutesPracticeDomain
from hitl_pmp.methods.belief_space.types.skill_belief import SkillHypothesis, WeightedHypothesis
from hitl_pmp.methods.belief_space.types.weighted_hypothesis_belief import (
    WeightedHypothesisBelief,
)

DOMAIN = RewardedRoutesPracticeDomain(
    deployment_horizon=1,
    route_rewards=(("RouteA", 5.0), ("RouteB", 10.0)),
    belief_sources=(),
)


def belief(*rows: tuple[float, float, float]) -> WeightedHypothesisBelief:
    return WeightedHypothesisBelief.of(
        hypotheses=tuple(
            WeightedHypothesis(
                hypothesis=SkillHypothesis(competence=c, learning_rate=r), probability=p
            )
            for c, r, p in rows
        )
    )


@pytest.fixture(autouse=True)
def empty_memo() -> None:
    RewardedRoutesPracticeDomain.clear_value_memo()


class TestDeploymentValue:
    @staticmethod
    def test_the_value_is_the_best_route_in_expectation_not_per_draw() -> None:
        # Route B is a coin flip between hopeless and perfect: worth 10 * 0.5 = 5 in
        # expectation. Route A is a sure 0.8: worth 4. Per draw, the better route would
        # average 0.5 * 10 + 0.5 * 4 = 7, which no deployed robot can earn.
        beliefs = {
            "RouteA": belief((0.8, 0.0, 1.0)),
            "RouteB": belief((0.0, 0.0, 0.5), (1.0, 0.0, 0.5)),
        }
        assert DOMAIN.expected_policy_value(beliefs=beliefs) == pytest.approx(5.0)

    @staticmethod
    def test_a_chain_is_worth_the_product_of_its_skills_means() -> None:
        domain = DOMAIN.model_copy(update={"route_prerequisites": (("RouteB", ("Gather",)),)})
        beliefs = {
            "RouteA": belief((0.1, 0.0, 1.0)),
            "RouteB": belief((0.2, 0.0, 0.5), (0.6, 0.0, 0.5)),
            "Gather": belief((0.5, 0.0, 1.0)),
        }
        assert domain.expected_policy_value(beliefs=beliefs) == pytest.approx(10 * 0.4 * 0.5)

    @staticmethod
    def test_a_repeated_belief_is_valued_from_the_memo(*, monkeypatch: pytest.MonkeyPatch) -> None:
        beliefs = {"RouteA": belief((0.5, 0.25, 1.0)), "RouteB": belief((0.1, 0.05, 1.0))}
        calls = []
        original = RewardedRoutesPracticeDomain.route_expected_values

        def counted(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(RewardedRoutesPracticeDomain, "route_expected_values", counted)
        first = DOMAIN.expected_policy_value(beliefs=beliefs)
        rebuilt = {
            name: WeightedHypothesisBelief.model_validate(b.model_dump())
            for name, b in beliefs.items()
        }
        assert DOMAIN.expected_policy_value(beliefs=rebuilt) == first
        assert len(calls) == 1
