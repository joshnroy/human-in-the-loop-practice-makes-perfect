"""`RewardedRoutesPracticeDomain.expected_policy_value` is a fixed-seed draw, so a belief
the search has already valued is answered from a memo: the same float, without drawing
again, and never for a belief whose draw could differ."""

import numpy as np
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


def fresh(*, domain: RewardedRoutesPracticeDomain, beliefs: dict) -> float:
    """The value as computed before the memo: one fixed-seed draw per call."""
    return float(
        domain.sample_policy_values(
            beliefs=beliefs, rng=np.random.default_rng(0), count=4096
        ).mean()
    )


@pytest.fixture(autouse=True)
def empty_memo() -> None:
    RewardedRoutesPracticeDomain.clear_value_memo()


class TestValueMemo:
    @staticmethod
    def test_a_repeated_belief_is_valued_without_drawing_again(
        *, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        beliefs = {"RouteA": belief((0.5, 0.25, 1.0)), "RouteB": belief((0.1, 0.05, 1.0))}
        draws = []
        original = RewardedRoutesPracticeDomain.sample_policy_values

        def counted(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            draws.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(RewardedRoutesPracticeDomain, "sample_policy_values", counted)
        first = DOMAIN.expected_policy_value(beliefs=beliefs)
        equal = {name: b.refit(training_examples=0) for name, b in beliefs.items()}
        rebuilt = {
            name: WeightedHypothesisBelief.model_validate(b.model_dump())
            for name, b in beliefs.items()
        }
        assert DOMAIN.expected_policy_value(beliefs=equal) == first
        assert DOMAIN.expected_policy_value(beliefs=rebuilt) == first
        assert len(draws) == 1

    @staticmethod
    def test_every_value_is_the_fresh_draws_float() -> None:
        meta = np.random.default_rng(7)
        cases = []
        for size in (1, 1, 2, 5, 441):
            for _ in range(4):
                beliefs = {}
                for name in ("RouteA", "RouteB"):
                    weights = meta.random(size) + 1e-3
                    beliefs[name] = belief(
                        *zip(
                            meta.random(size).tolist(),
                            (meta.random(size) * 0.2).tolist(),
                            (weights / weights.sum()).tolist(),
                            strict=True,
                        )
                    )
                cases.append(beliefs)
        # Value each twice, interleaved, so hits and misses are both checked.
        for beliefs in cases + cases:
            assert DOMAIN.expected_policy_value(beliefs=beliefs) == fresh(
                domain=DOMAIN, beliefs=beliefs
            )

    @staticmethod
    def test_beliefs_that_draw_differently_never_share_an_entry() -> None:
        base = {
            "RouteA": belief((0.5, 0.25, 0.5), (0.2, 0.1, 0.5)),
            "RouteB": belief((0.1, 0.0, 1.0)),
        }
        variants = [
            base,
            # Same hypotheses, other weights: other rows picked.
            base | {"RouteA": belief((0.5, 0.25, 0.25), (0.2, 0.1, 0.75))},
            # Same weights, other order: the rows each uniform picks swap.
            base | {"RouteA": belief((0.2, 0.1, 0.5), (0.5, 0.25, 0.5))},
            # Another competence.
            base | {"RouteB": belief((0.3, 0.0, 1.0))},
            # The routes' rewards swapped.
            {"RouteA": base["RouteB"], "RouteB": base["RouteA"]},
            # A competence of -0.0 rather than 0.0.
            base | {"RouteB": belief((-0.0, 0.0, 1.0))},
            base | {"RouteB": belief((0.0, 0.0, 1.0))},
        ]
        for beliefs in variants:
            assert DOMAIN.expected_policy_value(beliefs=beliefs) == fresh(
                domain=DOMAIN, beliefs=beliefs
            )
        other = DOMAIN.model_copy(update={"route_rewards": (("RouteA", 10.0), ("RouteB", 5.0))})
        assert other.expected_policy_value(beliefs=base) == fresh(domain=other, beliefs=base)

    @staticmethod
    def test_the_memo_is_bounded() -> None:
        for i in range(RewardedRoutesPracticeDomain.VALUE_MEMO_SIZE + 50):
            DOMAIN.expected_policy_value(
                beliefs={
                    "RouteA": belief((i / 1e5, 0.0, 1.0)),
                    "RouteB": belief((0.1, 0.0, 1.0)),
                }
            )
        assert RewardedRoutesPracticeDomain.value_memo_size() <= (
            RewardedRoutesPracticeDomain.VALUE_MEMO_SIZE
        )
