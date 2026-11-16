"""`WeightedHypothesisBelief.sample` draws the hypotheses `rng.choice` picks, as rows of
(competence, learning rate), and stays fast on a full prior grid."""

import time

import numpy as np

from hitl_pmp.methods.belief_space.types.skill_belief import SkillHypothesis, WeightedHypothesis
from hitl_pmp.methods.belief_space.types.weighted_hypothesis_belief import (
    WeightedHypothesisBelief,
)


def reference(*, belief: WeightedHypothesisBelief, seed: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probabilities = [item.probability for item in belief.hypotheses]
    indexes = np.atleast_1d(rng.choice(len(belief.hypotheses), size=count, p=probabilities))
    return np.array([
        [
            belief.hypotheses[int(i)].hypothesis.competence,
            belief.hypotheses[int(i)].hypothesis.learning_rate,
        ]
        for i in indexes
    ])


class TestUpdates:
    @staticmethod
    def test_refit_and_condition_equal_the_validated_construction() -> None:
        belief = WeightedHypothesisBelief.broad_prior()
        refit = belief.refit(training_examples=2)
        expected = WeightedHypothesisBelief(
            hypotheses=tuple(
                WeightedHypothesis(
                    hypothesis=SkillHypothesis(
                        competence=min(
                            1.0, item.hypothesis.competence + item.hypothesis.learning_rate * 2
                        ),
                        learning_rate=item.hypothesis.learning_rate,
                    ),
                    probability=item.probability,
                )
                for item in belief.hypotheses
            )
        )
        assert refit == expected
        assert refit.signature() == expected.signature()
        failed = refit.condition_outcome(success=False)
        masses = [
            (item.hypothesis, item.probability * (1.0 - item.hypothesis.competence))
            for item in refit.hypotheses
            if item.probability * (1.0 - item.hypothesis.competence) > 0.0
        ]
        total = sum(m for _, m in masses)
        assert failed == WeightedHypothesisBelief(
            hypotheses=tuple(
                WeightedHypothesis(hypothesis=h, probability=m / total) for h, m in masses
            )
        )


class TestSample:
    @staticmethod
    def test_rows_are_the_hypotheses_the_rng_picks() -> None:
        belief = WeightedHypothesisBelief.broad_prior().condition_outcome(success=True)
        for count in (1, 7, 512):
            drawn = belief.sample(rng=np.random.default_rng(3), count=count)
            expected = reference(belief=belief, seed=3, count=count)
            assert drawn.shape == expected.shape == (count, 2)
            assert drawn.dtype == expected.dtype
            assert np.array_equal(drawn, expected)

    @staticmethod
    def test_a_node_value_sized_draw_is_fast() -> None:
        # `RewardedRoutesPracticeDomain.expected_policy_value` draws 4096 per search node.
        belief = WeightedHypothesisBelief.broad_prior()
        rng = np.random.default_rng(0)
        started = time.perf_counter()
        for _ in range(50):
            belief.sample(rng=rng, count=4096)
        assert (time.perf_counter() - started) / 50 < 0.5e-3
