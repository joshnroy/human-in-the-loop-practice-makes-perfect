"""Every `WeightedHypothesisBelief` operation returns the very floats, and leaves the rng
in the very state, that the one-model-per-hypothesis implementation did, for one
hypothesis and for many; and a known curve's operations cost only microseconds."""

import time

import numpy as np
import pytest

from hitl_pmp.methods.belief_space.types.skill_belief import SkillHypothesis, WeightedHypothesis
from hitl_pmp.methods.belief_space.types.weighted_hypothesis_belief import (
    WeightedHypothesisBelief,
)


class Reference:
    """The per-hypothesis arithmetic of `WeightedHypothesisBelief` before it held arrays."""

    @staticmethod
    def mean_competence(*, items: tuple[WeightedHypothesis, ...]) -> float:
        return sum(item.probability * item.hypothesis.competence for item in items)

    @staticmethod
    def mean_learning_rate(*, items: tuple[WeightedHypothesis, ...]) -> float:
        return sum(item.probability * item.hypothesis.learning_rate for item in items)

    @staticmethod
    def outcome_probability(
        *, items: tuple[WeightedHypothesis, ...], successes: int, failures: int
    ) -> float:
        return sum(
            item.probability
            * item.hypothesis.competence**successes
            * (1 - item.hypothesis.competence) ** failures
            for item in items
        )

    @staticmethod
    def condition(
        *, items: tuple[WeightedHypothesis, ...], success: bool
    ) -> tuple[WeightedHypothesis, ...]:
        weighted = []
        for item in items:
            likelihood = item.hypothesis.competence if success else 1.0 - item.hypothesis.competence
            if (mass := item.probability * likelihood) > 0.0:
                weighted.append((item.hypothesis, mass))
        normalizer = sum(mass for _, mass in weighted)
        return tuple(
            WeightedHypothesis(hypothesis=h, probability=m / normalizer) for h, m in weighted
        )

    @staticmethod
    def refit(
        *, items: tuple[WeightedHypothesis, ...], training_examples: int
    ) -> tuple[WeightedHypothesis, ...]:
        return tuple(
            WeightedHypothesis(
                hypothesis=SkillHypothesis(
                    competence=min(
                        1.0,
                        item.hypothesis.competence
                        + item.hypothesis.learning_rate * training_examples,
                    ),
                    learning_rate=item.hypothesis.learning_rate,
                ),
                probability=item.probability,
            )
            for item in items
        )

    @staticmethod
    def sample(
        *, items: tuple[WeightedHypothesis, ...], rng: np.random.Generator, count: int
    ) -> np.ndarray:
        indexes = np.atleast_1d(
            rng.choice(len(items), size=count, p=[item.probability for item in items])
        )
        table = np.array([
            (item.hypothesis.competence, item.hypothesis.learning_rate) for item in items
        ])
        return table[indexes]


def random_items(*, rng: np.random.Generator, size: int) -> tuple[WeightedHypothesis, ...]:
    weights = rng.random(size) ** 3 + 1e-9
    probabilities = (weights / weights.sum()).tolist()
    competences = rng.random(size).tolist()
    # Some hypotheses at the bounds, where refit clips and conditioning drops mass.
    if size > 2:
        competences[0], competences[1] = 0.0, 1.0
    return tuple(
        WeightedHypothesis(
            hypothesis=SkillHypothesis(
                competence=c, learning_rate=float(r), cost=float(k) if size % 2 else 0.0
            ),
            probability=p,
        )
        for c, r, k, p in zip(
            competences, rng.random(size) * 0.3, rng.random(size), probabilities, strict=True
        )
    )


SIZES = (1, 2, 3, 15, 16, 17, 40, 441)


class TestEquivalence:
    @staticmethod
    @pytest.mark.parametrize("size", SIZES)
    def test_every_operation_returns_the_reference_floats(*, size: int) -> None:
        meta = np.random.default_rng(size)
        for _ in range(20):
            items = random_items(rng=meta, size=size)
            belief = WeightedHypothesisBelief.of(hypotheses=items)
            assert belief.mean_competence() == Reference.mean_competence(items=items)
            assert belief.mean_learning_rate() == Reference.mean_learning_rate(items=items)
            for successes, failures in ((0, 0), (1, 0), (0, 1), (3, 2)):
                assert belief.competence_outcome_probability(
                    successes=successes, failures=failures
                ) == Reference.outcome_probability(
                    items=items, successes=successes, failures=failures
                )
            for examples in (0, 1, 4, 50):
                refit = belief.refit(training_examples=examples)
                expected = (
                    items
                    if examples == 0
                    else Reference.refit(items=items, training_examples=examples)
                )
                assert refit.hypotheses == expected
                assert refit == WeightedHypothesisBelief.of(hypotheses=expected)
            for success in (True, False):
                conditioned = belief.condition_outcome(success=success)
                expected = Reference.condition(items=items, success=success)
                assert conditioned.hypotheses == expected
                assert conditioned == WeightedHypothesisBelief.of(hypotheses=expected)

    @staticmethod
    @pytest.mark.parametrize("size", SIZES)
    def test_sample_draws_the_reference_rows_and_leaves_the_rng_where_it_did(*, size: int) -> None:
        meta = np.random.default_rng(100 + size)
        for _ in range(10):
            items = random_items(rng=meta, size=size)
            belief = WeightedHypothesisBelief.of(hypotheses=items)
            seed = int(meta.integers(1 << 30))
            for count in (1, 7, 4096):
                ours, theirs = np.random.default_rng(seed), np.random.default_rng(seed)
                drawn = belief.sample(rng=ours, count=count)
                expected = Reference.sample(items=items, rng=theirs, count=count)
                assert drawn.shape == expected.shape == (count, 2)
                assert drawn.dtype == expected.dtype
                assert np.array_equal(drawn, expected)
                assert ours.bit_generator.state == theirs.bit_generator.state
                drawn[:, 0] *= 2.0  # still a writable array of its own


class TestKnownCurveSpeed:
    @staticmethod
    def test_one_hypothesis_operations_cost_microseconds() -> None:
        # A known curve is one hypothesis, and a search node refits, conditions and
        # samples it thousands of times: numpy's per-call overhead must not dominate.
        belief = WeightedHypothesisBelief.of(
            hypotheses=(
                WeightedHypothesis(
                    hypothesis=SkillHypothesis(competence=0.5, learning_rate=0.25),
                    probability=1.0,
                ),
            )
        )
        rng = np.random.default_rng(0)
        rounds = 2000
        started = time.perf_counter()
        for _ in range(rounds):
            belief.refit(training_examples=1).condition_outcome(success=True).mean_competence()
        per_update = (time.perf_counter() - started) / rounds
        started = time.perf_counter()
        for _ in range(rounds):
            belief.sample(rng=rng, count=4096)
        per_sample = (time.perf_counter() - started) / rounds
        assert per_update < 50e-6, per_update
        assert per_sample < 60e-6, per_sample
