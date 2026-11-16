"""How a weighted-hypothesis belief is stored: one read-only float64 array per quantity,
and a column an update leaves unchanged is the parent's own array, not a copy.

A search conditions a 441-hypothesis grid at every node. As four tuples of Python floats
each posterior was ~61 kB; the learning rates (21 distinct values) and the costs (all
0.0) were rebuilt at every node though no update changes them."""

import numpy as np

from hitl_pmp.methods.belief_space.belief_digest import BeliefDigest
from hitl_pmp.methods.belief_space.types.weighted_hypothesis_belief import (
    WeightedHypothesisBelief,
)

COLUMNS = ("competences", "learning_rates", "costs", "probabilities")


def grid() -> WeightedHypothesisBelief:
    # Competences strictly inside (0, 1), so no outcome drops a hypothesis.
    belief = WeightedHypothesisBelief.broad_prior()
    return WeightedHypothesisBelief(
        competences=np.clip(np.asarray(belief.competences), 0.01, 0.99),
        learning_rates=belief.learning_rates,
        costs=belief.costs,
        probabilities=belief.probabilities,
    )


class TestStorage:
    @staticmethod
    def test_every_column_is_a_read_only_float64_array() -> None:
        for belief in (
            grid(),
            grid().condition_outcome(success=True),
            grid().refit(training_examples=2),
        ):
            for name in COLUMNS:
                column = getattr(belief, name)
                assert isinstance(column, np.ndarray), name
                assert column.dtype == np.float64, name
                assert not column.flags.writeable, name

    @staticmethod
    def test_conditioning_shares_every_column_it_does_not_change() -> None:
        parent = grid()
        child = parent.condition_outcome(success=False)
        assert child.competences is parent.competences
        assert child.learning_rates is parent.learning_rates
        assert child.costs is parent.costs
        assert child.probabilities is not parent.probabilities

    @staticmethod
    def test_refitting_shares_the_rates_weights_and_a_zero_cost_column() -> None:
        parent = grid()
        first = parent.refit(training_examples=1)
        second = parent.refit(training_examples=2)
        assert first.learning_rates is parent.learning_rates
        assert first.probabilities is parent.probabilities
        assert first.costs is second.costs

    @staticmethod
    def test_equality_hash_and_digest_still_merge_minus_zero_with_zero() -> None:
        def of(*, competence: float) -> WeightedHypothesisBelief:
            return WeightedHypothesisBelief(
                competences=(competence, 0.5),
                learning_rates=(0.0, 0.1),
                costs=(0.0, 0.0),
                probabilities=(0.5, 0.5),
            )

        minus, plus, other = of(competence=-0.0), of(competence=0.0), of(competence=0.25)
        assert BeliefDigest.of(belief=minus) == BeliefDigest.digest(
            value=(type(minus).__module__, type(minus).__qualname__, minus.signature())
        )
        assert minus == plus and hash(minus) == hash(plus)
        assert BeliefDigest.of(belief=minus) == BeliefDigest.of(belief=plus)
        assert minus != other
        assert BeliefDigest.of(belief=minus) != BeliefDigest.of(belief=other)


class TestDigest:
    @staticmethod
    def test_the_column_digest_is_the_signature_digest_and_fills_no_float_table() -> None:
        BeliefDigest._float_encodings.clear()
        parent = grid().refit(training_examples=1)
        beliefs = [parent, parent.condition_outcome(success=True)]
        beliefs.append(beliefs[-1].condition_outcome(success=False))
        digests = [BeliefDigest.of(belief=belief) for belief in beliefs]
        assert not BeliefDigest._float_encodings
        assert digests == [
            BeliefDigest.digest(
                value=(type(belief).__module__, type(belief).__qualname__, belief.signature())
            )
            for belief in beliefs
        ]
