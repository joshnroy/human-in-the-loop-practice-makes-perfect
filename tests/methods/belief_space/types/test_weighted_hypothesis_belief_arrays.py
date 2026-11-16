"""A weighted-hypothesis belief is still built from, read as and serialised as weighted
hypotheses, keeps every field bound, and updates fast on a large grid."""

import time

import pytest
from pydantic import ValidationError

from hitl_pmp.methods.belief_space.types.skill_belief import SkillHypothesis, WeightedHypothesis
from hitl_pmp.methods.belief_space.types.weighted_hypothesis_belief import (
    WeightedHypothesisBelief,
)

HYPOTHESES = (
    WeightedHypothesis(
        hypothesis=SkillHypothesis(competence=0.2, learning_rate=0.1, cost=1.5), probability=0.25
    ),
    WeightedHypothesis(
        hypothesis=SkillHypothesis(competence=0.6, learning_rate=0.0), probability=0.75
    ),
)


class TestRepresentation:
    @staticmethod
    def test_built_from_hypotheses_and_read_back_as_them() -> None:
        belief = WeightedHypothesisBelief.of(hypotheses=HYPOTHESES)
        assert belief.hypotheses == HYPOTHESES
        assert belief == WeightedHypothesisBelief.of(hypotheses=HYPOTHESES)
        assert hash(belief) == hash(WeightedHypothesisBelief.of(hypotheses=HYPOTHESES))

    @staticmethod
    def test_serialised_as_weighted_hypotheses_and_loaded_back() -> None:
        belief = WeightedHypothesisBelief.of(hypotheses=HYPOTHESES)
        dumped = belief.model_dump(mode="json")
        assert dumped == {"hypotheses": [item.model_dump(mode="json") for item in HYPOTHESES]}
        assert WeightedHypothesisBelief.model_validate(dumped) == belief

    @staticmethod
    def test_every_field_bound_is_still_checked() -> None:
        bad = {"competence": 0.5, "learning_rate": 0.1, "cost": 0.0, "probability": 1.0}
        for field, value in (
            ("competence", 1.5),
            ("learning_rate", -0.1),
            ("cost", float("inf")),
            ("probability", 0.0),
        ):
            row = bad | {field: value}
            with pytest.raises(ValidationError):
                WeightedHypothesisBelief(
                    hypotheses=(
                        {
                            "hypothesis": {
                                k: row[k] for k in ("competence", "learning_rate", "cost")
                            },
                            "probability": row["probability"],
                        },
                    )
                )
        with pytest.raises(ValidationError):
            WeightedHypothesisBelief.of(hypotheses=HYPOTHESES[:1])

    @staticmethod
    def test_refit_and_condition_are_fast_on_a_large_grid() -> None:
        belief = WeightedHypothesisBelief.broad_prior(points_per_dimension=30)
        started = time.perf_counter()
        for _ in range(20):
            belief.refit(training_examples=3).condition_outcome(success=True)
        assert (time.perf_counter() - started) / 20 < 1e-3
