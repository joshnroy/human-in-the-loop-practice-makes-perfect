"""Finite weighted-hypothesis skill belief."""

import numpy as np
from pydantic import model_validator

from .skill_belief import (
    COMPETENCE_MAX,
    SkillBelief,
    SkillHypothesis,
    WeightedHypothesis,
)


class WeightedHypothesisBelief(SkillBelief):
    hypotheses: tuple[WeightedHypothesis, ...]

    @model_validator(mode="after")
    def validate_probabilities(self) -> "WeightedHypothesisBelief":
        if not self.hypotheses:
            raise ValueError("weighted beliefs require hypotheses")
        if not np.isclose(sum(item.probability for item in self.hypotheses), 1.0):
            raise ValueError("hypothesis probabilities must sum to one")
        return self

    @classmethod
    def broad_prior(cls, *, points_per_dimension: int = 21) -> "WeightedHypothesisBelief":
        assert points_per_dimension >= 2
        values = np.linspace(0.0, 1.0, points_per_dimension)
        probability = 1.0 / points_per_dimension**2
        return cls(
            hypotheses=tuple(
                WeightedHypothesis(
                    hypothesis=SkillHypothesis(competence=kappa, learning_rate=eta),
                    probability=probability,
                )
                for kappa in values
                for eta in values
            )
        )

    def mean_competence(self) -> float:
        return sum(item.probability * item.hypothesis.competence for item in self.hypotheses)

    def mean_learning_rate(self) -> float:
        return sum(item.probability * item.hypothesis.learning_rate for item in self.hypotheses)

    def sample(self, *, rng: np.random.Generator, count: int) -> np.ndarray:
        indexes = np.atleast_1d(
            rng.choice(
                len(self.hypotheses), size=count, p=[item.probability for item in self.hypotheses]
            )
        )
        return np.asarray([
            [
                self.hypotheses[int(i)].hypothesis.competence,
                self.hypotheses[int(i)].hypothesis.learning_rate,
            ]
            for i in indexes
        ])

    def condition_outcome(self, *, success: bool) -> "WeightedHypothesisBelief":
        weighted = []
        for item in self.hypotheses:
            likelihood = item.hypothesis.competence if success else 1.0 - item.hypothesis.competence
            if (mass := item.probability * likelihood) > 0.0:
                weighted.append((item.hypothesis, mass))
        normalizer = sum(mass for _, mass in weighted)
        if normalizer <= 0.0:
            raise ValueError(f"observation success={success} has zero probability")
        return type(self)(
            hypotheses=tuple(
                WeightedHypothesis(hypothesis=h, probability=m / normalizer) for h, m in weighted
            )
        )

    def condition_learning_rate(
        self, *, observed_learning_rate: float
    ) -> "WeightedHypothesisBelief":
        scale = 0.02
        masses = [
            item.probability
            / (1.0 + ((observed_learning_rate - item.hypothesis.learning_rate) / scale) ** 2 / 4.0)
            ** 2.5
            for item in self.hypotheses
        ]
        normalizer = sum(masses)
        return type(self)(
            hypotheses=tuple(
                WeightedHypothesis(hypothesis=item.hypothesis, probability=mass / normalizer)
                for item, mass in zip(self.hypotheses, masses, strict=True)
                if mass > 0.0
            )
        )

    def refit(self, *, training_examples: int) -> "WeightedHypothesisBelief":
        assert training_examples >= 0
        if training_examples == 0:
            return self
        return type(self)(
            hypotheses=tuple(
                WeightedHypothesis(
                    hypothesis=SkillHypothesis(
                        competence=min(
                            COMPETENCE_MAX,
                            item.hypothesis.competence
                            + item.hypothesis.learning_rate * training_examples,
                        ),
                        learning_rate=item.hypothesis.learning_rate,
                    ),
                    probability=item.probability,
                )
                for item in self.hypotheses
            )
        )

    def diagnostics(self) -> dict[str, object]:
        return {"representation": "weighted_hypotheses", "num_hypotheses": len(self.hypotheses)}

    def signature(self) -> tuple[object, ...]:
        return tuple(
            (item.hypothesis.competence, item.hypothesis.learning_rate, item.probability)
            for item in self.hypotheses
        )
