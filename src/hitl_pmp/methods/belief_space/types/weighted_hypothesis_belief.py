"""Finite weighted-hypothesis skill belief."""

from typing import Any

import numpy as np
from pydantic import model_serializer, model_validator

from .skill_belief import (
    COMPETENCE_MAX,
    COMPETENCE_MIN,
    LEARNING_RATE_MAX,
    LEARNING_RATE_MIN,
    SkillBelief,
    SkillHypothesis,
    WeightedHypothesis,
)


class WeightedHypothesisBelief(SkillBelief):
    """Hypothesis i is (competences[i], learning_rates[i], costs[i]) with probabilities[i].

    Stored as one tuple of floats per quantity, not one model per hypothesis: a search
    node refits and conditions beliefs of hundreds of hypotheses, and building that many
    models was most of a prior-spread search. It is still built from, read as
    (`hypotheses`) and serialised as weighted hypotheses, so callers and logs are
    unchanged; the field bounds are those of `SkillHypothesis` and `WeightedHypothesis`.
    """

    competences: tuple[float, ...]
    learning_rates: tuple[float, ...]
    costs: tuple[float, ...]
    probabilities: tuple[float, ...]

    @model_validator(mode="before")
    @classmethod
    def from_hypotheses(cls, data: Any) -> Any:  # noqa: PLR0917 -- pydantic's signature
        # The serialised form (and older checkpoints): a list of weighted hypotheses.
        if not isinstance(data, dict) or "hypotheses" not in data:
            return data
        return cls.fields_of(hypotheses=data["hypotheses"])

    @classmethod
    def of(cls, *, hypotheses: Any) -> "WeightedHypothesisBelief":
        return cls(**cls.fields_of(hypotheses=hypotheses))

    @staticmethod
    def fields_of(*, hypotheses: Any) -> dict[str, tuple[float, ...]]:
        items = [WeightedHypothesis.model_validate(item) for item in hypotheses]
        return {
            "competences": tuple(item.hypothesis.competence for item in items),
            "learning_rates": tuple(item.hypothesis.learning_rate for item in items),
            "costs": tuple(item.hypothesis.cost for item in items),
            "probabilities": tuple(item.probability for item in items),
        }

    @model_validator(mode="after")
    def validate_probabilities(self) -> "WeightedHypothesisBelief":
        if not self.probabilities:
            raise ValueError("weighted beliefs require hypotheses")
        n = len(self.probabilities)
        if not len(self.competences) == len(self.learning_rates) == len(self.costs) == n:
            raise ValueError("every hypothesis needs a competence, learning rate and cost")
        c, r = np.asarray(self.competences), np.asarray(self.learning_rates)
        k, p = np.asarray(self.costs), np.asarray(self.probabilities)
        if not np.all((c >= COMPETENCE_MIN) & (c <= COMPETENCE_MAX)):
            raise ValueError("competences must lie in [0, 1]")
        if not np.all((r >= LEARNING_RATE_MIN) & (r <= LEARNING_RATE_MAX)):
            raise ValueError("learning rates must lie in [0, 1]")
        if not np.all(np.isfinite(k) & (k >= 0.0)):
            raise ValueError("costs must be finite and non-negative")
        if not np.all((p > 0.0) & (p <= 1.0)):
            raise ValueError("hypothesis probabilities must lie in (0, 1]")
        if not np.isclose(sum(self.probabilities), 1.0):
            raise ValueError("hypothesis probabilities must sum to one")
        return self

    @model_serializer(mode="plain")
    def as_hypotheses(self) -> dict[str, Any]:
        return {"hypotheses": [item.model_dump() for item in self.hypotheses]}

    @property
    def hypotheses(self) -> tuple[WeightedHypothesis, ...]:
        return tuple(
            WeightedHypothesis(
                hypothesis=SkillHypothesis(competence=c, learning_rate=r, cost=k), probability=p
            )
            for c, r, k, p in zip(
                self.competences, self.learning_rates, self.costs, self.probabilities, strict=True
            )
        )

    @classmethod
    def broad_prior(cls, *, points_per_dimension: int = 21) -> "WeightedHypothesisBelief":
        assert points_per_dimension >= 2
        values = np.linspace(0.0, 1.0, points_per_dimension)
        probability = 1.0 / points_per_dimension**2
        size = points_per_dimension**2
        return cls(
            competences=tuple(np.repeat(values, points_per_dimension).tolist()),
            learning_rates=tuple(np.tile(values, points_per_dimension).tolist()),
            costs=(0.0,) * size,
            probabilities=(probability,) * size,
        )

    def mean_competence(self) -> float:
        # Python's left-to-right sum, not numpy's pairwise one: the same float as before.
        return sum((np.asarray(self.probabilities) * np.asarray(self.competences)).tolist())

    def competence_outcome_probability(self, *, successes: int, failures: int) -> float:
        assert successes >= 0 and failures >= 0
        return sum(
            p * c**successes * (1 - c) ** failures
            for c, p in zip(self.competences, self.probabilities, strict=True)
        )

    def mean_learning_rate(self) -> float:
        return sum((np.asarray(self.probabilities) * np.asarray(self.learning_rates)).tolist())

    def sample(self, *, rng: np.random.Generator, count: int) -> np.ndarray:
        indexes = np.atleast_1d(
            rng.choice(len(self.probabilities), size=count, p=list(self.probabilities))
        )
        # One row per hypothesis, then index: a search node draws thousands of samples.
        table = np.column_stack((self.competences, self.learning_rates))
        return table[indexes]

    def condition_outcome(self, *, success: bool) -> "WeightedHypothesisBelief":
        competences = np.asarray(self.competences)
        likelihood = competences if success else 1.0 - competences
        mass = np.asarray(self.probabilities) * likelihood
        kept = mass > 0.0
        masses = mass[kept]
        normalizer = sum(masses.tolist())
        if normalizer <= 0.0:
            raise ValueError(f"observation success={success} has zero probability")
        return type(self)(
            competences=tuple(competences[kept].tolist()),
            learning_rates=tuple(np.asarray(self.learning_rates)[kept].tolist()),
            costs=tuple(np.asarray(self.costs)[kept].tolist()),
            probabilities=tuple((masses / normalizer).tolist()),
        )

    def refit(self, *, training_examples: int) -> "WeightedHypothesisBelief":
        assert training_examples >= 0
        if training_examples == 0:
            return self
        rates = np.asarray(self.learning_rates)
        return type(self)(
            competences=tuple(
                np.minimum(
                    COMPETENCE_MAX, np.asarray(self.competences) + rates * training_examples
                ).tolist()
            ),
            learning_rates=self.learning_rates,
            # A refit hypothesis has always carried the default cost.
            costs=(0.0,) * len(self.costs),
            probabilities=self.probabilities,
        )

    def advance_learning_rate(self, *, process_noise_std: float) -> "WeightedHypothesisBelief":
        assert process_noise_std >= 0.0
        if process_noise_std > 0.0:
            raise NotImplementedError("process noise requires a particle-filter belief")
        return self

    def diagnostics(self) -> dict[str, object]:
        return {"representation": "weighted_hypotheses", "num_hypotheses": len(self.probabilities)}

    def signature(self) -> tuple[object, ...]:
        return tuple(zip(self.competences, self.learning_rates, self.probabilities, strict=True))
