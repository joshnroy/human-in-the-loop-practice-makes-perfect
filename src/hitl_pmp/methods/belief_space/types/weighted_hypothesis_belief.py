"""Finite weighted-hypothesis skill belief."""

import math
from typing import Any, Final

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

# Below this many hypotheses a Python loop beats numpy's per-call overhead; a known curve
# is one hypothesis, a prior grid hundreds. Both paths give the same floats.
VECTORISE_FROM: Final = 32
# np.isclose's defaults, and the tolerance rng.choice checks probabilities against.
SUM_ATOL: Final = 1e-08
SUM_RTOL: Final = 1e-05
CHOICE_ATOL: Final = float(np.sqrt(np.finfo(np.float64).eps))


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
        if n < VECTORISE_FROM:
            competences_ok = all(COMPETENCE_MIN <= c <= COMPETENCE_MAX for c in self.competences)
            rates_ok = all(LEARNING_RATE_MIN <= r <= LEARNING_RATE_MAX for r in self.learning_rates)
            costs_ok = all(math.isfinite(k) and k >= 0.0 for k in self.costs)
            probabilities_ok = all(0.0 < p <= 1.0 for p in self.probabilities)
        else:
            c, r = np.asarray(self.competences), np.asarray(self.learning_rates)
            k, p = np.asarray(self.costs), np.asarray(self.probabilities)
            competences_ok = bool(np.all((c >= COMPETENCE_MIN) & (c <= COMPETENCE_MAX)))
            rates_ok = bool(np.all((r >= LEARNING_RATE_MIN) & (r <= LEARNING_RATE_MAX)))
            costs_ok = bool(np.all(np.isfinite(k) & (k >= 0.0)))
            probabilities_ok = bool(np.all((p > 0.0) & (p <= 1.0)))
        if not competences_ok:
            raise ValueError("competences must lie in [0, 1]")
        if not rates_ok:
            raise ValueError("learning rates must lie in [0, 1]")
        if not costs_ok:
            raise ValueError("costs must be finite and non-negative")
        if not probabilities_ok:
            raise ValueError("hypothesis probabilities must lie in (0, 1]")
        # np.isclose(total, 1.0) with its default tolerances, without its per-call cost.
        if not abs(sum(self.probabilities) - 1.0) <= SUM_ATOL + SUM_RTOL * 1.0:
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
        return self.expectation(values=self.competences)

    def competence_outcome_probability(self, *, successes: int, failures: int) -> float:
        assert successes >= 0 and failures >= 0
        return sum(
            p * c**successes * (1 - c) ** failures
            for c, p in zip(self.competences, self.probabilities, strict=True)
        )

    def mean_learning_rate(self) -> float:
        return self.expectation(values=self.learning_rates)

    def expectation(self, *, values: tuple[float, ...]) -> float:
        # Python's left-to-right sum, never numpy's pairwise one: the same float as the
        # per-hypothesis sum this replaced. The products are equal either way.
        if len(values) < VECTORISE_FROM:
            return sum(p * v for p, v in zip(self.probabilities, values, strict=True))
        return sum((np.asarray(self.probabilities) * np.asarray(values)).tolist())

    def sample(self, *, rng: np.random.Generator, count: int) -> np.ndarray:
        if len(self.probabilities) > 1:
            indexes = np.atleast_1d(
                rng.choice(len(self.probabilities), size=count, p=list(self.probabilities))
            )
            # One row per hypothesis, then index: a search node draws thousands of samples.
            return np.column_stack((self.competences, self.learning_rates))[indexes]
        # A known curve. `rng.choice(1, size=count, p=p)` raises as below, otherwise draws
        # `count` uniforms and picks row 0 for every one; draw them too, so the rng ends
        # where it did, and skip choice's per-call cost.
        if abs(self.probabilities[0] - 1.0) > CHOICE_ATOL:
            raise ValueError("probabilities do not sum to 1")
        rng.random(count)
        rows = np.empty((count, 2))
        rows[:, 0] = self.competences[0]
        rows[:, 1] = self.learning_rates[0]
        return rows

    def condition_outcome(self, *, success: bool) -> "WeightedHypothesisBelief":
        if len(self.probabilities) < VECTORISE_FROM:
            kept = [
                (c, r, k, mass)
                for c, r, k, p in zip(
                    self.competences,
                    self.learning_rates,
                    self.costs,
                    self.probabilities,
                    strict=True,
                )
                if (mass := p * (c if success else 1.0 - c)) > 0.0
            ]
            normalizer = sum(mass for *_, mass in kept)
            if normalizer <= 0.0:
                raise ValueError(f"observation success={success} has zero probability")
            return type(self)(
                competences=tuple(c for c, *_ in kept),
                learning_rates=tuple(r for _, r, _, _ in kept),
                costs=tuple(k for _, _, k, _ in kept),
                probabilities=tuple(mass / normalizer for *_, mass in kept),
            )
        competences = np.asarray(self.competences)
        likelihood = competences if success else 1.0 - competences
        mass = np.asarray(self.probabilities) * likelihood
        keep = mass > 0.0
        masses = mass[keep]
        normalizer = sum(masses.tolist())
        if normalizer <= 0.0:
            raise ValueError(f"observation success={success} has zero probability")
        return type(self)(
            competences=tuple(competences[keep].tolist()),
            learning_rates=tuple(np.asarray(self.learning_rates)[keep].tolist()),
            costs=tuple(np.asarray(self.costs)[keep].tolist()),
            probabilities=tuple((masses / normalizer).tolist()),
        )

    def refit(self, *, training_examples: int) -> "WeightedHypothesisBelief":
        assert training_examples >= 0
        if training_examples == 0:
            return self
        if len(self.probabilities) < VECTORISE_FROM:
            competences = tuple(
                min(COMPETENCE_MAX, c + r * training_examples)
                for c, r in zip(self.competences, self.learning_rates, strict=True)
            )
        else:
            rates = np.asarray(self.learning_rates)
            competences = tuple(
                np.minimum(
                    COMPETENCE_MAX, np.asarray(self.competences) + rates * training_examples
                ).tolist()
            )
        return type(self)(
            competences=competences,
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
