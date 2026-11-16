"""Finite weighted-hypothesis skill belief."""

import math
from typing import Any, ClassVar, Final

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, field_validator, model_serializer, model_validator

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

Column = NDArray[np.float64]


class WeightedHypothesisBelief(SkillBelief):
    """Hypothesis i is (competences[i], learning_rates[i], costs[i]) with probabilities[i].

    Stored as one read-only float64 array per quantity, not one model per hypothesis nor
    a tuple of Python floats: a search keeps a posterior per open node, and as four
    tuples a 441-hypothesis grid was ~61 kB. A column an update leaves unchanged is the
    parent's own array -- conditioning changes only the weights, refitting only the
    competences -- and every zero cost column of one length is one shared array, so a
    conditioned posterior costs one new array. Read-only is what makes sharing safe.

    The arithmetic is unchanged to the bit: elementwise numpy is IEEE double like
    Python's floats, every sum is still Python's left-to-right `sum` over a list (never
    numpy's pairwise one), and powers are still Python's. Equality, hashing and the
    digest treat -0.0 and 0.0 as equal, as the tuples did. It is still built from, read
    as (`hypotheses`) and serialised as weighted hypotheses, so callers and logs are
    unchanged; the field bounds are those of `SkillHypothesis` and `WeightedHypothesis`.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    competences: Column
    learning_rates: Column
    costs: Column
    probabilities: Column

    _zero_columns: ClassVar[dict[int, Column]] = {}

    @model_validator(mode="before")
    @classmethod
    def from_hypotheses(cls, data: Any) -> Any:  # noqa: PLR0917 -- pydantic's signature
        # The serialised form (and older checkpoints): a list of weighted hypotheses.
        if not isinstance(data, dict) or "hypotheses" not in data:
            return data
        return cls.fields_of(hypotheses=data["hypotheses"])

    @field_validator("competences", "learning_rates", "costs", "probabilities", mode="before")
    @classmethod
    def as_column(cls, value: Any) -> Column:  # noqa: PLR0917 -- pydantic's signature
        """A read-only float64 vector: a read-only one is taken as is (so shared), any
        other value copied once and frozen."""
        if (
            isinstance(value, np.ndarray)
            and value.dtype == np.float64
            and value.ndim == 1
            and not value.flags.writeable
        ):
            return value
        column = np.array(value, dtype=np.float64)
        if column.ndim != 1:
            raise ValueError("a hypothesis column must be one-dimensional")
        return WeightedHypothesisBelief.frozen(column=column)

    @staticmethod
    def frozen(*, column: Column) -> Column:
        """`column`, which nothing else holds, made read-only."""
        column.flags.writeable = False
        return column

    @staticmethod
    def zeros(*, size: int) -> Column:
        """The one read-only zero column of this length."""
        column = WeightedHypothesisBelief._zero_columns.get(size)
        if column is None:
            column = WeightedHypothesisBelief.frozen(column=np.zeros(size))
            WeightedHypothesisBelief._zero_columns[size] = column
        return column

    @classmethod
    def of(cls, *, hypotheses: Any) -> "WeightedHypothesisBelief":
        return cls.model_validate(cls.fields_of(hypotheses=hypotheses))

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
        n = len(self.probabilities)
        if not n:
            raise ValueError("weighted beliefs require hypotheses")
        if not len(self.competences) == len(self.learning_rates) == len(self.costs) == n:
            raise ValueError("every hypothesis needs a competence, learning rate and cost")
        p = self.probabilities.tolist()
        if n < VECTORISE_FROM:
            competences_ok = all(
                COMPETENCE_MIN <= c <= COMPETENCE_MAX for c in self.competences.tolist()
            )
            rates_ok = all(
                LEARNING_RATE_MIN <= r <= LEARNING_RATE_MAX for r in self.learning_rates.tolist()
            )
            costs_ok = all(math.isfinite(k) and k >= 0.0 for k in self.costs.tolist())
            probabilities_ok = all(0.0 < q <= 1.0 for q in p)
        else:
            c, r = self.competences, self.learning_rates
            k, q = self.costs, self.probabilities
            competences_ok = bool(np.all((c >= COMPETENCE_MIN) & (c <= COMPETENCE_MAX)))
            rates_ok = bool(np.all((r >= LEARNING_RATE_MIN) & (r <= LEARNING_RATE_MAX)))
            costs_ok = bool(np.all(np.isfinite(k) & (k >= 0.0)))
            probabilities_ok = bool(np.all((q > 0.0) & (q <= 1.0)))
        if not competences_ok:
            raise ValueError("competences must lie in [0, 1]")
        if not rates_ok:
            raise ValueError("learning rates must lie in [0, 1]")
        if not costs_ok:
            raise ValueError("costs must be finite and non-negative")
        if not probabilities_ok:
            raise ValueError("hypothesis probabilities must lie in (0, 1]")
        # np.isclose(total, 1.0) with its default tolerances, without its per-call cost.
        if not abs(sum(p) - 1.0) <= SUM_ATOL + SUM_RTOL * 1.0:
            raise ValueError("hypothesis probabilities must sum to one")
        return self

    def __eq__(self, other: object) -> bool:  # noqa: PLR0917 -- a dunder
        if not isinstance(other, WeightedHypothesisBelief):
            return NotImplemented
        return type(other) is type(self) and all(
            np.array_equal(getattr(self, name), getattr(other, name))
            for name in ("probabilities", "competences", "learning_rates", "costs")
        )

    def __hash__(self) -> int:
        # Adding 0.0 turns -0.0 into 0.0, which the equality above does not tell apart.
        return hash(((self.probabilities + 0.0).tobytes(), (self.competences + 0.0).tobytes()))

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
                self.competences.tolist(),
                self.learning_rates.tolist(),
                self.costs.tolist(),
                self.probabilities.tolist(),
                strict=True,
            )
        )

    @classmethod
    def broad_prior(cls, *, points_per_dimension: int = 21) -> "WeightedHypothesisBelief":
        assert points_per_dimension >= 2
        values = np.linspace(0.0, 1.0, points_per_dimension)
        probability = 1.0 / points_per_dimension**2
        size = points_per_dimension**2
        return cls(
            competences=np.repeat(values, points_per_dimension),
            learning_rates=np.tile(values, points_per_dimension),
            costs=WeightedHypothesisBelief.zeros(size=size),
            probabilities=np.full(size, probability),
        )

    def mean_competence(self) -> float:
        return self.expectation(values=self.competences)

    def competence_outcome_probability(self, *, successes: int, failures: int) -> float:
        assert successes >= 0 and failures >= 0
        return sum(
            p * c**successes * (1 - c) ** failures
            for c, p in zip(self.competences.tolist(), self.probabilities.tolist(), strict=True)
        )

    def mean_learning_rate(self) -> float:
        return self.expectation(values=self.learning_rates)

    def expectation(self, *, values: Column) -> float:
        # Python's left-to-right sum, never numpy's pairwise one: the same float as the
        # per-hypothesis sum this replaced. The products are equal either way.
        if len(values) < VECTORISE_FROM:
            return sum(
                p * v for p, v in zip(self.probabilities.tolist(), values.tolist(), strict=True)
            )
        return sum((self.probabilities * values).tolist())

    def sample(self, *, rng: np.random.Generator, count: int) -> np.ndarray:
        if len(self.probabilities) > 1:
            indexes = np.atleast_1d(
                rng.choice(len(self.probabilities), size=count, p=self.probabilities)
            )
            # One row per hypothesis, then index: a search node draws thousands of samples.
            return np.column_stack((self.competences, self.learning_rates))[indexes]
        # A known curve. `rng.choice(1, size=count, p=p)` raises as below, otherwise draws
        # `count` uniforms and picks row 0 for every one; draw them too, so the rng ends
        # where it did, and skip choice's per-call cost.
        if abs(float(self.probabilities[0]) - 1.0) > CHOICE_ATOL:
            raise ValueError("probabilities do not sum to 1")
        rng.random(count)
        rows = np.empty((count, 2))
        rows[:, 0] = self.competences[0]
        rows[:, 1] = self.learning_rates[0]
        return rows

    def condition_outcome(self, *, success: bool) -> "WeightedHypothesisBelief":
        keep: Any
        if len(self.probabilities) < VECTORISE_FROM:
            mass = [
                p * (c if success else 1.0 - c)
                for c, p in zip(self.competences.tolist(), self.probabilities.tolist(), strict=True)
            ]
            keep = [index for index, m in enumerate(mass) if m > 0.0]
            normalizer = sum(mass[index] for index in keep)
            if normalizer <= 0.0:
                raise ValueError(f"observation success={success} has zero probability")
            probabilities = self.frozen(
                column=np.array([mass[index] / normalizer for index in keep])
            )
            kept_all = len(keep) == len(mass)
        else:
            likelihood = self.competences if success else 1.0 - self.competences
            masses = self.probabilities * likelihood
            keep = masses > 0.0
            masses = masses[keep]
            normalizer = sum(masses.tolist())
            if normalizer <= 0.0:
                raise ValueError(f"observation success={success} has zero probability")
            probabilities = self.frozen(column=masses / normalizer)
            kept_all = bool(keep.all())
        if kept_all:
            # Only the weights move: every other column is the parent's own.
            return type(self)(
                competences=self.competences,
                learning_rates=self.learning_rates,
                costs=self.costs,
                probabilities=probabilities,
            )
        kept = len(probabilities)
        return type(self)(
            competences=self.frozen(column=self.competences[keep]),
            learning_rates=self.frozen(column=self.learning_rates[keep]),
            costs=(
                self.zeros(size=kept)
                if self.costs is self.zeros(size=len(self.costs))
                else self.frozen(column=self.costs[keep])
            ),
            probabilities=probabilities,
        )

    def refit(self, *, training_examples: int) -> "WeightedHypothesisBelief":
        assert training_examples >= 0
        if training_examples == 0:
            return self
        if len(self.probabilities) < VECTORISE_FROM:
            competences = np.array([
                min(COMPETENCE_MAX, c + r * training_examples)
                for c, r in zip(
                    self.competences.tolist(), self.learning_rates.tolist(), strict=True
                )
            ])
        else:
            competences = np.minimum(
                COMPETENCE_MAX, self.competences + self.learning_rates * training_examples
            )
        return type(self)(
            competences=self.frozen(column=competences),
            learning_rates=self.learning_rates,
            # A refit hypothesis has always carried the default cost.
            costs=self.zeros(size=len(self.costs)),
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
        return tuple(
            zip(
                self.competences.tolist(),
                self.learning_rates.tolist(),
                self.probabilities.tolist(),
                strict=True,
            )
        )

    def signature_columns(self) -> tuple[Column, Column, Column]:
        """`signature()` by column, for `BeliefDigest`, which encodes it to the same bytes
        without building a row of Python floats per hypothesis. The first two columns
        are the ones updates share between beliefs."""
        return self.competences, self.learning_rates, self.probabilities
