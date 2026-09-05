"""Common interface and value types for skill beliefs."""

from abc import ABC, abstractmethod
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

COMPETENCE_MIN: Final = 0.0
COMPETENCE_MAX: Final = 1.0
LEARNING_RATE_MIN: Final = 0.0
LEARNING_RATE_MAX: Final = 1.0
COST_MIN: Final = 0.0
COST_MAX: Final = 0.01
COST_OBSERVATION_SCALE: Final = 0.0001


class SkillHypothesis(BaseModel):
    model_config = ConfigDict(frozen=True)
    competence: float = Field(ge=COMPETENCE_MIN, le=COMPETENCE_MAX)
    learning_rate: float = Field(ge=LEARNING_RATE_MIN, le=LEARNING_RATE_MAX)


class WeightedHypothesis(BaseModel):
    model_config = ConfigDict(frozen=True)
    hypothesis: SkillHypothesis
    probability: float = Field(gt=0.0, le=1.0)


class SkillBelief(BaseModel, ABC):
    """Representation-independent posterior over a skill's parameters."""

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def mean_competence(self) -> float: ...
    @abstractmethod
    def mean_learning_rate(self) -> float: ...
    @abstractmethod
    def sample(self, *, rng: np.random.Generator, count: int) -> np.ndarray: ...
    @abstractmethod
    def condition_outcome(self, *, success: bool) -> Self: ...
    @abstractmethod
    def condition_learning_rate(self, *, observed_learning_rate: float) -> Self: ...
    @abstractmethod
    def refit(self, *, training_examples: int) -> Self: ...
    @abstractmethod
    def diagnostics(self) -> dict[str, object]: ...
    @abstractmethod
    def signature(self) -> tuple[object, ...]: ...
