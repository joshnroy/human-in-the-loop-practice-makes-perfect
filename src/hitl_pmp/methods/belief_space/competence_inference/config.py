"""Shared model priors and numerical settings for the competence experiment."""

import math

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class InferenceConfig(BaseModel):
    """The same probabilistic specification is used by particles and the grid."""

    model_config = ConfigDict(frozen=True)

    competence_bins: int = Field(default=25, ge=2)
    learning_rate_bins: int = Field(default=16, ge=2)
    eta_max: float = Field(default=0.15, gt=0.0, le=1.0, allow_inf_nan=False)
    sigma_competence: float = Field(default=0.03, ge=0.0, allow_inf_nan=False)
    sigma_eta: float = Field(default=0.005, ge=0.0, allow_inf_nan=False)
    learning_rate_decay: float = Field(default=0.9, ge=0.0, le=1.0)
    initial_competence_alpha: float = Field(default=2.0, gt=0.0, allow_inf_nan=False)
    initial_competence_beta: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    initial_eta_sigma: float = Field(default=0.05, gt=0.0, allow_inf_nan=False)
    phi_initial: tuple[float, ...] = tuple(index / 10 for index in range(11))
    phi_plateau: tuple[float, ...] = tuple(index / 10 for index in range(11))
    phi_rates: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8)
    phi_concentrations: tuple[float, ...] = (6.0, 24.0, 96.0)
    resample_ess_fraction: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_curve_prior(self) -> Self:
        for values in (self.phi_initial, self.phi_plateau):
            if not values or any(
                not math.isfinite(value) or not 0 <= value <= 1 for value in values
            ):
                raise ValueError("curve initial and plateau values must lie in [0, 1]")
        if not any(
            plateau >= initial for initial in self.phi_initial for plateau in self.phi_plateau
        ):
            raise ValueError("the curve prior has no initial <= plateau configuration")
        if not self.phi_rates or any(
            not math.isfinite(rate) or rate < 0 for rate in self.phi_rates
        ):
            raise ValueError("curve rates must be finite and nonnegative")
        if not self.phi_concentrations or any(
            not math.isfinite(value) or value <= 2 for value in self.phi_concentrations
        ):
            raise ValueError("curve concentrations must exceed two")
        return self
