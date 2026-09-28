"""Shared model priors and numerical settings for the competence experiment."""

import math
from typing import Literal

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

    def scaled_learning_time(
        self, *, model: Literal["global_curve", "local_trend"], time_scale: float
    ) -> Self:
        """Stretch the model's learning clock without changing actual example counts.

        Values above one predict slower learning. Model A stretches its phi curve;
        Model B stretches learning-rate magnitudes and per-example decay together.
        Competence process noise and all non-learning priors stay unchanged. The
        `eta_max` division here scales only the GRID's discretization range (and,
        with it, where the top bin's absorbed tail begins) -- particle rates carry
        no cap, so for the particle engine the stretch acts through the prior and
        process-noise sigmas and the decay alone.
        """
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be finite and positive")
        if model not in ("global_curve", "local_trend"):
            raise ValueError(f"unknown competence model: {model}")
        if time_scale == 1.0:
            return self
        parameters = self.model_dump()
        if model == "global_curve":
            parameters["phi_rates"] = tuple(rate / time_scale for rate in self.phi_rates)
        else:
            parameters.update(
                eta_max=self.eta_max / time_scale,
                initial_eta_sigma=self.initial_eta_sigma / time_scale,
                sigma_eta=self.sigma_eta / time_scale,
                learning_rate_decay=self.learning_rate_decay ** (1.0 / time_scale),
            )
        return self.model_validate(parameters)

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
