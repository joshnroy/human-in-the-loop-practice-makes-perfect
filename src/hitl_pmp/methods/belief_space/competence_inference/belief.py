"""Immutable competence posteriors with a separate, unchanged execution-cost filter."""

from __future__ import annotations

import numpy as np
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from hitl_pmp.methods.belief_space.tossing3d_particle_filter import make_rng
from hitl_pmp.methods.belief_space.types.particle_filter_belief import (
    ParticleFilterBelief,
    create_fixed_performance_cost_prior,
)
from hitl_pmp.methods.belief_space.types.skill_belief import SkillBelief

from .config import InferenceConfig
from .models import (
    CompetenceModel,
    InferenceEngine,
    global_curve_learning_rate,
    grid_predict,
    grid_prior,
    grid_support,
    particle_prior,
    particle_transition,
)

FLOAT_DTYPE = np.dtype("<f8")
INDEX_DTYPE = np.dtype("<i8")


class BayesianSkillBelief(SkillBelief):
    """A filtered cycle state. Arrays are immutable byte buffers, including ancestry.

    S/F observations condition only competence. The learning rate is latent and
    is learned through its joint posterior with competence, never a constructed
    slope observation. Execution costs use the existing independent cost filter.
    """

    model_config = ConfigDict(frozen=True, ser_json_bytes="base64", val_json_bytes="base64")

    model_name: CompetenceModel
    engine: InferenceEngine
    config: InferenceConfig = InferenceConfig()
    seed: int = Field(ge=0)
    state_count: int = Field(ge=1)
    latent_values: bytes
    state_weights: bytes
    parent_indices: bytes = b""
    cost_belief: ParticleFilterBelief
    cycle_index: int = Field(default=0, ge=0)
    total_training_examples: int = Field(default=0, ge=0)
    incoming_training_examples: int = Field(default=0, ge=0)
    cycle_successes: int = Field(default=0, ge=0)
    cycle_failures: int = Field(default=0, ge=0)
    resampling_count: int = Field(default=0, ge=0)
    process_transition_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_distribution(self) -> Self:
        values, weights = self.arrays()
        if len(values) != self.state_count or weights.shape != (self.state_count,):
            raise ValueError("state buffers do not match state_count")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
            raise ValueError("state values and weights must be finite")
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("state weights must be nonnegative and sum to one")
        competence = values[:, -1] if self.model_name == "global_curve" else values[:, 0]
        if np.any((competence < 0) | (competence > 1)):
            raise ValueError("competence must lie in [0, 1]")
        if self.model_name == "global_curve":
            if np.any((values[:, 0] < 0) | (values[:, 0] > values[:, 1]) | (values[:, 1] > 1)):
                raise ValueError("curve parameters must satisfy 0 <= phi0 <= phi1 <= 1")
            if np.any(values[:, 2] < 0) or np.any(values[:, 3] <= 2):
                raise ValueError("curve rates must be nonnegative and concentration > 2")
        elif np.any((values[:, 1] < 0) | (values[:, 1] > self.config.eta_max)):
            raise ValueError("learning rates must respect the shared cap")
        if self.engine == "particle":
            ancestors = self.ancestors()
            if ancestors.shape != (self.state_count,) or np.any(
                (ancestors < 0) | (ancestors >= self.state_count)
            ):
                raise ValueError("invalid particle parent indices")
        return self

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        columns = 5 if self.model_name == "global_curve" else 2
        return (
            np.frombuffer(self.latent_values, dtype=FLOAT_DTYPE).reshape(-1, columns),
            np.frombuffer(self.state_weights, dtype=FLOAT_DTYPE),
        )

    def ancestors(self) -> np.ndarray:
        return np.frombuffer(self.parent_indices, dtype=INDEX_DTYPE)

    def competence_values(self) -> np.ndarray:
        values, _ = self.arrays()
        return values[:, -1] if self.model_name == "global_curve" else values[:, 0]

    def learning_rate_values(self) -> np.ndarray:
        values, _ = self.arrays()
        if self.model_name == "global_curve":
            return global_curve_learning_rate(
                phi=values[:, :4], training_examples=self.total_training_examples
            )
        return values[:, 1]

    def mean_competence(self) -> float:
        _, weights = self.arrays()
        return float(weights @ self.competence_values())

    def mean_learning_rate(self) -> float:
        _, weights = self.arrays()
        return float(weights @ self.learning_rate_values())

    def mean_cost(self) -> float:
        return self.cost_belief.mean_cost()

    def sample(self, *, rng: np.random.Generator, count: int) -> np.ndarray:
        if count < 0:
            raise ValueError("sample count must be nonnegative")
        _, weights = self.arrays()
        selected = rng.choice(self.state_count, size=count, p=weights)
        return np.column_stack((
            self.competence_values()[selected],
            self.learning_rate_values()[selected],
            self.cost_belief.sample(rng=rng, count=count)[:, 2],
        ))

    def condition_outcome(self, *, success: bool) -> Self:
        values, weights = self.arrays()
        competence = self.competence_values()
        masses = weights * (competence if success else 1.0 - competence)
        normalizer = float(masses.sum())
        if normalizer <= 0 or not np.isfinite(normalizer):
            raise ValueError("S/F observation has zero probability under the represented posterior")
        posterior = masses / normalizer
        updates: dict[str, object] = {
            "cycle_successes": self.cycle_successes + int(success),
            "cycle_failures": self.cycle_failures + int(not success),
        }
        if (
            self.engine == "particle"
            and 1.0 / float(posterior @ posterior)
            < self.config.resample_ess_fraction * self.state_count
        ):
            rng = make_rng(seed=self.seed, stream=self.resampling_count, tag=0x434F4D50524553)
            positions = (rng.random() + np.arange(self.state_count)) / self.state_count
            cumulative = np.cumsum(posterior)
            cumulative[-1] = 1.0
            selected = np.searchsorted(cumulative, positions, side="right")
            updates.update({
                "latent_values": np.ascontiguousarray(
                    values[selected], dtype=FLOAT_DTYPE
                ).tobytes(),
                "parent_indices": np.ascontiguousarray(
                    self.ancestors()[selected], dtype=INDEX_DTYPE
                ).tobytes(),
                "resampling_count": self.resampling_count + 1,
            })
            posterior = np.full(self.state_count, 1.0 / self.state_count)
        updates["state_weights"] = np.ascontiguousarray(posterior, dtype=FLOAT_DTYPE).tobytes()
        return self.model_copy(update=updates)

    def condition_execution(self, *, success: bool, observed_cost: float) -> Self:
        return self.condition_outcome(success=success).condition_cost(observed_cost=observed_cost)

    def condition_cost(self, *, observed_cost: float) -> Self:
        return self.model_copy(
            update={"cost_belief": self.cost_belief.condition_cost(observed_cost=observed_cost)}
        )

    def condition_learning_rate(self, *, observed_learning_rate: float) -> Self:
        del observed_learning_rate
        raise ValueError(
            "learning rate is inferred from S/F observations; direct observations are unsupported"
        )

    def advance_learning_rate(self, *, process_noise_std: float) -> Self:
        """Compatibility hook: refit already applies the full model transition once."""
        del process_noise_std
        return self

    def refit(self, *, training_examples: int) -> Self:
        """Pure forecast used by search, with no transition at zero added examples."""
        if training_examples < 0:
            raise ValueError("training_examples must be nonnegative")
        if training_examples == 0:
            return self
        return self.advance_cycle(training_examples=training_examples)

    def advance_cycle(self, *, training_examples: int) -> Self:
        """Start a recorded cycle, resetting evidence/ancestry even without training.

        Save this cycle's filtered belief in the real-run history BEFORE calling
        this method. A zero-example boundary changes only bookkeeping, with an
        identity latent transition: no fictitious process noise or learning.
        """
        if training_examples < 0:
            raise ValueError("training_examples must be nonnegative")
        values, weights = self.arrays()
        total = self.total_training_examples + training_examples
        updates: dict[str, object] = {
            "cycle_index": self.cycle_index + 1,
            "total_training_examples": total,
            "incoming_training_examples": training_examples,
            "cycle_successes": 0,
            "cycle_failures": 0,
            "process_transition_count": self.process_transition_count + int(training_examples > 0),
        }
        if self.engine == "particle":
            updates["parent_indices"] = np.arange(self.state_count, dtype=INDEX_DTYPE).tobytes()
        if training_examples == 0:
            return self.model_copy(update=updates)
        if self.engine == "grid":
            projected = grid_predict(
                model=self.model_name,
                config=self.config,
                weights=weights,
                training_examples=training_examples,
                total_training_examples=total,
            )
            updates["state_weights"] = np.ascontiguousarray(
                projected / projected.sum(), dtype=FLOAT_DTYPE
            ).tobytes()
        else:
            rng = make_rng(seed=self.seed, stream=self.process_transition_count, tag=0x434F4D505452)
            projected_values = particle_transition(
                model=self.model_name,
                config=self.config,
                values=values,
                training_examples=training_examples,
                total_training_examples=total,
                rng=rng,
            )
            updates["latent_values"] = np.ascontiguousarray(
                projected_values, dtype=FLOAT_DTYPE
            ).tobytes()
        return self.model_copy(update=updates)

    def diagnostics(self) -> dict[str, object]:
        values, weights = self.arrays()
        result: dict[str, object] = {
            "representation": self.engine,
            "competence_model": self.model_name,
            "num_states": self.state_count,
            "cycle_index": self.cycle_index,
            "total_training_examples": self.total_training_examples,
            "cycle_successes": self.cycle_successes,
            "cycle_failures": self.cycle_failures,
            "effective_sample_size": float(1.0 / (weights @ weights)),
            "resampling_count": self.resampling_count,
            "process_transition_count": self.process_transition_count,
            "cost": self.cost_belief.diagnostics(),
        }
        if self.engine == "particle":
            result["unique_parent_particles"] = len(np.unique(self.ancestors()))
            if self.model_name == "global_curve":
                result["unique_phi_configurations"] = len(np.unique(values[:, :4], axis=0))
        return result

    def signature(self) -> tuple[object, ...]:
        return (
            self.model_name,
            self.engine,
            self.config,
            self.seed,
            self.cycle_index,
            self.total_training_examples,
            self.incoming_training_examples,
            self.cycle_successes,
            self.cycle_failures,
            self.resampling_count,
            self.process_transition_count,
            self.latent_values,
            self.state_weights,
            self.parent_indices,
            self.cost_belief.signature(),
        )


def create_bayesian_prior(
    *,
    model: CompetenceModel,
    engine: InferenceEngine,
    seed: int,
    num_particles: int,
    config: InferenceConfig | None = None,
) -> BayesianSkillBelief:
    """Construct matched model priors and the same independent cost marginal."""
    if model not in ("global_curve", "local_trend") or engine not in ("particle", "grid"):
        raise ValueError("unknown competence model or inference engine")
    if num_particles < 1 or seed < 0:
        raise ValueError("num_particles must be positive and seed nonnegative")
    settings = config or InferenceConfig()
    if engine == "particle":
        values = particle_prior(
            model=model, config=settings, rng=np.random.default_rng(seed), count=num_particles
        )
        weights = np.full(num_particles, 1.0 / num_particles)
        parents = np.arange(num_particles, dtype=INDEX_DTYPE).tobytes()
    else:
        values = grid_support(model=model, config=settings)
        weights = grid_prior(model=model, config=settings)
        parents = b""
    return BayesianSkillBelief(
        model_name=model,
        engine=engine,
        config=settings,
        seed=seed,
        state_count=len(weights),
        latent_values=np.ascontiguousarray(values, dtype=FLOAT_DTYPE).tobytes(),
        state_weights=np.ascontiguousarray(weights, dtype=FLOAT_DTYPE).tobytes(),
        parent_indices=parents,
        cost_belief=create_fixed_performance_cost_prior(
            num_particles=num_particles, seed=seed, competence=1.0, learning_rate=0.0
        ),
    )
