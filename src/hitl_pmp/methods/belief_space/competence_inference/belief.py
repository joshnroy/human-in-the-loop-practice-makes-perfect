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
        elif np.any(values[:, 1] < 0):
            raise ValueError("learning rates must be nonnegative")
        elif self.engine == "grid" and np.any(values[:, 1] > self.config.eta_max):
            # Grid states are axis points by construction, so a rate above eta_max is
            # a corrupted buffer; particle rates are uncapped, notebook-exact.
            raise ValueError("grid learning rates must lie on the [0, eta_max] axis")
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

    def competence_outcome_probability(self, *, successes: int, failures: int) -> float:
        assert successes >= 0 and failures >= 0
        _, weights = self.arrays()
        competence = self.competence_values()
        return float(weights @ (competence**successes * (1 - competence) ** failures))

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
        """Multiply one S/F likelihood into the weights, never resampling.

        The notebook ingests a cycle as one (m, n, s) batch: weight by c^s (1-c)^(n-s),
        then resample once. Per-outcome weighting multiplies to exactly that batch
        likelihood, and the single systematic resample happens at the real cycle
        boundary (`advance_cycle`), so the filter is the notebook's.
        """
        _, weights = self.arrays()
        competence = self.competence_values()
        masses = weights * (competence if success else 1.0 - competence)
        normalizer = float(masses.sum())
        if normalizer <= 0 or not np.isfinite(normalizer):
            raise ValueError("S/F observation has zero probability under the represented posterior")
        posterior = masses / normalizer
        return self.model_copy(
            update={
                "cycle_successes": self.cycle_successes + int(success),
                "cycle_failures": self.cycle_failures + int(not success),
                "state_weights": np.ascontiguousarray(posterior, dtype=FLOAT_DTYPE).tobytes(),
            }
        )

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
        """The notebook's `extrapolate`: a pure predict step, including at zero examples.

        There is deliberately no zero-example identity (the notebook always predicts
        across a boundary). The identity used to be here to suppress a
        spurious-practice-value pathology in search; notebook semantics were chosen
        over it, so watch search values for that pathology rather than re-adding it.

        Particles are pushed through the transition with their weights intact and
        without resampling, so an imagined branch carries no resampling noise; the
        process noise comes from a stream keyed on (seed, process_transition_count),
        so the same forecast of the same belief is bit-identical.
        """
        if training_examples < 0:
            raise ValueError("training_examples must be nonnegative")
        return self.advance_cycle(training_examples=training_examples, resample=False)

    def advance_cycle(self, *, training_examples: int, resample: bool = True) -> Self:
        """Start a recorded cycle, resetting evidence/ancestry even without training.

        Save this cycle's filtered belief in the real-run history BEFORE calling
        this method. A zero-example boundary applies the notebook's n = 0
        transition -- the noise-only step for Model B (drift 0, decay^0 = 1, both
        noises), a redraw at the unchanged total for Model A.

        Particles are systematically resampled once here, on every real boundary,
        before the predict step: the notebook's weight-then-resample for the cycle
        just ended. `parent_indices` records the selection for ancestral smoothing.
        """
        if training_examples < 0:
            raise ValueError("training_examples must be nonnegative")
        values, weights = self.arrays()
        resampling_count = self.resampling_count
        parents = np.arange(self.state_count, dtype=INDEX_DTYPE)
        if resample and self.engine == "particle":
            rng = make_rng(seed=self.seed, stream=self.resampling_count, tag=0x434F4D50524553)
            positions = (rng.random() + np.arange(self.state_count)) / self.state_count
            cumulative = np.cumsum(weights)
            cumulative[-1] = 1.0
            parents = np.searchsorted(cumulative, positions, side="right").astype(INDEX_DTYPE)
            values = values[parents]
            weights = np.full(self.state_count, 1.0 / self.state_count)
            resampling_count += 1
        total = self.total_training_examples + training_examples
        updates: dict[str, object] = {
            "cycle_index": self.cycle_index + 1,
            "total_training_examples": total,
            "incoming_training_examples": training_examples,
            "cycle_successes": 0,
            "cycle_failures": 0,
            # Every real boundary consumes one process-noise stream now, including
            # the n = 0 step, so idle cycles draw distinct deterministic noise.
            "process_transition_count": self.process_transition_count + 1,
            "resampling_count": resampling_count,
        }
        if self.engine == "particle":
            updates["parent_indices"] = parents.tobytes()
            updates["state_weights"] = np.ascontiguousarray(weights, dtype=FLOAT_DTYPE).tobytes()
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
