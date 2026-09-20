"""Offline backward smoothing of recorded filtered cycle-end posteriors."""

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from .belief import BayesianSkillBelief
from .models import grid_backward


class SmoothedCycle(BaseModel):
    model_config = ConfigDict(frozen=True)

    cycle_index: int
    total_training_examples: int
    filtered_competence: float
    smoothed_competence: float
    filtered_learning_rate: float
    smoothed_learning_rate: float
    competence_std: float
    learning_rate_std: float
    effective_sample_size: float
    unique_ancestors: int | None = None


def _summarize(*, belief: BayesianSkillBelief, weights: np.ndarray) -> SmoothedCycle:
    competence, rate = belief.competence_values(), belief.learning_rate_values()
    c_mean, eta_mean = float(weights @ competence), float(weights @ rate)
    return SmoothedCycle(
        cycle_index=belief.cycle_index,
        total_training_examples=belief.total_training_examples,
        filtered_competence=belief.mean_competence(),
        smoothed_competence=c_mean,
        filtered_learning_rate=belief.mean_learning_rate(),
        smoothed_learning_rate=eta_mean,
        competence_std=float(np.sqrt(weights @ (competence - c_mean) ** 2)),
        learning_rate_std=float(np.sqrt(weights @ (rate - eta_mean) ** 2)),
        effective_sample_size=float(1.0 / (weights @ weights)),
        unique_ancestors=int(np.count_nonzero(weights)) if belief.engine == "particle" else None,
    )


def smooth_history(*, history: Sequence[BayesianSkillBelief]) -> list[SmoothedCycle]:
    """Smooth without changing any filtered state or consuming a live random stream.

    Each entry is saved BEFORE its real cycle boundary. Particles trace ancestry
    through every resampling step, carrying terminal importance weights backwards.
    The grid applies the backward transition with the next cycle's S/F likelihood.
    This is filtering/smoothing of one shared model, not a fit to posterior means.
    """
    if not history:
        return []
    first = history[0]
    for previous, current in zip(history[:-1], history[1:], strict=True):
        if (
            current.model_name,
            current.engine,
            current.config,
            current.seed,
            current.state_count,
        ) != (first.model_name, first.engine, first.config, first.seed, first.state_count):
            raise ValueError("smoothing requires one model, engine, prior configuration and seed")
        if current.cycle_index != previous.cycle_index + 1:
            raise ValueError("smoothing history must contain consecutive cycle boundaries")
        if (
            current.total_training_examples
            != previous.total_training_examples + current.incoming_training_examples
        ):
            raise ValueError("inconsistent training-example counts between cycle boundaries")
    _, terminal_weights = history[-1].arrays()
    smoothed_weights = terminal_weights
    summaries = [_summarize(belief=history[-1], weights=smoothed_weights)]
    backward_message = np.ones(first.state_count)
    for index in range(len(history) - 2, -1, -1):
        previous, current = history[index], history[index + 1]
        if first.engine == "particle":
            smoothed_weights = np.bincount(
                current.ancestors(), weights=smoothed_weights, minlength=previous.state_count
            )
        else:
            competence = current.competence_values()
            with np.errstate(divide="ignore", invalid="ignore"):
                log_likelihood = np.zeros(current.state_count)
                if current.cycle_successes:
                    log_likelihood += current.cycle_successes * np.log(competence)
                if current.cycle_failures:
                    log_likelihood += current.cycle_failures * np.log1p(-competence)
            likelihood = np.exp(log_likelihood - float(np.max(log_likelihood)))
            backward_message = grid_backward(
                model=first.model_name,
                config=first.config,
                message=likelihood * backward_message,
                training_examples=current.incoming_training_examples,
                total_training_examples=current.total_training_examples,
            )
            backward_message /= backward_message.max()
            _, filtered = previous.arrays()
            smoothed_weights = filtered * backward_message
        normalizer = float(smoothed_weights.sum())
        if not np.isfinite(normalizer) or normalizer <= 0:
            raise ValueError("smoothing encountered zero posterior mass")
        smoothed_weights = smoothed_weights / normalizer
        summaries.append(_summarize(belief=previous, weights=smoothed_weights))
    return list(reversed(summaries))
