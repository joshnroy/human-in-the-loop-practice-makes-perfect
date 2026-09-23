"""Sampling and fixed-grid operators for the two notebook competence models.

The spec is `docs/from_tom/competence_models.ipynb` (Model A: cell 9, Model B: cells
10-11), and this module now follows it exactly where it used to deviate: the particle
learning-rate prior and transition carry no upper cap (`eta_max` is only the grid's
discretization range, whose top bin absorbs the tail mass), Model A's particle prior
is the notebook's continuous sampler rather than the grid's finite support, the
zero-new-example cycle transition applies the n=0 noise step rather than the
identity, and the mode entering `beta_parameters` is clipped into [1e-3, 1 - 1e-3].
Each of those was a port deviation this change removes.
"""

from functools import lru_cache
from typing import Literal

import numpy as np
from scipy.special import betainc, ndtr

from .config import InferenceConfig

CompetenceModel = Literal["global_curve", "local_trend"]
InferenceEngine = Literal["particle", "grid"]


def global_curve_mode(*, phi: np.ndarray, training_examples: int) -> np.ndarray:
    """Model A's exponential mode curve, with phi constant across cycles."""
    return phi[..., 0] + (phi[..., 1] - phi[..., 0]) * (-np.expm1(-phi[..., 2] * training_examples))


def global_curve_learning_rate(*, phi: np.ndarray, training_examples: int) -> np.ndarray:
    """Derivative of expected competence, accounting for the mode-to-mean shift."""
    return (
        (phi[..., 1] - phi[..., 0])
        * phi[..., 2]
        * np.exp(-phi[..., 2] * training_examples)
        * (phi[..., 3] - 2.0)
        / phi[..., 3]
    )


def beta_parameters(*, phi: np.ndarray, training_examples: int) -> tuple[np.ndarray, np.ndarray]:
    # The notebook's beta_from_mode clips the mode into [1e-3, 1 - 1e-3] before
    # computing the parameters, which keeps flat curves at the extremes proper.
    mode = np.clip(global_curve_mode(phi=phi, training_examples=training_examples), 1e-3, 1 - 1e-3)
    return 1.0 + mode * (phi[..., 3] - 2.0), 1.0 + (1.0 - mode) * (phi[..., 3] - 2.0)


@lru_cache(maxsize=32)
def phi_support(*, config: InferenceConfig) -> np.ndarray:
    """The GRID engine's uniform finite support (the notebook's grid), including flat
    curves. The particle engine no longer draws from these atoms: its prior is the
    notebook's continuous sampler in `particle_prior`."""
    support = np.array(
        [
            (initial, plateau, rate, concentration)
            for initial in config.phi_initial
            for plateau in config.phi_plateau
            if plateau >= initial
            for rate in config.phi_rates
            for concentration in config.phi_concentrations
        ],
        dtype=np.float64,
    )
    support.setflags(write=False)
    return support


def competence_axis(*, model: CompetenceModel, config: InferenceConfig) -> np.ndarray:
    if model == "global_curve":
        return (np.arange(config.competence_bins) + 0.5) / config.competence_bins
    return np.linspace(0.0, 1.0, config.competence_bins)


def learning_rate_axis(*, config: InferenceConfig) -> np.ndarray:
    return np.linspace(0.0, config.eta_max, config.learning_rate_bins)


def bin_edges(*, axis: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.concatenate(([lower], (axis[:-1] + axis[1:]) / 2, [upper]))


@lru_cache(maxsize=64)
def curve_cycle_mass(*, config: InferenceConfig, training_examples: int) -> np.ndarray:
    phi = phi_support(config=config)
    alpha, beta = beta_parameters(phi=phi, training_examples=training_examples)
    edges = np.linspace(0.0, 1.0, config.competence_bins + 1)
    mass = np.diff(betainc(alpha[:, None], beta[:, None], edges[None, :]), axis=1)
    mass = np.maximum(mass, 0.0)
    mass /= mass.sum(axis=1, keepdims=True)
    mass.setflags(write=False)
    return mass


@lru_cache(maxsize=32)
def grid_support(*, model: CompetenceModel, config: InferenceConfig) -> np.ndarray:
    competence = competence_axis(model=model, config=config)
    if model == "global_curve":
        phi = phi_support(config=config)
        values = np.column_stack((
            np.repeat(phi, len(competence), axis=0),
            np.tile(competence, len(phi)),
        ))
    else:
        rate = learning_rate_axis(config=config)
        values = np.column_stack((np.repeat(competence, len(rate)), np.tile(rate, len(competence))))
    values.setflags(write=False)
    return values


def grid_prior(*, model: CompetenceModel, config: InferenceConfig) -> np.ndarray:
    if model == "global_curve":
        return (
            curve_cycle_mass(config=config, training_examples=0) / len(phi_support(config=config))
        ).ravel()
    competence = competence_axis(model=model, config=config)
    rate = learning_rate_axis(config=config)
    c_edges = bin_edges(axis=competence, lower=0.0, upper=1.0)
    c_mass = np.diff(
        betainc(config.initial_competence_alpha, config.initial_competence_beta, c_edges)
    )
    eta_edges = bin_edges(axis=rate, lower=0.0, upper=np.inf)
    eta_mass = np.diff(2.0 * ndtr(eta_edges / config.initial_eta_sigma) - 1.0)
    weights = (c_mass[:, None] * eta_mass[None, :]).ravel()
    return weights / weights.sum()


def particle_prior(
    *, model: CompetenceModel, config: InferenceConfig, rng: np.random.Generator, count: int
) -> np.ndarray:
    if model == "global_curve":
        # The notebook's continuous prior (cell 9 `sample_initial`): the grid atoms
        # are the discretization, not the model. Rate and concentration ranges come
        # from the config's own support extremes.
        phi0 = rng.uniform(0.0, 1.0, size=count)
        phi1 = phi0 + (1.0 - phi0) * rng.uniform(0.0, 1.0, size=count)
        rate_bounds = (min(config.phi_rates), max(config.phi_rates))
        kappa_bounds = (min(config.phi_concentrations), max(config.phi_concentrations))
        curve_rate = np.exp(rng.uniform(np.log(rate_bounds[0]), np.log(rate_bounds[1]), size=count))
        concentration = np.exp(
            rng.uniform(np.log(kappa_bounds[0]), np.log(kappa_bounds[1]), size=count)
        )
        phi = np.column_stack((phi0, phi1, curve_rate, concentration))
        alpha, beta = beta_parameters(phi=phi, training_examples=0)
        return np.column_stack((phi, rng.beta(alpha, beta)))
    competence = rng.beta(
        config.initial_competence_alpha, config.initial_competence_beta, size=count
    )
    # Untruncated half-normal, as the notebook draws it; eta_max caps nothing here.
    rate = np.abs(rng.normal(0.0, config.initial_eta_sigma, size=count))
    return np.column_stack((competence, rate))


def particle_transition(
    *,
    model: CompetenceModel,
    config: InferenceConfig,
    values: np.ndarray,
    training_examples: int,
    total_training_examples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    # No zero-example shortcut: the notebook's engines always predict across a cycle
    # boundary, so n = 0 is the noise-only step (drift 0, decay^0 = 1 for Model B; a
    # redraw at the unchanged total for Model A). The search-side identity at zero
    # lives in `BayesianSkillBelief.refit`, deliberately, not here.
    if model == "global_curve":
        alpha, beta = beta_parameters(phi=values[:, :4], training_examples=total_training_examples)
        return np.column_stack((values[:, :4], rng.beta(alpha, beta)))
    competence = np.clip(
        values[:, 0]
        + training_examples * values[:, 1]
        + rng.normal(0.0, config.sigma_competence, size=len(values)),
        0.0,
        1.0,
    )
    # Floor only, no upper clip: eta' = max(decay^n * eta + noise, 0), notebook-exact.
    rate = np.maximum(
        config.learning_rate_decay**training_examples * values[:, 1]
        + rng.normal(0.0, config.sigma_eta, size=len(values)),
        0.0,
    )
    return np.column_stack((competence, rate))


def gaussian_bin_mass(*, means: np.ndarray, sigma: float, axis: np.ndarray) -> np.ndarray:
    """CDF bin integrals with all clipped tail mass assigned to endpoint states."""
    edges = bin_edges(axis=axis, lower=-np.inf, upper=np.inf)
    if sigma == 0.0:
        # At zero process noise, quantize the deterministic clipped destination.
        index = np.searchsorted(edges[1:-1], means, side="right")
        return np.eye(len(axis))[index]
    mass = np.diff(ndtr((edges - means[..., None]) / sigma), axis=-1)
    mass = np.maximum(mass, 0.0)
    return mass / mass.sum(axis=-1, keepdims=True)


@lru_cache(maxsize=64)
def local_transition_kernels(
    *, config: InferenceConfig, training_examples: int
) -> tuple[np.ndarray, np.ndarray]:
    competence = competence_axis(model="local_trend", config=config)
    rate = learning_rate_axis(config=config)
    c_mass = gaussian_bin_mass(
        means=competence[:, None] + training_examples * rate[None, :],
        sigma=config.sigma_competence,
        axis=competence,
    )
    eta_mass = gaussian_bin_mass(
        means=config.learning_rate_decay**training_examples * rate,
        sigma=config.sigma_eta,
        axis=rate,
    )
    c_mass.setflags(write=False)
    eta_mass.setflags(write=False)
    return c_mass, eta_mass


def grid_predict(
    *,
    model: CompetenceModel,
    config: InferenceConfig,
    weights: np.ndarray,
    training_examples: int,
    total_training_examples: int,
) -> np.ndarray:
    # Like `particle_transition`, no zero-example shortcut: the n = 0 kernels apply
    # the noise-only step, and Model A redraws at the unchanged total.
    if model == "global_curve":
        phi_mass = weights.reshape(-1, config.competence_bins).sum(axis=1)
        return (
            phi_mass[:, None]
            * curve_cycle_mass(config=config, training_examples=total_training_examples)
        ).ravel()
    c_mass, eta_mass = local_transition_kernels(config=config, training_examples=training_examples)
    return np.einsum(
        "ce,cej,ef->jf",
        weights.reshape(config.competence_bins, config.learning_rate_bins),
        c_mass,
        eta_mass,
        optimize=True,
    ).ravel()


def grid_backward(
    *,
    model: CompetenceModel,
    config: InferenceConfig,
    message: np.ndarray,
    training_examples: int,
    total_training_examples: int,
) -> np.ndarray:
    """Apply the transpose of the exact same discrete transition used online.

    That contract is why there is no zero-example shortcut here either: real cycle
    boundaries now apply the n = 0 noise step forward, so smoothing must apply its
    transpose or the forward-backward pair would disagree.
    """
    if model == "global_curve":
        per_phi = np.sum(
            curve_cycle_mass(config=config, training_examples=total_training_examples)
            * message.reshape(-1, config.competence_bins),
            axis=1,
        )
        return np.repeat(per_phi, config.competence_bins)
    c_mass, eta_mass = local_transition_kernels(config=config, training_examples=training_examples)
    return np.einsum(
        "cej,ef,jf->ce",
        c_mass,
        eta_mass,
        message.reshape(config.competence_bins, config.learning_rate_bins),
        optimize=True,
    ).ravel()
