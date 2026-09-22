"""Numerical contracts for matched competence models and offline smoothing."""

import numpy as np
import pytest
from scipy.special import ndtr

from hitl_pmp.methods.belief_space.competence_inference import (
    BayesianSkillBelief,
    CompetenceModel,
    InferenceConfig,
    InferenceEngine,
    create_bayesian_prior,
    smooth_history,
)
from hitl_pmp.methods.belief_space.competence_inference.models import (
    beta_parameters,
    gaussian_bin_mass,
    global_curve_learning_rate,
    global_curve_mode,
    grid_backward,
    grid_predict,
    local_transition_kernels,
    particle_transition,
    phi_support,
)
from hitl_pmp.methods.belief_space.types.particle_filter_belief import (
    create_fixed_performance_cost_prior,
)


def _belief(
    *,
    model: CompetenceModel = "local_trend",
    engine: InferenceEngine = "grid",
    seed: int = 11,
    count: int = 1024,
    config: InferenceConfig | None = None,
) -> BayesianSkillBelief:
    return create_bayesian_prior(
        model=model, engine=engine, seed=seed, num_particles=count, config=config
    )


def test_global_curve_uses_mode_beta_and_derivative_of_expected_competence() -> None:
    phi = np.array([[0.2, 0.8, 0.1, 24.0], [0.4, 0.4, 0.8, 6.0]])
    mode = global_curve_mode(phi=phi, training_examples=10)
    assert mode == pytest.approx([0.2 + 0.6 * (1 - np.exp(-1)), 0.4])
    alpha, beta = beta_parameters(phi=phi, training_examples=10)
    assert alpha + beta == pytest.approx([24, 6])
    assert alpha == pytest.approx(1 + mode * np.array([22, 4]))
    expected_derivative = [0.6 * 0.1 * np.exp(-1) * 22 / 24, 0]
    assert global_curve_learning_rate(phi=phi, training_examples=10) == pytest.approx(
        expected_derivative
    )
    assert alpha / (alpha + beta) != pytest.approx(mode)


def test_global_prior_contains_the_same_discrete_phi_population() -> None:
    config = InferenceConfig()
    support = phi_support(config=config)
    assert support.shape == (66 * 6 * 3, 4)
    particle = _belief(model="global_curve", engine="particle", count=10000)
    values, _ = particle.arrays()
    support_set = {tuple(row) for row in support}
    assert all(tuple(row) in support_set for row in values[:, :4])
    assert any(initial == plateau for initial, plateau, _, _ in support_set)


@pytest.mark.parametrize("model", ["global_curve", "local_trend"])
def test_particle_and_grid_priors_and_sf_posteriors_agree(*, model: CompetenceModel) -> None:
    config = InferenceConfig(resample_ess_fraction=0.0)
    particle = _belief(model=model, engine="particle", count=50000, config=config)
    grid = _belief(model=model, config=config)
    for observations, training in [([True, False, True], 3), ([False, True, True], 4), ([True], 0)]:
        assert particle.mean_competence() == pytest.approx(grid.mean_competence(), abs=0.012)
        assert particle.mean_learning_rate() == pytest.approx(grid.mean_learning_rate(), abs=0.003)
        for success in observations:
            particle = particle.condition_outcome(success=success)
            grid = grid.condition_outcome(success=success)
        assert particle.mean_competence() == pytest.approx(grid.mean_competence(), abs=0.016)
        particle = particle.refit(training_examples=training)
        grid = grid.refit(training_examples=training)


def test_local_transition_scales_learning_and_decay_by_data_count() -> None:
    config = InferenceConfig(sigma_competence=0, sigma_eta=0)
    values = np.array([[0.2, 0.05], [0.99, 0.1], [0.1, 0.0]])
    predicted = particle_transition(
        model="local_trend",
        config=config,
        values=values,
        training_examples=4,
        total_training_examples=4,
        rng=np.random.default_rng(7),
    )
    assert predicted[:, 0] == pytest.approx([0.4, 1.0, 0.1])
    assert predicted[:, 1] == pytest.approx(config.learning_rate_decay**4 * values[:, 1])
    assert np.array_equal(
        particle_transition(
            model="local_trend",
            config=config,
            values=values,
            training_examples=0,
            total_training_examples=4,
            rng=np.random.default_rng(7),
        ),
        values,
    )


def test_local_grid_includes_clipped_boundary_mass_and_stochastic_rows() -> None:
    axis = np.array([0.0, 0.5, 1.0])
    result = gaussian_bin_mass(means=np.array([0.0, 1.0, -100.0, 100.0]), sigma=0.1, axis=axis)
    assert result.sum(axis=1) == pytest.approx(np.ones(4))
    assert result[0, 0] == pytest.approx(ndtr(2.5))
    assert result[1, -1] == pytest.approx(ndtr(2.5))
    assert result[2] == pytest.approx([1, 0, 0])
    assert result[3] == pytest.approx([0, 0, 1])
    for n in (1, 4, 20):
        c_mass, eta_mass = local_transition_kernels(config=InferenceConfig(), training_examples=n)
        assert c_mass.sum(axis=-1) == pytest.approx(np.ones((25, 16)))
        assert eta_mass.sum(axis=-1) == pytest.approx(np.ones(16))


def test_local_prior_integrates_beta_and_halfnormal_including_eta_cap() -> None:
    config = InferenceConfig(competence_bins=3, learning_rate_bins=3, eta_max=0.1)
    _, weights = _belief(config=config).arrays()
    joint = weights.reshape(3, 3)
    assert joint.sum(axis=1) == pytest.approx([0.25**2, 0.75**2 - 0.25**2, 1 - 0.75**2])
    assert joint.sum(axis=0) == pytest.approx([
        2 * ndtr(0.025 / 0.05) - 1,
        2 * (ndtr(0.075 / 0.05) - ndtr(0.025 / 0.05)),
        2 * (1 - ndtr(0.075 / 0.05)),
    ])
    particles = _belief(engine="particle", count=20000, config=config)
    values, _ = particles.arrays()
    assert values[:, 1].max() == config.eta_max
    assert np.mean(values[:, 1] == config.eta_max) == pytest.approx(2 * (1 - ndtr(2)), abs=0.006)


@pytest.mark.parametrize("model", ["global_curve", "local_trend"])
def test_grid_support_never_moves_or_resamples(*, model: CompetenceModel) -> None:
    belief = _belief(model=model)
    support = belief.latent_values
    for training in (1, 4, 20):
        belief = (
            belief
            .condition_outcome(success=False)
            .condition_outcome(success=True)
            .refit(training_examples=training)
        )
        assert belief.latent_values == support
        assert belief.resampling_count == 0
        assert belief.arrays()[1].sum() == pytest.approx(1)


@pytest.mark.parametrize("engine", ["particle", "grid"])
def test_learning_rate_is_implicitly_updated_from_cross_cycle_sf(
    *, engine: InferenceEngine
) -> None:
    belief = _belief(engine=engine, count=20000)
    for success in [False, False, True]:
        belief = belief.condition_outcome(success=success)
    predicted = belief.refit(training_examples=4)
    successful = predicted.condition_outcome(success=True)
    failed = predicted.condition_outcome(success=False)
    assert successful.mean_learning_rate() > predicted.mean_learning_rate()
    assert failed.mean_learning_rate() < predicted.mean_learning_rate()
    with pytest.raises(ValueError, match="direct observations are unsupported"):
        belief.condition_learning_rate(observed_learning_rate=0.5)


@pytest.mark.parametrize("model", ["global_curve", "local_trend"])
@pytest.mark.parametrize("engine", ["particle", "grid"])
def test_execution_cost_filter_is_identical_and_independent(
    *, model: CompetenceModel, engine: InferenceEngine
) -> None:
    belief = _belief(model=model, engine=engine)
    cost = create_fixed_performance_cost_prior(
        num_particles=1024, seed=11, competence=1.0, learning_rate=0.0
    )
    original_cost = cost.signature()
    belief = belief.condition_outcome(success=True).refit(training_examples=2)
    assert belief.cost_belief.signature() == original_cost
    for index, observation in enumerate([1.0, 1.2, 0.9, 1.1, 5.0, 1.0]):
        cost = cost.condition_cost(observed_cost=observation)
        belief = belief.condition_execution(success=index % 2 == 0, observed_cost=observation)
        assert belief.cost_belief.signature() == cost.signature()
        assert belief.mean_cost() == cost.mean_cost()
    assert cost.resampling_count > 0


@pytest.mark.parametrize("model", ["global_curve", "local_trend"])
@pytest.mark.parametrize("engine", ["particle", "grid"])
def test_forecasts_are_immutable_seeded_and_zero_examples_are_noop(
    *, model: CompetenceModel, engine: InferenceEngine
) -> None:
    original = _belief(model=model, engine=engine).condition_outcome(success=True)
    signature = original.signature()
    first = original.refit(training_examples=4)
    second = original.refit(training_examples=4)
    assert first.signature() == second.signature()
    assert original.signature() == signature
    assert original.refit(training_examples=0) is original
    assert original.advance_learning_rate(process_noise_std=1.0) is original
    boundary = original.advance_cycle(training_examples=0)
    assert boundary.latent_values == original.latent_values
    assert boundary.state_weights == original.state_weights
    assert boundary.arrays()[1] == pytest.approx(original.arrays()[1], abs=1e-15)
    assert boundary.cycle_index == 1 and boundary.cycle_successes == boundary.cycle_failures == 0
    assert boundary.process_transition_count == 0
    assert boundary.mean_cost() == original.mean_cost()
    if engine == "particle":
        assert np.array_equal(boundary.ancestors(), np.arange(original.state_count))
    with pytest.raises(ValueError):
        original.arrays()[0][0, 0] = 0.3
    assert original.sample(rng=np.random.default_rng(4), count=17).shape == (17, 3)
    assert original.sample(rng=np.random.default_rng(4), count=0).shape == (0, 3)
    assert (
        BayesianSkillBelief.model_validate_json(original.model_dump_json()).signature() == signature
    )


@pytest.mark.parametrize("model", ["global_curve", "local_trend"])
def test_backward_operator_matches_forward_operator(*, model: CompetenceModel) -> None:
    config = InferenceConfig(
        competence_bins=5,
        learning_rate_bins=4,
        phi_initial=(0.1, 0.5),
        phi_plateau=(0.5, 0.9),
        phi_rates=(0.1,),
        phi_concentrations=(6.0,),
    )
    belief = _belief(model=model, config=config)
    _, weights = belief.arrays()
    message = np.random.default_rng(3).random(belief.state_count)
    for n in (0, 1, 7):
        forward = grid_predict(
            model=model,
            config=config,
            weights=weights,
            training_examples=n,
            total_training_examples=n,
        )
        backward = grid_backward(
            model=model,
            config=config,
            message=message,
            training_examples=n,
            total_training_examples=n,
        )
        assert forward @ message == pytest.approx(weights @ backward, abs=1e-13)


@pytest.mark.parametrize("model", ["global_curve", "local_trend"])
def test_grid_smoothing_matches_enumeration_and_does_not_repeat_zero_cycle_evidence(
    *, model: CompetenceModel
) -> None:
    config = InferenceConfig(
        competence_bins=5,
        learning_rate_bins=4,
        phi_initial=(0.2,),
        phi_plateau=(0.2, 0.9),
        phi_rates=(0.2,),
        phi_concentrations=(6.0,),
    )
    first = _belief(model=model, config=config).condition_outcome(success=False)
    second = first.advance_cycle(training_examples=2).condition_outcome(success=True)
    third = second.advance_cycle(training_examples=0).condition_outcome(success=True)
    history = [first, second, third]
    signatures = [item.signature() for item in history]
    summaries = smooth_history(history=history)
    _, filtered = first.arrays()
    transition = np.stack([
        grid_predict(
            model=model, config=config, weights=row, training_examples=2, total_training_examples=2
        )
        for row in np.eye(first.state_count)
    ])
    # Two later successes, with an identity transition between their cycles.
    joint = filtered[:, None] * transition * second.competence_values()[None, :] ** 2
    posterior = joint.sum(axis=1) / joint.sum()
    assert summaries[0].smoothed_competence == pytest.approx(posterior @ first.competence_values())
    assert summaries[-1].smoothed_competence == summaries[-1].filtered_competence
    assert summaries[0].smoothed_learning_rate != pytest.approx(
        summaries[0].filtered_learning_rate, abs=1e-5
    )
    assert [item.signature() for item in history] == signatures
    assert summaries[0].model_dump(mode="json")["cycle_index"] == 0


def test_particle_smoothing_tracks_composed_ancestors_and_terminal_weights() -> None:
    first = _belief(engine="particle", count=12, config=InferenceConfig(resample_ess_fraction=0))
    values, _ = first.arrays()
    second = first.advance_cycle(training_examples=1)
    parents = np.array([0, 0, 1, 1, 1, 4, 5, 5, 8, 9, 10, 11], dtype="<i8")
    weights = np.arange(1, 13, dtype="<f8")
    weights /= weights.sum()
    second = second.model_copy(
        update={
            "latent_values": np.ascontiguousarray(values[parents]).tobytes(),
            "parent_indices": parents.tobytes(),
            "state_weights": weights.tobytes(),
        }
    )
    summaries = smooth_history(history=[first, second])
    expected = np.bincount(parents, weights=weights, minlength=12)
    assert summaries[0].smoothed_competence == pytest.approx(expected @ first.competence_values())
    assert summaries[0].unique_ancestors == len(np.unique(parents))
    assert summaries[-1].smoothed_competence == summaries[-1].filtered_competence


def test_particle_resampling_preserves_ancestry_through_multiple_observations() -> None:
    belief = _belief(engine="particle", count=128, config=InferenceConfig(resample_ess_fraction=1))
    initial_values, _ = belief.arrays()
    for _ in range(12):
        belief = belief.condition_outcome(success=False)
        values, _ = belief.arrays()
        np.testing.assert_array_equal(values, initial_values[belief.ancestors()])
    assert belief.resampling_count > 1
    assert len(np.unique(belief.ancestors())) < belief.state_count


@pytest.mark.parametrize("model", ["global_curve", "local_trend"])
def test_particle_and_grid_smoothed_history_agree(*, model: CompetenceModel) -> None:
    summaries = []
    for engine in ("particle", "grid"):
        belief = create_bayesian_prior(model=model, engine=engine, seed=17, num_particles=50000)
        history = []
        for outcomes, count in [
            ([False, False, True], 3),
            ([True, True, False], 4),
            ([True, True, True], 2),
        ]:
            for success in outcomes:
                belief = belief.condition_outcome(success=success)
            history.append(belief)
            belief = belief.advance_cycle(training_examples=count)
        summaries.append(smooth_history(history=history))
    for particle, grid in zip(*summaries, strict=True):
        assert particle.smoothed_competence == pytest.approx(grid.smoothed_competence, abs=0.015)
        assert particle.smoothed_learning_rate == pytest.approx(
            grid.smoothed_learning_rate, abs=0.004
        )


@pytest.mark.parametrize("engine", ["particle", "grid"])
def test_zero_training_cycles_smooth_without_changing_online_posteriors(
    *, engine: InferenceEngine
) -> None:
    first = _belief(engine=engine, count=20000).condition_outcome(success=False)
    second = first.advance_cycle(training_examples=0)
    for _ in range(6):
        second = second.condition_outcome(success=True)
    signature = first.signature()
    summary = smooth_history(history=[first, second])
    assert summary[0].smoothed_competence > summary[0].filtered_competence
    assert summary[0].smoothed_competence == pytest.approx(
        summary[1].filtered_competence, abs=1e-12
    )
    assert first.signature() == signature
    with pytest.raises(ValueError, match="consecutive"):
        smooth_history(history=[first, first])
