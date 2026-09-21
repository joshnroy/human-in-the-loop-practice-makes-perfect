"""Calibrated learning clocks alter forecasts without changing practice accounting."""

import argparse
from collections.abc import Callable

import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DOracle, Tossing3DSkillProvider
from hitl_pmp.methods.belief_space.competence_inference import (
    BayesianSkillBelief,
    CompetenceModel,
    InferenceConfig,
    InferenceEngine,
    create_bayesian_prior,
)
from hitl_pmp.methods.belief_space.competence_inference.models import (
    global_curve_learning_rate,
    global_curve_mode,
    particle_transition,
)
from hitl_pmp.methods.belief_space.tossing3d_constants import TOSS_SKILL
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    make_default_tossing3d_belief,
)
from hitl_pmp.methods.practice_makes_perfect.cli import Tossing3DPomdpCli


def _candidate_config(*, model: CompetenceModel) -> InferenceConfig:
    # Frozen development-selected settings, not formulas repeated from the transform.
    if model == "global_curve":
        return InferenceConfig(phi_rates=(0.005, 0.0125, 0.025, 0.05, 0.1, 0.2))
    return InferenceConfig(
        eta_max=0.009375,
        sigma_eta=0.0003125,
        learning_rate_decay=0.9934366015836147,
        initial_eta_sigma=0.003125,
    )


@pytest.mark.parametrize(("model", "time_scale"), [("global_curve", 4), ("local_trend", 16)])
def test_time_scale_matches_the_full_calibration_candidate(
    *, model: CompetenceModel, time_scale: float
) -> None:
    config = InferenceConfig()
    original = config.model_dump_json()
    scaled = config.scaled_learning_time(model=model, time_scale=time_scale)
    assert scaled.model_dump_json() == _candidate_config(model=model).model_dump_json()
    assert config.model_dump_json() == original


@pytest.mark.parametrize("model", ["global_curve", "local_trend"])
@pytest.mark.parametrize("engine", ["particle", "grid"])
def test_unit_time_scale_preserves_default_belief_bytes(
    *, model: CompetenceModel, engine: InferenceEngine
) -> None:
    env = Tossing3DEnvironment(scene_bg=False)
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env),
        seed=11,
        pomdp_num_particles=32,
        pomdp_competence_model=model,
        pomdp_inference_engine=engine,
    )
    baseline = make_default_tossing3d_belief(
        num_particles=32,
        seed=11,
        model=model,
        engine=engine,
        inference_config=InferenceConfig(),
        additional_skill_names=tuple(skill.name for skill in method.human_skills()),
    )
    assert method.pomdp_learning_time_scale == 1.0
    assert method.pomdp_state.model_dump_json() == baseline.model_dump_json()


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_nonpositive_or_nonfinite_time_scales_are_rejected(*, invalid: float) -> None:
    for model in ("global_curve", "local_trend"):
        with pytest.raises(ValueError, match="time_scale"):
            InferenceConfig().scaled_learning_time(model=model, time_scale=invalid)
    env = Tossing3DEnvironment(scene_bg=False)
    with pytest.raises(ValidationError, match="pomdp_learning_time_scale"):
        Tossing3DPomdpMethod(
            env=env,
            skill_provider=Tossing3DSkillProvider(env=env),
            seed=0,
            pomdp_learning_time_scale=invalid,
        )


def test_model_a_time_scale_stretches_the_curve_and_its_derivative() -> None:
    base_phi = np.array([[0.1, 0.8, 0.2, 24.0]])
    scaled = InferenceConfig(phi_rates=(0.2,)).scaled_learning_time(
        model="global_curve", time_scale=4
    )
    scaled_phi = base_phi.copy()
    scaled_phi[:, 2] = scaled.phi_rates[0]
    assert global_curve_mode(phi=scaled_phi, training_examples=32) == pytest.approx(
        global_curve_mode(phi=base_phi, training_examples=8)
    )
    assert global_curve_learning_rate(phi=scaled_phi, training_examples=32) == pytest.approx(
        global_curve_learning_rate(phi=base_phi, training_examples=8) / 4
    )


def test_model_b_time_scale_preserves_real_example_counts_and_stretches_dynamics() -> None:
    base = InferenceConfig(sigma_competence=0, sigma_eta=0)
    scaled = base.scaled_learning_time(model="local_trend", time_scale=16)
    initial = np.array([[0.2, 0.05]])
    slowed_initial = np.array([[0.2, 0.003125]])
    kwargs = {"model": "local_trend", "total_training_examples": 16}
    original = particle_transition(
        **kwargs,
        config=base,
        values=initial,
        training_examples=1,
        rng=np.random.default_rng(0),
    )
    slowed = particle_transition(
        **kwargs,
        config=scaled,
        values=slowed_initial,
        training_examples=16,
        rng=np.random.default_rng(0),
    )
    assert slowed[:, 0] == pytest.approx(original[:, 0])
    assert slowed[:, 1] * 16 == pytest.approx(original[:, 1])
    belief = create_bayesian_prior(
        model="local_trend", engine="grid", seed=0, num_particles=32, config=scaled
    ).refit(training_examples=35)
    assert belief.total_training_examples == 35


@pytest.mark.parametrize(("model", "time_scale"), [("global_curve", 4), ("local_trend", 16)])
@pytest.mark.parametrize("engine", ["particle", "grid"])
def test_cli_installs_candidate_config_with_physical_costs_unchanged(
    *,
    model: CompetenceModel,
    time_scale: float,
    engine: InferenceEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    Tossing3DPomdpCli.add_arguments(parser=parser)
    assert parser.parse_args([]).pomdp_learning_time_scale == 1.0
    args = parser.parse_args([
        "--pomdp-competence-model",
        model,
        "--pomdp-inference-engine",
        engine,
        "--pomdp-learning-time-scale",
        str(time_scale),
        "--pomdp-observation-probability-weight",
        "0",
        "--pomdp-linear-cost-lambda",
        "0.0003",
        "--pomdp-num-particles",
        "32",
    ])
    captured: list[Method] = []

    def capture_factory(
        *,
        args: argparse.Namespace,
        method_factory: Callable[[DomainContext], Method],
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> None:
        del args, num_cycles, max_steps_per_interaction
        env = Tossing3DEnvironment(scene_bg=False)
        captured.append(
            method_factory(
                DomainContext(
                    env=env,
                    skill_provider=Tossing3DSkillProvider(env=env, human_reset_practice_cost=5.0),
                    oracle=Tossing3DOracle(env=env),
                )
            )
        )

    monkeypatch.setattr(Tossing3DCli, "run_method", capture_factory)
    Tossing3DPomdpCli.run(args=args, env_cli=Tossing3DCli)
    method = captured[0]
    assert isinstance(method, Tossing3DPomdpMethod)
    belief = method.pomdp_state.skill_beliefs[TOSS_SKILL]
    assert isinstance(belief, BayesianSkillBelief)
    assert belief.config == _candidate_config(model=model)
    assert method.pomdp_learning_time_scale == time_scale
    assert method.pomdp_observation_probability_weight == 0
    assert method.pomdp_linear_cost_lambda == 0.0003
    assert {skill.evaluate_practice_cost() for skill in method.skills()} == {1.0}
    assert {skill.evaluate_practice_cost() for skill in method.human_skills()} == {5.0}
    assert method.env._backend is None  # noqa: SLF001
