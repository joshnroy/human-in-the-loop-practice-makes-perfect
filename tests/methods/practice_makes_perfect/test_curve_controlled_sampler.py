"""`--curve-controlled-sampler` (Josh, 2026-11-25): an oracle control for the planners.
A skill's parameters are drawn from its measured success region with probability
alpha(n) = min(1, 0.1 + 0.15 n), n its own practice tries so far, and from its measured
failure region otherwise, so its success rate follows the chained toy's known curve. It
replaces the learned sampler and its epsilon exploration, at practice and at test."""

import argparse
import pickle

import numpy as np
import pytest
from scipy.stats import binom

from hitl_pmp.core.method.types import KnownOutcomeRegions, SamplerConsultation
from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.skill_provider import RackServingSkillProvider
from hitl_pmp.environments.racktoserving3d.toy_environment import RackServingToyEnvironment
from hitl_pmp.environments.racktoserving3d.toy_interface import (
    GATHER,
    ROBOT_RESET,
    SWEEP3,
    SWEEP5,
)
from hitl_pmp.methods.practice_makes_perfect.cli import DpCli, EesCli
from hitl_pmp.methods.practice_makes_perfect.curve_controlled_sampler import (
    CurveControlledSampler,
)
from hitl_pmp.methods.practice_makes_perfect.dp_method import DpMethod
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.step_protocol import DeploymentSnapshot


def _provider(*, gather_delta_high: float | None = 0.2) -> RackServingSkillProvider:
    env = RackServingToyEnvironment(config=RackServingConfig(gather_delta_high=gather_delta_high))
    return RackServingSkillProvider(env=env)


def _method(
    *,
    curve: bool,
    seed: int = 0,
    cls: type[EesMethod] = EesMethod,
    dp_kwargs: dict[str, object] | None = None,
) -> EesMethod:
    provider = _provider()
    return cls(
        env=provider.env,
        skill_provider=provider,
        seed=seed,
        **({"dp_budget": 10.0, **(dp_kwargs or {})} if cls is DpMethod else {}),
        curve_controlled_sampler=(
            CurveControlledSampler(regions=provider.known_outcome_regions()) if curve else None
        ),
    )


def _inside(*, params: np.ndarray, box: tuple[tuple[float, float], ...]) -> bool:
    return all(low <= v <= high for v, (low, high) in zip(params, box, strict=True))


def _draw(*, method: EesMethod, name: str, explore: bool) -> tuple[np.ndarray, object]:
    ground = method.skill_provider.ground(name=name)
    # The serving provider's draws and rows read the parameters alone, never the state.
    _, record = method.execute_ground_skill(ground_skill=ground, state=None, explore=explore)
    assert record is not None
    return np.asarray(record.params), record


def test_alpha_is_the_toys_curve_capped_at_one() -> None:
    curve = CurveControlledSampler(regions=_provider().known_outcome_regions())
    assert [curve.alpha(n=n) for n in (0, 1, 2, 6, 7, 100)] == pytest.approx([
        0.1,
        0.25,
        0.4,
        1.0,
        1.0,
        1.0,
    ])


def test_the_serving_regions_are_the_measured_near_certain_cells() -> None:
    regions = _provider().known_outcome_regions()
    assert set(regions) == {GATHER, SWEEP3, SWEEP5}
    assert regions[GATHER] == KnownOutcomeRegions(
        success=((0.069, 0.071), (0.005, 0.015)), failure=((0.03, 0.11), (0.12, 0.2))
    )
    assert regions[SWEEP3] == KnownOutcomeRegions(success=((0.05, 0.12),), failure=((0.0, 0.03),))
    assert regions[SWEEP5] == KnownOutcomeRegions(success=((0.08, 0.12),), failure=((0.0, 0.03),))
    for name, region in regions.items():
        bounds = _provider().bounds(name=name)
        for box in (region.success, region.failure):
            assert all(lo <= a < b <= hi for (a, b), (lo, hi) in zip(box, bounds, strict=True)), (
                name
            )


def test_regions_outside_the_skills_bounds_are_refused() -> None:
    # Gather's failure region needs delta up to 0.20: without --gather-delta-high 0.2
    # the skill cannot reach it, so the oracle must not pretend to draw there.
    with pytest.raises(ValueError, match="gather"):
        _provider(gather_delta_high=None).known_outcome_regions()


@pytest.mark.parametrize("n", [0, 1, 3, 5])
def test_drawn_settings_fall_in_the_success_region_with_frequency_alpha(*, n: int) -> None:
    method = _method(curve=True, seed=7)
    curve = method.curve_controlled_sampler
    assert curve is not None
    curve.tries[GATHER] = n
    region = curve.regions[GATHER]
    draws = 2000
    in_success = 0
    for _ in range(draws):
        params, record = _draw(method=method, name=GATHER, explore=True)
        in_s = _inside(params=params, box=region.success)
        in_f = _inside(params=params, box=region.failure)
        assert in_s != in_f, params  # never the ambiguous middle, never both
        in_success += in_s
        assert record.was_random_exploration is False
        assert record.was_informed_choice is True
        assert record.consultation is SamplerConsultation.INFORMED
    alpha = 0.1 + 0.15 * n
    # Two-sided binomial test at the fixed seed: the count is consistent with alpha.
    p_value = 2 * min(binom.cdf(in_success, draws, alpha), binom.sf(in_success - 1, draws, alpha))
    assert p_value > 0.001, (in_success, draws, alpha)


def test_n_counts_the_skills_own_tries() -> None:
    method = _method(curve=True)
    curve = method.curve_controlled_sampler
    assert curve is not None
    for name in (GATHER, GATHER, SWEEP5, GATHER):
        method.record_practice_attempt(
            skill_name=name, success=False, consultation=SamplerConsultation.INFORMED
        )
    # A skill with no parameters is not one the oracle draws for.
    method.record_practice_attempt(
        skill_name=ROBOT_RESET, success=True, consultation=SamplerConsultation.NO_SAMPLER
    )
    assert curve.tries == {GATHER: 3, SWEEP5: 1}
    assert curve.tries.get(SWEEP3, 0) == 0
    assert curve.success_probability(name=GATHER) == pytest.approx(0.55)
    assert curve.success_probability(name=SWEEP3) == pytest.approx(0.1)


def test_a_dp_setup_run_does_not_count_as_a_try() -> None:
    # The toy's `execution_trains`: a DP run of a skill on the way to its practice
    # target does not train that skill.
    method = _method(curve=True, cls=DpMethod)
    curve = method.curve_controlled_sampler
    assert curve is not None
    method._dp_setup_dispatch = True
    method.record_practice_attempt(
        skill_name=GATHER, success=True, consultation=SamplerConsultation.INFORMED
    )
    method._dp_setup_dispatch = False
    method.record_practice_attempt(
        skill_name=SWEEP5, success=True, consultation=SamplerConsultation.INFORMED
    )
    assert curve.tries == {SWEEP5: 1}


def _setup_then_target(*, method: EesMethod) -> None:
    """One DP setup run of gather (it succeeds), then one target run of sweep5."""
    assert isinstance(method, DpMethod)
    gather = method.skill_provider.ground(name=GATHER)
    method._dp_setup_dispatch = True
    method.record_practice_attempt(
        skill_name=GATHER, success=True, consultation=SamplerConsultation.INFORMED
    )
    method.observe_outcome(ground_skill=gather, success=True)
    method._dp_setup_dispatch = False
    method.record_practice_attempt(
        skill_name=SWEEP5, success=True, consultation=SamplerConsultation.INFORMED
    )


def test_with_curve_count_setup_tries_a_dp_setup_run_counts_as_a_try() -> None:
    # Trial variant (Josh, 2026-11-26): DP's setup runs advance the curve's n too.
    method = _method(curve=True, cls=DpMethod, dp_kwargs={"curve_count_setup_tries": True})
    curve = method.curve_controlled_sampler
    assert curve is not None
    _setup_then_target(method=method)
    assert curve.tries == {GATHER: 1, SWEEP5: 1}


def test_curve_count_setup_tries_changes_nothing_but_the_curves_count() -> None:
    on = _method(curve=True, cls=DpMethod, dp_kwargs={"curve_count_setup_tries": True})
    off = _method(curve=True, cls=DpMethod)
    for method in (on, off):
        _setup_then_target(method=method)
    assert isinstance(on, DpMethod) and isinstance(off, DpMethod)
    on._dp_setup_dispatch = off._dp_setup_dispatch = True
    # A setup run still trains nothing: not the sampler, not competence.
    assert on.execution_trains_skill() is off.execution_trains_skill() is False
    assert on.practice_outcomes() == off.practice_outcomes()
    assert on._all_attempt_outcomes == off._all_attempt_outcomes == {}
    assert on._competence_models.keys() == off._competence_models.keys()
    assert on.dp_allocation() == off.dp_allocation()


def test_curve_count_setup_tries_stays_out_of_the_deployment_snapshot() -> None:
    on = _method(curve=True, cls=DpMethod, dp_kwargs={"curve_count_setup_tries": True})
    off = _method(curve=True, cls=DpMethod)
    for specific in (False, True):
        assert DeploymentSnapshot.encode(
            method=on, method_specific=specific
        ) == DeploymentSnapshot.encode(method=off, method_specific=specific)


def test_the_dp_cli_takes_curve_count_setup_tries_only_with_the_curve() -> None:
    parser = argparse.ArgumentParser()
    DpCli.add_arguments(parser=parser)
    assert parser.parse_args(["--dp-budget", "10"]).curve_count_setup_tries is False
    on = parser.parse_args(["--dp-budget", "10", "--curve-count-setup-tries"])
    assert on.curve_count_setup_tries is True
    with pytest.raises(ValueError, match="--curve-controlled-sampler"):
        DpCli.run(args=on, env_cli=None)  # type: ignore[arg-type]


def test_the_test_policy_draws_at_the_current_n() -> None:
    method = _method(curve=True, seed=3)
    curve = method.curve_controlled_sampler
    assert curve is not None
    for _ in range(6):  # alpha(6) = 1: every test draw lands in S
        method.record_practice_attempt(
            skill_name=SWEEP5, success=True, consultation=SamplerConsultation.INFORMED
        )
    # The deployment the evaluation runs is restored from a snapshot; it carries n.
    restored = DeploymentSnapshot.restore(
        raw=DeploymentSnapshot.encode(method=method, method_specific=False),
        env=method.env,
        provider=method.skill_provider,
    )
    assert restored.curve_controlled_sampler is not None
    assert restored.curve_controlled_sampler.tries == {SWEEP5: 6}
    region = curve.regions[SWEEP5]
    for _ in range(200):
        params, record = _draw(method=restored, name=SWEEP5, explore=False)
        assert _inside(params=params, box=region.success)
        assert record.records_training_row is False
    # Evaluation never counts a try.
    assert restored.curve_controlled_sampler.tries == {SWEEP5: 6}
    # And at n = 0 the test policy fails 9 in 10, as practice would.
    fresh = _method(curve=True, seed=3)
    hits = sum(
        _inside(params=_draw(method=fresh, name=SWEEP5, explore=False)[0], box=region.success)
        for _ in range(400)
    )
    assert binom.sf(hits - 1, 400, 0.1) > 0.001 and binom.cdf(hits, 400, 0.1) > 0.001


def test_the_classifier_still_records_the_outcome() -> None:
    method = _method(curve=True)
    _, record = _draw(method=method, name=GATHER, explore=True)
    assert record.records_training_row is True
    method.observe_sampler_outcome(
        skill_name=GATHER, param_dim=2, sampler_input=record.sampler_input, success=True
    )
    assert method.sampler(skill_name=GATHER, param_dim=2).num_observations == 1


def test_off_draws_exactly_what_the_learned_sampler_draws() -> None:
    off, default = _method(curve=False, seed=5), _method(curve=False, seed=5)
    assert off.curve_controlled_sampler is None
    for name in (GATHER, SWEEP3, SWEEP5) * 5:
        a, ra = _draw(method=off, name=name, explore=True)
        b, rb = _draw(method=default, name=name, explore=True)
        assert np.array_equal(a, b)
        assert ra.consultation is rb.consultation is SamplerConsultation.UNINFORMATIVE


def test_off_writes_the_deployment_snapshot_it_always_wrote() -> None:
    method = _method(curve=False)
    payload = pickle.loads(DeploymentSnapshot.encode(method=method, method_specific=False))
    assert "curve_controlled_sampler" not in payload["configuration"]


def test_the_cli_builds_it_only_when_asked() -> None:
    parser = argparse.ArgumentParser()
    EesCli.add_arguments(parser=parser)
    provider = _provider()
    off = parser.parse_args([])
    assert off.curve_controlled_sampler is False
    assert EesCli.curve_controlled_sampler(args=off, provider=provider) is None
    on = parser.parse_args(["--curve-controlled-sampler"])
    built = EesCli.curve_controlled_sampler(args=on, provider=provider)
    assert isinstance(built, CurveControlledSampler)
    assert built.regions == provider.known_outcome_regions()


def test_the_cli_refuses_a_domain_with_no_measured_regions() -> None:
    from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
    from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider

    parser = argparse.ArgumentParser()
    EesCli.add_arguments(parser=parser)
    env = LightSwitchEnvironment(grid_size=2)
    with pytest.raises(ValueError, match="measured"):
        EesCli.curve_controlled_sampler(
            args=parser.parse_args(["--curve-controlled-sampler"]),
            provider=LightSwitchSkillProvider(env=env),
        )
