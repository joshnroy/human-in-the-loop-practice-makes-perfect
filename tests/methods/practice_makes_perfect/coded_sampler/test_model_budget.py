"""The coded sampler's per-run model budget, accounted as Full-Agentic's is."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.methods.practice_makes_perfect.coded_sampler.coder import RobocodeCoder
from hitl_pmp.methods.practice_makes_perfect.coded_sampler.types import CodedSamplerMode

from .fakes import CodedLightSwitch


def _refits(*, path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in (path / "coded_sampler_refits.jsonl").read_text().splitlines()
    ]


def _cycles(*, method: object, count: int) -> None:
    for _ in range(count):
        method.end_cycle()  # type: ignore[attr-defined]


def test_each_refit_is_offered_only_what_the_run_has_left(*, tmp_path: Path) -> None:
    method, _, _, coder = CodedLightSwitch.build(
        tmp_path=tmp_path, mode=CodedSamplerMode.SCORER, model_budget_usd=20.0
    )
    coder.costs = [3.0, 5.5, 1.25]
    _cycles(method=method, count=3)
    offered = [call["max_budget_usd"] for call in coder.calls]
    assert offered == pytest.approx([20.0, 17.0, 11.5])


def test_a_rate_limit_retry_never_replenishes_the_budget() -> None:
    """A refit's cost is its final attempt plus every aborted attempt before it."""
    metrics = SimpleNamespace(aborted_cost_usd=4.0, rate_limit_retries=2)
    result = SimpleNamespace(total_cost_usd=1.5, generation_metrics=metrics)
    assert RobocodeCoder.invocation_cost(result=result) == pytest.approx(5.5)


def test_once_the_budget_is_spent_the_coder_is_never_invoked_again(*, tmp_path: Path) -> None:
    method, _, hub, coder = CodedLightSwitch.build(
        tmp_path=tmp_path, mode=CodedSamplerMode.SCORER, model_budget_usd=5.0
    )
    coder.costs = [2.0, 3.0]
    _cycles(method=method, count=4)
    assert len(coder.calls) == 2
    assert hub.current_version.index == 2
    refits = _refits(path=tmp_path)
    assert [r.get("coding_stopped") for r in refits] == [None, None, True, True]
    assert refits[2]["frozen_version"] == 2
    assert refits[1]["budget_exhausted_at_cycle"] == 1
    assert refits[1]["spent_usd"] == pytest.approx(5.0)


def test_a_refit_cut_off_by_the_budget_keeps_the_previous_version(*, tmp_path: Path) -> None:
    method, _, hub, coder = CodedLightSwitch.build(
        tmp_path=tmp_path, mode=CodedSamplerMode.SCORER, model_budget_usd=5.0
    )
    coder.costs = [2.0, 3.0]
    coder.cut_off = {1}
    _cycles(method=method, count=3)
    assert len(coder.calls) == 2
    assert hub.current_version.index == 1
    sampler = method.sampler(skill_name="TurnOnLight", param_dim=1)
    candidates = [np.array([float(v)]) for v in np.linspace(-1, 1, 10)]
    choice = sampler.sample(
        sampler_inputs=[[1.0, float(c[0])] for c in candidates],
        candidates=candidates,
        explore=False,
    )
    # Version 1's SCORER, not the cut-off refit's flat scores.
    assert choice.was_informed and choice.params is candidates[-1]
    refits = _refits(path=tmp_path)
    assert refits[1]["cut_off_by_budget"] is True
    assert refits[1]["frozen_version"] == 1


def test_zero_means_unlimited(*, tmp_path: Path) -> None:
    method, _, _, coder = CodedLightSwitch.build(tmp_path=tmp_path, mode=CodedSamplerMode.SCORER)
    coder.costs = [50.0, 50.0]
    _cycles(method=method, count=2)
    assert [call["max_budget_usd"] for call in coder.calls] == [0.0, 0.0]


def test_the_cli_defaults_to_full_agentics_twenty_dollars_and_passes_it_on(
    *, tmp_path: Path
) -> None:
    import argparse

    from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
    from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
    from hitl_pmp.methods.practice_makes_perfect.coded_sampler.cli import CodedSamplerCli

    parser = argparse.ArgumentParser()
    CodedSamplerCli.add_arguments(parser=parser)
    settings = tmp_path / "sandbox.json"
    settings.write_text(json.dumps({"image": "x", "robocode_checkout": str(tmp_path)}))
    args = parser.parse_args([
        "--sampler",
        "claude_code",
        "--coded-sampler-sandbox-settings",
        str(settings),
    ])
    assert args.coded_sampler_model_budget == 20.0
    args.output_dir = tmp_path / "run"
    env = LightSwitchEnvironment(grid_size=4)
    hub = CodedSamplerCli.build(args=args, provider=LightSwitchSkillProvider(env=env))
    assert hub is not None and hub.model_budget_usd == 20.0
