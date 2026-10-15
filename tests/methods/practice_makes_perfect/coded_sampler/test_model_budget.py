"""The coded sampler's per-run model budget, accounted as Full-Agentic's is.

The Claude CLI reports `total_cost_usd` cumulatively over a session, so a resumed
session's figure already contains every earlier refit: spend is the session's
cumulative cost, and one refit costs the rise in it."""

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.core.method.method import InteractionComplete
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.methods.practice_makes_perfect.coded_sampler.cli import CodedSamplerCli
from hitl_pmp.methods.practice_makes_perfect.coded_sampler.coder import StreamLog
from hitl_pmp.methods.practice_makes_perfect.coded_sampler.types import CodedSamplerMode
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

from .fakes import CodedLightSwitch


def _refits(*, path: Path) -> list[dict[str, object]]:
    lines = (path / "coded_sampler_refits.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def _practice_cycles(*, method: EesMethod, env: LightSwitchEnvironment, count: int) -> None:
    """`count` cycles, each with real practice attempts, so no refit is idle."""
    for cycle in range(count):
        task = LightSwitchTasks(env=env, seed=cycle).sample_train_task()
        env.set_state(state=task.initial_state)
        policy = method.get_practice_policy(task=task)
        state = env.get_current_state()
        for _ in range(3):
            try:
                state = env.take_action(action=policy(state).action)
            except InteractionComplete:
                break
        method.end_cycle()


def test_each_refit_is_offered_the_budget_less_the_sessions_cumulative_cost(
    *, tmp_path: Path
) -> None:
    method, env, _, coder = CodedLightSwitch.build(
        tmp_path=tmp_path, mode=CodedSamplerMode.SCORER, model_budget_usd=20.0
    )
    coder.costs = [3.0, 5.5, 1.25]
    _practice_cycles(method=method, env=env, count=3)
    offered = [call["max_budget_usd"] for call in coder.calls]
    assert offered == pytest.approx([20.0, 17.0, 11.5])
    refits = _refits(path=tmp_path)
    assert [r["cost_usd_this_refit"] for r in refits] == pytest.approx([3.0, 5.5, 1.25])
    assert [r["session_cumulative_usd"] for r in refits] == pytest.approx([3.0, 8.5, 9.75])
    assert [r["spent_usd"] for r in refits] == pytest.approx([3.0, 8.5, 9.75])


def test_a_refit_that_starts_a_new_session_counts_its_whole_cumulative(*, tmp_path: Path) -> None:
    method, env, _, coder = CodedLightSwitch.build(
        tmp_path=tmp_path, mode=CodedSamplerMode.SCORER, model_budget_usd=20.0
    )
    coder.costs = [3.0, 2.0, 1.0]
    coder.new_session_at = {1}
    _practice_cycles(method=method, env=env, count=3)
    assert [call["max_budget_usd"] for call in coder.calls] == pytest.approx([20.0, 17.0, 15.0])
    assert _refits(path=tmp_path)[-1]["spent_usd"] == pytest.approx(6.0)


def test_retried_attempts_in_one_session_are_counted_once() -> None:
    """Two CLI results in one refit (an aborted attempt, then its resumed retry) carry
    cumulative costs for the same session: the larger already contains the smaller."""
    lines = [
        json.dumps({"type": "system", "subtype": "init", "session_id": "A"}),
        json.dumps({"type": "result", "session_id": "A", "total_cost_usd": 2.0}),
        json.dumps({"type": "system", "subtype": "init", "session_id": "A"}),
        json.dumps({"type": "result", "session_id": "A", "total_cost_usd": 4.0}),
    ]
    assert StreamLog.summarize(lines=lines)["session_costs"] == {"A": 4.0}


def test_a_retry_whose_resume_failed_counts_both_sessions(*, tmp_path: Path) -> None:
    lines = [
        json.dumps({"type": "result", "session_id": "A", "total_cost_usd": 2.0}),
        json.dumps({"type": "result", "session_id": "B", "total_cost_usd": 1.5}),
    ]
    assert StreamLog.summarize(lines=lines)["session_costs"] == {"A": 2.0, "B": 1.5}


def test_once_the_budget_is_spent_the_coder_is_never_invoked_again(*, tmp_path: Path) -> None:
    method, env, hub, coder = CodedLightSwitch.build(
        tmp_path=tmp_path, mode=CodedSamplerMode.SCORER, model_budget_usd=5.0
    )
    coder.costs = [2.0, 3.0]
    _practice_cycles(method=method, env=env, count=4)
    assert len(coder.calls) == 2
    assert hub.current_version.index == 2
    refits = _refits(path=tmp_path)
    assert [r.get("coding_stopped") for r in refits] == [None, None, True, True]
    assert refits[2]["frozen_version"] == 2
    assert refits[1]["budget_exhausted_at_cycle"] == 1
    assert refits[1]["spent_usd"] == pytest.approx(5.0)


def test_a_missing_cost_stops_coding_under_a_cap(*, tmp_path: Path) -> None:
    method, env, hub, coder = CodedLightSwitch.build(
        tmp_path=tmp_path, mode=CodedSamplerMode.SCORER, model_budget_usd=5.0
    )
    coder.costs = [1.0, 1.0]
    coder.missing_cost = {1}
    _practice_cycles(method=method, env=env, count=3)
    assert len(coder.calls) == 2
    assert hub.current_version.index == 1


def test_a_refit_cut_off_by_the_budget_keeps_the_previous_version(*, tmp_path: Path) -> None:
    method, env, hub, coder = CodedLightSwitch.build(
        tmp_path=tmp_path, mode=CodedSamplerMode.SCORER, model_budget_usd=5.0
    )
    coder.costs = [2.0, 3.0]
    coder.cut_off = {1}
    _practice_cycles(method=method, env=env, count=3)
    assert len(coder.calls) == 2
    assert hub.current_version.index == 1
    sampler = method.sampler(skill_name="TurnOnLight", param_dim=1)
    candidates = [np.array([float(v)]) for v in np.linspace(-1, 1, 10)]
    choice = sampler.sample(
        sampler_inputs=[[1.0, float(c[0])] for c in candidates],
        candidates=candidates,
        explore=False,
    )
    # Version 1's scorer, not the cut-off refit's flat scores.
    assert choice.was_informed and choice.params is candidates[-1]
    refits = _refits(path=tmp_path)
    assert refits[1]["cut_off_by_budget"] is True
    assert refits[1]["frozen_version"] == 1


def test_zero_means_unlimited(*, tmp_path: Path) -> None:
    method, env, _, coder = CodedLightSwitch.build(tmp_path=tmp_path, mode=CodedSamplerMode.SCORER)
    coder.costs = [50.0, 50.0]
    _practice_cycles(method=method, env=env, count=2)
    assert [call["max_budget_usd"] for call in coder.calls] == [0.0, 0.0]


def test_the_cli_defaults_to_full_agentics_twenty_dollars_and_passes_it_on(
    *, tmp_path: Path
) -> None:
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
