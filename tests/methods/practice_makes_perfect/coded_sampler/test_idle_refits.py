"""A refit is skipped only when nothing was attempted since the last one.

An MLP refit from scratch on unchanged data with an unchanged seed reproduces its
weights, so it is a no-op; the coding agent's refit with nothing new is skipped to
match. Bookkeeping events (a period starting, an environment reset, a cycle ending)
are not practice data."""

import json
from pathlib import Path

from hitl_pmp.core.method.method import InteractionComplete
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.methods.practice_makes_perfect.coded_sampler.types import CodedSamplerMode
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

from .fakes import CodedLightSwitch


def _refits(*, path: Path) -> list[dict[str, object]]:
    lines = (path / "coded_sampler_refits.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def _cycle(*, method: EesMethod, env: LightSwitchEnvironment, steps: int, seed: int) -> None:
    task = LightSwitchTasks(env=env, seed=seed).sample_train_task()
    env.set_state(state=task.initial_state)
    policy = method.get_practice_policy(task=task)
    state = env.get_current_state()
    for _ in range(steps):
        try:
            state = env.take_action(action=policy(state).action)
        except InteractionComplete:
            break
    method.end_cycle()


def test_a_cycle_without_attempts_skips_the_refit_and_keeps_the_version(*, tmp_path: Path) -> None:
    method, env, hub, coder = CodedLightSwitch.build(
        tmp_path=tmp_path, mode=CodedSamplerMode.SCORER
    )
    _cycle(method=method, env=env, steps=4, seed=0)
    _cycle(method=method, env=env, steps=0, seed=1)
    assert len(coder.calls) == 1
    assert hub.current_version.index == 1
    refits = _refits(path=tmp_path)
    assert refits[1]["skipped"] == "no_new_practice_data"
    assert refits[1]["version_index"] == 1


def test_the_skipped_cycles_events_reach_the_next_refit(*, tmp_path: Path) -> None:
    method, env, hub, coder = CodedLightSwitch.build(
        tmp_path=tmp_path, mode=CodedSamplerMode.SCORER
    )
    _cycle(method=method, env=env, steps=4, seed=0)
    first_end = _refits(path=tmp_path)[0]["history_events"][1]
    _cycle(method=method, env=env, steps=0, seed=1)
    _cycle(method=method, env=env, steps=4, seed=2)
    assert len(coder.calls) == 2
    third = _refits(path=tmp_path)[2]
    assert third["history_events"][0] == first_end
    new = (hub.workspace.root / "data" / "new_since_refit_002.jsonl").read_text().splitlines()
    kinds = [json.loads(line)["kind"] for line in new]
    assert kinds.count("cycle_end") == 2  # the skipped cycle's and its own
    assert f"lines {first_end} to" in str(coder.calls[1]["prompt"])


def test_a_first_cycle_without_attempts_does_not_invoke_the_coder(*, tmp_path: Path) -> None:
    method, env, hub, coder = CodedLightSwitch.build(
        tmp_path=tmp_path, mode=CodedSamplerMode.SCORER
    )
    _cycle(method=method, env=env, steps=0, seed=0)
    assert coder.calls == []
    assert hub.current_version.index == 0


def test_a_cycle_with_attempts_always_refits(*, tmp_path: Path) -> None:
    method, env, _, coder = CodedLightSwitch.build(tmp_path=tmp_path, mode=CodedSamplerMode.SCORER)
    for seed in range(3):
        _cycle(method=method, env=env, steps=3, seed=seed)
    assert len(coder.calls) == 3
