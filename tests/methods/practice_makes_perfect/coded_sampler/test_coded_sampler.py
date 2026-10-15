import json
import re
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.core.method.method import InteractionComplete
from hitl_pmp.core.method.types import GroundSkill, ParameterSpec, SamplerConsultation
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skills import LightSwitchSkills
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.methods.practice_makes_perfect.coded_sampler.sampler import (
    CodedProposalSampler,
    CodedScorerSampler,
)
from hitl_pmp.methods.practice_makes_perfect.coded_sampler.types import CodedSamplerMode
from hitl_pmp.methods.practice_makes_perfect.coded_sampler.workspace import (
    GeneratedImports,
    ParameterCheck,
)
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.methods.practice_makes_perfect.wrapped_sampler import LearnedSkillSampler

from .fakes import SCORER, CodedLightSwitch


def _practice(*, method: EesMethod, env: LightSwitchEnvironment, steps: int, seed: int = 0) -> None:
    tasks = LightSwitchTasks(env=env, seed=seed)
    task = tasks.sample_train_task()
    env.set_state(state=task.initial_state)
    policy = method.get_practice_policy(task=task)
    state = env.get_current_state()
    for _ in range(steps):
        try:
            state = env.take_action(action=policy(state).action)
        except InteractionComplete:
            break


def _turn_on(*, env: LightSwitchEnvironment) -> GroundSkill:
    cells = env.get_cells()
    return GroundSkill(
        skill=LightSwitchSkills.TURN_ON_LIGHT, objects=(env.robot, cells[-1], env.light)
    )


def _lines(*, path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_mlp_is_the_default_and_builds_no_coded_sampler() -> None:
    env = LightSwitchEnvironment(grid_size=4)
    from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider

    method = EesMethod(env=env, skill_provider=LightSwitchSkillProvider(env=env), seed=0)
    assert method.coded_sampler is None
    assert type(method.sampler(skill_name="TurnOnLight", param_dim=1)) is LearnedSkillSampler


def test_before_the_first_refit_the_scorer_behaves_as_an_unfitted_mlp(*, tmp_path: Path) -> None:
    method, env, _, _ = CodedLightSwitch.build(tmp_path=tmp_path, mode=CodedSamplerMode.SCORER)
    sampler = method.sampler(skill_name="TurnOnLight", param_dim=1)
    assert isinstance(sampler, CodedScorerSampler)
    candidates = [np.array([float(v)]) for v in np.linspace(-1, 1, 10)]
    rows = [[1.0, float(c[0])] for c in candidates]
    choice = sampler.sample(sampler_inputs=rows, candidates=candidates, explore=True)
    assert (choice.was_random, choice.was_informed) == (False, False)
    del env


def test_refits_continue_one_session_and_the_scorer_runs_the_generated_code(
    *, tmp_path: Path
) -> None:
    method, env, hub, coder = CodedLightSwitch.build(
        tmp_path=tmp_path, mode=CodedSamplerMode.SCORER, epsilon=0.0
    )
    _practice(method=method, env=env, steps=6)
    method.end_cycle()
    _practice(method=method, env=env, steps=6, seed=1)
    method.end_cycle()
    assert [call["resume"] for call in coder.calls] == [False, True]
    refits = _lines(path=tmp_path / "coded_sampler_refits.jsonl")
    assert [r["version"]["index"] for r in refits] == [1, 2]
    sampler = method.sampler(skill_name="TurnOnLight", param_dim=1)
    candidates = [np.array([float(v)]) for v in np.linspace(-1, 1, 10)]
    rows = [[1.0, float(c[0])] for c in candidates]
    choice = sampler.sample(sampler_inputs=rows, candidates=candidates, explore=False)
    assert choice.was_informed
    assert choice.params is candidates[-1]  # the caller's own untransformed vector
    assert hub.current_version.index == 2


def test_history_spans_every_skill_and_refit_slices_neither_drop_nor_repeat(
    *, tmp_path: Path
) -> None:
    method, env, hub, _ = CodedLightSwitch.build(tmp_path=tmp_path, mode=CodedSamplerMode.SCORER)
    _practice(method=method, env=env, steps=8)
    method.end_cycle()
    _practice(method=method, env=env, steps=8, seed=1)
    method.end_cycle()
    data = hub.workspace.root / "data"
    whole = _lines(path=data / "trajectories.jsonl")
    slices = _lines(path=data / "new_since_refit_000.jsonl")
    slices += _lines(path=data / "new_since_refit_001.jsonl")
    assert [e["index"] for e in slices] == [e["index"] for e in whole] == list(range(len(whole)))
    dispatched = {e["data"]["skill"] for e in whole if e["kind"] == "skill_dispatched"}
    # MoveRobot has no parameters and no sampler, but its attempts are still history.
    assert "MoveRobot" in dispatched
    kinds = [e["kind"] for e in whole]
    assert kinds.count("cycle_end") == 2
    assert kinds[0] == "practice_period_start"
    for event in whole:
        if event["kind"] == "skill_dispatched":
            assert "state_before" in event["data"]
        if event["kind"] == "skill_outcome":
            assert "state_after" in event["data"]


def test_evaluation_runs_the_frozen_version_and_records_no_history(*, tmp_path: Path) -> None:
    method, env, hub, _ = CodedLightSwitch.build(tmp_path=tmp_path, mode=CodedSamplerMode.SCORER)
    _practice(method=method, env=env, steps=6)
    method.end_cycle()
    # Anything the workspace holds after the refit is not what runs.
    (hub.workspace.root / "samplers" / "TurnOnLight.py").write_text("raise SystemExit(3)\n")
    before = len(hub.history)
    tasks = LightSwitchTasks(env=env, seed=5)
    task = tasks.sample_test_task()
    env.set_state(state=task.initial_state)
    policy = method.get_task_policy(task=task)
    state = env.get_current_state()
    for _ in range(6):
        state = env.take_action(action=policy(state).action)
    assert len(hub.history) == before
    calls = [
        c for c in _lines(path=tmp_path / "coded_sampler_calls.jsonl") if c["phase"] == "evaluation"
    ]
    assert calls
    assert all(c["ok"] and c["version"] == 1 for c in calls if not c.get("note"))


def test_a_crashing_module_falls_back_uninformed_and_the_error_reaches_the_next_refit(
    *, tmp_path: Path
) -> None:
    crash = "def score(sampler_inputs):\n    raise RuntimeError('boom')\n"
    method, env, hub, coder = CodedLightSwitch.build(
        tmp_path=tmp_path,
        mode=CodedSamplerMode.SCORER,
        sources={"TurnOnLight": crash, "TurnOffLight": crash, "JumpToLight": crash},
    )
    _practice(method=method, env=env, steps=4)
    method.end_cycle()
    sampler = method.sampler(skill_name="TurnOnLight", param_dim=1)
    candidates = [np.array([float(v)]) for v in np.linspace(-1, 1, 10)]
    choice = sampler.sample(
        sampler_inputs=[[1.0, float(c[0])] for c in candidates],
        candidates=candidates,
        explore=False,
    )
    assert (choice.was_random, choice.was_informed) == (False, False)
    # A refit with no new attempts is skipped, so practise before the next one.
    _practice(method=method, env=env, steps=4, seed=1)
    method.end_cycle()
    errors = _lines(path=hub.workspace.root / "data" / "code_errors.jsonl")
    assert any("boom" in str(e["error"]) for e in errors)
    reported = re.search(r"(\d+) errors from your code", str(coder.calls[-1]["prompt"]))
    assert reported is not None and int(reported.group(1)) == len(errors) >= 1


def test_a_garbage_score_vector_is_rejected_not_used(*, tmp_path: Path) -> None:
    bad = "def score(sampler_inputs):\n    return [1.0]\n"
    method, env, _, _ = CodedLightSwitch.build(
        tmp_path=tmp_path, mode=CodedSamplerMode.SCORER, sources={"TurnOnLight": bad}
    )
    _practice(method=method, env=env, steps=2)
    method.end_cycle()
    sampler = method.sampler(skill_name="TurnOnLight", param_dim=1)
    candidates = [np.array([float(v)]) for v in np.linspace(-1, 1, 10)]
    choice = sampler.sample(
        sampler_inputs=[[1.0, float(c[0])] for c in candidates],
        candidates=candidates,
        explore=False,
    )
    assert not choice.was_informed


def test_a_disallowed_import_is_rejected_at_refit(*, tmp_path: Path) -> None:
    torch = "import torch\n" + SCORER
    method, env, hub, _ = CodedLightSwitch.build(
        tmp_path=tmp_path, mode=CodedSamplerMode.SCORER, sources={"TurnOnLight": torch}
    )
    _practice(method=method, env=env, steps=2)
    method.end_cycle()
    assert not hub.has_code(skill_name="TurnOnLight")
    refit = _lines(path=tmp_path / "coded_sampler_refits.jsonl")[0]
    assert refit["import_violations"] == {"TurnOnLight": ["torch"]}


def test_generated_imports_allow_numpy_scipy_sklearn_and_the_stdlib_only() -> None:
    source = (
        "import numpy\nimport scipy.stats\nfrom sklearn import svm\nimport json\nimport torch\n"
    )
    assert GeneratedImports.violations(source=source) == ["torch"]


@pytest.mark.parametrize(
    ("value", "ok"),
    [([0.5], True), ([1.5], False), ([0.1, 0.2], False), (["x"], False), ([float("nan")], False)],
)
def test_parameter_check_never_clips(*, value: object, ok: bool) -> None:
    spec = ParameterSpec(name="p", unit="u", meaning="m", lower=-1.0, upper=1.0)
    assert (ParameterCheck.violation(value=value, specs=[spec]) is None) == ok


def test_sampler_mode_returns_the_generated_vector_and_epsilon_skips_the_code(
    *, tmp_path: Path
) -> None:
    method, env, hub, _ = CodedLightSwitch.build(tmp_path=tmp_path, mode=CodedSamplerMode.SAMPLER)
    _practice(method=method, env=env, steps=4)
    method.end_cycle()
    sampler = method.sampler(skill_name="TurnOnLight", param_dim=1)
    assert isinstance(sampler, CodedProposalSampler)
    state = env.get_current_state()
    hub.begin_decision(ground_skill=_turn_on(env=env), state=state, explore=True)
    candidates = [np.array([-0.9]), np.array([0.9])]
    choices = [
        sampler.sample(
            sampler_inputs=[[1.0, -0.9], [1.0, 0.9]], candidates=candidates, explore=True
        )
        for _ in range(40)
    ]
    informed = [c for c in choices if c.was_informed]
    random = [c for c in choices if c.was_random]
    assert informed and random and len(informed) + len(random) == 40
    assert all(np.allclose(c.params, [0.25]) for c in informed)
    assert all(c.consultation == SamplerConsultation.EPSILON_RANDOM for c in random)


def test_sampler_mode_rejects_an_out_of_bounds_vector_without_clipping(*, tmp_path: Path) -> None:
    out = "def sample(state, grounding, rng):\n    return [3.0]\n"
    method, env, hub, _ = CodedLightSwitch.build(
        tmp_path=tmp_path, mode=CodedSamplerMode.SAMPLER, sources={"TurnOnLight": out}
    )
    _practice(method=method, env=env, steps=2)
    method.end_cycle()
    sampler = method.sampler(skill_name="TurnOnLight", param_dim=1)
    hub.begin_decision(ground_skill=_turn_on(env=env), state=env.get_current_state(), explore=False)
    candidates = [np.array([-0.9]), np.array([0.9])]
    choice = sampler.sample(
        sampler_inputs=[[1.0, -0.9], [1.0, 0.9]], candidates=candidates, explore=False
    )
    assert not choice.was_informed and not choice.was_random
    assert any(choice.params is c for c in candidates)
    calls = _lines(path=tmp_path / "coded_sampler_calls.jsonl")
    notes = [c for c in calls if c.get("bounds_violation")]
    assert notes and "outside" in notes[0]["bounds_violation"]


def test_the_default_layout_description_matches_the_row_the_method_builds(
    *, tmp_path: Path
) -> None:
    from hitl_pmp.methods.practice_makes_perfect.coded_sampler.workspace import (
        SamplerInputLayout,
    )

    method, env, hub, _ = CodedLightSwitch.build(tmp_path=tmp_path, mode=CodedSamplerMode.SCORER)
    ground = _turn_on(env=env)
    env.hard_reset()
    row = method.sampler_input_row(
        ground_skill=ground, state=env.get_current_state(), params=np.array([0.3])
    )
    description = SamplerInputLayout.describe(provider=hub.provider, skill=ground.skill)
    assert len(description["sampler_input"]) == len(row)
    assert description["sampler_input"][-1]["name"] == "param.dlight"


def test_a_module_may_import_its_own_helpers_under_samplers() -> None:
    source = "import toss_features\nfrom helpers.fit import model\nimport torch\n"
    assert GeneratedImports.violations(
        source=source, local_modules={"toss_features", "helpers"}
    ) == ["torch"]


def test_a_refit_keeps_a_module_that_imports_a_local_helper(*, tmp_path: Path) -> None:
    source = "import helper\ndef score(sampler_inputs):\n    return helper.score(sampler_inputs)\n"
    method, env, hub, coder = CodedLightSwitch.build(
        tmp_path=tmp_path, mode=CodedSamplerMode.SCORER, sources={"TurnOnLight": source}
    )
    coder.sources["helper"] = SCORER
    _practice(method=method, env=env, steps=2)
    method.end_cycle()
    assert hub.has_code(skill_name="TurnOnLight")
    sampler = method.sampler(skill_name="TurnOnLight", param_dim=1)
    candidates = [np.array([float(v)]) for v in np.linspace(-1, 1, 10)]
    choice = sampler.sample(
        sampler_inputs=[[1.0, float(c[0])] for c in candidates],
        candidates=candidates,
        explore=False,
    )
    assert choice.was_informed and choice.params is candidates[-1]
