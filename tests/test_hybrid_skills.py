"""The hybrid changes skill learning, not the physical budget or planner."""

# ruff: noqa: SLF001, PLR0917 -- fixtures and generated controller interface

import json
from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.hybrid_skills.artifacts import SessionBudget, SkillBundle
from hitl_pmp.hybrid_skills.method import HybridMethod
from hitl_pmp.methods.belief_space.tossing3d_constants import TOSS_SKILL
from hitl_pmp.planning.grounding import SkillGrounder


def test_bundle_is_frozen_and_rejects_paths_outside_submission(tmp_path):
    (tmp_path / "skills.py").write_text("class GeneratedSkills: pass\n")
    bundle = SkillBundle.read(directory=tmp_path)
    (tmp_path / "skills.py").write_text("class GeneratedSkills: changed = True\n")
    assert bundle.files["skills.py"] == "class GeneratedSkills: pass\n"
    with pytest.raises(ValueError):
        SkillBundle(files={"../escape.py": "pass", "skills.py": "pass"})


def test_resumed_costs_are_cumulative_and_missing_cost_fails_closed():
    budget = SessionBudget(limit=20)
    budget.record(cumulative=3)
    budget.record(cumulative=5)
    assert budget.spent == 5
    assert budget.remaining == 15
    with pytest.raises(ValueError):
        budget.record(cumulative=None)
    with pytest.raises(ValueError):
        budget.record(cumulative=4)
    budget.record(cumulative=20)
    assert budget.exhausted


def test_agentic_toss_bypasses_sampler_but_counts_learning_evidence():
    env = Tossing3DEnvironment(scene_bg=False)
    method = HybridMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env),
        seed=0,
        pomdp_inference_engine="grid",
        pomdp_search_depth=6,
        pomdp_observation_probability_weight=0,
    )
    universe = SkillGrounder.all_possible_ground_atoms(
        objects=method.objects(), predicates=method.predicates()
    )
    toss = next(
        s
        for s in SkillGrounder.applicable_ground_skills(
            skills=method.skills(), objects=method.objects(), true_atoms=universe
        )
        if s.skill.name == TOSS_SKILL
    )
    action, record = method.execute_ground_skill(ground_skill=toss, state=None, explore=True)
    assert int(action.action[0]) == env.move_to_toss_location_and_toss_id
    assert record is None
    assert action.label.startswith(TOSS_SKILL + "(")
    assert "preconditions" not in action.label
    assert not method._samplers
    method.observe_outcome(ground_skill=toss, success=False)
    assert method._pomdp_state.pending_examples[TOSS_SKILL] == 1
    # No binary-sampler warmup gate applies to generated controllers.
    assert method._pomdp_state.sampler_training == {}


def test_worker_serializes_numpy_without_changing_schedule(monkeypatch, tmp_path):
    from hitl_pmp.hybrid_skills.worker import run_controller

    actions = []

    class Controller:
        def __init__(self, *args):
            pass

        def reset(self, observation, skill):
            self.index = 0

        def get_action(self, observation):
            self.index += 1
            return np.ones((2, 18)) if self.index == 1 else None

    def request(*, payload):
        json.dumps(payload, allow_nan=False)
        if payload["operation"] == "step":
            actions.append(payload["action"])
        return {"action_spec": {}}

    assert run_controller(controller_type=Controller, request=request, skill="PickCube", limit=5)
    assert actions == [np.ones((2, 18)).tolist()]


def test_snapshot_restores_code_without_executing_it(tmp_path):
    from hitl_pmp.hybrid_skills.execution import HybridEnvironment, HybridSnapshot
    from hitl_pmp.hybrid_skills.method import HybridDeploymentMethod

    env = HybridEnvironment(scene_bg=False)
    env.bundle = SkillBundle(files={"skills.py": "raise RuntimeError('must stay isolated')"})
    provider = Tossing3DSkillProvider(env=env)
    method = HybridMethod(env=env, skill_provider=provider, seed=0, pomdp_inference_engine="grid")
    raw = HybridSnapshot.encode(method=method)
    env.bundle = SkillBundle(files={"skills.py": "pass"})
    restored = HybridSnapshot.restore(raw=raw, env=env, provider=provider)
    assert isinstance(restored, HybridDeploymentMethod)
    assert "must stay isolated" in env.bundle.files["skills.py"]
    assert not restored._samplers


def test_evidence_excludes_evaluation_and_host_diagnostics(tmp_path, monkeypatch):
    from hitl_pmp.environments.tossing3d.agentic_bridge import Tossing3DAgenticBridge
    from hitl_pmp.hybrid_skills.learner import SkillLearner

    monkeypatch.setattr(Tossing3DAgenticBridge, "observe", lambda self: {"objects": []})
    workspace = tmp_path / "coding/sandbox"
    workspace.mkdir(parents=True)
    trajectories = tmp_path / "practice_trajectories"
    trajectories.mkdir()
    (trajectories / "trajectory-1.jsonl").write_text('{"observation": {}}\n')
    (tmp_path / "step_events.jsonl").write_text(
        '{"event":"robot","cost":3}\n{"event":"measurement","num_solved":9}\n'
    )
    (tmp_path / "status.json").write_text('{"evaluations_completed":10}')
    learner = SkillLearner.__new__(SkillLearner)
    learner.env = Tossing3DEnvironment(scene_bg=False)
    learner.output, learner.workspace, learner.calls = tmp_path, workspace, 1
    learner.settings = SimpleNamespace(artifact_dir=trajectories)
    from hitl_pmp.core.practice_costs import PracticeCosts

    learner.costs = PracticeCosts()
    learner.budget = SessionBudget(limit=20)
    learner.prepare_evidence()
    evidence = "".join(p.read_text() for p in (workspace / "evidence").glob("*"))
    assert '"cost": 3' in evidence
    assert "num_solved" not in evidence
    assert not (workspace / "status.json").exists()


def test_prompt_preserves_original_hint_and_costs():
    from hitl_pmp.core.practice_costs import PracticeCosts
    from hitl_pmp.hybrid_skills.learner import hybrid_prompt

    prompt = hybrid_prompt(costs=PracticeCosts(), mat_size=1)
    assert "hint: in order to minimize human cost" in prompt
    assert "robot_spec.json" not in prompt
    assert "GeneratedApproach" not in prompt
    assert "env.finish_adaptation" not in prompt
    assert "fragile" in prompt
    assert "20 skill invocations" in prompt


def test_learner_resumes_one_session_and_does_not_replenish_budget(tmp_path, monkeypatch):
    from hitl_pmp.agentic_runtime.sandbox import SandboxSettings
    from hitl_pmp.core.practice_costs import PracticeCosts
    from hitl_pmp.environments.tossing3d.agentic_bridge import Tossing3DAgenticBridge
    from hitl_pmp.hybrid_skills import learner as module
    from hitl_pmp.hybrid_skills.artifacts import ModelBudgetExhausted
    from hitl_pmp.hybrid_skills.learner import SkillLearner

    monkeypatch.setattr(Tossing3DAgenticBridge, "observe", lambda self: {"objects": []})
    calls = []

    class Transport:
        def __init__(self, **kwargs):
            pass

        def modules(self):
            return (
                SimpleNamespace(SandboxConfig=lambda **kw: SimpleNamespace(**kw)),
                SimpleNamespace(create_backend=lambda config: None),
                SimpleNamespace(DictConfig=lambda x: x),
            )

        def run_with_recovery(self, **kwargs):
            calls.append(kwargs["config"])
            return SimpleNamespace(total_cost_usd=[2, 20][len(calls) - 1], success=True, error=None)

    monkeypatch.setattr(module, "DeadlineTransport", Transport)
    env = Tossing3DEnvironment(scene_bg=False)
    # This test exercises source publication without starting the simulator.
    from hitl_pmp.hybrid_skills.execution import HybridEnvironment

    env = HybridEnvironment(**env.model_dump())
    settings = SandboxSettings(
        image="fake", robocode_checkout=tmp_path, artifact_dir=tmp_path / "trajectories"
    )
    learner = SkillLearner(output=tmp_path, env=env, settings=settings, costs=PracticeCosts())
    # Even an exhausted initial coding call must leave an honest, evaluable bootstrap.
    assert env.bundle is not None
    assert env.bundle.files["skills.py"] == (learner.submission / "skills.py").read_text()
    learner.revise()
    learner.revise()
    assert [c.resume_previous_session for c in calls] == [False, True]
    assert [c.max_budget_usd for c in calls] == [20, 18]
    assert learner.budget.spent == 20
    with pytest.raises(ModelBudgetExhausted):
        learner.revise()
    assert len(calls) == 2
