"""Unit orchestration tests with scripted vision and no simulator execution."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from hitl_pmp.agentic_runtime.sandbox import PolicyExecution
from hitl_pmp.agentic_runtime.types import GeneratedLibrary, OptionContract
from hitl_pmp.core.method.method import HumanCubeBinResetRequested, InteractionComplete
from hitl_pmp.environments.tossing3d.agentic_bridge import Tossing3DAgenticBridge
from hitl_pmp.methods.agentic_options.cli import OptionExecutor
from hitl_pmp.methods.belief_space.tossing3d_constants import RESET_SKILL, TOSS_SKILL

from .support import ScriptedJudge, method, task


@pytest.fixture
def observed_frames(*, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    frame = {"number": 0}
    monkeypatch.setattr(
        Tossing3DAgenticBridge,
        "observe",
        lambda self: {"images": [], "proprioception": {}, "control_step": frame["number"]},
    )
    return frame


def test_final_action_is_settled_before_session_learning(
    *, tmp_path: Path, observed_frames: dict[str, int]
) -> None:
    actor = method(tmp_path=tmp_path, judge=ScriptedJudge(clusters=["near", "near"]))
    fixture = task()
    policy = actor.get_practice_policy(task=fixture)
    actor.observe_practice_action_budget(remaining_actions=1)
    selected = policy(fixture.initial_state)
    assert selected.label == "toss"
    assert actor.learner.events == ["begin"]
    assert actor.learner_belief.real_observations == []

    observed_frames["number"] = 1
    actor.end_cycle()

    assert actor.learner.events == ["begin", "record", "learn"]
    assert actor.learner.evidence[0]["after"]["control_step"] == 1
    assert actor.learner_belief.real_observations == [(TOSS_SKILL, True, 1.0)]
    assert actor.learner_belief.session_inputs[0].pending_examples[TOSS_SKILL] == 1
    assert actor.belief.pending_examples == {}
    actor.end_cycle()
    assert actor.learner.events == ["begin", "record", "learn"]


def test_new_session_resets_only_session_cost_not_the_world(
    *, tmp_path: Path, observed_frames: dict[str, int]
) -> None:
    actor = method(tmp_path=tmp_path, judge=ScriptedJudge(clusters=["near", "near", "near"]))
    fixture = task()
    policy = actor.get_practice_policy(task=fixture)
    policy(fixture.initial_state)
    observed_frames["number"] = 1
    actor.end_cycle()
    assert actor.belief.accumulated_cost == 1.0

    actor.get_practice_policy(task=fixture)

    assert actor.belief.accumulated_cost == 0.0
    assert observed_frames["number"] == 1
    assert actor.env._backend is None  # noqa: SLF001 -- no simulator reset in unit fixture


def test_unresolved_initial_cluster_stops_without_outcome_evidence(
    *, tmp_path: Path, observed_frames: dict[str, int]
) -> None:
    actor = method(tmp_path=tmp_path, judge=ScriptedJudge(clusters=[None]))
    fixture = task()
    policy = actor.get_practice_policy(task=fixture)
    with pytest.raises(InteractionComplete) as stopped:
        policy(fixture.initial_state)
    assert not stopped.value.planner_stop
    actor.end_cycle()
    assert actor.learner.evidence == []
    assert actor.learner_belief.real_observations == []
    assert actor.chain.counts == {}


@pytest.mark.parametrize("unknown", ["termination", "success", "destination"])
def test_unresolved_execution_is_saved_without_inventing_success_or_failure(
    *, tmp_path: Path, observed_frames: dict[str, int], unknown: str
) -> None:
    judge = ScriptedJudge(
        clusters=["near", None if unknown == "destination" else "near"],
        judgments={} if unknown == "destination" else {unknown: None},
    )
    actor = method(tmp_path=tmp_path, judge=judge)
    fixture = task()
    policy = actor.get_practice_policy(task=fixture)
    policy(fixture.initial_state)
    observed_frames["number"] = 1
    with pytest.raises(InteractionComplete):
        policy(fixture.initial_state)
    assert len(actor.learner.evidence) == 1
    assert actor.belief.accumulated_cost == 1.0
    assert actor.learner_belief.real_observations == []
    assert actor.chain.counts == {}


def test_human_transition_is_scored_after_help_not_the_pre_reset_hook(
    *, tmp_path: Path, observed_frames: dict[str, int]
) -> None:
    judge = ScriptedJudge(clusters=["blocked", "near"])
    actor = method(tmp_path=tmp_path, judge=judge)
    fixture = task()
    policy = actor.get_practice_policy(task=fixture)
    actor.observe_practice_action_budget(remaining_actions=3)
    with pytest.raises(HumanCubeBinResetRequested) as request:
        policy(fixture.initial_state)
    assert request.value.destination == "robot_side"
    assert request.value.cost == 5.0
    actor.observe_environment_reset(state=fixture.initial_state)
    assert actor.learner.evidence == []
    assert judge.phases == ["classify", "initiation"]

    observed_frames["number"] = 10
    actor.observe_help_granted(state=fixture.initial_state)

    assert actor.learner.evidence[0]["before"]["control_step"] == 0
    assert actor.learner.evidence[0]["after"]["control_step"] == 10
    assert actor.learner_belief.real_observations == [(RESET_SKILL, True, 5.0)]
    assert actor.belief.accumulated_cost == 5.0
    actor.end_cycle()
    assert len(actor.learner.evidence) == 1


def test_evaluation_does_not_train_or_use_practice_initiation_rejections(
    *, tmp_path: Path, observed_frames: dict[str, int]
) -> None:
    actor = method(tmp_path=tmp_path, judge=ScriptedJudge(clusters=["near"]))
    actor.chain.blocked_actions.add(("near", "toss"))
    before = actor.belief.model_dump_json()
    fixture = task()

    selected = actor.get_task_policy(task=fixture)(fixture.initial_state)

    assert selected.label == "toss"
    assert actor.learner.events == []
    assert actor.learner.evidence == []
    assert actor.learner_belief.real_observations == []
    assert actor.belief.model_dump_json() == before


def test_judgment_exception_preserves_paid_execution_without_fabricating_evidence(
    *, tmp_path: Path, observed_frames: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    judge = ScriptedJudge(clusters=["near"])
    actor = method(tmp_path=tmp_path, judge=judge)
    fixture = task()
    policy = actor.get_practice_policy(task=fixture)
    assert policy(fixture.initial_state).label == "toss"
    before_pending = dict(actor.belief.pending_examples)
    actor.execution_records["toss"] = {
        "steps": 7,
        "controller_done": True,
        "trajectory_path": "fixture-trajectory.jsonl",
    }
    observed_frames["number"] = 7

    def provider_failure(**kwargs: Any) -> None:
        raise RuntimeError("scripted provider failure after execution")

    monkeypatch.setattr(judge, "check", provider_failure)
    with pytest.raises(InteractionComplete):
        policy(fixture.initial_state)

    assert actor.belief.accumulated_cost == 1.0
    assert actor.belief.pending_examples == before_pending
    assert actor.learner_belief.real_observations == []
    assert actor.chain.counts == {}
    assert len(actor.learner.evidence) == 1
    evidence = actor.learner.evidence[0]
    assert evidence["resolved"] is False
    assert evidence["judgment_error"] == "RuntimeError"
    assert evidence["cost"] == 1.0
    assert evidence["execution"]["steps"] == 7
    assert evidence["trajectory_path"] == "fixture-trajectory.jsonl"
    assert "success" not in evidence
    audit = [json.loads(line) for line in actor.audit_path.read_text().splitlines()]
    assert [entry["event"] for entry in audit][-2:] == ["executed_unjudged", "judgment_error"]
    actor.end_cycle()
    assert len(actor.learner.evidence) == 1
    assert actor.belief.accumulated_cost == 1.0


def test_evaluation_callback_audits_visual_outcome_without_training(
    *, tmp_path: Path, observed_frames: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    judge = ScriptedJudge(clusters=[], judgments={"termination": True, "success": False})
    checks = Mock(wraps=judge.check)
    monkeypatch.setattr(judge, "check", checks)
    actor = method(tmp_path=tmp_path, judge=judge)
    final = {"images": [], "proprioception": {}, "control_step": 7}

    class ScriptedExecutor:
        @staticmethod
        def execute(
            *, library: GeneratedLibrary, option: OptionContract, bridge: Tossing3DAgenticBridge
        ) -> PolicyExecution:
            return PolicyExecution(
                steps=7,
                controller_done=True,
                final_observation=final,
                trajectory_path="fixture-evaluation.jsonl",
            )

    journal: dict[str, dict[str, Any]] = {}
    callback = OptionExecutor(
        env=actor.evaluation_env,
        option=actor.chain.options["toss"],
        executor=ScriptedExecutor(),
        learner=actor.learner,
        records=journal,
        evaluation_judge=judge,
        audit_path=tmp_path / "evaluation.jsonl",
    )
    before = actor.belief.model_dump_json()

    execution = callback(params=np.zeros(4))

    assert execution.terminated  # Controller completion remains distinct from visual success.
    assert execution.steps == 7
    assert actor.learner.events == []
    assert actor.learner.evidence == []
    assert actor.belief.model_dump_json() == before
    assert actor.execution_records == {}
    assert judge.phases == ["termination", "success"]
    for call in checks.call_args_list:
        # Stopping is real control evidence, but is never a task-success label.
        control = call.kwargs["observation"]["execution"]
        assert control["controller_done"] is True and control["steps"] == 7
        assert "success" not in control
    audit = json.loads((tmp_path / "evaluation.jsonl").read_text())
    assert audit["termination"]["value"] is True
    assert audit["success"]["value"] is False
    assert journal["toss"]["trajectory_path"] == "fixture-evaluation.jsonl"
