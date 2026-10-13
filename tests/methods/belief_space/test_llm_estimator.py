"""The LLM as competence/cost estimator: no Bayesian update, raw evidence in, the
model's stated estimates out through the same logging accessors."""

import json
import pickle
from pathlib import Path
from typing import Any

import pytest

from hitl_pmp.agentic_runtime.types import Decision, SkillEstimate
from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.belief_space.llm_planner import LlmPracticePlanner, MissingEstimatesError
from hitl_pmp.methods.belief_space.tossing3d_constants import PICK_SKILL, TOSS_SKILL
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.step_protocol import DeploymentSnapshot
from tests.methods.belief_space.test_llm_planner import (
    _build,
    _grounding,
    _last_decision,
    _practice_policy,
    _reset,
    _state,
)

ESTIMATE = {"competence": 0.4, "learning_rate": 0.05, "expected_cost": 3.0}


class EstimatingClient:
    """Answers scripted ids with a per-skill estimates block (or none, when told)."""

    def __init__(self, *, answers: list[Any], estimates: bool = True) -> None:
        self.answers = list(answers)
        self.estimates = estimates
        self.calls: list[dict[str, Any]] = []

    def decide(
        self,
        *,
        context: dict[str, Any],
        candidate_ids: list[str],
        session_key: str,
        require_estimates: bool = False,
        estimate_keys: list[str] | None = None,
    ) -> Decision:
        self.calls.append({
            "context": json.loads(json.dumps(context)),
            "candidate_ids": list(candidate_ids),
            "session_key": session_key,
            "require_estimates": require_estimates,
            "estimate_keys": estimate_keys,
        })
        answer = self.answers.pop(0)
        return Decision(
            action_id=answer,
            reason=f"fake chose {answer}",
            estimates=(
                {key: SkillEstimate(**ESTIMATE) for key in estimate_keys or []}
                if self.estimates
                else None
            ),
            cost_usd=0.01,
            call_index=len(self.calls) - 1,
        )


def _estimator(
    *, answers: list[Any], estimates: bool = True, **kwargs: Any
) -> tuple[Tossing3DPomdpMethod, EstimatingClient]:
    env = Tossing3DEnvironment(scene_bg=False)
    client = EstimatingClient(answers=answers, estimates=estimates)
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env, human_reset_practice_cost=5.0),
        seed=0,
        pomdp_solver="llm",
        pomdp_estimator="llm",
        planner_client=client,
        pomdp_num_particles=200,
        **kwargs,
    )
    return method, client


def _beliefs(*, method: Tossing3DPomdpMethod) -> str:
    return json.dumps(
        {
            name: belief.model_dump(mode="json")
            for name, belief in method.pomdp_state.skill_beliefs.items()
        },
        sort_keys=True,
    )


def _pin_end_cycle(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """end_cycle flushes the in-flight action against the live scene, which is not
    what these tests measure (test_llm_planner pins it the same way)."""
    monkeypatch.setattr(Tossing3DPomdpMethod, "observe_environment_reset", lambda self, **_: None)
    monkeypatch.setattr(Tossing3DEnvironment, "get_current_state", lambda self: object())


def test_the_estimator_requires_the_llm_solver() -> None:
    env = Tossing3DEnvironment(scene_bg=False)
    with pytest.raises(ValueError, match="pomdp_estimator='llm'"):
        Tossing3DPomdpMethod(
            env=env,
            skill_provider=Tossing3DSkillProvider(env=env),
            seed=0,
            pomdp_solver="expectimax",
            pomdp_estimator="llm",
            pomdp_num_particles=200,
        )


def test_observations_leave_the_bayesian_belief_byte_unchanged_while_ees_learns(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    method, _ = _estimator(answers=[])
    pick = _grounding(method=method, name=PICK_SKILL)
    reset = _reset(method=method, side="robot_side")
    before = _beliefs(method=method)
    competence_before = pickle.dumps(method._competence_models.get(pick))  # noqa: SLF001
    method.record_action_cost(ground_skill=pick)
    method.observe_outcome(ground_skill=pick, success=True)
    method.observe_sampler_outcome(
        skill_name=TOSS_SKILL, param_dim=2, sampler_input=[0.0, 0.0], success=False
    )
    method.record_action_cost(ground_skill=reset)
    method.observe_help_granted(state=_state(method=method))
    _pin_end_cycle(monkeypatch=monkeypatch)
    method.end_cycle()
    assert _beliefs(method=method) == before
    assert method._all_attempt_outcomes[pick] == [True]  # noqa: SLF001
    assert pickle.dumps(method._competence_models[pick]) != competence_before  # noqa: SLF001
    # The Bayesian arm, fed the same observation, does move its belief.
    bayesian, _ = _build(answers=[])
    bayesian.record_action_cost(ground_skill=pick)
    bayesian.observe_outcome(ground_skill=pick, success=True)
    assert _beliefs(method=bayesian) != before


def test_the_context_carries_raw_evidence_and_no_bayesian_numbers(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    method, client = _estimator(answers=["STOP", "STOP"])
    pick = _grounding(method=method, name=PICK_SKILL)
    pick_id = LlmPracticePlanner.candidate_id(ground=pick)
    _practice_policy(
        method=method,
        state=_state(method=method),
        atoms=pick.preconditions,
        monkeypatch=monkeypatch,
    )
    method.select_skill_to_practice(true_atoms=pick.preconditions)
    method.record_action_cost(ground_skill=pick)
    method.observe_symbolic_transition(
        ground_skill=pick,
        before_atoms=pick.preconditions,
        after_atoms=pick.preconditions,
        success=False,
        was_random_exploration=False,
    )
    method.observe_outcome(ground_skill=pick, success=False)
    method.observe_sampler_outcome(
        skill_name=TOSS_SKILL, param_dim=2, sampler_input=[0.0, 0.0], success=True
    )
    method.select_skill_to_practice(true_atoms=pick.preconditions)
    call = client.calls[1]
    assert call["require_estimates"] is True
    context = call["context"]
    skills = context["estimate_skills"]
    assert call["estimate_keys"] == skills
    assert {PICK_SKILL, TOSS_SKILL} <= set(skills)
    assert "stop_value" not in context
    for candidate in context["candidates"]:
        assert not {
            "posterior_mean_competence",
            "posterior_mean_learning_rate",
            "expected_cost",
        } & set(candidate)
    assert not any("posterior" in key or key == "stop_value" for key in context["guide"])
    evidence = context["evidence"]
    assert evidence["scope"] == "cumulative"
    assert evidence["attempt_outcomes"] == {pick_id: [False]}
    assert evidence["sampler_training_examples"] == {TOSS_SKILL: [True]}
    assert evidence["pending_training_examples"] == {TOSS_SKILL: 1}
    (effect,) = evidence["failure_effects"]
    assert (effect["skill"], effect["count"]) == (PICK_SKILL, 1)
    assert {"execution_durations", "measured_costs"} <= set(evidence)
    attempts = [e for e in context["history"]["events"] if e["type"] == "attempt"]
    assert attempts[-1]["action"] == pick_id
    assert attempts[-1]["success"] is False


def test_resume_sends_only_the_evidence_since_the_previous_call() -> None:
    method, client = _estimator(answers=["STOP"] * 3, planner_memory="resume")
    pick = _grounding(method=method, name=PICK_SKILL)
    pick_id = LlmPracticePlanner.candidate_id(ground=pick)
    for success in (True, False):
        method.select_skill_to_practice(true_atoms=pick.preconditions)
        method.record_action_cost(ground_skill=pick)
        method.observe_outcome(ground_skill=pick, success=success)
    method.select_skill_to_practice(true_atoms=pick.preconditions)
    first, second, third = (call["context"]["evidence"] for call in client.calls)
    assert first["scope"] == "cumulative"
    assert first["attempt_outcomes"] == {}
    assert second["scope"] == "since_previous_call"
    assert second["attempt_outcomes"] == {pick_id: [True]}
    assert third["scope"] == "since_previous_call"
    assert third["attempt_outcomes"] == {pick_id: [False]}
    assert [e["type"] for e in client.calls[2]["context"]["history"]["events"]] == ["attempt"]


def test_stateless_sends_the_cumulative_evidence_every_call() -> None:
    method, client = _estimator(answers=["STOP"] * 2)
    pick = _grounding(method=method, name=PICK_SKILL)
    pick_id = LlmPracticePlanner.candidate_id(ground=pick)
    for success in (True, False):
        method.record_action_cost(ground_skill=pick)
        method.observe_outcome(ground_skill=pick, success=success)
        method.select_skill_to_practice(true_atoms=pick.preconditions)
    second = client.calls[1]["context"]["evidence"]
    assert second["scope"] == "cumulative"
    assert second["attempt_outcomes"] == {pick_id: [True, False]}


def test_the_accessors_and_the_decision_record_carry_the_stated_estimates(
    *, tmp_path: Path
) -> None:
    method, client = _estimator(answers=["STOP"], decision_log=tmp_path / "decisions.jsonl")
    assert method.practice_skill_competences() == {}
    pick = _grounding(method=method, name=PICK_SKILL)
    method.select_skill_to_practice(true_atoms=pick.preconditions)
    skills = client.calls[0]["estimate_keys"]
    assert method.practice_skill_competences() == {
        f"{skill} (llm estimate)": ESTIMATE["competence"] for skill in skills
    }
    assert method.practice_skill_learning_rates() == {
        f"{skill} (llm estimate)": ESTIMATE["learning_rate"] for skill in skills
    }
    assert method.practice_skill_costs() == {
        f"{skill} (llm estimate)": ESTIMATE["expected_cost"] for skill in skills
    }
    assert method.practice_skill_improvement_potentials() == {}
    assert method.practice_action_values() == {}
    decision = _last_decision(method=method)
    assert decision["estimator"] == "llm"
    assert decision["llm_estimates"] == {skill: ESTIMATE for skill in skills}
    assert decision["competences"] == method.practice_skill_competences()
    assert decision["estimated_costs"] == method.practice_skill_costs()
    assert decision["improvement_potentials"] == {}
    assert not [e for e in decision["search"] if e["event"] == "stop_value"]
    summary = next(e for e in decision["search"] if e["event"] == "search_summary")
    assert summary["stop_value_source"] is None
    assert summary["estimator"] == "llm"


def test_a_missing_estimates_block_is_an_error_never_filled_from_the_posterior() -> None:
    method, _ = _estimator(answers=["STOP"], estimates=False)
    pick = _grounding(method=method, name=PICK_SKILL)
    with pytest.raises(MissingEstimatesError):
        method.select_skill_to_practice(true_atoms=pick.preconditions)
    assert method.practice_skill_competences() == {}
    assert method.practice_skill_costs() == {}


def test_the_bayesian_estimator_does_not_ask_for_estimates() -> None:
    method, client = _build(answers=["STOP"])
    pick = _grounding(method=method, name=PICK_SKILL)
    method.select_skill_to_practice(true_atoms=pick.preconditions)
    assert client.calls[0]["require_estimates"] is False
    assert "stop_value" in client.calls[0]["context"]
    assert "evidence" not in client.calls[0]["context"]


def test_the_protocol_records_the_estimator() -> None:
    method, _ = _estimator(answers=[])
    planner = method.pomdp_planner
    assert isinstance(planner, LlmPracticePlanner)
    assert planner.protocol()["estimator"] == "llm"


def _observe(*, method: Tossing3DPomdpMethod, pick: GroundSkill) -> None:
    for success in (True, False, True):
        method.record_action_cost(ground_skill=pick)
        method.observe_outcome(ground_skill=pick, success=success)


def test_snapshots_after_the_same_outcomes_match_the_bayesian_llm_arm() -> None:
    estimator, _ = _estimator(answers=[])
    bayesian, _ = _build(answers=[])
    pick = _grounding(method=estimator, name=PICK_SKILL)
    _observe(method=estimator, pick=pick)
    _observe(method=bayesian, pick=pick)
    assert DeploymentSnapshot.encode(method=estimator) == DeploymentSnapshot.encode(method=bayesian)
