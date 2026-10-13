"""The LLM practice planner at the belief-space decision seam, with a fake client."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hitl_pmp.agentic_runtime.types import Decision, PlannerMemory
from hitl_pmp.core.method.method import HumanCubeBinResetRequested, InteractionComplete
from hitl_pmp.core.method.types import GroundSkill, LabeledAction
from hitl_pmp.core.problem.tasks.types import Goal, Task
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.types import Tossing3DState
from hitl_pmp.hybrid_skills.artifacts import ModelBudgetExhausted
from hitl_pmp.methods.belief_space.llm_planner import (
    LlmPracticePlanner,
    UnknownDecisionError,
)
from hitl_pmp.methods.belief_space.tossing3d_constants import PICK_SKILL, RESET_SKILL, TOSS_SKILL
from hitl_pmp.methods.belief_space.tossing3d_method import (
    Tossing3DPomdpMethod,
    make_belief_space_planner,
)
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    mean_competence,
    mean_learning_rate,
)
from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
    estimated_action_cost,
    make_tossing3d_search_state,
)
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.planning.grounding import SkillGrounder
from hitl_pmp.step_protocol import DeploymentSnapshot


class FakeClient:
    """Answers scripted action ids; never validates them, so the planner must."""

    def __init__(self, *, answers: list[Any]) -> None:
        self.answers = list(answers)
        self.calls: list[dict[str, Any]] = []

    def decide(
        self,
        *,
        context: dict[str, Any],
        candidate_ids: list[str],
        session_key: str,
        require_estimates: bool = False,
    ) -> Decision:
        self.calls.append({
            "context": json.loads(json.dumps(context)),
            "candidate_ids": list(candidate_ids),
            "session_key": session_key,
            "require_estimates": require_estimates,
        })
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return Decision(
            action_id=answer,
            reason=f"fake chose {answer}",
            estimates=None,
            cost_usd=0.01,
            call_index=len(self.calls) - 1,
        )


def _build(*, answers: list[Any], **kwargs: Any) -> tuple[Tossing3DPomdpMethod, FakeClient]:
    env = Tossing3DEnvironment(scene_bg=False)
    client = FakeClient(answers=answers)
    config: dict[str, Any] = {
        "pomdp_solver": "llm",
        "planner_client": client,
        "pomdp_num_particles": 200,
        **kwargs,
    }
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env, human_reset_practice_cost=5.0),
        seed=0,
        **config,
    )
    return method, client


def _grounding(*, method: Tossing3DPomdpMethod, name: str) -> GroundSkill:
    universe = SkillGrounder.all_possible_ground_atoms(
        objects=method.objects(), predicates=method.predicates()
    )
    return next(
        skill
        for skill in SkillGrounder.applicable_ground_skills(
            skills=method.skills(), objects=method.objects(), true_atoms=universe
        )
        if skill.skill.name == name
    )


def _reset(*, method: Tossing3DPomdpMethod, side: str) -> GroundSkill:
    return next(
        reset
        for reset in method.skill_provider.human_cube_bin_reset_skills()
        if reset.objects[-1].name == side
    )


def _state(*, method: Tossing3DPomdpMethod) -> Tossing3DState:
    return Tossing3DState(
        data={obj: np.zeros(obj.type.dim) for obj in method.objects()},
        abstract_atoms=frozenset(),
    )


def _practice_policy(
    *,
    method: Tossing3DPomdpMethod,
    state: Tossing3DState,
    atoms: frozenset[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Tossing3D's predicates are upstream's classifiers and need a live scene, so the
    symbolic state the policy sees is pinned directly."""
    monkeypatch.setattr(Tossing3DPomdpMethod, "abstract_state", lambda self, *, state: atoms)
    toss = _grounding(method=method, name=TOSS_SKILL)
    goal = Goal(atoms=frozenset(a for a in toss.add_effects if a.predicate.name == "InBin"))
    return method.get_practice_policy(task=Task(initial_state=state, goal=goal))


def _last_decision(*, method: Tossing3DPomdpMethod) -> dict[str, Any]:
    assert method.decision_log is not None
    records = [json.loads(line) for line in method.decision_log.read_text().splitlines()]
    return next(record for record in reversed(records) if record["event"] == "decision")


def test_make_belief_space_planner_rejects_an_unknown_solver_name() -> None:
    with pytest.raises(ValueError, match="nonsense"):
        make_belief_space_planner(
            planner=None,
            solver="nonsense",  # type: ignore[arg-type]
            max_iterations=1,
            seed=0,
            observation_probability_weight=0.0,
        )


def test_the_llm_solver_requires_a_client_or_sandbox() -> None:
    env = Tossing3DEnvironment(scene_bg=False)
    with pytest.raises(ValueError, match="planner_sandbox"):
        Tossing3DPomdpMethod(
            env=env,
            skill_provider=Tossing3DSkillProvider(env=env),
            seed=0,
            pomdp_solver="llm",
            pomdp_num_particles=200,
        )


def test_a_stop_answer_ends_the_period_as_a_planner_stop(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method, client = _build(answers=["STOP"], decision_log=tmp_path / "decisions.jsonl")
    pick = _grounding(method=method, name=PICK_SKILL)
    state = _state(method=method)
    policy = _practice_policy(
        method=method, state=state, atoms=pick.preconditions, monkeypatch=monkeypatch
    )
    with pytest.raises(InteractionComplete) as stop:
        policy(state)
    assert stop.value.planner_stop is True
    assert len(client.calls) == 1
    decision = _last_decision(method=method)
    assert decision["action"] == "STOP"
    assert decision["solver"] == "llm"
    assert decision["llm_reason"] == "fake chose STOP"
    assert decision["llm_call_index"] == 0
    assert decision["llm_cost_usd"] == 0.01
    assert decision["planner_memory"] == "stateless"
    assert decision["horizon"] is None
    assert method.practice_action_values() == {"STOP": decision["value"]}
    assert method.practice_skill_improvement_potentials() == {}


@pytest.mark.parametrize(
    ("side", "human_name"),
    [("robot_side", "reset_cube_and_bin_near"), ("opposite_side", "reset_cube_far")],
)
def test_a_reset_answer_requests_that_destination_under_its_human_name(
    *, side: str, human_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_method, _ = _build(answers=[])
    reset = _reset(method=seed_method, side=side)
    answer = LlmPracticePlanner.candidate_id(ground=reset)
    method, client = _build(answers=[answer])
    pick = _grounding(method=method, name=PICK_SKILL)
    state = _state(method=method)
    policy = _practice_policy(
        method=method,
        state=state,
        atoms=pick.preconditions | reset.preconditions,
        monkeypatch=monkeypatch,
    )
    with pytest.raises(HumanCubeBinResetRequested) as request:
        policy(state)
    assert request.value.destination == side
    assert request.value.cost == 5.0
    candidates = {c["id"]: c for c in client.calls[0]["context"]["candidates"]}
    assert answer in client.calls[0]["candidate_ids"]
    assert candidates[answer]["human_name"] == human_name
    assert candidates[answer]["skill"] == RESET_SKILL
    assert candidates[answer]["objects"][-1] == side
    offered = {c["human_name"] for c in candidates.values() if c["human_name"] is not None}
    assert offered == {"reset_cube_and_bin_near", "reset_cube_far"}


def test_a_robot_skill_answer_is_dispatched_as_the_whole_plan(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Belief-space candidates are applicable by construction, so the Fast Downward
    prefix is empty for every solver; the chosen skill is the plan."""
    seed_method, _ = _build(answers=[])
    pick = _grounding(method=seed_method, name=PICK_SKILL)
    method, client = _build(answers=[LlmPracticePlanner.candidate_id(ground=pick)])
    executed: list[GroundSkill] = []

    def execute(  # noqa: PLR0917 - stands in for a method, so `self` is positional
        self: Tossing3DPomdpMethod, *, ground_skill: GroundSkill, state: Any, explore: bool
    ) -> tuple[LabeledAction, None]:
        del self, state, explore
        executed.append(ground_skill)
        return LabeledAction(action=np.zeros(1), label=ground_skill.skill.name), None

    monkeypatch.setattr(Tossing3DPomdpMethod, "execute_ground_skill", execute)
    state = _state(method=method)
    policy = _practice_policy(
        method=method, state=state, atoms=pick.preconditions, monkeypatch=monkeypatch
    )
    labeled = policy(state)
    assert labeled.label == PICK_SKILL
    assert executed == [pick]
    episode = method._practice_episode  # noqa: SLF001
    assert episode is not None
    assert episode._plan == []  # noqa: SLF001
    assert client.calls[0]["candidate_ids"][-1] == "STOP"
    assert pick.skill.name in {c["skill"] for c in client.calls[0]["context"]["candidates"]}


def test_the_context_carries_candidates_values_schedule_and_costs(*, tmp_path: Path) -> None:
    method, client = _build(answers=["STOP"], decision_log=tmp_path / "decisions.jsonl")
    pick = _grounding(method=method, name=PICK_SKILL)
    method.observe_practice_action_budget(remaining_actions=7)
    method.observe_practice_schedule(
        step_budget=85_000,
        measurement_interval_steps=1_700,
        steps_used=3_400,
        model_budget_remaining_usd=12.5,
    )
    method.select_skill_to_practice(true_atoms=pick.preconditions)
    call = client.calls[0]
    context = call["context"]
    assert call["session_key"] == "session-0"
    assert context["schema"] == LlmPracticePlanner.SCHEMA
    assert context["atoms"] == sorted(str(atom) for atom in pick.preconditions)

    ids = [candidate["id"] for candidate in context["candidates"]]
    assert ids == call["candidate_ids"]
    assert ids[-1] == "STOP"
    assert len(ids) == len(set(ids))
    search_state = make_tossing3d_search_state(
        state=method.pomdp_state, true_atoms=pick.preconditions
    )
    model = method._pomdp_model  # noqa: SLF001
    valid = model.get_valid_actions(environment_state=search_state)
    assert ids[:-1] == [LlmPracticePlanner.candidate_id(ground=ground) for ground in valid]
    expected_keys = {
        "id",
        "skill",
        "objects",
        "human_name",
        "expected_cost",
        "posterior_mean_competence",
        "posterior_mean_learning_rate",
        "attempts",
        "successes",
    }
    for candidate, ground in zip(context["candidates"][:-1], valid, strict=True):
        assert expected_keys <= set(candidate)
        belief = method.pomdp_state.skill_beliefs[ground.skill.name]
        assert candidate["posterior_mean_competence"] == pytest.approx(
            mean_competence(belief=belief)
        )
        assert candidate["posterior_mean_learning_rate"] == pytest.approx(
            mean_learning_rate(belief=belief)
        )
        # The same certainty-equivalent cost expectimax searches with.
        assert candidate["expected_cost"] == pytest.approx(
            estimated_action_cost(state=method.pomdp_state, action=ground)
        )
        assert candidate["expected_cost"] == pytest.approx(
            method.practice_skill_costs()[f"{ground.skill.name} (belief mean)"]
        )
        assert (candidate["attempts"], candidate["successes"]) == (0, 0)
    stop = context["candidates"][-1]
    assert stop["skill"] == "STOP"
    assert context["stop_value"] == pytest.approx(
        model.J(belief_state=method.pomdp_state, summed_cost=0.0, num_samples=1)
    )
    assert context["stop_value"] == pytest.approx(method.practice_action_values()["STOP"])
    assert context["session_cost"] == 0.0
    assert context["remaining_session_actions"] == 7
    assert context["session_index"] == 0
    assert context["decision_index"] == 1
    assert context["schedule"] == {
        "step_budget": 85_000,
        "measurement_interval_steps": 1_700,
        "steps_used": 3_400,
        "model_budget_remaining_usd": 12.5,
    }
    costs = context["configured_costs"]
    assert costs["objective_lambda"] is None
    assert costs["hard_practice_budget"] == 20.0
    assert costs["human_reset_cost"] == 5.0
    assert context["history"]["scope"] == "last_20_decisions"
    assert context["history"] == {"scope": "last_20_decisions", "events": [], "skill_totals": {}}
    assert context["planner"] == {
        "model": "claude-opus-5-5",
        "effort": "low",
        "memory": "stateless",
    }
    decision = _last_decision(method=method)
    assert decision["practice_step_budget"] == 85_000
    assert decision["model_budget_remaining_usd"] == 12.5


def _drive_two_decisions_then_a_new_session(
    *, method: Tossing3DPomdpMethod, pick: GroundSkill, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(method=method)
    # end_cycle flushes the in-flight action against the live scene, which is not
    # what these tests measure (the existing end_cycle test pins it the same way).
    monkeypatch.setattr(Tossing3DPomdpMethod, "observe_environment_reset", lambda self, **_: None)
    monkeypatch.setattr(Tossing3DEnvironment, "get_current_state", lambda self: object())
    _practice_policy(method=method, state=state, atoms=pick.preconditions, monkeypatch=monkeypatch)
    method.select_skill_to_practice(true_atoms=pick.preconditions)
    method.record_action_cost(ground_skill=pick)
    method.observe_outcome(ground_skill=pick, success=True)
    method.select_skill_to_practice(true_atoms=pick.preconditions)
    method.end_cycle()
    _practice_policy(method=method, state=state, atoms=pick.preconditions, monkeypatch=monkeypatch)
    method.select_skill_to_practice(true_atoms=pick.preconditions)


def test_stateless_memory_sends_the_full_recent_window_every_call(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    method, client = _build(answers=["STOP"] * 3, planner_memory="stateless")
    pick = _grounding(method=method, name=PICK_SKILL)
    pick_id = LlmPracticePlanner.candidate_id(ground=pick)
    _drive_two_decisions_then_a_new_session(method=method, pick=pick, monkeypatch=monkeypatch)
    assert [call["session_key"] for call in client.calls] == [
        "session-0",
        "session-0",
        "session-1",
    ]
    histories = [call["context"]["history"] for call in client.calls]
    assert {history["scope"] for history in histories} == {"last_20_decisions"}
    second = histories[1]["events"]
    assert [event["type"] for event in second] == ["session_start", "decision", "attempt"]
    assert second[1]["action"] == "STOP"
    assert second[2] == {
        "type": "attempt",
        "session": 0,
        "decision": 1,
        "action": pick_id,
        "skill": PICK_SKILL,
        "success": True,
        "random_exploration": False,
        "cost": 1.0,
    }
    third = histories[2]["events"]
    assert [event["type"] for event in third] == [
        "session_start",
        "decision",
        "attempt",
        "decision",
        "session_start",
    ]
    assert histories[2]["skill_totals"][pick_id] == {"attempts": 1, "successes": 1}
    candidates = {c["id"]: c for c in client.calls[2]["context"]["candidates"]}
    assert (candidates[pick_id]["attempts"], candidates[pick_id]["successes"]) == (1, 1)


def test_the_stateless_window_is_bounded_by_decisions(*, monkeypatch: pytest.MonkeyPatch) -> None:
    method, client = _build(answers=["STOP"] * 3, planner_history_window=1)
    pick = _grounding(method=method, name=PICK_SKILL)
    _drive_two_decisions_then_a_new_session(method=method, pick=pick, monkeypatch=monkeypatch)
    third = client.calls[2]["context"]["history"]
    assert third["scope"] == "last_1_decisions"
    assert [event["type"] for event in third["events"]] == ["decision", "session_start"]


def test_resume_memory_sends_only_the_delta_within_a_session(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    method, client = _build(answers=["STOP"] * 3, planner_memory="resume")
    pick = _grounding(method=method, name=PICK_SKILL)
    _drive_two_decisions_then_a_new_session(method=method, pick=pick, monkeypatch=monkeypatch)
    assert [call["session_key"] for call in client.calls] == [
        "session-0",
        "session-0",
        "session-1",
    ]
    first, second, third = (call["context"]["history"] for call in client.calls)
    assert first["scope"] == "last_20_decisions"
    assert [event["type"] for event in first["events"]] == ["session_start"]
    assert second["scope"] == "since_previous_call"
    assert [event["type"] for event in second["events"]] == ["attempt"]
    # A new practice session is a new Claude session: it gets the window again.
    assert third["scope"] == "last_20_decisions"
    assert [event["type"] for event in third["events"]] == [
        "session_start",
        "decision",
        "attempt",
        "decision",
        "session_start",
    ]
    assert client.calls[1]["context"]["decision_index"] == 2


def test_an_unknown_action_id_is_an_error_not_a_fallback() -> None:
    method, _ = _build(answers=["nonsense"])
    pick = _grounding(method=method, name=PICK_SKILL)
    with pytest.raises(UnknownDecisionError, match="nonsense"):
        method.select_skill_to_practice(true_atoms=pick.preconditions)


def test_budget_exhaustion_raised_by_the_client_reaches_the_practice_policy(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    method, _ = _build(answers=[ModelBudgetExhausted()])
    pick = _grounding(method=method, name=PICK_SKILL)
    state = _state(method=method)
    policy = _practice_policy(
        method=method, state=state, atoms=pick.preconditions, monkeypatch=monkeypatch
    )
    with pytest.raises(ModelBudgetExhausted):
        policy(state)


def test_the_planner_requires_a_situation() -> None:
    planner = LlmPracticePlanner(
        client=FakeClient(answers=[]),
        memory=PlannerMemory.stateless,
        model="m",
        effort="low",
        budget_limit_usd=1.0,
    )
    with pytest.raises(ValueError, match="situation"):
        planner.solve(
            environment_state=None,  # type: ignore[arg-type]
            summed_cost=0.0,
            belief_state=None,  # type: ignore[arg-type]
            horizon=1,
            model=None,  # type: ignore[arg-type]
            num_samples=1,
        )


def test_the_trace_marks_the_choice_without_inventing_values(*, tmp_path: Path) -> None:
    seed_method, _ = _build(answers=[])
    pick = _grounding(method=seed_method, name=PICK_SKILL)
    method, _ = _build(
        answers=[LlmPracticePlanner.candidate_id(ground=pick)],
        decision_log=tmp_path / "decisions.jsonl",
    )
    method.select_skill_to_practice(true_atoms=pick.preconditions)
    decision = _last_decision(method=method)
    events = decision["search"]
    stop_values = [e for e in events if e["event"] == "stop_value" and e["node"] == 0]
    assert len(stop_values) == 1
    action_values = [e for e in events if e["event"] == "action_value" and e["node"] == 0]
    assert len(action_values) >= 1
    assert all(event["value"] is None for event in action_values)
    assert [e["action"]["skill"]["name"] for e in action_values if e["chosen"]] == [PICK_SKILL]
    choice = next(e for e in events if e["event"] == "choice")
    assert choice["action"]["skill"]["name"] == PICK_SKILL
    summary = next(e for e in events if e["event"] == "search_summary")
    assert summary["solver"] == "llm"
    assert decision["value"] == "NaN"
    assert decision["action"]["skill"]["name"] == PICK_SKILL
    assert decision["improvement_potentials"] == {}
    assert set(method.practice_action_values()) == {"STOP"}


def test_the_protocol_describes_the_arm() -> None:
    method, _ = _build(answers=[], planner_model="m", planner_effort="e", planner_memory="resume")
    planner = method.pomdp_planner
    assert isinstance(planner, LlmPracticePlanner)
    assert planner.protocol() == {
        "solver": "llm",
        "model": "m",
        "effort": "e",
        "memory": "resume",
        "budget_limit_usd": 20.0,
        "ledger_scope": "practice_only",
        "estimator": "bayesian",
    }


def test_snapshots_and_the_evaluation_path_match_the_expectimax_arm() -> None:
    llm, _ = _build(answers=[])
    env = Tossing3DEnvironment(scene_bg=False)
    expectimax = Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env, human_reset_practice_cost=5.0),
        seed=0,
        pomdp_solver="expectimax",
        pomdp_num_particles=200,
    )
    raw = DeploymentSnapshot.encode(method=llm)
    assert raw == DeploymentSnapshot.encode(method=expectimax)
    fresh = Tossing3DEnvironment(scene_bg=False)
    restored = DeploymentSnapshot.restore(
        raw=raw, env=fresh, provider=Tossing3DSkillProvider(env=fresh, offer_human_reset=False)
    )
    assert type(restored) is EesMethod
    assert restored.may_request_human_help() is False
    assert all(
        ground.skill.name != RESET_SKILL
        for ground in restored._competence_models  # noqa: SLF001
    )
