# ruff: noqa: PLR0917 -- pytest fixtures
"""One structured practice decision per Claude call, faked exactly like the classifier."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import hitl_pmp.agentic_runtime.decision_client as module
from hitl_pmp.agentic_runtime.decision_client import DecisionClient, DecisionError
from hitl_pmp.agentic_runtime.docker_cli import RobocodeDockerTransport
from hitl_pmp.agentic_runtime.types import Decision, PlannerMemory
from hitl_pmp.agentic_states.budget import SharedBudget
from hitl_pmp.hybrid_skills.artifacts import ModelBudgetExhausted

CANDIDATES = ["PickCube", "MoveToTossLocationAndToss", "reset_cube_far"]
GOOD = {"action_id": "PickCube", "reason": "lowest competence, cheapest to practice"}
ESTIMATES = {
    "PickCube": {"competence": 0.3, "learning_rate": 0.1, "expected_cost": 12.5},
    "MoveToTossLocationAndToss": {"competence": 0.6, "learning_rate": 0.05, "expected_cost": 30},
}


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Script the Docker CLI call: each entry is one transport invocation's outcome.

    An entry is ``(decision_text_or_None, cost, session_id)``; ``None`` writes no
    decision.json at all, and ``cost`` is what the CLI reports as ``total_cost_usd``.
    """
    calls = []
    script = []
    constructed = []

    class Transport(RobocodeDockerTransport):
        def __init__(self, **kwargs):  # noqa: PLR0917 - never calls the real transport
            constructed.append(kwargs)

        def modules(self):
            return (
                SimpleNamespace(SandboxConfig=SimpleNamespace),
                SimpleNamespace(create_backend=lambda settings: None),
                SimpleNamespace(DictConfig=dict),
            )

        def run_with_recovery(self, *, config, **kwargs):
            calls.append(config)
            text, cost, session_id = script.pop(0)
            if text is not None:
                (config.sandbox_dir / "decision.json").write_text(text)
            call_dir = config.sandbox_dir.parent
            with (call_dir / "stream.jsonl").open("a") as stream:
                stream.write(
                    json.dumps({"type": "system", "subtype": "init", "session_id": session_id})
                    + "\n"
                )
                stream.write(json.dumps({"type": "result", "session_id": session_id}) + "\n")
            (call_dir / "broker.jsonl").write_text("{}\n")
            (call_dir / "container_command.json").write_text("[]\n")
            sessions = config.sandbox_dir / ".agent_sessions" / "claude" / "-sandbox"
            sessions.mkdir(parents=True, exist_ok=True)
            (sessions / f"{session_id}.jsonl").write_text("{}\n")
            return SimpleNamespace(success=True, error=None, total_cost_usd=cost)

    monkeypatch.setattr(module, "RobocodeDockerTransport", Transport)
    return SimpleNamespace(
        calls=calls,
        script=script,
        constructed=constructed,
        settings=SimpleNamespace(robocode_checkout=tmp_path / "robocode"),
        output=tmp_path / "run",
    )


def client(fake, **overrides):
    options = dict(output=fake.output, settings=fake.settings)
    options.update(overrides)
    return DecisionClient(**options)


def decide(instance, **overrides):
    options = dict(context={"step": 1}, candidate_ids=CANDIDATES, session_key="s1")
    options.update(overrides)
    return instance.decide(**options)


def test_well_formed_decision_is_parsed_and_the_call_is_shaped_like_the_classifier(fake):
    fake.script.append((json.dumps(GOOD), 0.05, "sid-a"))
    decision = decide(client(fake), context={"skills": ["PickCube"], "step": 7})
    assert decision == Decision(
        action_id="PickCube",
        reason=GOOD["reason"],
        estimates=None,
        cost_usd=0.05,
        call_index=0,
    )
    assert fake.constructed == [
        dict(sandbox=fake.settings, backend="claude", subagent_policy="disallowed")
    ]
    (config,) = fake.calls
    assert config.sandbox_dir == fake.output / "planner_calls/0000/sandbox"
    assert json.loads((config.sandbox_dir / "decision_context.json").read_text()) == {
        "skills": ["PickCube"],
        "step": 7,
    }
    assert config.output_filename == "decision.json"
    assert config.model == "claude-opus-5-5"
    assert config.effort == "low"
    assert config.max_output_tokens == 4096
    assert config.max_turns == 0
    assert config.blackbox is True
    assert config.mcp_tools == ()
    assert config.resume_previous_session is False
    assert config.max_budget_usd == 20
    for candidate in CANDIDATES:
        assert candidate in config.prompt
    assert "decision_context.json" in config.prompt and "decision.json" in config.prompt
    assert "one practice action" in config.system_prompt
    assert "must not execute" in config.system_prompt
    # The client's own words carry no task semantics; only the caller's ids and
    # context do, so check the prompt built around neutral ids.
    own_words = config.system_prompt + DecisionClient.prompt(
        candidate_ids=["a", "b"], require_estimates=True, resumed=True, rejection="r"
    )
    for human_text in ("reset", "cube", "bin", "toss", "gripper"):
        assert human_text not in own_words.lower()


def test_constructor_knobs_reach_the_sandbox_config(fake):
    fake.script.append((json.dumps(GOOD), 0.05, "sid-a"))
    decide(
        client(
            fake,
            model="claude-sonnet-4-5",
            effort="high",
            max_output_tokens=512,
            budget_limit=3.5,
        )
    )
    (config,) = fake.calls
    assert (config.model, config.effort, config.max_output_tokens) == (
        "claude-sonnet-4-5",
        "high",
        512,
    )
    assert config.max_budget_usd == 3.5


@pytest.mark.parametrize(
    "bad,rejection",
    [
        (json.dumps({"action_id": "fly_to_moon", "reason": "why not"}), "unknown action"),
        ("{not json", "malformed"),
        (json.dumps({"action_id": "PickCube"}), "malformed"),
        (json.dumps({**GOOD, "extra": 1}), "malformed"),
        (None, "not written"),
    ],
)
def test_bad_output_is_retried_once_then_raises(fake, bad, rejection):
    fake.script.extend([(bad, 0.05, "sid-a"), (bad, 0.07, "sid-b")])
    with pytest.raises(DecisionError) as exc:
        decide(client(fake))
    assert rejection in str(exc.value)
    assert len(fake.calls) == 2
    assert "rejected" in fake.calls[1].prompt
    records = [
        json.loads((fake.output / "planner_calls" / name / "result.json").read_text())
        for name in ("0000", "0001")
    ]
    assert [r["attempt"] for r in records] == [1, 2]
    assert [r["decision_index"] for r in records] == [0, 0]
    assert [r["accepted"] for r in records] == [False, False]
    assert all(rejection in r["rejection"] for r in records)
    assert [r["cost_usd"] for r in records] == [0.05, 0.07]
    ledger = json.loads((fake.output / "model_budget.json").read_text())
    assert ledger["spent"] == pytest.approx(0.12)


def test_retry_that_succeeds_returns_the_second_attempt(fake):
    fake.script.extend([("{not json", 0.05, "sid-a"), (json.dumps(GOOD), 0.07, "sid-b")])
    instance = client(fake)
    decision = decide(instance)
    assert (decision.action_id, decision.cost_usd, decision.call_index) == ("PickCube", 0.07, 1)
    fake.script.append((json.dumps(GOOD), 0.01, "sid-c"))
    assert decide(instance).call_index == 2
    record = json.loads((fake.output / "planner_calls/0002/result.json").read_text())
    assert (record["attempt"], record["decision_index"]) == (1, 1)


def test_transport_failure_counts_as_a_failed_attempt(fake, monkeypatch):
    fake.script.extend([(None, 0.02, "sid-a"), (json.dumps(GOOD), 0.03, "sid-b")])
    original = module.RobocodeDockerTransport.run_with_recovery

    def failing(self, *, config, **kwargs):
        result = original(self, config=config, **kwargs)
        return (
            result
            if len(fake.calls) > 1
            else SimpleNamespace(**{
                **vars(result),
                "success": False,
                "error": "boom",
            })
        )

    monkeypatch.setattr(module.RobocodeDockerTransport, "run_with_recovery", failing)
    assert decide(client(fake)).call_index == 1
    record = json.loads((fake.output / "planner_calls/0000/result.json").read_text())
    assert record["success"] is False and record["error"] == "boom"
    assert "boom" in record["rejection"]


def test_ledger_records_each_call_and_an_exhausted_ledger_never_calls(fake):
    fake.script.extend([(json.dumps(GOOD), 0.05, "sid-a"), (json.dumps(GOOD), 0.07, "sid-b")])
    instance = client(fake, budget_limit=1.0)
    decide(instance, session_key="s1")
    decide(instance, session_key="s2")
    ledger = json.loads((fake.output / "model_budget.json").read_text())
    assert ledger["sessions"] == {"planner-0000": 0.05, "planner-0001": 0.07}
    assert ledger["spent"] == pytest.approx(0.12)
    # Another ledger user (the coding learner) spends the rest before the next call.
    other = SharedBudget(path=fake.output / "model_budget.json", limit=1.0, session="coding")
    with other.lock():
        other.record(cumulative=0.88)
    before = (fake.output / "model_budget.json").read_bytes()
    fake.script.append((json.dumps(GOOD), 0.01, "sid-c"))
    with pytest.raises(ModelBudgetExhausted):
        decide(instance, session_key="s3")
    assert len(fake.calls) == 2
    assert (fake.output / "model_budget.json").read_bytes() == before
    assert not (fake.output / "planner_calls/0002/stream.jsonl").exists()


def test_ledger_path_can_be_shared_with_other_users(fake, tmp_path):
    ledger = tmp_path / "elsewhere/model_budget.json"
    fake.script.append((json.dumps(GOOD), 0.05, "sid-a"))
    decide(client(fake, ledger_path=ledger))
    assert json.loads(ledger.read_text())["sessions"] == {"planner-0000": 0.05}
    assert not (fake.output / "model_budget.json").exists()


def test_resume_reuses_the_saved_session_within_a_session_key(fake):
    fake.script.extend([
        (json.dumps(GOOD), 0.10, "sid-a"),
        (json.dumps(GOOD), 0.25, "sid-a"),
        (json.dumps(GOOD), 0.04, "sid-b"),
        (json.dumps(GOOD), 0.09, "sid-b"),
    ])
    instance = client(fake, memory=PlannerMemory.resume)
    first = decide(instance, session_key="practice-1")
    second = decide(instance, session_key="practice-1", context={"delta": True})
    third = decide(instance, session_key="practice-2")
    fourth = decide(instance, session_key="practice-2")
    assert [c.resume_previous_session for c in fake.calls] == [False, True, False, True]
    calls = fake.output / "planner_calls"
    assert not (calls / "0000/resume-session.json").exists()
    assert json.loads((calls / "0001/resume-session.json").read_text()) == {"session_id": "sid-a"}
    assert not (calls / "0002/resume-session.json").exists()
    assert json.loads((calls / "0003/resume-session.json").read_text()) == {"session_id": "sid-b"}
    # The CLI finds a session by id in its mounted project store, carried forward.
    assert (calls / "0001/sandbox/.agent_sessions/claude/-sandbox/sid-a.jsonl").exists()
    assert not (calls / "0002/sandbox/.agent_sessions/claude/-sandbox/sid-a.jsonl").exists()
    assert json.loads((calls / "0001/sandbox/decision_context.json").read_text()) == {"delta": True}
    # The CLI reports a resumed session's cost cumulatively; charge only the delta.
    assert [d.cost_usd for d in (first, second, third, fourth)] == pytest.approx([
        0.10,
        0.15,
        0.04,
        0.05,
    ])
    ledger = json.loads((fake.output / "model_budget.json").read_text())
    assert ledger["sessions"] == {"planner-practice-1": 0.25, "planner-practice-2": 0.09}
    assert ledger["spent"] == pytest.approx(0.34)
    records = [json.loads((calls / f"000{i}/result.json").read_text()) for i in range(4)]
    assert [r["resumed"] for r in records] == [False, True, False, True]
    assert [r["memory"] for r in records] == ["resume"] * 4
    assert [r["session_id"] for r in records] == ["sid-a", "sid-a", "sid-b", "sid-b"]
    assert "continues" in fake.calls[1].prompt


def test_resume_retry_continues_the_same_session(fake):
    fake.script.extend([("{not json", 0.10, "sid-a"), (json.dumps(GOOD), 0.16, "sid-a")])
    decision = decide(client(fake, memory=PlannerMemory.resume))
    assert [c.resume_previous_session for c in fake.calls] == [False, True]
    assert decision.cost_usd == pytest.approx(0.06)


def test_resume_without_a_recorded_session_id_is_an_error(fake):
    fake.script.append((json.dumps(GOOD), 0.10, None))
    with pytest.raises(DecisionError):
        decide(client(fake, memory=PlannerMemory.resume))


def test_stateless_never_resumes(fake):
    fake.script.extend([(json.dumps(GOOD), 0.10, "sid-a"), (json.dumps(GOOD), 0.10, "sid-a")])
    instance = client(fake, memory=PlannerMemory.stateless)
    first = decide(instance, session_key="practice-1")
    second = decide(instance, session_key="practice-1")
    assert [c.resume_previous_session for c in fake.calls] == [False, False]
    assert not (fake.output / "planner_calls/0001/resume-session.json").exists()
    assert [d.cost_usd for d in (first, second)] == [0.10, 0.10]
    records = [
        json.loads((fake.output / "planner_calls" / f"000{i}/result.json").read_text())
        for i in range(2)
    ]
    assert [r["memory"] for r in records] == ["stateless"] * 2
    assert [r["resumed"] for r in records] == [False, False]


@pytest.mark.parametrize(
    "estimates",
    [
        None,
        {},
        {"PickCube": {"competence": 0.3, "learning_rate": 0.1}},
        {"PickCube": {"competence": "high", "learning_rate": 0.1, "expected_cost": 1}},
        {"PickCube": {"competence": 0.3, "learning_rate": 0.1, "expected_cost": None}},
    ],
)
def test_require_estimates_rejects_an_incomplete_block_and_never_fills_it(fake, estimates):
    text = json.dumps(GOOD if estimates is None else {**GOOD, "estimates": estimates})
    fake.script.extend([(text, 0.05, "sid-a"), (text, 0.05, "sid-b")])
    with pytest.raises(DecisionError) as exc:
        decide(client(fake), require_estimates=True)
    assert "estimates" in str(exc.value)
    assert len(fake.calls) == 2
    assert "estimates" in fake.calls[0].prompt


def test_complete_estimates_are_returned_as_floats(fake):
    fake.script.append((json.dumps({**GOOD, "estimates": ESTIMATES}), 0.05, "sid-a"))
    decision = decide(client(fake), require_estimates=True)
    assert decision.estimates is not None
    assert set(decision.estimates) == set(ESTIMATES)
    pick = decision.estimates["PickCube"]
    assert (pick.competence, pick.learning_rate, pick.expected_cost) == (0.3, 0.1, 12.5)
    assert decision.estimates["MoveToTossLocationAndToss"].expected_cost == 30.0


def test_estimates_are_optional_when_not_required(fake):
    fake.script.append((json.dumps(GOOD), 0.05, "sid-a"))
    assert decide(client(fake)).estimates is None
    assert "estimates" not in fake.calls[0].prompt


def test_call_records_are_numbered_and_carry_the_transport_files(fake):
    fake.script.extend([(json.dumps(GOOD), 0.05, "sid-a"), (json.dumps(GOOD), 0.07, "sid-b")])
    instance = client(fake)
    decide(instance, session_key="k1")
    decide(instance, session_key="k2")
    calls = fake.output / "planner_calls"
    assert sorted(p.name for p in calls.iterdir()) == ["0000", "0001"]
    for name in ("stream.jsonl", "broker.jsonl", "container_command.json", "result.json"):
        assert (calls / "0001" / name).exists()
    record = json.loads((calls / "0001/result.json").read_text())
    assert record == {
        "success": True,
        "error": None,
        "cost_usd": 0.07,
        "reported_cost_usd": 0.07,
        "memory": "stateless",
        "session_key": "k2",
        "session_id": "sid-b",
        "ledger_session": "planner-0001",
        "resumed": False,
        "attempt": 1,
        "decision_index": 1,
        "candidate_ids": CANDIDATES,
        "accepted": True,
        "rejection": None,
    }


def test_candidates_must_be_nonempty_and_unique(fake):
    instance = client(fake)
    with pytest.raises(ValueError):
        decide(instance, candidate_ids=[])
    with pytest.raises(ValueError):
        decide(instance, candidate_ids=["a", "a"])
    with pytest.raises(ValueError):
        decide(instance, session_key="")
    assert fake.calls == []
    assert not (fake.output / "planner_calls").exists()


def test_robocode_source_is_put_on_sys_path_like_the_classifier(fake, monkeypatch):
    import sys

    fake.script.append((json.dumps(GOOD), 0.05, "sid-a"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    decide(client(fake))
    assert sys.path[0] == str(Path(fake.settings.robocode_checkout).resolve() / "src")
