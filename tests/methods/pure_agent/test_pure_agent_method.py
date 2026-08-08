"""The acting-agent baseline, driven end to end through the real PracticeLoop against a
scripted transport. Everything except the network call is the production path."""

import argparse
import json
import tempfile
from pathlib import Path

import pytest

from hitl_pmp.core.method.method import InteractionComplete
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.environments.tossingroom.cli import TossingRoomCli
from hitl_pmp.environments.tossingroom.skill_provider import TossingRoomSkillProvider
from hitl_pmp.methods.pure_agent.agent_backend import (
    FirstApplicableAgentBackend,
    ScriptedAgentBackend,
)
from hitl_pmp.methods.pure_agent.prompts import DIGEST_REQUEST
from hitl_pmp.methods.pure_agent.pure_agent_method import PureAgentMethod
from hitl_pmp.methods.pure_agent.types import AgentCallKind, AgentPhase, PureAgentLedger
from hitl_pmp.planning.grounding import SkillGrounder
from hitl_pmp.practice_loop import PracticeLoop

# Small everywhere: these tests are about the wiring and the firewall, and every extra
# step is another scripted call to account for by hand in an exact-count assertion.
NUM_CYCLES = 2
MAX_STEPS = 4
NUM_TEST_TASKS = 3
# Enough of the digest request to recognise it, taken from the prompt itself so the two
# cannot drift apart silently.
DIGEST_MARKER = DIGEST_REQUEST.splitlines()[0]


def build_args(**overrides):
    """A TossingRoom namespace at its own defaults, which is what its cli.py reads."""
    parser = argparse.ArgumentParser()
    TossingRoomCli.add_arguments(parser=parser)
    args = parser.parse_args([])
    args.seed = 0
    args.num_test_tasks = NUM_TEST_TASKS
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def build_method(
    *,
    practice_replies=None,
    evaluation_replies=None,
    digest_reply=None,
    cost_per_call=0.0,
    **method_overrides,
):
    """A PureAgentMethod on Tossing Room, with a scripted transport on each side."""
    args = build_args()
    problem = TossingRoomCli.build_problem(
        args=args, num_cycles=NUM_CYCLES, max_steps_per_interaction=MAX_STEPS
    )
    evaluation_problem = TossingRoomCli.build_problem(
        args=args, num_cycles=NUM_CYCLES, max_steps_per_interaction=MAX_STEPS
    )
    markers = {DIGEST_MARKER: digest_reply} if digest_reply is not None else {}
    # `None` means "act legally at every decision point", which is what most of these
    # tests want; a fixed reply tuple is for the tests that are *about* a bad reply.
    practice_backend = (
        FirstApplicableAgentBackend(replies_by_marker=markers, cost_usd_per_query=cost_per_call)
        if practice_replies is None
        else ScriptedAgentBackend(
            replies=practice_replies,
            replies_by_marker=markers,
            cost_usd_per_query=cost_per_call,
        )
    )
    evaluation_backend = (
        FirstApplicableAgentBackend(cost_usd_per_query=cost_per_call)
        if evaluation_replies is None
        else ScriptedAgentBackend(replies=evaluation_replies, cost_usd_per_query=cost_per_call)
    )
    method = PureAgentMethod(
        env=problem.env,
        skill_provider=TossingRoomSkillProvider(env=problem.env),
        practice_backend=practice_backend,
        evaluation_backend=evaluation_backend,
        **method_overrides,
    )
    return method, problem, evaluation_problem, practice_backend, evaluation_backend


def run_loop(*, method, problem, evaluation_problem, num_cycles=NUM_CYCLES):
    metrics = Metrics()
    PracticeLoop.run(
        problem=problem,
        evaluation_problem=evaluation_problem,
        method=method,
        metrics=metrics,
        num_cycles=num_cycles,
        max_steps_per_interaction=MAX_STEPS,
        num_test_tasks=NUM_TEST_TASKS,
    )
    return metrics


def test_no_evaluation_step_ever_reaches_the_practice_agent():
    """The firewall, asserted as an exact count rather than as an absence.

    The practice backend must see exactly one opening per period, one decision per
    practice step taken, and one digest per cycle -- and NOTHING else. Routing any part of
    evaluation through it (the natural mistake, since `decide` is shared by both phases)
    changes this number by the evaluation call count, so the test fails by a large
    margin rather than on a judgement call."""
    method, problem, evaluation_problem, practice, evaluation = build_method()
    run_loop(method=method, problem=problem, evaluation_problem=evaluation_problem)

    ledger = PureAgentLedger(records=method.call_records())
    practice_decisions = ledger.num_decisions(phase=AgentPhase.PRACTICE)
    expected_practice_calls = (
        practice_decisions  # one per practice step actually taken
        + NUM_CYCLES  # one opening per period
        + NUM_CYCLES  # one digest per cycle
    )
    assert len(practice.prompts_seen()) == expected_practice_calls
    assert ledger.num_calls(phase=AgentPhase.PRACTICE) == expected_practice_calls
    # And the evaluation side really did run, so the count above is not trivially right
    # because nothing happened.
    assert ledger.num_decisions(phase=AgentPhase.EVALUATION) > 0
    assert len(evaluation.prompts_seen()) == ledger.num_calls(phase=AgentPhase.EVALUATION)


def test_the_evaluation_conversation_is_reset_once_per_held_out_task():
    """Without this, the agent carries test task k into test task k+1, which is learning
    across held-out tasks -- training on the test set by a slower route. One reset per
    `get_task_policy` call, i.e. per task per sweep."""
    method, problem, evaluation_problem, _practice, evaluation = build_method()
    run_loop(method=method, problem=problem, evaluation_problem=evaluation_problem)
    assert evaluation.reset_count() == (NUM_CYCLES + 1) * NUM_TEST_TASKS


def test_the_practice_conversation_is_reset_once_per_period():
    method, problem, evaluation_problem, practice, _evaluation = build_method()
    run_loop(method=method, problem=problem, evaluation_problem=evaluation_problem)
    assert practice.reset_count() == NUM_CYCLES


def test_no_evaluation_prompt_carries_an_outcome_or_a_tally():
    """The other half of the firewall: not merely that evaluation runs on its own backend,
    but that what that backend is sent contains nothing measurement produced."""
    method, problem, evaluation_problem, _practice, evaluation = build_method()
    run_loop(method=method, problem=problem, evaluation_problem=evaluation_problem)
    for prompt in evaluation.prompts_seen():
        assert "achieve its declared add effects" not in prompt
        assert "Cumulatively over practice" not in prompt


def test_practice_prompts_do_carry_the_outcome_of_the_previous_action():
    """The complement of the test above: the practice side is *supposed* to be told, and a
    firewall that leaked nothing because nothing was ever reported would pass every test
    above while measuring a method that cannot learn."""
    method, problem, evaluation_problem, practice, _evaluation = build_method()
    run_loop(method=method, problem=problem, evaluation_problem=evaluation_problem)
    outcome_prompts = [
        prompt for prompt in practice.prompts_seen() if "achieve its declared add effects" in prompt
    ]
    assert outcome_prompts


def test_the_digest_carries_practice_knowledge_forward_and_only_after_practice():
    """The one channel from practice to evaluation, in both directions of the claim.

    The sweep that runs *before* any practice must not see the note -- otherwise the
    untrained baseline point on the learning curve is not untrained. Every sweep after a
    practice period must see it -- otherwise the arm cannot learn at all and a flat curve
    would be an artefact of the plumbing rather than a result."""
    note = "SENTINEL-NOTE-abc123: the required force is twice the item weight."
    method, problem, evaluation_problem, _practice, evaluation = build_method(digest_reply=note)
    run_loop(method=method, problem=problem, evaluation_problem=evaluation_problem, num_cycles=1)

    openings = [prompt for prompt in evaluation.prompts_seen() if "READY" in prompt]
    assert len(openings) == 2 * NUM_TEST_TASKS
    before_any_practice, after_one_period = openings[:NUM_TEST_TASKS], openings[NUM_TEST_TASKS:]
    assert all(note not in prompt for prompt in before_any_practice)
    assert all(note in prompt for prompt in after_one_period)


def test_the_spend_ceiling_stops_the_run_querying_and_says_so():
    """The guard that matters. A run makes one call per environment step against a weekly
    allowance with no overflow, so it must be able to stop itself -- and a run that stops
    must finish and write its results rather than crash, with the stop visible."""
    method, problem, evaluation_problem, practice, evaluation = build_method(
        max_total_cost_usd=1.0, cost_per_call=1.0
    )
    run_loop(method=method, problem=problem, evaluation_problem=evaluation_problem)
    # One call spends the whole ceiling, so exactly one is made and then nothing -- over a
    # run that would otherwise have made dozens.
    assert len(method.call_records()) == 1
    assert method.spend_usd() == 1.0
    assert method.budget_exhausted()
    assert len(practice.prompts_seen()) + len(evaluation.prompts_seen()) == 1


def test_a_zero_ceiling_is_a_dry_run_that_makes_no_calls_at_all():
    """Distinct from a small one: `--pure-agent-max-total-cost-usd 0` is how an operator
    checks the whole wiring, on the real CLI, without spending anything."""
    method, problem, evaluation_problem, _practice, _evaluation = build_method(
        max_total_cost_usd=0.0, cost_per_call=1.0
    )
    run_loop(method=method, problem=problem, evaluation_problem=evaluation_problem)
    assert method.call_records() == []
    assert method.spend_usd() == 0.0


def test_no_ceiling_means_no_ceiling():
    """The tests above all run uncapped, so the disabled path is the one every other
    assertion in this file depends on."""
    method, problem, evaluation_problem, _practice, _evaluation = build_method()
    assert method.max_total_cost_usd is None
    run_loop(method=method, problem=problem, evaluation_problem=evaluation_problem)
    assert not method.budget_exhausted()
    assert len(method.call_records()) > 1


def test_a_malformed_reply_becomes_a_counted_no_op_rather_than_a_crash():
    """An agent-as-policy baseline malforms replies sometimes, and how often is a result.
    A run must therefore survive one, and must say so."""
    method, problem, evaluation_problem, _practice, _evaluation = build_method(
        evaluation_replies=("I think we should pick up the trash first.",),
    )
    run_loop(method=method, problem=problem, evaluation_problem=evaluation_problem, num_cycles=0)
    ledger = PureAgentLedger(records=method.call_records())
    assert ledger.num_malformed_decisions() == ledger.num_decisions(phase=AgentPhase.EVALUATION)
    assert ledger.num_malformed_decisions() > 0


def test_an_out_of_range_skill_index_is_refused_rather_than_clamped():
    """Quietly repairing an illegal index would invent a choice the agent did not make,
    and would hide the one failure mode indexing the applicable set exists to prevent."""
    method, problem, evaluation_problem, _practice, _evaluation = build_method(
        evaluation_replies=('{"skill_index": 999, "params": []}',),
    )
    run_loop(method=method, problem=problem, evaluation_problem=evaluation_problem, num_cycles=0)
    ledger = PureAgentLedger(records=method.call_records())
    assert ledger.num_malformed_decisions() > 0
    assert all(
        record.skill_index is None
        for record in method.call_records()
        if record.kind is AgentCallKind.DECISION
    )


def test_the_ledger_is_journalled_line_by_line():
    """Spend is a deliverable and a run is hours long, so the record must survive the
    process dying part-way. One JSON line per call, written as it happens, into a
    directory the method creates itself."""
    with tempfile.TemporaryDirectory() as root:
        ledger_path = Path(root) / "nested" / "agent_calls.jsonl"
        method, problem, evaluation_problem, _practice, _evaluation = build_method(
            ledger_path=ledger_path
        )
        run_loop(method=method, problem=problem, evaluation_problem=evaluation_problem)
        lines = ledger_path.read_text().splitlines()
    assert len(lines) == len(method.call_records())
    ledger = PureAgentLedger(records=[json.loads(line) for line in lines])
    assert ledger.num_calls() == len(method.call_records())
    assert (
        ledger.num_calls(phase=AgentPhase.PRACTICE) + ledger.num_calls(phase=AgentPhase.EVALUATION)
        == ledger.num_calls()
    )


def test_every_decision_record_carries_the_digest_of_what_it_decided_against():
    """The digest is what a replay checks itself against; a decision recorded without one
    could not be verified, only assumed."""
    method, problem, evaluation_problem, _practice, _evaluation = build_method()
    run_loop(method=method, problem=problem, evaluation_problem=evaluation_problem)
    decisions = [r for r in method.call_records() if r.kind is AgentCallKind.DECISION]
    assert decisions
    assert all(len(record.observation_digest) == 64 for record in decisions)
    # Openings and digests have no observation behind them, so theirs is empty rather
    # than a hash of nothing.
    assert all(
        record.observation_digest == ""
        for record in method.call_records()
        if record.kind is not AgentCallKind.DECISION
    )


def test_a_dead_end_costs_no_agent_call(monkeypatch):  # noqa: PLR0917 (pytest fixture)
    """The applicable set is computed before the agent is asked, so a state with nothing
    applicable is a no-op that spends nothing. At a per-step price that is the difference
    between a dead-ended period being free and it being the most expensive part of a run
    -- `RandomSkillsMethod`'s docstring records ~145 of 150 dead-ended steps on one
    domain, which here would be 145 paid calls producing nothing."""
    method, problem, _evaluation_problem, practice, evaluation = build_method()
    problem.hard_reset()
    task = problem.sample_train_task()
    state = problem.reset_to_task(task=task)
    # Patched at the grounder rather than on the method, so the method's own code path --
    # ask what is applicable, then decide whether to query -- is the one under test.
    monkeypatch.setattr(
        SkillGrounder, "applicable_ground_skills", staticmethod(lambda **_kwargs: [])
    )

    labeled = method.evaluation_step(state=state, task=task)
    assert "no applicable skills" in labeled.label
    with pytest.raises(InteractionComplete):
        method.practice_step(state=state, task=task)
    assert practice.prompts_seen() == []
    assert evaluation.prompts_seen() == []
    assert method.call_records() == []


def test_practice_outcomes_are_recorded_so_the_arm_shares_the_ees_axis():
    method, problem, evaluation_problem, _practice, _evaluation = build_method()
    run_loop(method=method, problem=problem, evaluation_problem=evaluation_problem)
    outcomes = method.practice_outcomes()
    assert outcomes, "practice executed skills but reported no tally"
    assert all(tally.num_attempts > 0 for tally in outcomes.values())


def test_unreachable_method_hooks_say_why_rather_than_silently_doing_nothing():
    method, _problem, _evaluation_problem, _practice, _evaluation = build_method()
    with pytest.raises(NotImplementedError):
        method.generate_train_task(tbd_inputs=None)
    with pytest.raises(NotImplementedError):
        method.execute_skill(skill=None)
