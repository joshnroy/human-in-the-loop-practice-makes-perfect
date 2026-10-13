"""One structured practice decision per Claude call, through the Docker CLI transport."""

import shutil
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from hitl_pmp.agentic_states.budget import SharedBudget
from hitl_pmp.hybrid_skills.artifacts import ModelBudgetExhausted
from hitl_pmp.step_protocol import StepFiles

from .docker_cli import RobocodeDockerTransport
from .sandbox import SandboxSettings
from .types import Decision, DecisionOutput, PlannerMemory


class DecisionError(Exception):
    """The model did not produce an acceptable decision in two attempts."""


class DecisionClient:
    """Ask Claude to pick one practice action; the caller's context carries all semantics.

    Every transport invocation gets its own ``planner_calls/NNNN/`` record (a rejected
    attempt and its retry are two records), charged to the ``model_budget.json`` ledger
    shared with every other model user in the run. With ``PlannerMemory.resume`` the
    CLI reports a session's cost cumulatively, so the ledger keeps one entry per
    ``session_key`` and each call is charged the increase.
    """

    SYSTEM_PROMPT = (
        "You choose one practice action for a robot from the candidates you are given. "
        "You must not execute, simulate, or control anything: read the supplied context file "
        "and write only the requested decision file. Do not use other agents."
    )

    def __init__(
        self,
        *,
        output: Path,
        settings: SandboxSettings,
        model: str = "claude-opus-5-5",
        effort: str = "low",
        max_output_tokens: int = 4096,
        budget_limit: float = 20,
        memory: PlannerMemory = PlannerMemory.stateless,
        ledger_path: Path | None = None,
        transport: RobocodeDockerTransport | None = None,
    ) -> None:
        self.output, self.settings = output, settings
        self.model, self.effort, self.max_output_tokens = model, effort, max_output_tokens
        self.budget_limit, self.memory = budget_limit, PlannerMemory(memory)
        self.ledger_path = output / "model_budget.json" if ledger_path is None else ledger_path
        self.transport = (
            RobocodeDockerTransport(
                sandbox=settings, backend="claude", subagent_policy="disallowed"
            )
            if transport is None
            else transport
        )
        self.calls = 0
        self.decisions = 0
        self._session_ids: dict[str, str] = {}
        self._last_call: dict[str, Path] = {}

    def decide(
        self,
        *,
        context: dict[str, Any],
        candidate_ids: list[str],
        session_key: str,
        require_estimates: bool = False,
    ) -> Decision:
        if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Candidate action ids must be nonempty and unique")
        if not session_key:
            raise ValueError("A session key is required")
        decision_index = self.decisions
        self.decisions += 1
        rejection: str | None = None
        for attempt in (1, 2):
            outcome = self.invoke(
                context=context,
                candidate_ids=candidate_ids,
                session_key=session_key,
                require_estimates=require_estimates,
                attempt=attempt,
                decision_index=decision_index,
                rejection=rejection,
            )
            if isinstance(outcome, Decision):
                return outcome
            rejection = outcome
        raise DecisionError(f"No acceptable decision after two attempts: {rejection}")

    def invoke(  # noqa: PLR0914 - one call record is one bookkeeping pass
        self,
        *,
        context: dict[str, Any],
        candidate_ids: list[str],
        session_key: str,
        require_estimates: bool,
        attempt: int,
        decision_index: int,
        rejection: str | None,
    ) -> Decision | str:
        """Run one transport invocation and return the decision or why it was rejected."""
        index = self.calls
        self.calls += 1
        call_dir = self.output / "planner_calls" / f"{index:04d}"
        work = call_dir / "sandbox"
        work.mkdir(parents=True)
        StepFiles.json(path=work / "decision_context.json", value=context)
        previous = self._last_call.get(session_key) if self.memory is PlannerMemory.resume else None
        resumed = previous is not None
        if previous is not None:
            StepFiles.json(
                path=call_dir / "resume-session.json",
                value={"session_id": self._session_ids[session_key]},
            )
            # `--resume <id>` looks the transcript up in the CLI's project store, which
            # is mounted from the sandbox; carry the previous call's store forward.
            for source, target in (
                (previous / "sandbox" / ".agent_sessions", work / ".agent_sessions"),
                (previous / "cli-state", call_dir / "cli-state"),
            ):
                if source.is_dir():
                    shutil.copytree(source, target)
        ledger_session = (
            f"planner-{session_key}"
            if self.memory is PlannerMemory.resume
            else f"planner-{index:04d}"
        )
        budget = SharedBudget(
            path=self.ledger_path, limit=self.budget_limit, session=ledger_session
        )
        checkout = str(self.settings.robocode_checkout.resolve() / "src")
        if checkout not in sys.path:
            sys.path.insert(0, checkout)
        launcher, backends, omega = self.transport.modules()
        with budget.lock():
            if budget.exhausted:
                raise ModelBudgetExhausted()
            already = float(budget.data()["sessions"].get(ledger_session, 0.0))
            config = launcher.SandboxConfig(
                sandbox_dir=work,
                init_files={},
                prompt=self.prompt(
                    candidate_ids=candidate_ids,
                    require_estimates=require_estimates,
                    resumed=resumed,
                    rejection=rejection,
                ),
                system_prompt=self.SYSTEM_PROMPT,
                output_filename="decision.json",
                model=self.model,
                effort=self.effort,
                max_budget_usd=budget.remaining,
                max_turns=0,
                max_output_tokens=self.max_output_tokens,
                autocompact_pct=80,
                blackbox=True,
                mcp_tools=(),
                resume_previous_session=resumed,
            )
            backend = backends.create_backend(
                omega.DictConfig(dict(backend="claude", model=self.model, effort=self.effort))
            )
            result = self.transport.run_with_recovery(
                config=config, backend=backend, launcher=launcher
            )
            reported = result.total_cost_usd
            if reported is not None:
                budget.record(cumulative=reported)
            cost = 0.0 if reported is None else float(reported) - already
            session_id = self.transport.last_session_id(path=call_dir / "stream.jsonl")
            verdict = self.verdict(
                result=result,
                path=work / "decision.json",
                candidate_ids=candidate_ids,
                require_estimates=require_estimates,
            )
            StepFiles.json(
                path=call_dir / "result.json",
                value=dict(
                    success=bool(result.success),
                    error=result.error,
                    cost_usd=cost,
                    reported_cost_usd=reported,
                    memory=self.memory.value,
                    session_key=session_key,
                    session_id=session_id,
                    ledger_session=ledger_session,
                    resumed=resumed,
                    attempt=attempt,
                    decision_index=decision_index,
                    candidate_ids=list(candidate_ids),
                    accepted=isinstance(verdict, DecisionOutput),
                    rejection=None if isinstance(verdict, DecisionOutput) else verdict,
                ),
            )
            if not result.success and budget.exhausted:
                raise ModelBudgetExhausted()
        if self.memory is PlannerMemory.resume:
            if session_id is None:
                raise DecisionError("The CLI stream carries no session id to resume from")
            self._session_ids[session_key] = session_id
            self._last_call[session_key] = call_dir
        if isinstance(verdict, DecisionOutput):
            return Decision(
                action_id=verdict.action_id,
                reason=verdict.reason,
                estimates=verdict.estimates,
                cost_usd=cost,
                call_index=index,
            )
        return verdict

    @staticmethod
    def verdict(
        *, result: Any, path: Path, candidate_ids: list[str], require_estimates: bool
    ) -> DecisionOutput | str:
        """Validate decision.json, or say why it is rejected; never repair it."""
        if not result.success:
            return f"the model call failed: {result.error}"
        if not path.exists():
            return "decision.json was not written"
        try:
            output = DecisionOutput.model_validate_json(path.read_text())
        except ValidationError as exc:
            return f"decision.json is malformed: {exc}"
        if output.action_id not in candidate_ids:
            return f"unknown action id {output.action_id!r}"
        if require_estimates and not output.estimates:
            return "decision.json lacks the required estimates block"
        return output

    @staticmethod
    def prompt(
        *,
        candidate_ids: list[str],
        require_estimates: bool,
        resumed: bool,
        rejection: str | None,
    ) -> str:
        parts = []
        if rejection is not None:
            parts.append(f"Your previous decision.json was rejected: {rejection}")
        if resumed:
            parts.append(
                "This continues your earlier session. A new decision_context.json is in this "
                "directory; it carries what the caller chose to send since your last decision."
            )
        else:
            parts.append("Read decision_context.json in this directory.")
        parts.append(
            "Choose exactly one practice action for the robot from these candidate action "
            f"ids: {', '.join(candidate_ids)}. Write decision.json with exactly the keys "
            "action_id (one of the candidates) and reason (one short sentence)"
        )
        if require_estimates:
            parts.append(
                "and estimates: an object keyed by skill name whose values each have the "
                "numeric keys competence, learning_rate and expected_cost, with an entry for "
                "every skill the context describes."
            )
        else:
            parts[-1] += "."
        parts.append(
            "Do not execute, simulate or modify anything, do not edit the input file, and do "
            "not use other agents."
        )
        return " ".join(parts)
