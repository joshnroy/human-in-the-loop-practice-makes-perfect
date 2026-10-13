"""An LLM choosing the practice action at the belief-space decision seam.

Everything around the decision stays as it is for expectimax: the same candidate
set (`model.get_valid_actions`), the same Bayesian beliefs, the same stopping value
(`model.J`), the same trace events, the same decision log. Only the choice itself
is delegated, to one structured Claude call per decision through the Docker CLI
client. The planner never estimates a value for a non-STOP action: it logs every
candidate with `value: null` and marks the one the model chose.

With `estimator="llm"` the model is also the competence/cost estimator: no Bayesian
quantity (posterior means, the posterior cost forecast, `model.J`) is computed or
shown; the context carries raw outcome evidence instead, and every decision must
return per-skill competence, learning-rate and cost estimates, which the method's
logging accessors then report in place of the posterior.
"""

import math
import time
from pathlib import Path
from typing import Any, Literal, Protocol

from hitl_pmp.agentic_runtime.types import Decision, PlannerMemory, SkillEstimate
from hitl_pmp.core.method.types import GroundSkill

from .planner import BeliefSpacePlanner
from .tossing3d_observation_model import mean_competence, mean_learning_rate
from .tossing3d_transition_model import estimated_action_cost, render_atoms
from .types.belief_state import Tossing3DBeliefState
from .types.practice_situation import PracticeSituation
from .types.protocol import BeliefSpaceModel
from .types.search_state import Tossing3DSearchState
from .types.search_trace import SearchTrace
from .types.stop_action import NUM_SAMPLES, STOP_ACTION, StopAction
from .types.theta import Tossing3DTheta


class PracticeDecisionClient(Protocol):
    """What the planner needs from `agentic_runtime.decision_client.DecisionClient`."""

    def decide(
        self,
        *,
        context: dict[str, Any],
        candidate_ids: list[str],
        session_key: str,
        require_estimates: bool = False,
        estimate_keys: list[str] | None = None,
    ) -> Decision: ...


class UnknownDecisionError(ValueError):
    """The client returned an action id that is not one of this decision's candidates.

    The real client validates ids before returning, so this guards an injected or
    misbehaving one; there is deliberately no fallback to a classical solver."""


class MissingEstimatesError(ValueError):
    """The `llm` estimator's decision came back without its estimates block.

    The real client has already retried by then; nothing is filled in from the
    Bayesian posterior, which this arm never updates."""


class LlmPracticePlanner(
    BeliefSpacePlanner[Tossing3DSearchState, Tossing3DBeliefState, Tossing3DTheta, GroundSkill]
):
    name = "llm"
    SCHEMA = "hitl-pmp/llm-practice-planner/v1"
    STOP_ID = "STOP"
    GUIDE = {
        "candidates": "Every action applicable in the current symbolic state, plus STOP.",
        "candidates[].expected_cost": "Posterior-mean execution cost charged to the session.",
        "candidates[].posterior_mean_competence": "Mean of the Bayesian success-probability "
        "belief for the skill; None for actions without a belief.",
        "candidates[].posterior_mean_learning_rate": "Mean of the belief over how much one "
        "practice attempt improves that competence.",
        "candidates[].human_name": "Set when a person executes the action (a reset).",
        "stop_value": "Objective value of stopping practice now (G of the forecast deployment "
        "success rate minus the session cost); the same quantity expectimax compares "
        "every action against.",
        "session_cost": "Cost accumulated in this practice session so far.",
        "schedule": "The whole run's control-step budget, how much is used, how often the "
        "deployment policy is measured, and the model budget left for your own calls.",
        "history": "Scoped recent events: decisions, attempt outcomes, human resets.",
    }
    # The `llm` estimator's guide: no Bayesian quantity is computed or shown; the
    # model reasons from raw outcomes and states its own per-skill estimates.
    ESTIMATOR_GUIDE = {
        "candidates": "Every action applicable in the current symbolic state, plus STOP.",
        "candidates[].attempts": "Attempts of that action so far (all sessions).",
        "candidates[].successes": "Successful attempts of that action so far.",
        "candidates[].human_name": "Set when a person executes the action (a reset).",
        "estimate_skills": "The skill names your estimates block must cover: for each, "
        "your current estimate of its success probability (competence), how much one "
        "more practice attempt raises that probability (learning_rate), and the cost "
        "one attempt will be charged (expected_cost).",
        "evidence": "Raw outcome records. attempt_outcomes, execution_durations and "
        "measured_costs are per-action lists in observation order; failure_effects counts "
        "the symbolic effects of failed attempts; sampler_training_examples are the "
        "success labels of examples added to a learned sampler's training data, and "
        "pending_training_examples how many of those are not yet fitted (the sampler "
        "refits between practice sessions). evidence.scope says whether these are "
        "cumulative or only what was observed since your previous decision.",
        "session_cost": "Cost accumulated in this practice session so far.",
        "configured_costs": "The prices practice is charged under.",
        "schedule": "The whole run's control-step budget, how much is used, how often the "
        "deployment policy is measured, and the model budget left for your own calls.",
        "history": "Scoped recent events: decisions, attempt outcomes, human resets.",
    }

    def __init__(
        self,
        *,
        client: PracticeDecisionClient,
        memory: PlannerMemory,
        model: str,
        effort: str,
        budget_limit_usd: float,
        ledger: Any | None = None,
        estimator: Literal["bayesian", "llm"] = "bayesian",
    ) -> None:
        self.client = client
        self.estimator = estimator
        # The `llm` estimator's latest stated estimates, keyed by skill name.
        self.last_estimates: dict[str, SkillEstimate] = {}
        self.memory = PlannerMemory(memory)
        self.model_name = model
        self.effort = effort
        self.budget_limit_usd = float(budget_limit_usd)
        # The run's model-cost ledger (anything with `.remaining`), handed to the
        # step protocol so the schedule reports remaining dollars.
        self.ledger = ledger
        self.last_decision: Decision | None = None

    @staticmethod
    def candidate_id(*, ground: GroundSkill) -> str:
        return f"{ground.skill.name}({', '.join(obj.name for obj in ground.objects)})"

    def protocol(self) -> dict[str, Any]:
        return {
            "solver": self.name,
            "model": self.model_name,
            "effort": self.effort,
            "memory": self.memory.value,
            "budget_limit_usd": self.budget_limit_usd,
            "ledger_scope": "practice_only",
            "estimator": self.estimator,
        }

    def solve(
        self,
        *,
        environment_state: Tossing3DSearchState,
        summed_cost: float,
        belief_state: Tossing3DBeliefState,
        horizon: int,
        model: BeliefSpaceModel[
            Tossing3DSearchState, Tossing3DBeliefState, Tossing3DTheta, GroundSkill
        ],
        num_samples: int = NUM_SAMPLES,
        trace: SearchTrace | None = None,
        situation: PracticeSituation | None = None,
    ) -> tuple[float, GroundSkill | StopAction]:
        """Ask the client for one candidate id. Under the Bayesian estimator STOP's value
        is `model.J` and a chosen action's is NaN because the model does not estimate
        one; under the `llm` estimator `model.J` is never computed, every value is NaN,
        and the decision must carry the model's own per-skill estimates."""
        if situation is None:
            raise ValueError("The LLM planner needs the method's practice situation")
        started_at = time.perf_counter()
        candidates = model.get_valid_actions(environment_state=environment_state)
        ids = [self.candidate_id(ground=ground) for ground in candidates]
        assert len(set(ids)) == len(ids), f"candidate ids collide: {ids}"
        estimating = self.estimator == "llm"
        stop_value = (
            None
            if estimating
            else model.J(
                belief_state=belief_state, summed_cost=summed_cost, num_samples=num_samples
            )
        )
        candidate_ids = [*ids, self.STOP_ID]
        context = self.context(
            environment_state=environment_state,
            belief_state=belief_state,
            summed_cost=summed_cost,
            stop_value=stop_value,
            candidates=candidates,
            ids=ids,
            situation=situation,
        )
        if estimating:
            decision = self.client.decide(
                context=context,
                candidate_ids=candidate_ids,
                session_key=situation.session_key,
                require_estimates=True,
                estimate_keys=list(situation.estimate_skills),
            )
        else:
            decision = self.client.decide(
                context=context, candidate_ids=candidate_ids, session_key=situation.session_key
            )
        if decision.action_id not in candidate_ids:
            raise UnknownDecisionError(
                f"The planner client chose {decision.action_id!r}, not one of {candidate_ids}"
            )
        if estimating:
            if not decision.estimates:
                raise MissingEstimatesError(
                    f"Decision {decision.call_index} carries no estimates block"
                )
            self.last_estimates = dict(decision.estimates)
        self.last_decision = decision
        chosen: GroundSkill | StopAction = (
            STOP_ACTION
            if decision.action_id == self.STOP_ID
            else candidates[ids.index(decision.action_id)]
        )
        value = stop_value if chosen == STOP_ACTION and stop_value is not None else math.nan
        if trace is not None:
            if stop_value is not None:
                trace.record(event="stop_value", node=0, value=stop_value)
            for ground, candidate_id in zip(candidates, ids, strict=True):
                trace.record(
                    event="action_value",
                    node=0,
                    action=ground.model_dump(mode="json", fallback=str),
                    value=None,
                    chosen=candidate_id == decision.action_id,
                    value_source="llm_choice",
                )
            trace.record(
                event="choice",
                node=0,
                action=(
                    "STOP"
                    if chosen == STOP_ACTION
                    else chosen.model_dump(mode="json", fallback=str)
                ),
                value=value,
                reason=decision.reason,
            )
            trace.record(
                event="search_summary",
                node=0,
                solver=self.name,
                stop_value_source=None if estimating else "model_J",
                estimator=self.estimator,
                horizon=horizon,
                candidates=len(candidates),
                history_scope=situation.history.scope,
                history_events=len(situation.history.events),
                memory=self.memory.value,
                model=self.model_name,
                effort=self.effort,
                llm_call_index=decision.call_index,
                llm_cost_usd=decision.cost_usd,
                search_elapsed_seconds=time.perf_counter() - started_at,
                termination_reason="llm_decision",
            )
        return value, chosen

    def posterior_fields(
        self, *, belief_state: Tossing3DBeliefState, ground: GroundSkill | None
    ) -> dict[str, Any]:
        """A candidate's Bayesian numbers, or nothing at all under the `llm` estimator
        (whose posterior is never updated, so it would show the prior). `ground` None
        is STOP."""
        if self.estimator == "llm":
            return {}
        if ground is None:
            return {
                "expected_cost": 0.0,
                "posterior_mean_competence": None,
                "posterior_mean_learning_rate": None,
            }
        belief = belief_state.skill_beliefs.get(ground.skill.name)
        return {
            "expected_cost": estimated_action_cost(state=belief_state, action=ground),
            "posterior_mean_competence": (
                None if belief is None else mean_competence(belief=belief)
            ),
            "posterior_mean_learning_rate": (
                None if belief is None else mean_learning_rate(belief=belief)
            ),
        }

    def context(  # noqa: PLR0913 - one JSON document, every field named
        self,
        *,
        environment_state: Tossing3DSearchState,
        belief_state: Tossing3DBeliefState,
        summed_cost: float,
        stop_value: float | None,
        candidates: list[GroundSkill],
        ids: list[str],
        situation: PracticeSituation,
    ) -> dict[str, Any]:
        """`stop_value` is None exactly when the `llm` estimator is on: then no
        posterior quantity enters the document and the raw evidence does instead."""
        totals = situation.history.skill_totals
        described = []
        for ground, candidate_id in zip(candidates, ids, strict=True):
            counts = totals.get(candidate_id, {})
            described.append({
                "id": candidate_id,
                "skill": ground.skill.name,
                "objects": [obj.name for obj in ground.objects],
                "human_name": situation.human_names.get(candidate_id),
                **self.posterior_fields(belief_state=belief_state, ground=ground),
                "attempts": int(counts.get("attempts", 0)),
                "successes": int(counts.get("successes", 0)),
            })
        described.append({
            "id": self.STOP_ID,
            "skill": self.STOP_ID,
            "objects": [],
            "human_name": None,
            **self.posterior_fields(belief_state=belief_state, ground=None),
            "attempts": 0,
            "successes": 0,
            "ends_practice_session": True,
        })
        schedule = situation.schedule
        estimating = self.estimator == "llm"
        if estimating and situation.evidence is None:
            raise ValueError("The llm estimator needs the method's raw evidence")
        estimator_fields: dict[str, Any] = (
            {
                "estimate_skills": list(situation.estimate_skills),
                "evidence": situation.evidence.model_dump(mode="json")
                if situation.evidence is not None
                else None,
            }
            if estimating
            else {"stop_value": stop_value}
        )
        return {
            "schema": self.SCHEMA,
            "guide": dict(self.ESTIMATOR_GUIDE if estimating else self.GUIDE),
            "atoms": list(render_atoms(true_atoms=environment_state.true_atoms)),
            "candidates": described,
            **estimator_fields,
            "session_cost": summed_cost,
            "remaining_session_actions": situation.remaining_session_actions,
            "session_index": situation.session_index,
            "decision_index": situation.decision_index,
            "schedule": None if schedule is None else schedule.model_dump(mode="json"),
            "configured_costs": dict(situation.configured_costs),
            "history": {
                "scope": situation.history.scope,
                "events": list(situation.history.events),
                "skill_totals": dict(situation.history.skill_totals),
            },
            "planner": {
                "model": self.model_name,
                "effort": self.effort,
                "memory": self.memory.value,
            },
        }


def make_llm_practice_planner(
    *,
    sandbox: Path,
    output: Path,
    model: str,
    effort: str,
    budget_limit_usd: float,
    memory: PlannerMemory,
    estimator: Literal["bayesian", "llm"] = "bayesian",
) -> LlmPracticePlanner:
    """Build the planner over the real Docker Claude CLI client.

    Imported lazily: the client pulls in the step protocol and the shared ledger,
    which the classical solvers never need. Nothing here touches Docker; the first
    `decide` does."""
    import json

    from hitl_pmp.agentic_runtime.decision_client import DecisionClient
    from hitl_pmp.agentic_runtime.sandbox import SandboxSettings
    from hitl_pmp.agentic_states.budget import SharedBudget

    settings = SandboxSettings.model_validate(json.loads(sandbox.read_text()))
    client = DecisionClient(
        output=output,
        settings=settings,
        model=model,
        effort=effort,
        budget_limit=budget_limit_usd,
        memory=memory,
    )
    ledger = SharedBudget(
        path=client.ledger_path, limit=client.budget_limit, session="step-protocol-reader"
    )
    return LlmPracticePlanner(
        client=client,
        memory=memory,
        model=model,
        effort=effort,
        budget_limit_usd=budget_limit_usd,
        ledger=ledger,
        estimator=estimator,
    )
