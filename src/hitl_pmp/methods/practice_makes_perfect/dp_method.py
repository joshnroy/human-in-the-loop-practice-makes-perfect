"""Deliberate Practice with explicit solver, budget unit and allocation schedule."""

import hashlib
import json
import math
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field, PrivateAttr, model_validator

from hitl_pmp.core.method.types import GroundSkill, Policy
from hitl_pmp.core.practice_costs import PracticeAccounting
from hitl_pmp.core.problem.tasks.types import GroundAtom, Task
from hitl_pmp.results_writer.jsonl_log import JsonlLog

from .competence_models import LinearSkillCompetenceModel, OptimisticSkillCompetenceModel
from .dp_allocator import DpAllocator
from .dp_gurobi import DpAllocationResult, DpGurobiAllocator, DpOptimizationError
from .dp_mdp import DpTaskModel
from .dp_priors import DpPriorCalibration, DpSkillPrior
from .ees_method import STOP_SKILL, EesMethod


class DpMethod(EesMethod):
    """DP's linear learning model and globally optimal integer allocation.

    Each allocation is scored against the optimal policy of the full reachable
    symbolic deployment MDP. Successful skills apply their declared effects;
    failures absorb with reward zero. Human skills are available for practice
    setup, not deployment, so receive no allocation from this objective.

    The episode-budget/allocate-once and cost-budget/reallocate arms share the
    existing EES samplers and execution protocol, not CMA-ES. Cost-mode attempt
    prices include setup and failed setup but no human charge: a baseline never
    plans with a human skill's cost, which only our methods learn (Josh,
    2026-11-08). The cost file's human price is still charged to the run and its
    budget. These are empirical cost estimates, not our state-dependent
    belief-space lookahead over future practice resets.

    The curriculum follows the optimized policy's predecessor order, executing
    positive allocations only. Gurobi must certify the task-value optimum;
    enumeration is an explicitly selected reference, never a silent fallback.
    """

    # The Beta strength alpha + beta of a directly set initial competence: the same weak
    # strength scripts/tossing3d_dp_priors.py gives a calibrated one.
    DP_PRIOR_STRENGTH: ClassVar[float] = 2.0

    dp_budget: float = Field(gt=0.0, allow_inf_nan=False)
    # `steps` (Josh, 2026-11-13) is the episode arm under a step-capped run: an episode
    # still counts as one practice attempt for learning, but the allocation must fit the
    # budget in practice steps, each skill's episode priced at its own measured length
    # (dp_skill_cost_steps_for), and the budget is spent in the run's own practice steps.
    # Our adaptation, like `cost`: the paper's DP counts every episode equally.
    dp_budget_unit: Literal["cost", "episodes", "steps"] = "cost"
    dp_allocation_schedule: Literal["once", "period"] = "period"
    dp_solver: Literal["gurobi", "enumeration"] = "gurobi"
    dp_solver_time_limit: float = Field(default=60.0, gt=0.0, allow_inf_nan=False)
    dp_solver_log: Path | None = None
    dp_prior_calibration: Path | None = None
    # A starting competence per named robot skill, set directly rather than calibrated:
    # Beta(2p, 2(1-p)), the calibration path's prior strength, with the default Delta.
    dp_initial_competence: dict[str, float] = Field(default_factory=dict)
    dp_initial_learning_rate: float = Field(default=0.1, ge=0.0, allow_inf_nan=False)
    # Per-skill Delta priors for named robot skills; others keep dp_initial_learning_rate.
    dp_initial_learning_rate_for: dict[str, float] = Field(default_factory=dict)
    # Opt-in (Josh, 2026-11-12): a skill's Delta prior is dp_initial_learning_rate over
    # its number of continuous parameters, and 0 for a skill with none (nothing to tune,
    # so practice is not expected to improve it). A skill named in
    # dp_initial_learning_rate_for keeps that. Not from the paper.
    dp_learning_rate_by_parameters: bool = False
    dp_smoothing: float = Field(default=0.5, ge=0.0, le=1.0)
    # How competence and its learning rate are read off the outcomes: see
    # LinearSkillCompetenceModel. `cumulative` reproduces results from before 2026-11-12.
    dp_competence_estimator: Literal["recent_rounds", "cumulative"] = "recent_rounds"
    # A floor on every fitted Delta the allocator sees; None keeps them signed. Under
    # per-period reallocation a negative Delta fitted from a few noisy outcomes makes
    # practising that skill look harmful, so a re-solve allocates it nothing and no
    # outcome ever corrects the fit. Not from the paper, which is silent on both.
    dp_min_learning_rate: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    dp_max_skills: int = Field(default=8, ge=1)
    dp_max_allocations: int = Field(default=200_000, ge=1)
    dp_max_states: int = Field(default=4096, ge=1)
    skill_cost_prior_steps: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    # Control steps a named robot skill's attempt is priced at until DP has observed
    # its own cost for it; skills not named keep skill_cost_prior_steps. Like the
    # cost-unit budget itself this is this project's adaptation: the paper's DP counts
    # episodes and prices nothing.
    dp_skill_cost_prior_steps_for: dict[str, float] = Field(default_factory=dict)
    # Control steps a named robot skill's attempt is priced at for the whole run: a known
    # input like the goal rewards, never replaced by DP's own observation. A skill takes
    # either this or a starting value above, not both. Also this project's adaptation.
    dp_skill_cost_steps_for: dict[str, float] = Field(default_factory=dict)

    _dp_costs: dict[GroundSkill, list[float]] = PrivateAttr(default_factory=dict)
    _dp_pending_costs: dict[GroundSkill, float] = PrivateAttr(default_factory=dict)
    _dp_active_target: GroundSkill | None = PrivateAttr(default=None)
    _dp_unassigned_cost: float = PrivateAttr(default=0.0)
    _dp_forced_reset_pending: bool = PrivateAttr(default=False)
    _dp_allocation: dict[GroundSkill, int] = PrivateAttr(default_factory=dict)
    _dp_curriculum: list[GroundSkill] = PrivateAttr(default_factory=list)
    _dp_fitted_stages: set[GroundSkill] = PrivateAttr(default_factory=set)
    _dp_executed_since_solve: dict[GroundSkill, int] = PrivateAttr(default_factory=dict)
    _dp_solved_period: int | None = PrivateAttr(default=None)
    _dp_no_allocation: bool = PrivateAttr(default=False)
    _dp_episodes: int = PrivateAttr(default=0)
    _dp_solver_result: DpAllocationResult[GroundSkill] | None = PrivateAttr(default=None)
    _dp_skill_priors: dict[str, DpSkillPrior] = PrivateAttr(default_factory=dict)
    _dp_prior_sha256: str | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _budget_and_calibration_are_consistent(self) -> "DpMethod":
        if self.dp_budget_unit == "episodes" and not self.dp_budget.is_integer():
            raise ValueError("DP's episode budget must be an integer number of target rollouts")
        if self.dp_budget_unit == "steps":
            unpriced = {skill.name for skill in self.skills()} - set(self.dp_skill_cost_steps_for)
            if unpriced:
                raise ValueError(
                    "DP's step budget prices an episode at its skill's measured length, and "
                    f"dp_skill_cost_steps_for lacks one for {sorted(unpriced)}"
                )
        if self.dp_prior_calibration is not None:
            calibration = DpPriorCalibration.load(path=self.dp_prior_calibration)
            calibration.verify_simulators()
            calibration.verify_robot_skills(skills=self.skills())
            self._dp_skill_priors = dict(calibration.skills)
            self._dp_prior_sha256 = hashlib.sha256(
                self.dp_prior_calibration.read_bytes()
            ).hexdigest()
        if self.dp_initial_learning_rate_for:
            if self.dp_prior_calibration is not None:
                raise ValueError(
                    "dp_initial_learning_rate_for and dp_prior_calibration are mutually exclusive"
                )
            unknown = set(self.dp_initial_learning_rate_for) - {
                skill.name for skill in self.skills()
            }
            if unknown:
                raise ValueError(f"initial learning rates name not robot skills: {sorted(unknown)}")
            for name, rate in self.dp_initial_learning_rate_for.items():
                if not math.isfinite(rate) or rate < 0.0:
                    raise ValueError(
                        f"initial learning rate of {name} must be finite and non-negative"
                    )
        if self.dp_skill_cost_prior_steps_for:
            unknown = set(self.dp_skill_cost_prior_steps_for) - {
                skill.name for skill in self.skills()
            }
            if unknown:
                raise ValueError(f"cost prior steps name not robot skills: {sorted(unknown)}")
            for name, steps in self.dp_skill_cost_prior_steps_for.items():
                if not math.isfinite(steps) or steps <= 0.0:
                    raise ValueError(f"cost prior steps of {name} must be finite and positive")
        if self.dp_skill_cost_steps_for:
            unknown = set(self.dp_skill_cost_steps_for) - {skill.name for skill in self.skills()}
            if unknown:
                raise ValueError(f"fixed cost steps name not robot skills: {sorted(unknown)}")
            both = set(self.dp_skill_cost_steps_for) & set(self.dp_skill_cost_prior_steps_for)
            if both:
                raise ValueError(
                    f"fixed cost steps and cost prior steps both name {sorted(both)}: a skill's "
                    "price is either fixed or a starting value"
                )
            for name, steps in self.dp_skill_cost_steps_for.items():
                if not math.isfinite(steps) or steps <= 0.0:
                    raise ValueError(f"fixed cost steps of {name} must be finite and positive")
        if self.dp_initial_competence:
            if self.dp_prior_calibration is not None:
                raise ValueError(
                    "dp_initial_competence and dp_prior_calibration are mutually exclusive"
                )
            unknown = set(self.dp_initial_competence) - {skill.name for skill in self.skills()}
            if unknown:
                raise ValueError(f"initial competences name not robot skills: {sorted(unknown)}")
            for name, competence in self.dp_initial_competence.items():
                if not 0.0 < competence < 1.0:
                    raise ValueError(
                        f"initial competence of {name} must be strictly between 0 and 1"
                    )
                self._dp_skill_priors[name] = DpSkillPrior(
                    alpha=self.DP_PRIOR_STRENGTH * competence,
                    beta=self.DP_PRIOR_STRENGTH * (1.0 - competence),
                    initial_learning_rate=self.initial_learning_rate(skill_name=name),
                )
        return self

    def initial_learning_rate(self, *, skill_name: str | None) -> float:
        """A skill's Delta prior: its own override, else the global one."""
        if skill_name is None:
            return self.dp_initial_learning_rate
        if skill_name in self.dp_initial_learning_rate_for:
            return self.dp_initial_learning_rate_for[skill_name]
        if self.dp_learning_rate_by_parameters:
            parameters = {skill.name: skill.param_dim for skill in self.skills()}.get(skill_name, 0)
            return self.dp_initial_learning_rate / parameters if parameters else 0.0
        return self.dp_initial_learning_rate

    def competence_model(self, *, ground_skill: GroundSkill) -> OptimisticSkillCompetenceModel:
        prior = self._dp_skill_priors.get(ground_skill.skill.name)
        if prior is not None and ground_skill not in self._competence_models:
            self._competence_models[ground_skill] = OptimisticSkillCompetenceModel(
                alpha=prior.alpha,
                beta=prior.beta,
                window_size=self.competence_window_size,
                recency_size=self.competence_recency_size,
            )
        return super().competence_model(ground_skill=ground_skill)

    def practice_complete(self) -> bool:
        return self.practice_completion_reason() is not None

    def practice_completion_reason(self) -> str | None:
        exhausted = (
            self.dp_allocation_schedule == "once"
            and self._dp_solved_period is not None
            and not any(self.dp_remaining_allocation().values())
        )
        spent = (
            self._dp_episodes
            if self.dp_budget_unit == "episodes"
            else (0 if self._practice_accounting is None else self._spent_from_accounting())
        )
        if spent >= self.dp_budget:
            return "budget_spent"
        if self._dp_no_allocation:
            return "zero_allocation"
        if exhausted:
            return "once_curriculum_consumed"
        return None

    def dp_episodes(self) -> int:
        return self._dp_episodes

    def end_cycle(self) -> None:
        super().end_cycle()
        self._write_solver_record(
            record={
                "event": "period_end",
                "remaining_budget": self._remaining_budget(),
            }
        )

    def dp_solver_result(self) -> DpAllocationResult[GroundSkill] | None:
        return self._dp_solver_result

    def _remaining_budget(self) -> float:
        self._require_accounting()
        spent = (
            self._dp_episodes
            if self.dp_budget_unit == "episodes"
            else self._spent_from_accounting()
        )
        return max(0.0, self.dp_budget - spent)

    def _spent_from_accounting(self) -> float:
        accounting = self._require_accounting()
        return (
            accounting.practice_steps if self.dp_budget_unit == "steps" else accounting.total_cost
        )

    def _attempt_price(self, *, ground_skill: GroundSkill) -> float:
        """What one allocated attempt takes from the budget, in the budget's own unit."""
        if self.dp_budget_unit == "episodes":
            return 1.0
        if self.dp_budget_unit == "steps":
            # A human skill is no robot skill, so has no measured length; it is on no
            # deployment transition and takes skill_cost_prior_steps, as under `cost`.
            return self.dp_skill_cost_steps_for.get(
                ground_skill.skill.name, self.skill_cost_prior_steps
            )
        return self.dp_expected_attempt_cost(ground_skill=ground_skill)

    def _needs_allocation(self) -> bool:
        return self._dp_solved_period is None or (
            self.dp_allocation_schedule == "period"
            and self._dp_solved_period != self._practice_periods
        )

    def pursues_goal_in_period(self, *, period: int) -> bool:
        return False

    def get_practice_policy(self, *, task: Task) -> Policy:
        policy = super().get_practice_policy(task=task)
        self._dp_active_target = None
        remaining = self._remaining_budget()
        if remaining > 0 and self._needs_allocation():
            self._solve(budget=remaining)
        return policy

    def dp_allocation(self) -> dict[GroundSkill, int]:
        """The current period's allocated target attempts, including zero counts."""
        return dict(self._dp_allocation)

    def dp_remaining_allocation(self) -> dict[GroundSkill, int]:
        return {
            ground: max(0, count - self._dp_executed_since_solve.get(ground, 0))
            for ground, count in self._dp_allocation.items()
        }

    def begin_practice_attempt(self, *, ground_skill: GroundSkill) -> None:
        self._dp_active_target = ground_skill
        self._dp_pending_costs[ground_skill] = (
            self._dp_pending_costs.get(ground_skill, 0.0) + self._dp_unassigned_cost
        )
        self._dp_unassigned_cost = 0.0

    def observe_execution_cost(self, *, cost: float, steps: int, complete: bool) -> None:
        super().observe_execution_cost(cost=cost, steps=steps, complete=complete)
        if self._dp_forced_reset_pending:
            # A forced reset is the human's, so its charge is never DP's estimate.
            self._dp_forced_reset_pending = False
            return
        if self._execution_ground_skill in self.skill_provider.human_cube_bin_reset_skills():
            cost = 0.0
        target = self._dp_active_target
        if target is not None and complete and self._execution_ground_skill == target:
            costs = self._dp_costs.setdefault(target, [])
            costs.append(self._dp_pending_costs.pop(target, 0.0) + cost)
            self._dp_active_target = None
            self._dp_episodes += 1
            self._dp_executed_since_solve[target] = self._dp_executed_since_solve.get(target, 0) + 1
            return
        if target is None:
            self._dp_unassigned_cost += cost
        else:
            self._dp_pending_costs[target] = self._dp_pending_costs.get(target, 0.0) + cost

    def record_forced_reset(self, *, reason: str, cost: float, consecutive_stranded: int) -> None:
        super().record_forced_reset(
            reason=reason, cost=cost, consecutive_stranded=consecutive_stranded
        )
        self._dp_forced_reset_pending = True

    def dp_expected_attempt_cost(self, *, ground_skill: GroundSkill) -> float:
        fixed = self.dp_skill_cost_steps_for.get(ground_skill.skill.name)
        if fixed is not None:
            return max(1e-12, fixed * self._require_accounting().robot_charge().cost)
        observed = self._dp_costs.get(ground_skill, [])
        pending = self._dp_pending_costs.get(ground_skill, 0.0)
        if observed:
            return max(1e-12, (sum(observed) + pending) / len(observed))
        accounting = self._require_accounting()
        steps = self.dp_skill_cost_prior_steps_for.get(
            ground_skill.skill.name, self.skill_cost_prior_steps
        )
        prior = steps * accounting.robot_charge().cost
        return max(1e-12, prior + pending)

    def select_skill_to_practice(self, *, true_atoms: frozenset[GroundAtom]) -> list[GroundSkill]:
        """Use DP's curriculum without activating EES-only selector behavior."""
        del true_atoms
        return self.choose_practice_target()

    def choose_practice_target(self) -> list[GroundSkill]:
        remaining = self._remaining_budget()
        if self.practice_complete():
            return [STOP_SKILL]
        if self._needs_allocation():
            self._solve(budget=remaining)
        left = self.dp_remaining_allocation()
        completed = {
            candidate
            for candidate, count in self._dp_allocation.items()
            if count > 0 and left[candidate] == 0
        }
        if completed - self._dp_fitted_stages:
            self.fit_samplers()
            self._dp_fitted_stages.update(completed)
        ranked = [candidate for candidate in self._dp_curriculum if left[candidate] > 0]
        for candidate in ranked[:1]:
            self.record_practice_target(name=candidate.skill.name, field="scored")
        return [*ranked[:1], STOP_SKILL]

    def _solve(self, *, budget: float) -> None:
        mdp = DpTaskModel.build(
            tasks=self.planning_progress_tasks(),
            skills=self.skills(),
            objects=self.objects(),
            max_states=self.dp_max_states,
            goal_rewards=self.skill_provider.goal_rewards(),
        )
        if self.dp_prior_calibration is not None:
            missing = {edge.skill.skill.name for edge in mdp.transitions} - set(
                self._dp_skill_priors
            )
            if missing:
                raise ValueError(f"deployment skills lack calibrated DP priors: {sorted(missing)}")
        humans = self.offered_partial_resets()
        candidates = list(
            dict.fromkeys([
                *(
                    self.practice_candidates()
                    if self.practice_groundings == "all"
                    else self._competence_models
                ),
                *(transition.skill for transition in mdp.transitions),
                *humans,
            ])
        )
        competences: dict[GroundSkill, float] = {}
        learning_rates: dict[GroundSkill, float] = {}
        for candidate in candidates:
            history = (
                self._human_competence_models.get(candidate)
                if candidate in humans
                else self._competence_models.get(candidate)
            )
            model = self._linear_model(history=history, ground_skill=candidate)
            competences[candidate] = model.get_current_competence()
            learning_rates[candidate] = (
                model.learning_rate()
                if self.dp_min_learning_rate is None
                else max(model.learning_rate(), self.dp_min_learning_rate)
            )
        costs = {candidate: self._attempt_price(ground_skill=candidate) for candidate in candidates}
        try:
            if self.dp_solver == "gurobi":
                result = DpGurobiAllocator.allocate(
                    mdp=mdp,
                    competences=competences,
                    learning_rates=learning_rates,
                    costs=costs,
                    budget=budget,
                    time_limit=self.dp_solver_time_limit,
                )
            else:
                allocation = DpAllocator.allocate(
                    mdp=mdp,
                    competences=competences,
                    learning_rates=learning_rates,
                    costs=costs,
                    budget=budget,
                    max_skills=self.dp_max_skills,
                    max_allocations=self.dp_max_allocations,
                )
                value = DpAllocator.predicted_success(
                    mdp=mdp,
                    competences=competences,
                    learning_rates=learning_rates,
                    allocation=allocation,
                )
                result = DpAllocationResult[GroundSkill](
                    allocation=allocation,
                    solver="enumeration",
                    status="ENUMERATED",
                    objective=value,
                    objective_bound=value,
                    relative_gap=0,
                    runtime_seconds=0,
                )
        except DpOptimizationError as error:
            self._write_solver_record(record={"error": error.diagnostics, "budget": budget})
            raise
        self._dp_solver_result = result
        self._dp_allocation = dict(result.allocation)
        self._dp_executed_since_solve = {}
        self._dp_no_allocation = not any(self._dp_allocation.values())
        self._dp_solved_period = self._practice_periods
        self._write_solver_record(
            record={
                **result.model_dump(exclude={"allocation"}),
                "budget": budget,
                "allocation": [
                    {
                        "skill": ground.skill.name,
                        "objects": [obj.name for obj in ground.objects],
                        "attempts": count,
                        "attempt_price": costs[ground],
                        "competence": competences[ground],
                        "learning_rate": learning_rates[ground],
                    }
                    for ground, count in result.allocation.items()
                ],
                "groups": [
                    {
                        "skill": ground.skill.name,
                        "objects": [obj.name for obj in ground.objects],
                        "competence": competences[ground],
                        "learning_rate": learning_rates[ground],
                    }
                    for ground in candidates
                ],
            }
        )
        predicted = DpAllocator.predicted_competences(
            competences=competences,
            learning_rates=learning_rates,
            allocation=self._dp_allocation,
        )
        self._dp_curriculum = list(
            dict.fromkeys([*mdp.curriculum(competences=predicted), *self._dp_allocation])
        )
        self._dp_fitted_stages = set()

    def _write_solver_record(self, *, record: dict[str, object]) -> None:
        if self.dp_solver_log is not None:
            self.dp_solver_log.parent.mkdir(parents=True, exist_ok=True)
            JsonlLog.append(
                path=self.dp_solver_log,
                line=json.dumps({
                    "event": "allocation",
                    "budget_unit": self.dp_budget_unit,
                    "allocation_schedule": self.dp_allocation_schedule,
                    "allocation_schedule_status": "unverified_author_protocol",
                    "learning_rate_update": "per_retained_episode_closed_cycles",
                    "completion_reason": self.practice_completion_reason(),
                    "remaining_budget": self._remaining_budget(),
                    "period": self._practice_periods,
                    "episodes": self._dp_episodes,
                    "physical_cost": self._require_accounting().total_cost,
                    "prior_calibration_sha256": self._dp_prior_sha256,
                    "initial_competence": dict(self.dp_initial_competence),
                    "initial_learning_rate_for": dict(self.dp_initial_learning_rate_for),
                    "practice_groundings": self.practice_groundings,
                    "executed_since_solve": [
                        {
                            "skill": ground.skill.name,
                            "objects": [obj.name for obj in ground.objects],
                            "attempts": count,
                        }
                        for ground, count in self._dp_executed_since_solve.items()
                    ],
                    **record,
                })
                + "\n",
            )

    def _linear_model(
        self,
        *,
        history: OptimisticSkillCompetenceModel | None,
        ground_skill: GroundSkill | None = None,
    ) -> LinearSkillCompetenceModel:
        prior = (
            self._dp_skill_priors.get(ground_skill.skill.name) if ground_skill is not None else None
        )
        return LinearSkillCompetenceModel(
            alpha=prior.alpha if prior is not None else self.prior_alpha,
            beta=prior.beta if prior is not None else self.prior_beta,
            initial_delta=(
                prior.initial_learning_rate
                if prior is not None
                else self.initial_learning_rate(
                    skill_name=ground_skill.skill.name if ground_skill is not None else None
                )
            ),
            smoothing=self.dp_smoothing,
            estimator=self.dp_competence_estimator,
            cycle_observations=(
                [list(cycle) for cycle in history.cycle_observations]
                if history is not None
                else [[]]
            ),
        )

    def _require_accounting(self) -> PracticeAccounting:
        accounting = self._practice_accounting
        if accounting is None:
            raise RuntimeError(
                "DP's budget and costs are PracticeAccounting.total_cost; run it under "
                "the step protocol with --practice-cost-config"
            )
        return accounting
