"""Deliberate Practice with explicit solver, budget unit and allocation schedule."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, PrivateAttr, model_validator

from hitl_pmp.core.method.types import GroundSkill, Policy
from hitl_pmp.core.practice_costs import PracticeAccounting
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.results_writer.jsonl_log import JsonlLog

from .always_ask_reset import AlwaysAskReset
from .competence_models import LinearSkillCompetenceModel, OptimisticSkillCompetenceModel
from .dp_allocator import DpAllocator
from .dp_gurobi import DpAllocationResult, DpGurobiAllocator, DpOptimizationError
from .dp_mdp import DpMdp, DpTaskModel
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
    prices include setup, failed setup, human assistance and
    trailing forced resets. These are empirical cost estimates, not our
    state-dependent belief-space lookahead over future practice resets.

    The curriculum follows the optimized policy's predecessor order, executing
    positive allocations only. Gurobi must certify both optimization stages;
    enumeration is an explicitly selected reference, never a silent fallback.
    """

    dp_budget: float = Field(gt=0.0, allow_inf_nan=False)
    dp_budget_unit: Literal["cost", "episodes"] = "cost"
    dp_allocation_schedule: Literal["once", "period"] = "period"
    dp_solver: Literal["gurobi", "enumeration"] = "gurobi"
    dp_solver_time_limit: float = Field(default=60.0, gt=0.0, allow_inf_nan=False)
    dp_solver_log: Path | None = None
    dp_prior_calibration: Path | None = None
    dp_initial_learning_rate: float = Field(default=0.1, ge=0.0, allow_inf_nan=False)
    dp_smoothing: float = Field(default=0.5, ge=0.0, le=1.0)
    dp_max_skills: int = Field(default=8, ge=1)
    dp_max_allocations: int = Field(default=200_000, ge=1)
    dp_max_states: int = Field(default=4096, ge=1)
    skill_cost_prior_steps: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    score_human_skills: bool = True

    _dp_costs: dict[GroundSkill, list[float]] = PrivateAttr(default_factory=dict)
    _dp_pending_costs: dict[GroundSkill, float] = PrivateAttr(default_factory=dict)
    _dp_active_target: GroundSkill | None = PrivateAttr(default=None)
    _dp_last_attempt: tuple[GroundSkill, int] | None = PrivateAttr(default=None)
    _dp_unassigned_cost: float = PrivateAttr(default=0.0)
    _dp_forced_reset_pending: bool = PrivateAttr(default=False)
    _dp_allocation: dict[GroundSkill, int] = PrivateAttr(default_factory=dict)
    _dp_groups: dict[GroundSkill, GroundSkill] = PrivateAttr(default_factory=dict)
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
    def _human_skills_are_first_class(self) -> "DpMethod":
        if not self.score_human_skills:
            raise ValueError("DpMethod needs score_human_skills: human skills are first-class")
        if self.dp_budget_unit == "episodes" and not self.dp_budget.is_integer():
            raise ValueError("DP's episode budget must be an integer number of target rollouts")
        if self.dp_prior_calibration is not None:
            calibration = DpPriorCalibration.load(path=self.dp_prior_calibration)
            calibration.verify_simulators()
            calibration.verify_robot_skills(skills=self.skills())
            self._dp_skill_priors = dict(calibration.skills)
            self._dp_prior_sha256 = hashlib.sha256(
                self.dp_prior_calibration.read_bytes()
            ).hexdigest()
        return self

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

    def pooled_competence_model(
        self, *, ground_skill: GroundSkill
    ) -> OptimisticSkillCompetenceModel:
        model = super().pooled_competence_model(ground_skill=ground_skill)
        prior = self._dp_skill_priors.get(ground_skill.skill.name)
        if prior is not None:
            model.alpha = prior.alpha
            model.beta = prior.beta
        return model

    def practice_complete(self) -> bool:
        exhausted = (
            self.dp_allocation_schedule == "once"
            and self._dp_solved_period is not None
            and not any(self.dp_remaining_allocation().values())
        )
        spent = (
            self._dp_episodes
            if self.dp_budget_unit == "episodes"
            else (
                self._practice_accounting.total_cost if self._practice_accounting is not None else 0
            )
        )
        return self._dp_no_allocation or exhausted or spent >= self.dp_budget

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
        accounting = self._require_accounting()
        spent = self._dp_episodes if self.dp_budget_unit == "episodes" else accounting.total_cost
        return max(0.0, self.dp_budget - spent)

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
        self._dp_last_attempt = None
        remaining = self._remaining_budget()
        if remaining > 0 and self._needs_allocation():
            self._solve(budget=remaining)
        return policy

    def dp_allocation(self) -> dict[GroundSkill, int]:
        """Logical target attempts, keyed by one representative per shared parent."""
        return dict(self._dp_allocation)

    def dp_remaining_allocation(self) -> dict[GroundSkill, int]:
        executed: dict[GroundSkill, int] = {}
        for ground, count in self._dp_executed_since_solve.items():
            group = self._dp_groups.get(ground, ground)
            executed[group] = executed.get(group, 0) + count
        return {
            ground: max(0, count - executed.get(ground, 0))
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
        target = self._dp_active_target
        if self._dp_forced_reset_pending:
            self._dp_forced_reset_pending = False
            if target is None and self._dp_last_attempt is not None:
                ground, index = self._dp_last_attempt
                self._dp_costs[ground][index] += cost
                return
        elif target is not None and complete and self._execution_ground_skill == target:
            costs = self._dp_costs.setdefault(target, [])
            costs.append(self._dp_pending_costs.pop(target, 0.0) + cost)
            self._dp_last_attempt = (target, len(costs) - 1)
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
        observed = self._dp_costs.get(ground_skill, [])
        pending = self._dp_pending_costs.get(ground_skill, 0.0)
        if observed:
            return max(1e-12, (sum(observed) + pending) / len(observed))
        accounting = self._require_accounting()
        prior = self.skill_cost_prior_steps * accounting.robot_charge().cost
        if self.always_ask_reset is not None and AlwaysAskReset.triggers(ground_skill=ground_skill):
            prior += self.start_reset_cost()
        return max(1e-12, prior + pending)

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
        ranked = [
            candidate
            for candidate in self._dp_curriculum
            if left[self._dp_groups.get(candidate, candidate)] > 0
        ]
        for candidate in ranked[:1]:
            self.record_practice_target(name=candidate.skill.name, field="scored")
        return [*ranked[:1], STOP_SKILL]

    def _solve(self, *, budget: float) -> None:
        mdp = DpTaskModel.build(
            tasks=self.planning_progress_tasks(),
            skills=self.skills(),
            objects=self.objects(),
            max_states=self.dp_max_states,
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
                    if self.competence_sharing != "grounded" or self.practice_groundings == "all"
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
                else (
                    self.forecast_competence_model(ground_skill=candidate)
                    if self.competence_sharing != "grounded"
                    else self._competence_models.get(candidate)
                )
            )
            model = self._linear_model(history=history, ground_skill=candidate)
            competences[candidate] = model.get_current_competence()
            if candidate not in humans and self.competence_sharing in {"parent", "rate"}:
                model = self._linear_model(
                    history=self.pooled_competence_model(ground_skill=candidate),
                    ground_skill=candidate,
                )
            learning_rates[candidate] = model.learning_rate()
        groups = self._allocation_groups(mdp=mdp, candidates=candidates, competences=competences)
        self._dp_groups = groups if groups is not None else {}
        costs = {
            candidate: 1.0
            if self.dp_budget_unit == "episodes"
            else self._group_attempt_price(ground_skill=candidate)
            for candidate in dict.fromkeys(
                self._dp_groups.get(ground, ground) for ground in candidates
            )
        }
        try:
            if self.dp_solver == "gurobi":
                result = DpGurobiAllocator.allocate(
                    mdp=mdp,
                    competences=competences,
                    learning_rates=learning_rates,
                    costs=costs,
                    budget=budget,
                    time_limit=self.dp_solver_time_limit,
                    groups=groups,
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
                    groups=groups,
                )
                value = DpAllocator.predicted_success(
                    mdp=mdp,
                    competences=competences,
                    learning_rates=learning_rates,
                    allocation=allocation,
                    groups=groups,
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
                        "group": self._allocation_identity(ground_skill=ground),
                    }
                    for ground, count in result.allocation.items()
                ],
                "groups": [
                    {
                        "skill": ground.skill.name,
                        "objects": [obj.name for obj in ground.objects],
                        "group": self._allocation_identity(ground_skill=ground),
                        "competence": competences[ground],
                        "learning_rate": learning_rates[ground],
                    }
                    for ground in candidates
                ],
                "group_price_rule": "representative_or_minimum_shared_all_alternative",
            }
        )
        predicted = DpAllocator.predicted_competences(
            competences=competences,
            learning_rates=learning_rates,
            allocation=self._dp_allocation,
            groups=groups,
        )
        curriculum: dict[GroundSkill, GroundSkill] = {}
        for candidate in [*mdp.curriculum(competences=predicted), *self._dp_allocation]:
            curriculum.setdefault(self._dp_groups.get(candidate, candidate), candidate)
        self._dp_curriculum = list(curriculum.values())
        self._dp_no_allocation = not any(self._dp_allocation.values())
        self._dp_fitted_stages = set()
        self._dp_solved_period = self._practice_periods

    def _allocation_groups(
        self,
        *,
        mdp: DpMdp[GroundSkill],
        candidates: list[GroundSkill],
        competences: dict[GroundSkill, float],
    ) -> dict[GroundSkill, GroundSkill] | None:
        if self.competence_sharing == "grounded":
            return None
        humans = set(self.offered_partial_resets())
        representatives: dict[str, GroundSkill] = {}
        for candidate in [*mdp.curriculum(competences=competences), *candidates]:
            if candidate not in humans:
                representatives.setdefault(candidate.skill.name, candidate)
        return {
            candidate: candidate if candidate in humans else representatives[candidate.skill.name]
            for candidate in candidates
        }

    def _group_attempt_price(self, *, ground_skill: GroundSkill) -> float:
        """Certainty-equivalent price of the representative's directed attempt.

        Shared/all permits the cheapest estimated alternative. Other modes use
        the representative's own estimate, never a pooled mean across contexts.
        The estimate is frozen for this solve; realized setup, target and reset
        costs remain attributed to the actual grounding and physical accounting.
        This does not predict state-dependent future resets or guarantee that
        another curriculum grounding will realize the representative's price.
        """
        alternatives = (
            self.practice_alternatives(ground_skill=ground_skill)
            if self.competence_sharing != "grounded"
            and self.practice_groundings == "all"
            and ground_skill not in self.offered_partial_resets()
            else [ground_skill]
        )
        return min(
            self.dp_expected_attempt_cost(ground_skill=alternative) for alternative in alternatives
        )

    def _allocation_identity(self, *, ground_skill: GroundSkill) -> dict[str, object]:
        shared = self.competence_sharing != "grounded" and (
            ground_skill not in self.offered_partial_resets()
        )
        return {
            "kind": "parent" if shared else "grounded",
            "skill": ground_skill.skill.name,
            "objects": None if shared else [obj.name for obj in ground_skill.objects],
        }

    def _write_solver_record(self, *, record: dict[str, object]) -> None:
        if self.dp_solver_log is not None:
            self.dp_solver_log.parent.mkdir(parents=True, exist_ok=True)
            JsonlLog.append(
                path=self.dp_solver_log,
                line=json.dumps({
                    "event": "allocation",
                    "budget_unit": self.dp_budget_unit,
                    "allocation_schedule": self.dp_allocation_schedule,
                    "period": self._practice_periods,
                    "episodes": self._dp_episodes,
                    "physical_cost": self._require_accounting().total_cost,
                    "prior_calibration_sha256": self._dp_prior_sha256,
                    "competence_sharing": self.competence_sharing,
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
                prior.initial_learning_rate if prior is not None else self.dp_initial_learning_rate
            ),
            smoothing=self.dp_smoothing,
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
