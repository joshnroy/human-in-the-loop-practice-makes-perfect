from pydantic import Field, PrivateAttr, model_validator

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.practice_costs import PracticeAccounting

from .always_ask_reset import AlwaysAskReset
from .competence_models import LinearSkillCompetenceModel, OptimisticSkillCompetenceModel
from .dp_allocator import DpAllocator
from .ees_method import STOP_SKILL, EesMethod


class DpMethod(EesMethod):
    """The Deliberate Practice baseline (Vats & Konidaris, *Deliberate Practice:
    Learning Robot Skills under a Budget*), adapted to this project's practice protocol.

    DP splits a practice budget across skills before practising: it maximises predicted
    task success, with each skill's competence predicted by the linear model
    min(1, p_u + Delta_u * b_u) (`LinearSkillCompetenceModel`), subject to the budget.
    `DpAllocator` finds that optimum exactly.

    **What this class changes, and nothing else.** It overrides only which skill to
    practise (`choose_practice_target`) and observes what each execution cost
    (`observe_execution_cost`, which EES ignores). The goal phase, the samplers, cost
    accounting, the always-ask wrapper, the practice loop, planning and the planner's own
    `-log(competence)` prices are EES's, unchanged; skills, predicates and operators come
    from the domain's `SkillProvider` exactly as for EES.

    **Every practice period** it re-solves the allocation from the current competence
    estimates and the budget that remains (`dp_budget` minus
    `PracticeAccounting.total_cost`), then ranks candidates by allocation left this
    period: allocated attempts minus executions since the solve, with skills allocated
    0 last. The runner tries them in order until one is reachable. Once the budget is
    spent, `practice_complete` ends practice: the run spends nothing more.

    **Inputs to the allocation.**
    - Plans: the seen tasks' cached plans EES situates against
      (`planning_progress_plans`), the same plan-cost signal EES uses.
    - p_u, Delta_u: replayed into a linear model from the outcome history EES already
      keeps per ground skill (random-exploration attempts excluded, as for EES), from
      EES's Beta prior and `dp_initial_learning_rate`.
    - c_u: the expected cost of one attempt, in `PracticeAccounting.total_cost` units:
      the running mean of the observed cost of each completed execution of that
      grounding, from a prior of `skill_cost_prior_steps` robot steps (the belief-space
      method's own prior). Under the always-ask wrapper a forced reset is charged to the
      skill that last ran, so an attempt's cost includes the reset it triggered; the
      prior of a skill that triggers one includes its price too. The cost of a prefix
      that navigates to a skill's preconditions is not part of that skill's c_u.

    **Human skills are first-class** (`score_human_skills` is required). By default
    every partial human reset the domain offers is a candidate with its own competence
    model, allocation and cost, under exactly the rule a robot skill gets. The
    objective is predicted success on the evaluation tasks' plans, which never offer a
    human, so a human skill's competence never enters it and the rule always allocates
    it 0: DP's objective cannot value practising help, by design. Asking for help still
    spends budget. Under `--always-ask-reset` no partial reset is offered, so the
    allocation is over robot skills only.

    **Ours, not the paper's:** the per-attempt cost c_u (the paper counts every practice
    episode alike, sum_u b_u <= B); re-solving every period (the paper describes one
    allocation, then a curriculum); the objective as a mean over plans of each plan's
    product of competences (the paper values its abstract MDP); and the human-skill
    handling above. The paper's Gurobi bilinear program is replaced by exact
    enumeration, which reaches the same optimum and raises above its caps instead of
    approximating."""

    # In PracticeAccounting.total_cost units.
    dp_budget: float = Field(gt=0.0, allow_inf_nan=False)
    # The paper gives no experimental value; 0.1 per attempt is its illustrative one.
    dp_initial_learning_rate: float = 0.1
    # The paper's smoothing factor; it gives no experimental value.
    dp_smoothing: float = Field(default=0.5, ge=0.0, le=1.0)
    dp_max_skills: int = Field(default=8, ge=1)
    dp_max_allocations: int = Field(default=200_000, ge=1)
    skill_cost_prior_steps: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    score_human_skills: bool = True

    _dp_costs: dict[GroundSkill, list[float]] = PrivateAttr(default_factory=dict)
    _dp_forced_reset_pending: bool = PrivateAttr(default=False)
    _dp_allocation: dict[GroundSkill, int] = PrivateAttr(default_factory=dict)
    _dp_executed_since_solve: dict[GroundSkill, int] = PrivateAttr(default_factory=dict)
    _dp_solved_period: int | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _human_skills_are_first_class(self) -> "DpMethod":
        if not self.score_human_skills:
            raise ValueError("DpMethod needs score_human_skills: human skills are first-class")
        return self

    # ------------------------------------------------------------------ budget

    def practice_complete(self) -> bool:
        accounting = self._practice_accounting
        return accounting is not None and accounting.total_cost >= self.dp_budget

    def dp_allocation(self) -> dict[GroundSkill, int]:
        """The attempts the current period's solve allocated, per candidate."""
        return dict(self._dp_allocation)

    def dp_remaining_allocation(self) -> dict[GroundSkill, int]:
        return {
            ground: max(0, count - self._dp_executed_since_solve.get(ground, 0))
            for ground, count in self._dp_allocation.items()
        }

    # -------------------------------------------------------------------- cost

    def observe_execution_cost(self, *, cost: float, steps: int, complete: bool) -> None:
        super().observe_execution_cost(cost=cost, steps=steps, complete=complete)
        ground = self._execution_ground_skill
        if self._dp_forced_reset_pending:
            self._dp_forced_reset_pending = False
            observed = self._dp_costs.get(ground) if ground is not None else None
            if complete and observed:
                observed[-1] += cost
            return
        if ground is None or not complete:
            return
        self._dp_costs.setdefault(ground, []).append(cost)
        self._dp_executed_since_solve[ground] = self._dp_executed_since_solve.get(ground, 0) + 1

    def record_forced_reset(self, *, reason: str, cost: float, consecutive_stranded: int) -> None:
        super().record_forced_reset(
            reason=reason, cost=cost, consecutive_stranded=consecutive_stranded
        )
        # Its cost arrives through the next observe_execution_cost.
        self._dp_forced_reset_pending = True

    def dp_expected_attempt_cost(self, *, ground_skill: GroundSkill) -> float:
        observed = self._dp_costs.get(ground_skill)
        if observed:
            return sum(observed) / len(observed)
        accounting = self._require_accounting()
        prior = self.skill_cost_prior_steps * accounting.robot_charge().cost
        if self.always_ask_reset is not None and AlwaysAskReset.triggers(ground_skill=ground_skill):
            prior += self.start_reset_cost()
        return prior

    # ---------------------------------------------------------------- selection

    def choose_practice_target(self) -> list[GroundSkill]:
        accounting = self._require_accounting()
        remaining = self.dp_budget - accounting.total_cost
        if remaining <= 0.0:
            return [STOP_SKILL]
        if self._dp_solved_period != self._practice_periods:
            self._solve(budget=remaining)
        left = self.dp_remaining_allocation()
        ranked: list[tuple[int, float, GroundSkill]] = []
        for candidate, count in left.items():
            self.record_practice_target(name=candidate.skill.name, field="scored")
            ranked.append((count, float(self._rng.uniform()), candidate))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [candidate for _count, _tiebreak, candidate in ranked]

    def _solve(self, *, budget: float) -> None:
        plans = self.planning_progress_plans()
        humans = self.offered_partial_resets()
        candidates = list(
            dict.fromkeys([
                *self._competence_models,
                *(step for plan in plans for step in plan),
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
            model = self._linear_model(history=history)
            competences[candidate] = model.get_current_competence()
            learning_rates[candidate] = model.learning_rate()
        self._dp_allocation = DpAllocator.allocate(
            plans=plans,
            competences=competences,
            learning_rates=learning_rates,
            costs={
                candidate: self.dp_expected_attempt_cost(ground_skill=candidate)
                for candidate in candidates
            },
            budget=budget,
            max_skills=self.dp_max_skills,
            max_allocations=self.dp_max_allocations,
        )
        self._dp_executed_since_solve = {}
        self._dp_solved_period = self._practice_periods

    def _linear_model(
        self, *, history: OptimisticSkillCompetenceModel | None
    ) -> LinearSkillCompetenceModel:
        return LinearSkillCompetenceModel(
            alpha=self.prior_alpha,
            beta=self.prior_beta,
            initial_delta=self.dp_initial_learning_rate,
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
