import math
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from hitl_pmp.core.method.method import InteractionComplete, Method
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Policy, Rollout, SetupCommand
from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate, Task
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.predicates import (
    ADJACENT,
    LIGHT_IN_CELL,
    LIGHT_OFF,
    LIGHT_ON,
    ROBOT_IN_CELL,
)
from hitl_pmp.environments.lightswitch.skills import LightSwitchSkills
from hitl_pmp.planning.fast_downward import FastDownwardPlanner, PlanningFailure
from hitl_pmp.planning.grounding import SkillGrounder

from .competence_models import OptimisticSkillCompetenceModel
from .wrapped_sampler import LearnedSkillSampler


class EesMethod(Method):
    """EES (Estimate / Extrapolate / Situate) -- the "Practice Makes Perfect"
    paper's own method, ported from predicators' `active_sampler_learning`
    approach + `active_sampler` explorer with
    `active_sampler_explore_task_strategy=planning_progress` (the combination the
    paper's own `scripts/configs/active_sampler_learning.yaml` runs).

    The three named steps, and where each lives here:

    - **Estimate** -- `competence_model()` keeps one
      `OptimisticSkillCompetenceModel` per *ground* skill ever executed (matching
      predicators' `_ground_op_hist` keying), updated by `observe_outcome()` with
      whether the skill's own `add_effects` actually held afterward.
    - **Extrapolate** -- `score_ground_skill()` asks that model
      `predict_competence(num_additional_data=competence_lookahead)`: "how
      competent would this skill be after a bit more practice?"
    - **Situate** -- the extrapolated competence is substituted into the cost
      dict and the *seen tasks'* plans are re-priced. The skill whose
      hypothetical improvement most reduces the total cost of the plans the robot
      actually needs is the one worth practicing, and EES then plans to that
      skill's preconditions in order to practice it where it's actually
      executable.

    The load-bearing identity is that plan cost is `sum(-log(competence))`, so
    minimizing it maximizes `prod(competence)` -- the paper's `J_task`, the
    probability a plan executes without replanning. That is exactly why the
    planner has to be cost-aware and optimal (`seq-opt-lmcut` via real Fast
    Downward, see `planning/fast_downward.py`), and why predicators' own built-in
    A* planner is not a substitute: it ignores per-operator costs entirely.

    Deviations from predicators, all deliberate:
    1. `skip_perfect` and the UCB `num_tries` are computed from competence
       observations, which exclude epsilon-greedy random attempts. predicators
       reads `_ground_op_hist`, appended on *every* execution including random
       ones (`active_sampler_explorer.py:400`), so a skill here reaches a
       measured rate of 1.0 sooner and is dropped as a practice target earlier.
    2. The outcome of the *last* skill in an interaction period is never observed
       (there is no subsequent state to check `add_effects` against). predicators
       observes at option termination instead. This loses at most one datapoint
       per period.
    3. predicators double-counts one `observe()` call per non-exploratory attempt
       (`active_sampler_learning_approach.py` calls it at both line 407 and 443);
       that is a bug, and is not reproduced here. The *suppression* those same
       lines implement -- no competence update when the epsilon-greedy random
       branch fired -- IS reproduced; see `_SkillAttempt`.
    4. Candidate practice targets are scored against cached plans that are
       refreshed only every `replan_frequency` scoring calls -- predicators'
       own optimization (`active_sampler_explorer_replan_frequency`), and the
       reason scoring is cheap enough to do per candidate per step.
    """

    env: LightSwitchEnvironment
    seed: int = 0

    # --- EES hyperparameters, defaulted to predicators'/the paper's own values ---
    # CFG.skill_competence_model_lookahead
    competence_lookahead: int = 1
    # CFG.active_sampler_explore_bonus / _use_ucb_bonus / _skip_perfect
    explore_bonus: float = 1e-1
    use_ucb_bonus: bool = True
    skip_perfect: bool = True
    # CFG.active_sampler_explorer_planning_progress_max_tasks. The paper text says
    # "the 10 most recently seen tasks"; the reference code instead takes
    # `sorted(seen_idxs)[:10]` ("Don't randomize: would lead to noisy estimates").
    # This follows the text. On this domain the two coincide in effect, since every
    # Light Switch task differs only in the light's target value.
    planning_progress_max_tasks: int = 10
    # CFG.active_sampler_explorer_replan_frequency -- the paper: "cache last plan
    # per task, re-run planner once per 100 calls".
    replan_frequency: int = 100
    # CFG.active_sampler_learning_exploration_epsilon -- the paper: "epsilon-greedy
    # with epsilon = 0.5".
    exploration_epsilon: float = 0.5
    # CFG.active_sampler_learning_num_samples
    num_candidates: int = 100
    # Beta(10, 1), the paper's stated initial-cycle prior.
    prior_alpha: float = 10.0
    prior_beta: float = 1.0

    # Not a hyperparameter -- an ablation switch that restores predicators' own
    # double-`observe()` bug (see deviation 3 above), so its cost can be measured
    # rather than argued about. The paper's published curve contains the bug, so
    # this is the setting that makes our numbers directly comparable to it.
    reproduce_predicators_double_observe: bool = False
    planning_timeout: float = 10.0
    sampler_max_train_iters: int = 1000

    _rng: np.random.Generator = PrivateAttr()
    _competence_models: dict[GroundSkill, OptimisticSkillCompetenceModel] = PrivateAttr()
    _samplers: dict[str, LearnedSkillSampler] = PrivateAttr()
    # (init_atoms, goal) for each task EES has been handed, newest last.
    _seen_tasks: list[tuple[frozenset[GroundAtom], frozenset[GroundAtom]]] = PrivateAttr()
    _cached_plans: list[list[GroundSkill]] = PrivateAttr()
    _score_calls: int = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._competence_models = {}
        self._samplers = {}
        self._seen_tasks = []
        self._cached_plans = []
        self._score_calls = 0

    # ------------------------------------------------------------------ domain

    def skills(self) -> tuple:  # type: ignore[type-arg]
        """This domain's four lifted skills, including the deliberately impossible
        JumpToLight -- EES is supposed to *discover* that it never works."""
        return (
            LightSwitchSkills.MOVE_ROBOT,
            LightSwitchSkills.TURN_ON_LIGHT,
            LightSwitchSkills.TURN_OFF_LIGHT,
            LightSwitchSkills.JUMP_TO_LIGHT,
        )

    def predicates(self) -> tuple[Predicate, ...]:
        return (LIGHT_ON, LIGHT_OFF, ROBOT_IN_CELL, LIGHT_IN_CELL, ADJACENT)

    def types(self) -> tuple[Type, ...]:
        return (
            LightSwitchEnvironment.robot_type,
            LightSwitchEnvironment.light_type,
            LightSwitchEnvironment.cell_type,
        )

    def objects(self) -> tuple[Object, ...]:
        return (self.env.robot, self.env.light, *self.env.get_cells())

    def abstract_state(self, *, state: State) -> frozenset[GroundAtom]:
        return SkillGrounder.abstract_state(
            state=state, objects=self.objects(), predicates=self.predicates()
        )

    # ------------------------------------------------------- estimate (competence)

    def competence_model(self, *, ground_skill: GroundSkill) -> OptimisticSkillCompetenceModel:
        """Lazily created per ground skill, so the candidate set is exactly "every
        ground skill ever executed" -- predicators keys `_ground_op_hist` the same
        way, which matters because the number of *possible* groundings is
        quadratic in grid_size while the number ever tried stays small."""
        if ground_skill not in self._competence_models:
            self._competence_models[ground_skill] = OptimisticSkillCompetenceModel(
                alpha=self.prior_alpha, beta=self.prior_beta
            )
        return self._competence_models[ground_skill]

    def observe_outcome(
        self, *, ground_skill: GroundSkill, success: bool, was_random_exploration: bool = False
    ) -> None:
        """Records one practice outcome against a skill's competence model.

        An epsilon-greedy *random* attempt is not recorded: at the paper's
        epsilon = 0.5 half of all attempts are coin flips by construction, so
        counting them would make competence measure how often a coin flip works
        rather than how good the skill is when the robot actually tries. The
        sampler's own training data keeps those attempts regardless -- a
        deliberately random parameter that failed is exactly the negative example
        the classifier needs.

        `reproduce_predicators_double_observe` restores predicators' literal
        control flow instead; see that field for why the flag exists.
        """
        model = self.competence_model(ground_skill=ground_skill)
        if self.reproduce_predicators_double_observe:
            model.observe(success=success)  # active_sampler_explorer.py:407
            if not was_random_exploration:  # :442-443
                model.observe(success=success)
            return
        if not was_random_exploration:
            model.observe(success=success)

    def total_observations(self) -> int:
        return sum(model.num_observations for model in self._competence_models.values())

    def measured_success_rate(self, *, ground_skill: GroundSkill) -> float:
        """Raw (prior-free) success fraction, which is what predicators' own
        `skip_perfect` check uses -- deliberately not the posterior mean, which
        can never reach exactly 1.0 under a Beta prior."""
        model = self.competence_model(ground_skill=ground_skill)
        outcomes = [outcome for cycle in model.cycle_observations for outcome in cycle]
        if not outcomes:
            return 0.0
        return sum(outcomes) / len(outcomes)

    # ------------------------------------------------------------------- costs

    def default_cost(self) -> float:
        """`-log` of the Beta(alpha, beta) prior mean: the cost assigned to a
        ground skill that has never been executed, so it is neither optimistically
        free nor pessimistically impossible."""
        return -math.log(self.prior_alpha / (self.prior_alpha + self.prior_beta))

    def skill_costs(self) -> dict[GroundSkill, float]:
        """`-log(competence)` per ground skill: summing these over a plan and
        minimizing is exactly maximizing the product of competences."""
        return {
            ground_skill: -math.log(max(model.get_current_competence(), 1e-12))
            for ground_skill, model in self._competence_models.items()
        }

    # ---------------------------------------------------------------- planning

    def plan_to(
        self,
        *,
        init_atoms: frozenset[GroundAtom],
        goal: frozenset[GroundAtom],
        costs: dict[GroundSkill, float],
    ) -> list[GroundSkill]:
        return FastDownwardPlanner.plan(
            skills=self.skills(),
            predicates=self.predicates(),
            types=self.types(),
            objects=self.objects(),
            init_atoms=init_atoms,
            goal=goal,
            ground_skill_costs=costs,
            default_cost=self.default_cost(),
            timeout=self.planning_timeout,
        )

    def record_seen_task(
        self, *, init_atoms: frozenset[GroundAtom], goal: frozenset[GroundAtom]
    ) -> None:
        """The empirical task distribution EES situates against -- the paper's
        `J_tasks` expectation is taken over the tasks actually seen, not over some
        assumed prior."""
        self._seen_tasks.append((init_atoms, goal))

    def planning_progress_plans(self) -> list[list[GroundSkill]]:
        """Cached plans for the most recent seen tasks, refreshed only every
        `replan_frequency` calls (predicators' own optimization). Without the
        cache, scoring every candidate against every seen task would mean a fresh
        Fast Downward invocation per candidate per task per step."""
        if self._score_calls % self.replan_frequency == 0:
            self.refresh_planning_progress_plans()
        self._score_calls += 1
        return self._cached_plans

    def refresh_planning_progress_plans(self) -> None:
        costs = self.skill_costs()
        plans: list[list[GroundSkill]] = []
        for init_atoms, goal in self._seen_tasks[-self.planning_progress_max_tasks :]:
            try:
                plans.append(self.plan_to(init_atoms=init_atoms, goal=goal, costs=costs))
            except PlanningFailure:
                continue
        self._cached_plans = plans

    # ------------------------------------------------- extrapolate + situate (score)

    def score_ground_skill(self, *, ground_skill: GroundSkill) -> float:
        """Planning progress: how much cheaper do the seen tasks' plans get if
        *this* skill improves by one cycle's worth of practice? Ported from
        predicators' `_score_ground_op_planning_progress`."""
        if self.skip_perfect and self.measured_success_rate(ground_skill=ground_skill) == 1.0:
            return -math.inf
        model = self.competence_model(ground_skill=ground_skill)
        extrapolated = model.predict_competence(num_additional_data=self.competence_lookahead)
        costs = self.skill_costs()
        costs[ground_skill] = -math.log(max(extrapolated, 1e-12))

        plans = self.planning_progress_plans()
        if not plans:
            # Nothing seen yet to situate against: fall back to pure exploration
            # value, so the UCB bonus alone breaks ties.
            score = 0.0
        else:
            total = sum(
                sum(costs.get(step, self.default_cost()) for step in plan) for plan in plans
            )
            score = -total / len(plans)

        if self.use_ucb_bonus:
            # predicators' exact form: c * sqrt(log(total_trials) / num_tries). The
            # max(..., 1) guards only divide-by-zero / log(0) for a skill or a run
            # with no attempts yet; with one attempt log(1) = 0, i.e. no bonus,
            # which is predicators' behavior too.
            num_tries = max(model.num_observations, 1)
            total_trials = max(self.total_observations(), 1)
            score += self.explore_bonus * math.sqrt(math.log(total_trials) / num_tries)
        return score

    def choose_practice_target(self) -> list[GroundSkill]:
        """Candidates in descending score order -- the explorer tries them in turn
        until one's preconditions are actually reachable. Skills that scored
        `-inf` (already perfect, per `skip_perfect`) are dropped entirely."""
        scored: list[tuple[float, float, GroundSkill]] = []
        # list(...) because scoring can lazily create competence models, which
        # would otherwise mutate the dict mid-iteration.
        for candidate in list(self._competence_models):
            score = self.score_ground_skill(ground_skill=candidate)
            if score == -math.inf:
                continue
            # Ties broken randomly, matching predicators' own rng.uniform tiebreak.
            scored.append((score, float(self._rng.uniform()), candidate))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [candidate for _score, _tiebreak, candidate in scored]

    def random_choice(self, *, ground_skills: list[GroundSkill]) -> GroundSkill:
        """Uniform pick from this Method's own RNG stream, so a seeded EesMethod
        is fully reproducible."""
        return ground_skills[int(self._rng.integers(len(ground_skills)))]

    # ---------------------------------------------------------------- sampling

    def sampler(self, *, skill_name: str, param_dim: int) -> LearnedSkillSampler:
        if skill_name not in self._samplers:
            self._samplers[skill_name] = LearnedSkillSampler(
                skill_name=skill_name,
                param_dim=param_dim,
                num_candidates=self.num_candidates,
                exploration_epsilon=self.exploration_epsilon,
                seed=self.seed,
                max_train_iters=self.sampler_max_train_iters,
            )
        return self._samplers[skill_name]

    def state_features(self, *, ground_skill: GroundSkill, state: State) -> list[float]:
        """`concat(state[obj] for obj in ground_skill.objects)` -- predicators'
        `construct_active_sampler_input` under `feature_selection="all"`. The
        leading 1.0 bias term is added by the sampler itself."""
        features: list[float] = []
        for obj in ground_skill.objects:
            features.extend(float(value) for value in state[obj])
        return features

    def observe_sampler_outcome(
        self, *, skill_name: str, features: list[float], params: np.ndarray, success: bool
    ) -> None:
        param_dim = len(params)
        self.sampler(skill_name=skill_name, param_dim=param_dim).observe(
            features=features, params=params, success=success
        )

    # ---------------------------------------------------------------- lifecycle

    def advance_competence_cycle(self) -> None:
        for model in self._competence_models.values():
            model.advance_cycle()

    def fit_samplers(self) -> None:
        for sampler in self._samplers.values():
            sampler.fit()

    def end_cycle(self) -> None:
        """Retrain, then start a new competence cycle -- predicators' per-cycle
        order (`_update_sampler_data` -> `_learn_wrapped_samplers` ->
        `advance_cycle`). Called by practice_loop.py before the evaluation sweep
        that measures this cycle."""
        self.fit_samplers()
        self.advance_competence_cycle()
        # Competence has changed, so every cached plan's price is stale.
        self.refresh_planning_progress_plans()

    # ------------------------------------------------------------------ policy

    def get_task_policy(self, *, task: Task) -> Policy:
        """Evaluation: plan to the goal with current competences and execute
        greedily. Records nothing -- these are held-out test tasks."""
        episode = _EesEpisode(method=self, goal=task.goal.atoms, practicing=False)
        return lambda state: episode.step(state=state)

    def get_practice_policy(self, *, task: Task) -> Policy:
        """Practice: pursue the assigned goal first (predicators'
        `pursue_task_goal_first`), then spend the rest of the period practicing
        whichever skill scores best."""
        init_atoms = self.abstract_state(state=task.initial_state)
        self.record_seen_task(init_atoms=init_atoms, goal=task.goal.atoms)
        episode = _EesEpisode(method=self, goal=task.goal.atoms, practicing=True)
        return lambda state: episode.step(state=state)

    def execute_ground_skill(
        self, *, ground_skill: GroundSkill, state: State, explore: bool
    ) -> tuple[LabeledAction, "_SkillAttempt | None"]:
        """Returns the action plus, when this skill has continuous parameters and
        we're practicing, the record to label with the outcome once it's
        observed."""
        skill = ground_skill.skill
        if skill.param_dim == 0:
            params: np.ndarray = np.zeros(0)
            record = None
        else:
            features = self.state_features(ground_skill=ground_skill, state=state)
            candidates = [
                LightSwitchSkills.sample_params(ground_skill=ground_skill, rng=self._rng)
                for _ in range(self.num_candidates)
            ]
            sampler = self.sampler(skill_name=skill.name, param_dim=skill.param_dim)
            params, was_random = sampler.sample(
                features=features, candidates=candidates, explore=explore
            )
            record = (
                _SkillAttempt(
                    skill_name=skill.name,
                    features=features,
                    params=params,
                    was_random_exploration=was_random,
                )
                if explore
                else None
            )

        action = LightSwitchSkills.compute_action(
            ground_skill=ground_skill, params=params, state=state
        )
        objects_desc = ", ".join(obj.name for obj in ground_skill.objects)
        label = f"{skill.name}({objects_desc})"
        if params.size > 0:
            label += f", params={[round(float(p), 2) for p in params]}"
        return LabeledAction(action=action, label=label), record

    # ------------------------------------------- unreachable Method surface area

    def reset_environment(self, *, start_state: State) -> bool:
        """No irreversible actions exist in Light Switch and the base PMP paper has
        no human-in-the-loop layer at all (matches SkillOracleMethod's reasoning)."""
        self.env.set_state(state=start_state)
        return True

    def generate_train_task(self, *, tbd_inputs: Any) -> Task:
        raise NotImplementedError(
            "EesMethod.generate_train_task is unreachable: PracticeLoop hands it "
            "sampled train tasks rather than asking it to invent them."
        )

    def execute_setup_command(self, *, setup_command: SetupCommand) -> None:
        raise NotImplementedError(
            "EesMethod.execute_setup_command is unreachable: no HumanOracle is "
            "ever used in this reproduction."
        )

    def execute_skill(self, *, skill: GroundSkill) -> Rollout:
        raise NotImplementedError(
            "EesMethod.execute_skill is unreachable: skill execution happens "
            "inside the policy returned by get_practice_policy."
        )

    def improve_skill_parameters(self, *, skill: GroundSkill, rollout: Rollout) -> None:
        raise NotImplementedError(
            "EesMethod.improve_skill_parameters is unreachable: sampler retraining "
            "is per-cycle (end_cycle), not per-execution."
        )


class _SkillAttempt(BaseModel):
    """One executed skill whose outcome isn't known yet: what the sampler was
    asked, what it chose, and whether that choice came from the epsilon-greedy
    *random* branch rather than the learned argmax.

    That last flag matters: predicators suppresses the *competence* update for
    randomly-explored attempts (`active_sampler_learning_approach.py` lines
    442-443) while still keeping them as sampler training data. Without it,
    competence measures "how often does a coin flip work" rather than "how good
    is this skill when the robot actually tries" -- at the paper's epsilon=0.5
    that roughly halves the apparent competence of a skill the robot has in fact
    mastered, which then corrupts both the plan costs and the practice-selection
    scores computed from it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    skill_name: str
    features: list[float]
    params: np.ndarray
    was_random_exploration: bool


class _EesEpisode:
    """Per-episode mutable scratch for one EES rollout: the remaining plan, and the
    skill whose outcome hasn't been observed yet.

    A plain class rather than a pydantic model: this is short-lived per-episode
    bookkeeping owned by exactly one policy closure, not configuration or
    persisted state, and it holds a back-reference to its EesMethod (which
    pydantic would try to validate/copy)."""

    def __init__(self, *, method: EesMethod, goal: frozenset[GroundAtom], practicing: bool) -> None:
        self._method = method
        self._goal = goal
        self._practicing = practicing
        self._plan: list[GroundSkill] = []
        self._pending: GroundSkill | None = None
        self._pending_sampler_record: _SkillAttempt | None = None
        self._goal_phase_done = False

    def step(self, *, state: State) -> LabeledAction:
        method = self._method
        true_atoms = method.abstract_state(state=state)
        self._observe_pending(true_atoms=true_atoms)

        if not self._plan:
            self._plan = self._next_plan(true_atoms=true_atoms)
        if not self._plan:
            if self._practicing:
                # Nothing left worth practicing (no candidate reachable and no
                # applicable skill to bootstrap from). Ending the period here
                # rather than burning the remaining budget on no-ops is what keeps
                # the online-transition count data-driven -- see
                # InteractionComplete.
                raise InteractionComplete
            # Evaluation: run_task_episode owns termination (goal check + horizon),
            # so degrade to a no-op rather than ending its episode from in here.
            return LabeledAction(action=np.zeros(2), label="no-op (no plan)")

        ground_skill = self._plan.pop(0)
        labeled, record = method.execute_ground_skill(
            ground_skill=ground_skill, state=state, explore=self._practicing
        )
        self._pending = ground_skill
        self._pending_sampler_record = record
        return labeled

    def _observe_pending(self, *, true_atoms: frozenset[GroundAtom]) -> None:
        if self._pending is None:
            return
        if self._practicing:
            success = self._pending.add_effects <= true_atoms
            attempt = self._pending_sampler_record
            # observe_outcome() owns what an epsilon-greedy random attempt does to
            # competence; sampler data below is kept either way.
            self._method.observe_outcome(
                ground_skill=self._pending,
                success=success,
                was_random_exploration=attempt is not None and attempt.was_random_exploration,
            )
            if attempt is not None:
                self._method.observe_sampler_outcome(
                    skill_name=attempt.skill_name,
                    features=attempt.features,
                    params=attempt.params,
                    success=success,
                )
        self._pending = None
        self._pending_sampler_record = None

    def _next_plan(self, *, true_atoms: frozenset[GroundAtom]) -> list[GroundSkill]:
        method = self._method
        if not self._goal_phase_done:
            if self._goal <= true_atoms:
                self._goal_phase_done = True
            else:
                try:
                    plan = method.plan_to(
                        init_atoms=true_atoms, goal=self._goal, costs=method.skill_costs()
                    )
                except PlanningFailure:
                    plan = []
                if plan:
                    return plan
                self._goal_phase_done = True
        if not self._practicing:
            return []
        return self._practice_plan(true_atoms=true_atoms)

    def _practice_plan(self, *, true_atoms: frozenset[GroundAtom]) -> list[GroundSkill]:
        """Situate: plan to the preconditions of the best-scoring candidate, then
        execute that candidate there. Falls back to a uniformly random applicable
        skill while no candidate has been tried yet -- that bootstrap is what fills
        the candidate set in the first place."""
        method = self._method
        for candidate in method.choose_practice_target():
            if candidate.preconditions <= true_atoms:
                return [candidate]
            try:
                prefix = method.plan_to(
                    init_atoms=true_atoms,
                    goal=candidate.preconditions,
                    costs=method.skill_costs(),
                )
            except PlanningFailure:
                continue
            return [*prefix, candidate]

        applicable = SkillGrounder.applicable_ground_skills(
            skills=method.skills(), objects=method.objects(), true_atoms=true_atoms
        )
        if not applicable:
            return []
        return [method.random_choice(ground_skills=applicable)]
