import math
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from hitl_pmp.core.method.method import InteractionComplete, Method
from hitl_pmp.core.method.skill_provider import SkillProvider
from hitl_pmp.core.method.types import (
    GroundSkill,
    LabeledAction,
    Policy,
    Rollout,
    SetupCommand,
    Skill,
)
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate, Task
from hitl_pmp.planning.fast_downward import FastDownwardPlanner, PlanningFailure
from hitl_pmp.planning.grounding import SkillGrounder
from hitl_pmp.planning.types import TranslationCache

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
       `reproduce_predicators_practice_target_history` restores predicators'
       all-attempts bookkeeping for exactly these two quantities (competence stays
       random-excluding either way); see that field for why the flag exists.
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

    env: Environment
    skill_provider: SkillProvider
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
    # CFG.skill_competence_model_optimistic_{window,recency}_size. predicators'
    # settings.py default is 5 (Light Switch uses it); the paper's own
    # active_sampler_learning.yaml overrides both to 2 for the simulated Ball-Ring,
    # so a Ball-Ring run passes --competence-window-size 2 --competence-recency-size 2.
    competence_window_size: int = 5
    competence_recency_size: int = 5

    # Kept default FALSE, unlike the other two reproduce_* flags. This restores
    # predicators' own double-`observe()` bug (deviation 3): it counts each random
    # attempt once toward competence, which corrupts a mastered skill's estimate down
    # to ~0.67 and mis-prices the planner's edge costs. Measured NULL on the success
    # curve, so leaving it off does not hurt matching predicators' results, and it
    # keeps competence clean. Pass --reproduce-predicators-double-observe for a
    # bit-exact-faithful (bug-included) run.
    reproduce_predicators_double_observe: bool = False

    # A second, independent ablation switch (see deviation 1 above). predicators
    # computes `skip_perfect`'s success-rate check and the UCB `num_tries`/`total`
    # from `_ground_op_hist`, appended on *every* execution including epsilon-random
    # ones (`active_sampler_explorer.py:400`). This port instead reads the
    # competence model's history, which *excludes* random attempts -- correct for
    # competence (at epsilon=0.5 counting coin flips would make competence measure
    # how often a coin flip works), but that random-excluding history was then
    # reused for the practice-target bookkeeping too, which predicators does not do.
    # Default TRUE (match predicators). ON, the practice-target quantities read an
    # all-attempts history (greedy + random) matching `_ground_op_hist`, while
    # competence keeps reading its own clean random-excluding history unchanged. Two
    # separable decisions the port had accidentally coupled; turn OFF to ablate.
    reproduce_predicators_practice_target_history: bool = True
    # Kept default FALSE, and this is a *coupled* deviation, not an independent one.
    # ON, exploration (epsilon-greedy) fires only on the practice-target skill, greedy
    # for the prefix -- matching predicators (active_sampler_explorer.py fires its
    # exploration sampler only once next_practice_nsrt's preconditions hold). But
    # predicators pairs that with a HORIZON CAP on goal-pursuit (audit D7), which this
    # port lacks: our goal-pursuit is greedy and runs until the goal is achieved or
    # planning fails. Turning this ON alone deadlocks a goal-directed domain like Light
    # Switch -- a bad initial sampler can never achieve the goal greedily, so practice
    # (where the target would be explored) never begins. Our explore-EVERYTHING default
    # is what compensates for the missing horizon cap. So ON only faithfully matches
    # predicators together with a goal-pursuit horizon cap; on its own it is an ablation
    # that HELPS long multi-skill plans (Ball-Ring) but STARVES short goal-directed ones.
    reproduce_predicators_explore_target_only: bool = False

    # predicators' `CFG.horizon`, read by active_sampler_explorer as
    # `assigned_task_horizon`: how many skills it will spend pursuing the assigned
    # train task's goal before giving up and practicing for the rest of the period
    # (explorer lines 191-198 -- decrement per skill, and at <= 0 mark the assigned
    # task finished and force a replan). This port originally had no such cap: its
    # goal phase ran until the goal was achieved or planning failed.
    #
    # This is the OTHER HALF of reproduce_predicators_explore_target_only's coupling
    # (see that field). Target-only exploration is only sensible once goal-pursuit is
    # bounded -- uncapped, a greedy goal phase can eat the whole period, so practice,
    # the only place exploration would then fire, never starts. Measured: scope alone
    # did not close the gap to predicators on Ball-Ring, and deadlocks Light Switch.
    #
    # No single faithful default exists because predicators sets this PER ENVIRONMENT
    # rather than globally (`ball_and_cup_sticky_table: horizon: 8`, `grid_row:
    # grid_row_num_cells + 2`, defaulting to 100). So None keeps this port's original
    # uncapped behavior, and a Ball-Ring run passes --goal-pursuit-horizon 8 to match
    # the paper's own config for that domain.
    goal_pursuit_horizon: int | None = None
    planning_timeout: float = 10.0
    # Two reasons for 10000 over the old default of 1000, neither of which is a score
    # comparison:
    #
    # 1. STRUCTURAL. 1000 sits below `n_iter_no_change = 5000`, so the early-stopping
    #    branch in MlpBinaryClassifier._fit provably never fired -- not rarely, never.
    #    Every refit ran exactly 1000 full-batch steps whether or not the loss had
    #    stopped moving, making a mechanism ported from predicators dead code.
    # 2. IT IS PREDICATORS' OWN DEFAULT. `predicators/settings.py` L572 sets
    #    `sampler_mlp_classifier_max_itr = 10000`. The paper's launch configs override
    #    it to 100000 (`scripts/configs/active_sampler_learning.yaml` L112), which is
    #    why 100000 is the number this codebase used to contrast against -- but the
    #    library default a caller gets without a config is 10000, so matching the
    #    reference argues for exactly this value.
    #
    # The Ball-Ring sweep is consistent with 10000 but does NOT establish it as an
    # optimum, and is deliberately not the justification. Endpoints, 10 seeds/arm
    # (docs/experiment-logs/2026-08-03-ballring-iters.md): 1000 -> 83.0% +- 22.1,
    # 3000 -> 90.0 +- 28.3, 10000 -> 99.0 +- 3.2, 30000 -> 91.0 +- 12.0,
    # 100000 -> 89.0 +- 16.0. The point estimates trace an inverted U, but NO pairwise
    # difference is significant at n=10 (paired, vs 10000: 1000 p=0.057, 30000 p=0.070,
    # 100000 p=0.085, 3000 p=0.350). Resolving any of those pairs needs ~19-23 seeds.
    # Do not cite this sweep as having established an optimum.
    #
    # Running at the paper config's 100000 reproduces predicators' own score
    # (89.0 +- 16.0 against its 91.0 +- 12.0) -- a positive control on the port.
    sampler_max_train_iters: int = 10000

    _rng: np.random.Generator = PrivateAttr()
    _competence_models: dict[GroundSkill, OptimisticSkillCompetenceModel] = PrivateAttr()
    # Per ground skill, one bool per execution regardless of the epsilon-greedy
    # random branch -- predicators' `_ground_op_hist`. Only consulted when
    # `reproduce_predicators_practice_target_history` is on; competence always reads
    # its own random-excluding history in `_competence_models` instead.
    _all_attempt_outcomes: dict[GroundSkill, list[bool]] = PrivateAttr()
    _samplers: dict[str, LearnedSkillSampler] = PrivateAttr()
    # (init_atoms, goal) for each task EES has been handed, newest last.
    _seen_tasks: list[tuple[frozenset[GroundAtom], frozenset[GroundAtom]]] = PrivateAttr()
    _cached_plans: list[list[GroundSkill]] = PrivateAttr()
    _score_calls: int = PrivateAttr()
    # Cumulative over the run. The failure counter is incremented in plan_to's own
    # except-clause, so no catch site can be the one that forgets; the attempt counter
    # is its mandatory denominator. Surfaced as a pair through
    # Method.planning_outcomes into stats.json.
    _planning_failures: int = PrivateAttr()
    _planning_attempts: int = PrivateAttr()
    # Scoped to this Method instance, so it lives exactly as long as one run and is
    # never shared between runs, processes, or tests. See `plan_to`.
    _translation_cache: TranslationCache = PrivateAttr()
    # The practice episode currently in flight, so observe_environment_reset can
    # settle its unobserved skill before the harness discards the state that would
    # have judged it. None until the first practice period, and never an
    # *evaluation* episode: those observe nothing at all.
    _practice_episode: "_EesEpisode | None" = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._competence_models = {}
        self._all_attempt_outcomes = {}
        self._samplers = {}
        self._seen_tasks = []
        self._cached_plans = []
        self._score_calls = 0
        self._planning_failures = 0
        self._planning_attempts = 0
        self._translation_cache = TranslationCache()
        self._practice_episode = None

    # ------------------------------------------------------------------ domain

    def skills(self) -> tuple[Skill, ...]:
        """This domain's lifted skills (e.g. Light Switch's four, including the
        deliberately impossible JumpToLight -- EES is supposed to *discover* it never
        works). Delegated to the injected SkillProvider so EES is domain-agnostic."""
        return self.skill_provider.skills()

    def predicates(self) -> tuple[Predicate, ...]:
        return self.skill_provider.predicates()

    def types(self) -> tuple[Type, ...]:
        return self.skill_provider.types()

    def objects(self) -> tuple[Object, ...]:
        return self.skill_provider.objects()

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
                alpha=self.prior_alpha,
                beta=self.prior_beta,
                window_size=self.competence_window_size,
                recency_size=self.competence_recency_size,
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

        Independently of competence, every execution is recorded (greedy *and*
        random) into `_all_attempt_outcomes` -- predicators' `_ground_op_hist`.
        `skip_perfect`/UCB read it only when
        `reproduce_predicators_practice_target_history` is on.
        """
        self._all_attempt_outcomes.setdefault(ground_skill, []).append(success)
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

    def practice_target_num_tries(self, *, ground_skill: GroundSkill) -> int:
        """The per-skill trial count `skip_perfect`/UCB reason about. Reads the
        all-attempts (`_ground_op_hist`) history when
        `reproduce_predicators_practice_target_history` is on, else the competence
        history (which excludes epsilon-random attempts)."""
        if self.reproduce_predicators_practice_target_history:
            return len(self._all_attempt_outcomes.get(ground_skill, []))
        return self.competence_model(ground_skill=ground_skill).num_observations

    def practice_target_total_tries(self) -> int:
        """The UCB `total` across all skills, matching whichever per-skill history
        `practice_target_num_tries` reads."""
        if self.reproduce_predicators_practice_target_history:
            return sum(len(outcomes) for outcomes in self._all_attempt_outcomes.values())
        return self.total_observations()

    def measured_success_rate(self, *, ground_skill: GroundSkill) -> float:
        """Raw (prior-free) success fraction, which is what predicators' own
        `skip_perfect` check uses -- deliberately not the posterior mean, which
        can never reach exactly 1.0 under a Beta prior. Reads the all-attempts
        (`_ground_op_hist`) history when
        `reproduce_predicators_practice_target_history` is on, else the competence
        history (which excludes epsilon-random attempts)."""
        if self.reproduce_predicators_practice_target_history:
            outcomes: list[bool] = self._all_attempt_outcomes.get(ground_skill, [])
        else:
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
        """Costs change on nearly every call, but `init_atoms`/`goal` repeat heavily
        -- the test set is fixed for the whole run, and practice replans toward the
        same few candidates' preconditions -- so this hands the planner a per-run
        `TranslationCache` to skip re-translating PDDL it has already seen. Costs are
        patched into the SAS *after* translation, so caching that stage cannot change
        the plan; see `TranslationCache`."""
        self._planning_attempts += 1
        try:
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
                translation_cache=self._translation_cache,
            )
        except PlanningFailure:
            # Counted HERE rather than at the three `except PlanningFailure:` sites
            # that catch it, so that no catch site -- present or future -- can be the
            # one that forgets. This is the single place a plan is ever requested,
            # which makes the counted quantity exactly "asked the planner for a plan
            # and got none", with no judgement about which asks matter.
            #
            # That deliberately includes refresh_planning_progress_plans' scoring
            # pass and _practice_plan's per-candidate loop, where a failure is routine
            # and simply drops that task or candidate. Including them keeps the
            # definition one sentence long, and the failure mode this exists to catch
            # inflates every caller at once anyway: when the PDDL itself is malformed,
            # nothing plans. It is also exactly why the attempt count above is
            # mandatory -- against a speculative workload, a bare failure count says
            # nothing.
            self._planning_failures += 1
            raise

    def planning_outcomes(self) -> tuple[int, int]:
        """(failures, attempts), cumulative over the run; method_runner.py differences
        them per window. See Method.planning_outcomes and plan_to for what is counted."""
        return (self._planning_failures, self._planning_attempts)

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
            num_tries = max(self.practice_target_num_tries(ground_skill=ground_skill), 1)
            total_trials = max(self.practice_target_total_tries(), 1)
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
        leading 1.0 bias term is added by `build_sampler_input`."""
        features: list[float] = []
        for obj in ground_skill.objects:
            features.extend(float(value) for value in state[obj])
        return features

    def sampler_input_row(
        self, *, ground_skill: GroundSkill, state: State, params: np.ndarray
    ) -> list[float]:
        """The full classifier input row for one (ground skill, state, params) --
        predicators' `construct_active_sampler_input`, which is domain-aware. Asks the
        domain's `SkillProvider` for an oracle row first (`feature_selection="oracle"`);
        if it declines (returns `None`), falls back to the default `"all"` layout
        `[1.0] + concat(state[obj]) + params`. A pure function of its arguments, so the
        row built to *score* a candidate and the row later *observed* for the chosen
        candidate are identical as long as they are built at the same state -- which is
        why the caller snapshots this at decision time rather than rebuilding it once
        the state has moved on."""
        oracle = self.skill_provider.oracle_sampler_input(
            ground_skill=ground_skill, state=state, params=params
        )
        if oracle is not None:
            return oracle
        return LearnedSkillSampler.build_sampler_input(
            state_features=self.state_features(ground_skill=ground_skill, state=state),
            params=params,
        )

    def observe_sampler_outcome(
        self, *, skill_name: str, param_dim: int, sampler_input: list[float], success: bool
    ) -> None:
        self.sampler(skill_name=skill_name, param_dim=param_dim).observe(
            sampler_input=sampler_input, success=success
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
        self._practice_episode = episode
        return lambda state: episode.step(state=state)

    def observe_environment_reset(self, *, state: State) -> None:
        """Score the in-flight skill against the state the harness is about to
        discard, instead of against the initial state it is about to be reset to.

        Only reachable with practice_reset_interval set (practice_loop.py never
        calls this otherwise), and it is what keeps that knob from confounding
        itself: an outcome is normally read on the *next* step, so a reset in
        between would check add_effects against a freshly reset environment --
        where InBin/RobotInRoom-style effects essentially never hold -- and record
        a spurious failure into both the competence model and the sampler's
        training data, once per reset. That mislabelling would scale exactly with
        reset frequency.

        Flushing here rather than at the period boundary is deliberate: the last
        skill of a period still goes unobserved (this class's deviation 2), so
        every interaction period loses exactly one observation no matter how often
        the harness resets inside it."""
        if self._practice_episode is None:
            return
        self._practice_episode.observe_pending(true_atoms=self.abstract_state(state=state))

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
            candidates = [
                self.skill_provider.sample_params(ground_skill=ground_skill, rng=self._rng)
                for _ in range(self.num_candidates)
            ]
            sampler_inputs = [
                self.sampler_input_row(ground_skill=ground_skill, state=state, params=candidate)
                for candidate in candidates
            ]
            sampler = self.sampler(skill_name=skill.name, param_dim=skill.param_dim)
            choice = sampler.sample(
                sampler_inputs=sampler_inputs, candidates=candidates, explore=explore
            )
            params = choice.params
            record = (
                _SkillAttempt(
                    skill_name=skill.name,
                    param_dim=skill.param_dim,
                    # Snapshot the chosen candidate's row at *this* state -- observing
                    # it later, after the state has changed, would build a different
                    # (desynced) row. Same-state, same-params rebuild is deterministic.
                    sampler_input=self.sampler_input_row(
                        ground_skill=ground_skill, state=state, params=params
                    ),
                    was_random_exploration=choice.was_random,
                    was_informed_choice=choice.was_informed,
                )
                if explore
                else None
            )

        action = self.skill_provider.compute_action(
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
    param_dim: int
    # The classifier input row built at decision time (bias + features + params, or a
    # domain's oracle row). Stored rather than rederived so the observed training row
    # is exactly the row that was scored -- see EesMethod.sampler_input_row.
    sampler_input: list[float]
    was_random_exploration: bool
    # `SamplerChoice.was_informed`: the classifier's scores actually ranked the
    # candidates, so these parameters reflect something it learned. Orthogonal to
    # the flag above, and recorded for a different reader: `was_random_exploration`
    # is what the competence models key on, while this one exists so an analysis can
    # tell a trained classifier's greedy draw from the uniform fallback `sample`
    # takes on a degenerate score vector. Pooling those two is what made the
    # greedy-versus-random split in
    # docs/experiment-logs/2026-08-05-tossingroomsplit-throw-rates.md provisional.
    was_informed_choice: bool


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
        # The last skill of the current practice plan -- the one actually being
        # practiced (the prefix just navigates to its preconditions). Only consulted
        # under reproduce_predicators_explore_target_only, to explore that skill alone.
        self._practice_target: GroundSkill | None = None
        # Remaining goal-pursuit budget (predicators' `assigned_task_horizon`); None
        # means uncapped. Counts down one per skill while the goal phase runs.
        self._goal_pursuit_remaining: int | None = method.goal_pursuit_horizon

    def _noop_action(self) -> np.ndarray:
        """Ask the environment what inaction means, rather than assuming zeros.

        This used to be `np.zeros(env.action_space.shape)`, which is a *real* action
        on two of the six domains here: `pick_shelf` at distance 0.0 on Tossing3D
        (`pick_id == 0`) and "navigate to (0, 0)" on Ball-Ring. There, a method that
        failed to plan quietly acted anyway and the run reported whatever that action
        caused. Light Switch and the Tossing Rooms happened to be unaffected -- see
        `Environment.noop_action` for why that was luck rather than design."""
        return self._method.env.noop_action()

    def step(self, *, state: State) -> LabeledAction:
        method = self._method
        true_atoms = method.abstract_state(state=state)
        self.observe_pending(true_atoms=true_atoms)
        self._tick_goal_pursuit_horizon()

        # Closed-loop execution: if the next queued skill's preconditions no longer
        # hold, the plan has diverged (a stochastic outcome -- e.g. a bare ball
        # placed on a table falling to the floor -- broke a downstream skill's
        # preconditions) and executing it anyway would drive an inapplicable action
        # into the env. Discard the stale plan and replan instead. This mirrors
        # predicators, whose option policy raises OptionExecutionFailure when a step
        # is not initiable and re-plans (active_sampler_explorer.py:340-343) rather
        # than ever feeding simulate() an inapplicable option. The earlier open-loop
        # execution is what tripped Ball-Ring's pick/place obj_type_id asserts.
        if self._plan and not (self._plan[0].preconditions <= true_atoms):
            self._plan = []

        if not self._plan:
            self._plan = self._next_plan(true_atoms=true_atoms)
            # The practice target is the last skill of a practice plan (a goal-pursuit
            # plan has none); the prefix just reaches its preconditions.
            self._practice_target = (
                self._plan[-1]
                if (self._practicing and self._goal_phase_done and self._plan)
                else None
            )
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
            # The environment supplies the action, since only it knows what inaction
            # means in its own action space -- see Environment.noop_action.
            return LabeledAction(action=self._noop_action(), label="no-op (no plan)")

        ground_skill = self._plan.pop(0)
        # By default every skill executed during practice explores (epsilon-greedy).
        # Under reproduce_predicators_explore_target_only, only the practice target
        # does -- the prefix that navigates to it uses the greedy learned sampler,
        # matching predicators (active_sampler_explorer.py fires its exploration
        # sampler only once next_practice_nsrt's preconditions hold). On this domain's
        # long multi-skill plans, exploring every step spends ~half of all actions on
        # off-target random params; this flag measures that cost.
        explore = self._practicing and (
            not method.reproduce_predicators_explore_target_only
            or ground_skill is self._practice_target
        )
        labeled, record = method.execute_ground_skill(
            ground_skill=ground_skill, state=state, explore=explore
        )
        self._pending = ground_skill
        self._pending_sampler_record = record
        return labeled

    def _tick_goal_pursuit_horizon(self) -> None:
        """Spend one step of the goal-pursuit budget, and end the goal phase once it
        runs out -- predicators' `assigned_task_horizon` countdown
        (active_sampler_explorer.py:191-198). Only meaningful while practicing: an
        evaluation episode has no practice phase to fall back to, and its termination
        is run_task_episode's job. Exhausting the budget also drops any in-flight
        goal plan, matching predicators clearing `current_policy` so it replans -- the
        queued skills were chosen to reach a goal we've just stopped pursuing."""
        if not self._practicing or self._goal_phase_done:
            return
        if self._goal_pursuit_remaining is None:
            return
        if self._goal_pursuit_remaining <= 0:
            self._goal_phase_done = True
            self._plan = []
        else:
            self._goal_pursuit_remaining -= 1

    def observe_pending(self, *, true_atoms: frozenset[GroundAtom]) -> None:
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
                    param_dim=attempt.param_dim,
                    sampler_input=attempt.sampler_input,
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
