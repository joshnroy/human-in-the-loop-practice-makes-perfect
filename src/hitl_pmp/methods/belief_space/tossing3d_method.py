"""Tossing3D method backed by situated belief-space expectimax."""

import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, PrivateAttr

from hitl_pmp.agentic_runtime.types import PlannerMemory
from hitl_pmp.core.log_timing import LogTiming
from hitl_pmp.core.method.types import (
    GroundSkill,
    ParameterSamplingDiagnostics,
    Policy,
    SamplerConsultation,
    Skill,
)
from hitl_pmp.core.practice_costs import PracticeAccounting
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import GroundAtom, Task
from hitl_pmp.methods.practice_makes_perfect.ees_method import (
    STOP_SKILL,
    EesMethod,
)
from hitl_pmp.planning.grounding import SkillGrounder

from .competence_inference import BayesianSkillBelief, InferenceConfig, smooth_history
from .determinized import DeterminizedAStarPlanner
from .expectimax import ExpectimaxPlanner
from .failure_effect_model import EmpiricalFailureEffects
from .llm_planner import LlmPracticePlanner, make_llm_practice_planner
from .planner import BeliefSpacePlanner
from .tossing3d_constants import (
    LEARNING_RATE_PROCESS_NOISE_STD,
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    PICK_SKILLS,
    PRACTICE_BUDGET,
    RESET_SKILLS,
    TOSS_SKILL,
)
from .tossing3d_model import Tossing3DPracticeModel
from .tossing3d_observation_model import (
    make_default_tossing3d_belief,
    mean_competence,
    mean_cost,
    mean_learning_rate,
    refit_belief_state,
)
from .tossing3d_transition_model import make_tossing3d_search_state
from .types.belief_state import Tossing3DBeliefState
from .types.competence_evidence import CompetenceEvidence
from .types.particle_filter_belief import ParticleFilterBelief
from .types.practice_situation import PracticeHistory, PracticeSituation
from .types.protocol import BeliefSpaceModel
from .types.search_state import Tossing3DSearchState
from .types.search_trace import SearchTrace
from .types.skill_belief import COST_MAX
from .types.stop_action import STOP_ACTION, StopAction
from .types.theta import Tossing3DTheta


def make_belief_space_planner(
    *,
    planner: (
        BeliefSpacePlanner[Tossing3DSearchState, Tossing3DBeliefState, Tossing3DTheta, GroundSkill]
        | None
    ),
    solver: Literal["expectimax", "determinized_astar", "llm"],
    max_iterations: int,
    seed: int,
    observation_probability_weight: float,
    log_full_search_tree: bool = False,
    llm_planner: Callable[[], LlmPracticePlanner] | None = None,
) -> BeliefSpacePlanner[Tossing3DSearchState, Tossing3DBeliefState, Tossing3DTheta, GroundSkill]:
    """Return the injected planner or construct the configured planner once.

    An unknown name is an error: this used to fall through to determinized A*, so a
    misspelt solver silently ran a different algorithm."""
    if planner is not None:
        return planner
    if solver == "expectimax":
        return ExpectimaxPlanner(
            use_model_j=True,
            observation_probability_weight=observation_probability_weight,
        )
    if solver == "determinized_astar":
        return DeterminizedAStarPlanner(
            max_iterations=max_iterations,
            seed=seed,
            observation_probability_weight=observation_probability_weight,
            log_full_search_tree=log_full_search_tree,
        )
    if solver == "llm":
        assert llm_planner is not None, "the llm solver needs its planner factory"
        return llm_planner()
    raise ValueError(f"Unknown belief-space solver {solver!r}")


class Tossing3DPomdpMethod(EesMethod):
    """EES learner/executor with situated belief-space practice decisions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pomdp_search_depth: int = Field(default=3, ge=0)
    pomdp_log_full_search_tree: bool = False
    pomdp_solver: Literal["expectimax", "determinized_astar", "llm"] = "expectimax"
    # The `llm` solver: a Claude call per decision through the Docker CLI client. The
    # sandbox is the client's `SandboxSettings` JSON; the output directory holds the
    # call records and the `model_budget.json` ledger, which practice alone charges.
    planner_sandbox: Path | None = None
    planner_output: Path | None = None
    planner_model_budget_usd: float = Field(default=20.0, gt=0.0, allow_inf_nan=False)
    planner_memory: PlannerMemory = PlannerMemory.stateless
    planner_model: str = "claude-opus-5-5"
    planner_effort: str = "low"
    planner_history_window: int = Field(default=20, ge=1)
    # Injected in place of the real client (tests); anything with `.decide`.
    planner_client: Any | None = Field(default=None, exclude=True, repr=False)
    pomdp_max_search_iterations: int = Field(default=100, ge=1)
    pomdp_observation_probability_weight: float = Field(default=0.1, ge=0.0, allow_inf_nan=False)
    pomdp_planner: (
        BeliefSpacePlanner[Tossing3DSearchState, Tossing3DBeliefState, Tossing3DTheta, GroundSkill]
        | None
    ) = Field(default=None, exclude=True, repr=False)
    pomdp_num_samples: int = Field(default=100, ge=1)
    pomdp_num_particles: int = Field(default=20000, ge=1)
    pomdp_competence_model: Literal["global_curve", "local_trend"] = "local_trend"
    pomdp_inference_engine: Literal["particle", "grid"] = "particle"
    pomdp_competence_evidence: CompetenceEvidence = CompetenceEvidence.NON_EPSILON
    pomdp_grid_competence_bins: int = Field(default=25, ge=3)
    pomdp_grid_learning_rate_bins: int = Field(default=16, ge=2)
    pomdp_competence_process_noise_std: float = Field(default=0.03, ge=0.0)
    pomdp_learning_rate_decay: float = Field(default=0.9, ge=0.0, le=1.0)
    pomdp_learning_rate_max: float = Field(default=0.15, gt=0.0, le=1.0)
    pomdp_learning_rate_process_noise_std: float = Field(
        default=LEARNING_RATE_PROCESS_NOISE_STD, ge=0.0
    )
    pomdp_learning_time_scale: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    pomdp_linear_cost_lambda: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    robot_duration_prior_steps: float = Field(default=1, gt=0, allow_inf_nan=False)
    goal_pursuit_horizon: int | None = 0
    decision_log: Path | None = None

    _pomdp_state: Tossing3DBeliefState = PrivateAttr()
    _pomdp_model: Tossing3DPracticeModel = PrivateAttr()
    _decision_index: int = PrivateAttr(default=0)
    _cycle_index: int = PrivateAttr(default=0)
    _practice_values: dict[str, float] = PrivateAttr(default_factory=dict)
    _belief_history: dict[str, list[BayesianSkillBelief]] = PrivateAttr(default_factory=dict)
    _execution_durations: dict[GroundSkill, list[int]] = PrivateAttr(default_factory=dict)
    _pending_reset: GroundSkill | None = PrivateAttr(default=None)
    _remaining_practice_actions: int | None = PrivateAttr(default=None)
    _pending_consultation: tuple[str, SamplerConsultation] | None = PrivateAttr(default=None)
    # The compact session log an LLM planner is shown, and how much of it a resumed
    # Claude session has already seen. Cost of the attempt whose outcome is pending.
    _llm_history: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _llm_history_sent: int = PrivateAttr(default=0)
    _llm_calls_this_session: int = PrivateAttr(default=0)
    _last_attempt_cost: dict[GroundSkill, float] = PrivateAttr(default_factory=dict)

    def configure_practice_accounting(self, *, accounting: PracticeAccounting) -> None:
        super().configure_practice_accounting(accounting=accounting)
        self.pomdp_linear_cost_lambda = accounting.costs.objective_lambda
        self._pomdp_model = self._pomdp_model.model_copy(
            update={"linear_cost_lambda": self.pomdp_linear_cost_lambda}
        )
        self._refresh_execution_forecasts()

    def _refresh_execution_forecasts(self) -> None:
        accounting = self._practice_accounting
        if accounting is None:
            return
        price = accounting.robot_charge().cost
        humans = self.skill_provider.human_cube_bin_reset_skills()
        estimates = {}
        for ground in self._pomdp_model.ground_skills:
            durations = self._execution_durations.get(ground, [])
            duration = (
                sum(durations) / len(durations) if durations else self.robot_duration_prior_steps
            )
            estimates[str(ground)] = (
                self.practice_action_cost(ground_skill=ground)
                if ground in humans
                else (
                    sum(self._measured_execution_costs[ground])
                    / len(self._measured_execution_costs[ground])
                    if self._measured_execution_costs.get(ground)
                    else duration * price
                )
            )
        self._pomdp_state = self._pomdp_state.model_copy(
            update={"expected_execution_costs": estimates}
        )

    def observe_execution_cost(self, *, cost: float, steps: int, complete: bool) -> None:
        super().observe_execution_cost(cost=cost, steps=steps, complete=complete)
        if self._practice_accounting is None:
            return
        ground = self._execution_ground_skill
        assert ground is not None, "execution without dispatch"
        self._last_attempt_cost[ground] = cost
        if complete and ground not in self.skill_provider.human_cube_bin_reset_skills():
            self._execution_durations.setdefault(ground, []).append(steps)
        self._pomdp_state = self._pomdp_state.model_copy(
            update={"accumulated_cost": self._pomdp_state.accumulated_cost + cost}
        )
        self._refresh_execution_forecasts()
        self.record_diagnostic(
            event="execution_cost",
            skill=str(ground),
            cost=cost,
            steps=steps,
            complete=complete,
            summed_cost=self._pomdp_state.accumulated_cost,
        )

    def practice_action_values(self) -> dict[str, float]:
        """Values from the last real decision, never an extra search for rendering."""
        return dict(self._practice_values)

    def practice_skill_competences(self) -> dict[str, float]:
        return {
            skill_name + " (belief mean)": mean_competence(belief=belief)
            for skill_name, belief in self._pomdp_state.skill_beliefs.items()
        }

    def practice_skill_learning_rates(self) -> dict[str, float]:
        return {
            skill_name + " (belief mean)": mean_learning_rate(belief=belief)
            for skill_name, belief in self._pomdp_state.skill_beliefs.items()
        }

    def practice_skill_costs(self) -> dict[str, float]:
        if self._practice_accounting is not None:
            return {
                name + " (execution forecast)": cost
                for name, cost in self._pomdp_state.expected_execution_costs.items()
            }
        return {
            skill_name + " (belief mean)": mean_cost(belief=belief)
            for skill_name, belief in self._pomdp_state.skill_beliefs.items()
            if isinstance(belief, (ParticleFilterBelief, BayesianSkillBelief))
        }

    def practice_skill_improvement_potentials(self) -> dict[str, float]:
        """Expected advantage of each applicable skill over stopping now."""
        stop_value = self._practice_values.get("STOP")
        if stop_value is None:
            return {}
        return {
            name: value - stop_value
            for name, value in self._practice_values.items()
            if name != "STOP"
        }

    def belief_diagnostics(self) -> dict[str, dict[str, object]]:
        return {
            skill_name: belief.diagnostics()
            for skill_name, belief in self._pomdp_state.skill_beliefs.items()
        }

    def record_diagnostic(self, *, event: str, **fields: Any) -> None:
        if self.decision_log is None:
            return
        self.decision_log.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "event": event,
            "seed": self.seed,
            "cycle": self._cycle_index,
            "decision": self._decision_index,
            **fields,
        }
        with self.decision_log.open("a", encoding="utf-8") as stream:
            stream.write(LogTiming.encode(record=record))

    def human_skills(self) -> tuple[Skill, ...]:
        """Offer every provider-owned reset mechanism to belief-space planning.

        Empty when the provider offers none (`--no-human-reset`): the robot then has no
        way out of a stuck scene, which is what that baseline measures."""
        resets = self.skill_provider.movables_reset_skills()
        return tuple(dict.fromkeys(reset.skill for reset in resets))

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        self._pomdp_state = make_default_tossing3d_belief(
            num_particles=self.pomdp_num_particles,
            seed=self.seed,
            model=self.pomdp_competence_model,
            engine=self.pomdp_inference_engine,
            inference_config=InferenceConfig(
                competence_bins=self.pomdp_grid_competence_bins,
                learning_rate_bins=self.pomdp_grid_learning_rate_bins,
                sigma_competence=self.pomdp_competence_process_noise_std,
                sigma_eta=self.pomdp_learning_rate_process_noise_std,
                learning_rate_decay=self.pomdp_learning_rate_decay,
                eta_max=self.pomdp_learning_rate_max,
            ).scaled_learning_time(
                model=self.pomdp_competence_model, time_scale=self.pomdp_learning_time_scale
            ),
            additional_skill_names=tuple(skill.name for skill in self.human_skills()),
        )
        robot_skills = self.skills()
        human_skills = self.human_skills()
        practice_skills = (*robot_skills, *human_skills)
        ground_skills = SkillGrounder.applicable_ground_skills(
            skills=practice_skills,
            objects=self.objects(),
            true_atoms=SkillGrounder.all_possible_ground_atoms(
                objects=self.objects(), predicates=self.predicates()
            ),
        )
        self._pomdp_model = Tossing3DPracticeModel(
            seed=self.seed,
            exploration_epsilon=self.exploration_epsilon,
            competence_evidence=self.pomdp_competence_evidence,
            ground_skills=tuple(ground_skills),
            linear_cost_lambda=self.pomdp_linear_cost_lambda,
        )
        self.pomdp_planner = make_belief_space_planner(
            planner=self.pomdp_planner,
            solver=self.pomdp_solver,
            max_iterations=self.pomdp_max_search_iterations,
            seed=self.seed,
            observation_probability_weight=self.pomdp_observation_probability_weight,
            log_full_search_tree=self.pomdp_log_full_search_tree,
            llm_planner=self._make_llm_planner,
        )
        available = {ground_skill.skill.name for ground_skill in ground_skills}
        missing = {TOSS_SKILL, OPEN_GRIPPER_SKILL} - available
        if not (available & PICK_SKILLS):
            missing.add(PICK_SKILL)
        assert not missing, (
            "Tossing3DPomdpMethod requires canonical Tossing3D skills; missing "
            f"{sorted(missing)} from {sorted(available)}"
        )
        assert all(skill.skill.practice_cost is not None for skill in ground_skills)
        assert all(skill.evaluate_practice_cost() <= COST_MAX for skill in ground_skills), (
            f"POMDP cost observations must be at most {COST_MAX}"
        )

    def _make_llm_planner(self) -> LlmPracticePlanner:
        if self.planner_client is not None:
            return LlmPracticePlanner(
                client=self.planner_client,
                memory=self.planner_memory,
                model=self.planner_model,
                effort=self.planner_effort,
                budget_limit_usd=self.planner_model_budget_usd,
            )
        if self.planner_sandbox is None or self.planner_output is None:
            raise ValueError(
                "pomdp_solver='llm' needs planner_sandbox and planner_output, "
                "or an injected planner_client"
            )
        return make_llm_practice_planner(
            sandbox=self.planner_sandbox,
            output=self.planner_output,
            model=self.planner_model,
            effort=self.planner_effort,
            budget_limit_usd=self.planner_model_budget_usd,
            memory=self.planner_memory,
        )

    @property
    def pomdp_state(self) -> Tossing3DBeliefState:
        return self._pomdp_state

    def get_practice_policy(self, *, task: Task) -> Policy:
        """Start a practice session without resetting the learned skill state."""
        self._remaining_practice_actions = None
        self._llm_calls_this_session = 0
        self._llm_history.append({"type": "session_start", "session": self._cycle_index})
        # G scores the current session, so its accumulated cost starts at zero.
        previous_session_cost = self._pomdp_state.accumulated_cost
        self._pomdp_state = self._pomdp_state.model_copy(update={"accumulated_cost": 0.0})
        self._practice_values.clear()
        self.record_diagnostic(
            event="session_start",
            previous_session_cost=previous_session_cost,
            summed_cost=0.0,
            estimated_costs=self.practice_skill_costs(),
        )
        return super().get_practice_policy(task=task)

    def goal_failure_enters_random_mode(self) -> bool:
        """Never: a failed goal plan falls through to this planner's own practice
        choice, whatever --reproduce-predicators-random-when-stranded says."""
        return False

    def clear_starved_parameter_pools(self) -> None:
        """Mirror the base registry's clearing in the search model's action mask, so a
        new session or a movables reset leaves the planner unmasked too."""
        super().clear_starved_parameter_pools()
        self._pomdp_model = self._pomdp_model.model_copy(update={"starved_pools": ()})

    def record_starved_parameter_pool(
        self, *, ground_skill: GroundSkill, true_atoms: frozenset[GroundAtom]
    ) -> None:
        """Also mask the pair in the belief-space search model: the base registry
        filters *selected* candidates, but this planner deterministically re-derives
        its single choice, so deselection has to reach the search's own action
        enumeration for it to genuinely choose again."""
        super().record_starved_parameter_pool(ground_skill=ground_skill, true_atoms=true_atoms)
        pair = (true_atoms, ground_skill)
        if pair not in self._pomdp_model.starved_pools:
            self._pomdp_model = self._pomdp_model.model_copy(
                update={"starved_pools": (*self._pomdp_model.starved_pools, pair)}
            )

    def observe_practice_action_budget(self, *, remaining_actions: int) -> None:
        assert remaining_actions >= 0, "remaining_actions must be non-negative"
        self._remaining_practice_actions = remaining_actions

    def record_practice_attempt(
        self, *, skill_name: str, success: bool, consultation: SamplerConsultation
    ) -> None:
        """Remember the consultation of the outcome `observe_outcome` receives next.

        `_EesEpisode.observe_pending` calls this immediately before
        `observe_outcome`, which deliberately does not carry the consultation."""
        super().record_practice_attempt(
            skill_name=skill_name, success=success, consultation=consultation
        )
        self._pending_consultation = (skill_name, consultation)

    def observe_outcome(
        self, *, ground_skill: GroundSkill, success: bool, was_random_exploration: bool = False
    ) -> None:
        super().observe_outcome(
            ground_skill=ground_skill,
            success=success,
            was_random_exploration=was_random_exploration,
        )
        pending, self._pending_consultation = self._pending_consultation, None
        consultation = None
        if pending is not None:
            skill_name, consultation = pending
            assert skill_name == ground_skill.skill.name, (skill_name, ground_skill.skill.name)
            assert (consultation is SamplerConsultation.EPSILON_RANDOM) == was_random_exploration
        configured_cost_observation = (
            ground_skill.evaluate_practice_cost() if self._practice_accounting is None else None
        )
        self._pomdp_state = self._pomdp_model.observe_outcome(
            state=self._pomdp_state,
            ground_skill=ground_skill,
            success=success,
            was_random_exploration=was_random_exploration,
            observed_cost=configured_cost_observation,
            consultation=consultation,
        )
        self._llm_history.append({
            "type": "attempt",
            "session": self._cycle_index,
            "decision": self._decision_index,
            "action": LlmPracticePlanner.candidate_id(ground=ground_skill),
            "skill": ground_skill.skill.name,
            "success": success,
            "random_exploration": was_random_exploration,
            "cost": self._last_attempt_cost.pop(ground_skill, None),
        })
        self.record_diagnostic(
            event="outcome",
            skill=ground_skill.skill.name,
            success=success,
            random_exploration=was_random_exploration,
            consultation=None if consultation is None else consultation.value,
            competence_evidence=self.pomdp_competence_evidence.value,
            competence_conditioned=(
                not was_random_exploration
                if consultation is None
                else self.pomdp_competence_evidence.admits(consultation=consultation)
            ),
            belief=self._pomdp_state.model_dump(mode="json"),
            beliefs=self.belief_diagnostics(),
            configured_cost_observation=configured_cost_observation,
            cost_observation_source=(
                "configured_practice_cost"
                if self._practice_accounting is None
                else "host_control_steps"
            ),
            estimated_costs=self.practice_skill_costs(),
        )

    def record_parameter_sampling(
        self,
        *,
        ground_skill: GroundSkill,
        explore: bool,
        diagnostics: ParameterSamplingDiagnostics,
    ) -> None:
        super().record_parameter_sampling(
            ground_skill=ground_skill, explore=explore, diagnostics=diagnostics
        )
        self.record_diagnostic(
            event="parameter_sampling",
            skill=ground_skill.skill.name,
            explore=explore,
            **diagnostics.model_dump(mode="json"),
        )

    def record_action_cost(self, *, ground_skill: GroundSkill) -> None:
        """Charge each attempted action immediately, including a final-step reset."""
        super().record_action_cost(ground_skill=ground_skill)
        action_cost = (
            ground_skill.evaluate_practice_cost() if self._practice_accounting is None else 0.0
        )
        updates: dict[str, object] = {
            "accumulated_cost": self._pomdp_state.accumulated_cost + action_cost
        }
        if self._practice_accounting is None:
            self._last_attempt_cost[ground_skill] = action_cost
        if ground_skill.skill.name in RESET_SKILLS:
            self._pending_reset = ground_skill
        self._pomdp_state = self._pomdp_state.model_copy(update=updates)
        self.record_diagnostic(
            event="dispatch",
            skill=ground_skill.skill.name,
            cost=action_cost,
            summed_cost=self._pomdp_state.accumulated_cost,
            estimated_costs=self.practice_skill_costs(),
        )

    def observe_symbolic_transition(
        self,
        *,
        ground_skill: GroundSkill,
        before_atoms: frozenset[GroundAtom],
        after_atoms: frozenset[GroundAtom],
        success: bool,
        was_random_exploration: bool,
    ) -> None:
        if success or ground_skill.skill.name not in {*PICK_SKILLS, TOSS_SKILL, OPEN_GRIPPER_SKILL}:
            return
        counts = EmpiricalFailureEffects.observe(
            counts=self._pomdp_model.failure_effect_counts,
            ground_skill=ground_skill,
            before_atoms=before_atoms,
            after_atoms=after_atoms,
            was_random_exploration=was_random_exploration,
        )
        self._pomdp_model = self._pomdp_model.model_copy(update={"failure_effect_counts": counts})
        self.record_diagnostic(
            event="failure_transition",
            skill=ground_skill.skill.name,
            objects=[obj.name for obj in ground_skill.objects],
            before_atoms=sorted(str(atom) for atom in before_atoms),
            after_atoms=sorted(str(atom) for atom in after_atoms),
            was_random_exploration=was_random_exploration,
            add_effects=sorted(str(atom) for atom in after_atoms - before_atoms),
            delete_effects=sorted(str(atom) for atom in before_atoms - after_atoms),
            failure_effect_counts=EmpiricalFailureEffects.diagnostics(counts=counts),
        )

    def observe_help_granted(self, *, state: State) -> None:
        """Condition one completed reset on its joint success and cost observation."""
        super().observe_help_granted(state=state)
        reset = self._pending_reset
        assert reset is not None, "human reset completion observed without a dispatched reset"
        observed_cost = (
            reset.evaluate_practice_cost() if self._practice_accounting is None else None
        )
        self._pomdp_state = self._pomdp_model.observe_outcome(
            state=self._pomdp_state,
            ground_skill=reset,
            success=True,
            was_random_exploration=False,
            observed_cost=observed_cost,
        )
        self._pending_reset = None
        reset_id = LlmPracticePlanner.candidate_id(ground=reset)
        self._llm_history.append({
            "type": "human_reset",
            "session": self._cycle_index,
            "decision": self._decision_index,
            "action": reset_id,
            "human_name": self._human_names().get(reset_id),
            "success": True,
            "cost": self._last_attempt_cost.pop(reset, None),
        })
        self.record_diagnostic(
            event="outcome",
            skill=reset.skill.name,
            success=True,
            random_exploration=False,
            belief=self._pomdp_state.model_dump(mode="json"),
            beliefs=self.belief_diagnostics(),
            configured_cost_observation=observed_cost,
            cost_observation_source=(
                "configured_practice_cost"
                if self._practice_accounting is None
                else "host_control_steps"
            ),
            estimated_costs=self.practice_skill_costs(),
        )

    def observe_sampler_outcome(
        self, *, skill_name: str, param_dim: int, sampler_input: list[float], success: bool
    ) -> None:
        super().observe_sampler_outcome(
            skill_name=skill_name,
            param_dim=param_dim,
            sampler_input=sampler_input,
            success=success,
        )
        self._pomdp_state = self._pomdp_model.observe_training_example(
            state=self._pomdp_state, skill_name=skill_name, success=success
        )

    def end_cycle(self) -> None:
        """Advance inferred learning curves at the session boundary.

        Every attempt that entered the sampler's training data advances the
        training clock, one-class fits included. Fixed controllers have no
        training examples, so theirs is the zero-example step.
        """
        # Flush the in-flight EES action against the pre-reset state before refitting.
        self.observe_environment_reset(state=self.env.get_current_state())
        super().end_cycle()
        training_examples = dict(self._pomdp_state.pending_examples)
        for skill_name, belief in self._pomdp_state.skill_beliefs.items():
            if isinstance(belief, BayesianSkillBelief):
                self._belief_history.setdefault(skill_name, []).append(belief)
        if self.decision_log is not None:
            histories = {
                skill_name: [
                    cycle.model_dump(mode="json") for cycle in smooth_history(history=history)
                ]
                for skill_name, history in self._belief_history.items()
            }
            self.record_diagnostic(
                event="smoothing",
                competence_model=self.pomdp_competence_model,
                inference_engine=self.pomdp_inference_engine,
                history=histories,
                use="retrospective_diagnostics_only",
            )
        self._pomdp_state = refit_belief_state(
            state=self._pomdp_state,
            learning_rate_process_noise_std=self.pomdp_learning_rate_process_noise_std,
            advance_cycle=True,
        )
        self.record_diagnostic(
            event="refit",
            belief=self._pomdp_state.model_dump(mode="json"),
            beliefs=self.belief_diagnostics(),
            estimated_costs=self.practice_skill_costs(),
            training_examples=training_examples,
            learning_rate_evidence="success_failure_only",
            competence_model=self.pomdp_competence_model,
            inference_engine=self.pomdp_inference_engine,
        )
        self._cycle_index += 1

    def select_skill_to_practice(self, *, true_atoms: frozenset[GroundAtom]) -> list[GroundSkill]:
        self._refresh_execution_forecasts()
        self._decision_index += 1
        trace = SearchTrace()
        model: BeliefSpaceModel[
            Tossing3DSearchState,
            Tossing3DBeliefState,
            Tossing3DTheta,
            GroundSkill,
        ] = self._pomdp_model
        planner = self.pomdp_planner
        assert planner is not None
        horizon = self.pomdp_search_depth
        if self._remaining_practice_actions is not None:
            horizon = min(horizon, self._remaining_practice_actions)
        search_started_at = time.perf_counter()
        try:
            search_state = make_tossing3d_search_state(
                state=self._pomdp_state, true_atoms=true_atoms
            )
            if isinstance(planner, DeterminizedAStarPlanner):
                value, action = planner.solve(
                    environment_state=search_state,
                    belief_state=self._pomdp_state,
                    summed_cost=self._pomdp_state.accumulated_cost,
                    horizon=horizon,
                    model=model,
                    trace=trace,
                    num_samples=self.pomdp_num_samples,
                    remaining_actions=self._remaining_practice_actions,
                )
            elif isinstance(planner, LlmPracticePlanner):
                value, action = planner.solve(
                    environment_state=search_state,
                    belief_state=self._pomdp_state,
                    summed_cost=self._pomdp_state.accumulated_cost,
                    horizon=horizon,
                    model=model,
                    trace=trace,
                    num_samples=self.pomdp_num_samples,
                    situation=self._llm_situation(),
                )
            else:
                value, action = planner.solve(
                    environment_state=search_state,
                    belief_state=self._pomdp_state,
                    summed_cost=self._pomdp_state.accumulated_cost,
                    horizon=horizon,
                    model=model,
                    trace=trace,
                    num_samples=self.pomdp_num_samples,
                )
        finally:
            trace.close()
        search_duration_seconds = time.perf_counter() - search_started_at
        schedule = self.practice_schedule()
        self._practice_values = {}
        action_value_events = [
            event
            for event in trace.events
            if event["node"] == 0 and event["event"] == "action_value"
        ]
        action_name_counts = Counter(
            event["action"]["skill"]["name"] for event in action_value_events
        )
        for event in trace.events:
            if event["node"] == 0 and event["event"] == "stop_value":
                self._practice_values["STOP"] = event["value"]
            elif (
                event["node"] == 0
                and event["event"] == "action_value"
                # The LLM planner logs its candidates without a value.
                and event["value"] is not None
            ):
                skill_name = event["action"]["skill"]["name"]
                key = skill_name
                if action_name_counts[skill_name] > 1:
                    objects = ", ".join(obj["name"] for obj in event["action"]["objects"])
                    key = f"{skill_name}({objects})"
                self._practice_values[key] = event["value"]
        llm_fields: dict[str, Any] = {}
        if isinstance(planner, LlmPracticePlanner):
            decision = planner.last_decision
            assert decision is not None, "the LLM planner returned without a decision"
            llm_fields = {
                "llm_reason": decision.reason,
                "llm_call_index": decision.call_index,
                "llm_cost_usd": decision.cost_usd,
                "planner_memory": planner.memory.value,
            }
            self._llm_history.append({
                "type": "decision",
                "session": self._cycle_index,
                "decision": self._decision_index,
                "action": decision.action_id,
                "reason": decision.reason,
            })
            self._llm_history_sent = len(self._llm_history)
            self._llm_calls_this_session += 1
        self.record_diagnostic(
            event="decision",
            competences=self.practice_skill_competences(),
            learning_rates=self.practice_skill_learning_rates(),
            estimated_costs=self.practice_skill_costs(),
            improvement_potentials=self.practice_skill_improvement_potentials(),
            summed_cost=self._pomdp_state.accumulated_cost,
            search_duration_seconds=search_duration_seconds,
            num_samples=self.pomdp_num_samples,
            atoms=sorted(str(atom) for atom in true_atoms),
            action=(
                "STOP" if action == STOP_ACTION else action.model_dump(mode="json", fallback=str)
            ),
            value=value,
            horizon=horizon if planner.name == "expectimax" else None,
            remaining_practice_actions=self._remaining_practice_actions,
            practice_step_budget=schedule.step_budget if schedule else None,
            measurement_interval_steps=schedule.measurement_interval_steps if schedule else None,
            practice_steps_used=schedule.steps_used if schedule else None,
            model_budget_remaining_usd=schedule.model_budget_remaining_usd if schedule else None,
            solver=planner.name,
            max_search_iterations=(
                planner.max_iterations if isinstance(planner, DeterminizedAStarPlanner) else None
            ),
            observation_probability_weight=(
                planner.observation_probability_weight
                if isinstance(planner, (DeterminizedAStarPlanner, ExpectimaxPlanner))
                else None
            ),
            model=self._pomdp_model.model_dump(mode="json"),
            failure_effect_counts=EmpiricalFailureEffects.diagnostics(
                counts=self._pomdp_model.failure_effect_counts
            ),
            search=trace.events,
            **llm_fields,
        )
        if isinstance(action, StopAction):
            assert action == STOP_ACTION
            return [STOP_SKILL]

        self.record_practice_target(name=action.skill.name, field="scored")
        return [action]

    def _human_names(self) -> dict[str, str]:
        """Candidate id -> what the step protocol calls that human reset."""
        names = {}
        for reset in self.skill_provider.human_cube_bin_reset_skills():
            side = self.skill_provider.movables_reset_destination(ground_skill=reset)
            names[LlmPracticePlanner.candidate_id(ground=reset)] = (
                "reset_cube_and_bin_near" if side == "robot_side" else "reset_cube_far"
            )
        return names

    def _configured_costs(self) -> dict[str, Any]:
        """The prices a decision is charged under, as configured for this run."""
        accounting = self._practice_accounting
        if accounting is not None:
            return {
                "source": "practice_cost_config",
                **accounting.costs.model_dump(mode="json"),
            }
        resets = self.skill_provider.human_cube_bin_reset_skills()
        return {
            "source": "configured_practice_costs",
            "objective_lambda": self.pomdp_linear_cost_lambda,
            "hard_practice_budget": (
                None if self.pomdp_linear_cost_lambda is not None else PRACTICE_BUDGET
            ),
            "human_reset_cost": resets[0].evaluate_practice_cost() if resets else None,
            "robot_skill_costs": {
                ground.skill.name: ground.evaluate_practice_cost()
                for ground in self._pomdp_model.ground_skills
                if ground.skill.name not in RESET_SKILLS
            },
        }

    def _llm_history_slice(self) -> PracticeHistory:
        """Stateless: the last `planner_history_window` decisions, every call. Resume:
        the same window on a session's first call (a fresh Claude session), then only
        what happened since the previous call."""
        events = self._llm_history
        resumed = self.planner_memory is PlannerMemory.resume and self._llm_calls_this_session > 0
        if resumed:
            scope, start = "since_previous_call", self._llm_history_sent
        else:
            decisions = [index for index, event in enumerate(events) if event["type"] == "decision"]
            window = self.planner_history_window
            start = decisions[-window] if len(decisions) > window else 0
            scope = f"last_{window}_decisions"
        totals: dict[str, dict[str, int]] = {}
        for ground, outcomes in self._all_attempt_outcomes.items():
            totals[LlmPracticePlanner.candidate_id(ground=ground)] = {
                "attempts": len(outcomes),
                "successes": sum(outcomes),
            }
        for event in events:
            if event["type"] == "human_reset":
                counts = totals.setdefault(event["action"], {"attempts": 0, "successes": 0})
                counts["attempts"] += 1
                counts["successes"] += int(event["success"])
        return PracticeHistory(scope=scope, events=tuple(events[start:]), skill_totals=totals)

    def _llm_situation(self) -> PracticeSituation:
        return PracticeSituation(
            session_index=self._cycle_index,
            decision_index=self._decision_index,
            session_key=f"session-{self._cycle_index}",
            remaining_session_actions=self._remaining_practice_actions,
            schedule=self.practice_schedule(),
            configured_costs=self._configured_costs(),
            human_names=self._human_names(),
            history=self._llm_history_slice(),
        )
