"""Tossing3D method backed by situated belief-space expectimax."""

import time
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, PrivateAttr

from hitl_pmp.core.log_timing import LogTiming
from hitl_pmp.core.method.types import GroundSkill, Policy, Skill
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import GroundAtom, Task
from hitl_pmp.methods.practice_makes_perfect.ees_method import (
    STOP_SKILL,
    EesMethod,
)
from hitl_pmp.planning.grounding import SkillGrounder

from .determinized import DeterminizedAStarPlanner
from .expectimax import ExpectimaxPlanner
from .planner import BeliefSpacePlanner
from .tossing3d_constants import (
    LEARNING_RATE_PROCESS_NOISE_STD,
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    RESET_SKILLS,
    TOSS_SKILL,
)
from .tossing3d_model import Tossing3DPracticeModel
from .tossing3d_observation_model import (
    make_default_tossing3d_belief,
    mean_competence,
    mean_cost,
    mean_learning_rate,
    observed_learning_rates,
    refit_belief_state,
)
from .tossing3d_transition_model import make_tossing3d_search_state
from .types.belief_state import Tossing3DBeliefState
from .types.particle_filter_belief import ParticleFilterBelief
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
    solver: Literal["expectimax", "determinized_astar"],
    max_iterations: int,
    seed: int,
    observation_probability_weight: float,
) -> BeliefSpacePlanner[Tossing3DSearchState, Tossing3DBeliefState, Tossing3DTheta, GroundSkill]:
    """Return the injected planner or construct the configured planner once."""
    if planner is not None:
        return planner
    if solver == "expectimax":
        return ExpectimaxPlanner()
    return DeterminizedAStarPlanner(
        max_iterations=max_iterations,
        seed=seed,
        observation_probability_weight=observation_probability_weight,
    )


class Tossing3DPomdpMethod(EesMethod):
    """EES learner/executor with situated belief-space practice decisions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pomdp_search_depth: int = Field(default=3, ge=0)
    pomdp_solver: Literal["expectimax", "determinized_astar"] = "expectimax"
    pomdp_max_search_iterations: int = Field(default=100, ge=1)
    pomdp_observation_probability_weight: float = Field(default=0.1, ge=0.0, allow_inf_nan=False)
    pomdp_planner: (
        BeliefSpacePlanner[Tossing3DSearchState, Tossing3DBeliefState, Tossing3DTheta, GroundSkill]
        | None
    ) = Field(default=None, exclude=True, repr=False)
    pomdp_num_samples: int = Field(default=100, ge=1)
    pomdp_num_particles: int = Field(default=1024, ge=1)
    pomdp_learning_rate_process_noise_std: float = Field(
        default=LEARNING_RATE_PROCESS_NOISE_STD, ge=0.0
    )
    pomdp_linear_cost_lambda: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    goal_pursuit_horizon: int | None = 0
    decision_log: Path | None = None

    _pomdp_state: Tossing3DBeliefState = PrivateAttr()
    _pomdp_model: Tossing3DPracticeModel = PrivateAttr()
    _decision_index: int = PrivateAttr(default=0)
    _cycle_index: int = PrivateAttr(default=0)
    _practice_values: dict[str, float] = PrivateAttr(default_factory=dict)
    _cycle_start_competences: dict[str, float] = PrivateAttr(default_factory=dict)
    _pending_reset: GroundSkill | None = PrivateAttr(default=None)

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
        return {
            skill_name + " (belief mean)": mean_cost(belief=belief)
            for skill_name, belief in self._pomdp_state.skill_beliefs.items()
            if isinstance(belief, ParticleFilterBelief)
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
        """Offer every provider-owned reset mechanism to belief-space planning."""
        return tuple(reset.skill for reset in self.skill_provider.movables_reset_skills())

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        self._pomdp_state = make_default_tossing3d_belief(
            num_particles=self.pomdp_num_particles,
            seed=self.seed,
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
            ground_skills=tuple(ground_skills),
            linear_cost_lambda=self.pomdp_linear_cost_lambda,
        )
        self.pomdp_planner = make_belief_space_planner(
            planner=self.pomdp_planner,
            solver=self.pomdp_solver,
            max_iterations=self.pomdp_max_search_iterations,
            seed=self.seed,
            observation_probability_weight=self.pomdp_observation_probability_weight,
        )
        available = {ground_skill.skill.name for ground_skill in ground_skills}
        missing = {PICK_SKILL, TOSS_SKILL, OPEN_GRIPPER_SKILL} - available
        assert not missing, (
            "Tossing3DPomdpMethod requires canonical Tossing3D skills; missing "
            f"{sorted(missing)} from {sorted(available)}"
        )
        assert all(skill.skill.practice_cost is not None for skill in ground_skills)
        assert all(skill.evaluate_practice_cost() <= COST_MAX for skill in ground_skills), (
            f"POMDP cost observations must be at most {COST_MAX}"
        )

    @property
    def pomdp_state(self) -> Tossing3DBeliefState:
        return self._pomdp_state

    def get_practice_policy(self, *, task: Task) -> Policy:
        """Start a practice session without resetting the learned skill state."""
        # G scores the current session, so its accumulated cost starts at zero.
        previous_session_cost = self._pomdp_state.accumulated_cost
        self._pomdp_state = self._pomdp_state.model_copy(update={"accumulated_cost": 0.0})
        self._cycle_start_competences = {
            skill_name: mean_competence(belief=belief)
            for skill_name, belief in self._pomdp_state.skill_beliefs.items()
        }
        self._practice_values.clear()
        self.record_diagnostic(
            event="session_start",
            previous_session_cost=previous_session_cost,
            summed_cost=0.0,
            estimated_costs=self.practice_skill_costs(),
        )
        return super().get_practice_policy(task=task)

    def observe_outcome(
        self, *, ground_skill: GroundSkill, success: bool, was_random_exploration: bool = False
    ) -> None:
        super().observe_outcome(
            ground_skill=ground_skill,
            success=success,
            was_random_exploration=was_random_exploration,
        )
        configured_cost_observation = ground_skill.evaluate_practice_cost()
        self._pomdp_state = self._pomdp_model.observe_outcome(
            state=self._pomdp_state,
            ground_skill=ground_skill,
            success=success,
            was_random_exploration=was_random_exploration,
            observed_cost=configured_cost_observation,
        )
        self.record_diagnostic(
            event="outcome",
            skill=ground_skill.skill.name,
            success=success,
            random_exploration=was_random_exploration,
            belief=self._pomdp_state.model_dump(mode="json"),
            beliefs=self.belief_diagnostics(),
            configured_cost_observation=configured_cost_observation,
            cost_observation_source="configured_practice_cost",
            estimated_costs=self.practice_skill_costs(),
        )

    def record_action_cost(self, *, ground_skill: GroundSkill) -> None:
        """Charge each attempted action immediately, including a final-step reset."""
        action_cost = ground_skill.evaluate_practice_cost()
        updates: dict[str, object] = {
            "accumulated_cost": self._pomdp_state.accumulated_cost + action_cost
        }
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

    def observe_help_granted(self, *, state: State) -> None:
        """Condition one completed reset on its joint success and cost observation."""
        super().observe_help_granted(state=state)
        reset = self._pending_reset
        assert reset is not None, "human reset completion observed without a dispatched reset"
        observed_cost = reset.evaluate_practice_cost()
        self._pomdp_state = self._pomdp_model.observe_outcome(
            state=self._pomdp_state,
            ground_skill=reset,
            success=True,
            was_random_exploration=False,
            observed_cost=observed_cost,
        )
        self._pending_reset = None
        self.record_diagnostic(
            event="outcome",
            skill=reset.skill.name,
            success=True,
            random_exploration=False,
            belief=self._pomdp_state.model_dump(mode="json"),
            beliefs=self.belief_diagnostics(),
            configured_cost_observation=observed_cost,
            cost_observation_source="configured_practice_cost",
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
            state=self._pomdp_state, skill_name=skill_name
        )

    def end_cycle(self) -> None:
        """Advance inferred learning curves at the session boundary.

        For parameter-free skills this is a forecast, not a controller update.
        Subsequent outcomes reweight improving versus stationary hypotheses.
        """
        # Flush the in-flight EES action against the pre-reset state before refitting.
        self.observe_environment_reset(state=self.env.get_current_state())
        super().end_cycle()
        learning_rate_observations = observed_learning_rates(
            state=self._pomdp_state,
            cycle_start_competences=self._cycle_start_competences,
        )
        learning_rate_observation_counts = {
            skill_name: self._pomdp_state.pending_examples[skill_name]
            for skill_name in learning_rate_observations
        }
        self._pomdp_state = refit_belief_state(
            state=self._pomdp_state,
            cycle_start_competences=self._cycle_start_competences,
            learning_rate_process_noise_std=self.pomdp_learning_rate_process_noise_std,
        )
        self.record_diagnostic(
            event="refit",
            belief=self._pomdp_state.model_dump(mode="json"),
            beliefs=self.belief_diagnostics(),
            estimated_costs=self.practice_skill_costs(),
            learning_rate_observations=learning_rate_observations,
            learning_rate_observation_counts=learning_rate_observation_counts,
        )
        self._cycle_index += 1

    def select_skill_to_practice(self, *, true_atoms: frozenset[GroundAtom]) -> list[GroundSkill]:
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
        search_started_at = time.perf_counter()
        try:
            search_state = make_tossing3d_search_state(
                state=self._pomdp_state, true_atoms=true_atoms
            )
            value, action = planner.solve(
                environment_state=search_state,
                belief_state=self._pomdp_state,
                summed_cost=self._pomdp_state.accumulated_cost,
                horizon=self.pomdp_search_depth,
                model=model,
                trace=trace,
                num_samples=self.pomdp_num_samples,
            )
        finally:
            trace.close()
        search_duration_seconds = time.perf_counter() - search_started_at
        self._practice_values = {}
        for event in trace.events:
            if event["node"] == 0 and event["event"] == "stop_value":
                self._practice_values["STOP"] = event["value"]
            elif event["node"] == 0 and event["event"] == "action_value":
                self._practice_values[event["action"]["skill"]["name"]] = event["value"]
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
            action="STOP"
            if action == STOP_ACTION
            else action.model_dump(mode="json", fallback=str),
            value=value,
            horizon=self.pomdp_search_depth if planner.name == "expectimax" else None,
            solver=planner.name,
            max_search_iterations=(
                planner.max_iterations if isinstance(planner, DeterminizedAStarPlanner) else None
            ),
            observation_probability_weight=(
                planner.observation_probability_weight
                if isinstance(planner, DeterminizedAStarPlanner)
                else None
            ),
            model=self._pomdp_model.model_dump(mode="json"),
            search=trace.events,
        )
        if isinstance(action, StopAction):
            assert action == STOP_ACTION
            return [STOP_SKILL]

        self.record_practice_target(name=action.skill.name, field="scored")
        return [action]
