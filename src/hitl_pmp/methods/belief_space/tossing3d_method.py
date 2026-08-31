"""Tossing3D method backed by situated belief-space expectimax."""

import time
from pathlib import Path
from typing import Any

from pydantic import Field, PrivateAttr

from hitl_pmp.core.log_timing import LogTiming
from hitl_pmp.core.method.types import GroundSkill, Policy, Skill
from hitl_pmp.core.problem.tasks.types import GroundAtom, Task
from hitl_pmp.methods.practice_makes_perfect.ees_method import (
    STOP_SKILL,
    EesMethod,
)
from hitl_pmp.planning.grounding import SkillGrounder

from .expectimax import solve_belief_space_expectimax
from .tossing3d_constants import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    RESET_SKILL,
    TOSS_SKILL,
)
from .tossing3d_model import Tossing3DPracticeModel
from .tossing3d_observation_model import (
    make_default_tossing3d_belief,
    mean_competence,
    refit_belief_state,
)
from .tossing3d_transition_model import make_tossing3d_search_state
from .types.belief_state import Tossing3DBeliefState
from .types.search_trace import SearchTrace
from .types.stop_action import STOP_ACTION, StopAction


class Tossing3DPomdpMethod(EesMethod):
    """EES learner/executor with situated belief-space practice decisions."""

    pomdp_search_depth: int = Field(default=3, ge=0)
    pomdp_num_samples: int = Field(default=100, ge=1)
    goal_pursuit_horizon: int | None = 0
    decision_log: Path | None = None

    _pomdp_state: Tossing3DBeliefState = PrivateAttr()
    _pomdp_model: Tossing3DPracticeModel = PrivateAttr()
    _decision_index: int = PrivateAttr(default=0)
    _cycle_index: int = PrivateAttr(default=0)
    _practice_values: dict[str, float] = PrivateAttr(default_factory=dict)

    def practice_action_values(self) -> dict[str, float]:
        """Values from the last real decision, never an extra search for rendering."""
        return dict(self._practice_values)

    def practice_skill_competences(self) -> dict[str, float]:
        estimates = {
            PICK_SKILL + " (belief mean)": mean_competence(
                belief=self._pomdp_state.skill_beliefs[PICK_SKILL]
            ),
            TOSS_SKILL + " (belief mean)": mean_competence(
                belief=self._pomdp_state.skill_beliefs[TOSS_SKILL]
            ),
            OPEN_GRIPPER_SKILL + " (belief mean)": mean_competence(
                belief=self._pomdp_state.skill_beliefs[OPEN_GRIPPER_SKILL]
            ),
        }
        if self.ask_for_reset_cube_bin_cost is not None:
            estimates[RESET_SKILL + " (fixed)"] = 1.0
        return estimates

    def practice_skill_learning_rates(self) -> dict[str, float]:
        estimates = {
            PICK_SKILL + " (belief mean)": self._pomdp_state.pick_belief.mean_learning_rate,
            TOSS_SKILL + " (belief mean)": self._pomdp_state.toss_belief.mean_learning_rate,
            OPEN_GRIPPER_SKILL
            + " (belief mean)": self._pomdp_state.open_gripper_belief.mean_learning_rate,
        }
        if self._pomdp_model.reset_cost is not None:
            estimates[RESET_SKILL + " (fixed)"] = 0.0
        return estimates

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
        """Configured human-executed skills available during practice."""
        if self.ask_for_reset_cube_bin_cost is None:
            return ()
        reset = self.skill_provider.human_cube_bin_reset_skill()
        assert reset is not None
        return (reset.skill.model_copy(update={"practice_cost": self.ask_for_reset_cube_bin_cost}),)

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        self._pomdp_state = make_default_tossing3d_belief()
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
        )
        available = {ground_skill.skill.name for ground_skill in ground_skills}
        missing = {PICK_SKILL, TOSS_SKILL, OPEN_GRIPPER_SKILL} - available
        assert not missing, (
            "Tossing3DPomdpMethod requires canonical Tossing3D skills; missing "
            f"{sorted(missing)} from {sorted(available)}"
        )
        assert all(skill.skill.practice_cost is not None for skill in ground_skills)

    @property
    def pomdp_state(self) -> Tossing3DBeliefState:
        return self._pomdp_state

    def get_practice_policy(self, *, task: Task) -> Policy:
        """Start a practice session without resetting the learned skill state."""
        # G scores the current session, so its accumulated cost starts at zero.
        previous_session_cost = self._pomdp_state.accumulated_cost
        self._pomdp_state = self._pomdp_state.model_copy(update={"accumulated_cost": 0.0})
        self._practice_values.clear()
        self.record_diagnostic(
            event="session_start",
            previous_session_cost=previous_session_cost,
            summed_cost=0.0,
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
        self._pomdp_state = self._pomdp_model.observe_outcome(
            state=self._pomdp_state,
            ground_skill=ground_skill,
            success=success,
            was_random_exploration=was_random_exploration,
        )
        self.record_diagnostic(
            event="outcome",
            skill=ground_skill.skill.name,
            success=success,
            random_exploration=was_random_exploration,
            belief=self._pomdp_state.model_dump(mode="json"),
        )

    def record_action_cost(self, *, ground_skill: GroundSkill) -> None:
        """Charge each attempted action immediately, including a final-step reset."""
        action_cost = ground_skill.evaluate_practice_cost()
        self._pomdp_state = self._pomdp_state.model_copy(
            update={"accumulated_cost": self._pomdp_state.accumulated_cost + action_cost}
        )
        self.record_diagnostic(
            event="dispatch",
            skill=ground_skill.skill.name,
            cost=action_cost,
            summed_cost=self._pomdp_state.accumulated_cost,
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
        self._pomdp_state = refit_belief_state(state=self._pomdp_state)
        self.record_diagnostic(event="refit", belief=self._pomdp_state.model_dump(mode="json"))
        self._cycle_index += 1

    def select_skill_to_practice(self, *, true_atoms: frozenset[GroundAtom]) -> list[GroundSkill]:
        self._decision_index += 1
        trace_path = (
            None
            if self.decision_log is None
            else self.decision_log.parent
            / "search_traces"
            / f"cycle_{self._cycle_index:04d}_decision_{self._decision_index:04d}.jsonl.gz"
        )
        trace = (
            SearchTrace(path=trace_path, retain_events=False) if trace_path is not None else None
        )
        search_started_at = time.perf_counter()
        try:
            search_result = solve_belief_space_expectimax(
                environment_state=make_tossing3d_search_state(
                    state=self._pomdp_state, true_atoms=true_atoms
                ),
                belief_state=self._pomdp_state,
                summed_cost=self._pomdp_state.accumulated_cost,
                horizon=self.pomdp_search_depth,
                model=self._pomdp_model,
                trace=trace,
                num_samples=self.pomdp_num_samples,
            )
        finally:
            if trace is not None:
                trace.close()
        search_duration_seconds = time.perf_counter() - search_started_at
        value, action = search_result
        self._practice_values = {}
        if trace is not None:
            for event in trace.events:
                if event["node"] == 0 and event["event"] == "stop_value":
                    self._practice_values["STOP"] = event["value"]
                elif event["node"] == 0 and event["event"] == "action_value":
                    self._practice_values[event["action"]["skill"]["name"]] = event["value"]
        self.record_diagnostic(
            event="decision",
            competences=self.practice_skill_competences(),
            search_duration_seconds=search_duration_seconds,
            num_samples=self.pomdp_num_samples,
            atoms=sorted(str(atom) for atom in true_atoms),
            action="STOP"
            if action == STOP_ACTION
            else action.model_dump(mode="json", fallback=str),
            value=value,
            horizon=self.pomdp_search_depth,
            model=self._pomdp_model.model_dump(mode="json"),
            search=[] if trace is None else trace.events,
            search_trace=None
            if trace_path is None or self.decision_log is None
            else str(trace_path.relative_to(self.decision_log.parent)),
        )
        if isinstance(action, StopAction):
            assert action == STOP_ACTION
            return [STOP_SKILL]

        self.record_practice_target(name=action.skill.name, field="scored")
        return [action]
