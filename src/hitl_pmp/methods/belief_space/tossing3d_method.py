"""CLI-facing Tossing3D method backed by situated belief-space expectimax."""

import time
from pathlib import Path
from typing import Any

from pydantic import Field, PrivateAttr

from hitl_pmp.core.log_timing import LogTiming
from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.methods.practice_makes_perfect.ees_method import (
    STOP_SKILL,
    EesMethod,
)
from hitl_pmp.planning.grounding import SkillGrounder

from .expectimax import solve_belief_space_expectimax
from .tossing3d_model import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    RESET_SKILL,
    TOSS_SKILL,
    Tossing3DAction,
    Tossing3DBeliefState,
    Tossing3DEnvironmentState,
    Tossing3DPracticeModel,
    Tossing3DSearchState,
    make_default_tossing3d_belief,
)
from .types import STOP_ACTION, SearchTrace


class Tossing3DPomdpMethod(EesMethod):
    """EES learner/executor with situated belief-space practice decisions."""

    pomdp_horizon: int = Field(default=3, ge=0)
    pomdp_num_samples: int = Field(default=100, ge=1)
    pomdp_practice_cost: float = Field(default=0.001, ge=0.0, allow_inf_nan=False)
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
            PICK_SKILL + " (fixed)": self._pomdp_model.pick_competence,
            TOSS_SKILL + " (belief mean)": self._pomdp_state.toss_belief.mean_competence,
            OPEN_GRIPPER_SKILL + " (fixed)": 1.0,
        }
        if self._pomdp_model.reset_cost is not None:
            estimates[RESET_SKILL + " (fixed)"] = 1.0
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

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        self._pomdp_state = make_default_tossing3d_belief()
        self._pomdp_model = Tossing3DPracticeModel(
            seed=self.seed,
            practice_cost=self.pomdp_practice_cost,
            exploration_epsilon=self.exploration_epsilon,
            reset_cost=self.ask_for_reset_cube_bin_cost,
        )
        available = {skill.name for skill in self.skills()}
        missing = set(self._pomdp_model.required_skills) - available
        if missing:
            raise ValueError(
                "Tossing3DPomdpMethod requires canonical Tossing3D skills; missing "
                f"{sorted(missing)} from {sorted(available)}"
            )

    @property
    def pomdp_state(self) -> Tossing3DBeliefState:
        return self._pomdp_state

    def observe_outcome(
        self, *, ground_skill: GroundSkill, success: bool, was_random_exploration: bool = False
    ) -> None:
        super().observe_outcome(
            ground_skill=ground_skill,
            success=success,
            was_random_exploration=was_random_exploration,
        )
        name = ground_skill.skill.name
        if name == TOSS_SKILL:
            self._pomdp_state = self._pomdp_model.observe_toss(
                state=self._pomdp_state,
                success=success,
                was_random_exploration=was_random_exploration,
            )
        self.record_diagnostic(
            event="outcome",
            skill=name,
            success=success,
            random_exploration=was_random_exploration,
            belief=self._pomdp_state.model_dump(mode="json"),
        )

    def record_action_cost(self, *, ground_skill: GroundSkill) -> None:
        """Charge each attempted action immediately, including a final-step reset."""
        action_cost = {
            PICK_SKILL: self._pomdp_model.pick_cost,
            TOSS_SKILL: self._pomdp_model.toss_cost,
            OPEN_GRIPPER_SKILL: self._pomdp_model.open_gripper_cost,
            RESET_SKILL: self._pomdp_model.reset_cost,
        }.get(ground_skill.skill.name)
        if action_cost is not None:
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
        if skill_name == TOSS_SKILL:
            self._pomdp_state = self._pomdp_model.record_training_example(state=self._pomdp_state)

    def end_cycle(self) -> None:
        """Apply latent improvement only after the real sampler has been refit."""
        self.observe_environment_reset(state=self.env.get_current_state())
        super().end_cycle()
        self._pomdp_state = self._pomdp_state.after_refit()
        self.record_diagnostic(event="refit", belief=self._pomdp_state.model_dump(mode="json"))
        self._cycle_index += 1

    @staticmethod
    def _environment_state(*, true_atoms: frozenset[GroundAtom]) -> Tossing3DEnvironmentState:
        predicates = {atom.predicate.name for atom in true_atoms}
        hand_empty = "HandEmpty" in predicates
        if "InBin" in predicates:
            return (
                Tossing3DEnvironmentState.SOLVED
                if hand_empty
                else Tossing3DEnvironmentState.CLOSED_SOLVED
            )
        if "Holding" in predicates:
            return (
                Tossing3DEnvironmentState.HOLDING
                if "Reachable" in predicates
                else Tossing3DEnvironmentState.UNREACHABLE_HOLDING
            )
        if {"Reachable", "OnGround", "HandEmpty"} <= predicates:
            return Tossing3DEnvironmentState.READY
        if not hand_empty:
            return (
                Tossing3DEnvironmentState.GRIPPER_CLOSED
                if {"Reachable", "OnGround"} <= predicates
                else Tossing3DEnvironmentState.CLOSED_STRANDED
            )
        return Tossing3DEnvironmentState.STRANDED

    def select_skill_to_practice(self, *, true_atoms: frozenset[GroundAtom]) -> list[GroundSkill]:
        self._decision_index += 1
        environment_state = self._environment_state(true_atoms=true_atoms)
        self._pomdp_state = self._pomdp_state.model_copy(
            update={"environment_state": environment_state}
        )
        trace = SearchTrace() if self.decision_log is not None else None
        search_started_at = time.perf_counter()
        value, action = solve_belief_space_expectimax(
            environment_state=Tossing3DSearchState(state=self._pomdp_state),
            belief_state=self._pomdp_state,
            summed_cost=self._pomdp_state.accumulated_cost,
            horizon=self.pomdp_horizon,
            model=self._pomdp_model,
            trace=trace,
            num_samples=self.pomdp_num_samples,
        )
        search_duration_seconds = time.perf_counter() - search_started_at
        self._practice_values = {}
        if trace is not None:
            for event in trace.events:
                if event["node"] == 0 and event["event"] == "stop_value":
                    self._practice_values["STOP"] = event["value"]
                elif event["node"] == 0 and event["event"] == "action_value":
                    self._practice_values[event["action"]["name"]] = event["value"]
        self.record_diagnostic(
            event="decision",
            competences=self.practice_skill_competences(),
            search_duration_seconds=search_duration_seconds,
            num_samples=self.pomdp_num_samples,
            atoms=sorted(str(atom) for atom in true_atoms),
            action="STOP" if action == STOP_ACTION else action.model_dump(mode="json"),
            value=value,
            horizon=self.pomdp_horizon,
            model=self._pomdp_model.model_dump(mode="json"),
            search=[] if trace is None else trace.events,
        )
        if action == STOP_ACTION:
            return [STOP_SKILL]
        assert isinstance(action, Tossing3DAction)

        skills = self.skills()
        if self.ask_for_reset_cube_bin_cost is not None:
            reset = self.skill_provider.human_cube_bin_reset_skill()
            assert reset is not None
            skills = (*skills, reset.skill)
        groundings = SkillGrounder.applicable_ground_skills(
            skills=skills,
            objects=self.objects(),
            true_atoms=true_atoms,
        )
        candidates = [grounding for grounding in groundings if grounding.skill.name == action.name]
        if not candidates:
            raise RuntimeError(
                f"POMDP selected inapplicable Tossing3D skill {action.name!r} in "
                f"{environment_state.value}"
            )
        return candidates
