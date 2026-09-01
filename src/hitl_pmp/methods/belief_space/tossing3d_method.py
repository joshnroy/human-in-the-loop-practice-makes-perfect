"""CLI-facing Tossing3D method backed by situated belief-space expectimax."""

from pydantic import Field, PrivateAttr

from hitl_pmp.core.method.types import GroundSkill, Policy
from hitl_pmp.core.problem.tasks.types import GroundAtom, Task
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
    Tossing3DPracticeModel,
    Tossing3DSearchState,
    make_default_tossing3d_belief,
)
from .types import STOP_ACTION


class Tossing3DPomdpMethod(EesMethod):
    """EES learner/executor with situated belief-space practice decisions."""

    pomdp_horizon: int = Field(default=3, ge=0)
    pomdp_num_samples: int = Field(default=100, ge=1)
    pomdp_hard_budget: int | None = Field(default=None, ge=0)
    pomdp_practice_cost: float = Field(default=0.001, ge=0.0, allow_inf_nan=False)
    goal_pursuit_horizon: int | None = 0

    _pomdp_state: Tossing3DBeliefState = PrivateAttr()
    _pomdp_model: Tossing3DPracticeModel = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        self._pomdp_state = make_default_tossing3d_belief()
        practice_skills = self.skills()
        if self.ask_for_reset_cube_bin_cost is not None:
            reset = self.skill_provider.human_cube_bin_reset_skill()
            assert reset is not None
            practice_skills = (*practice_skills, reset.skill)
        ground_skills = SkillGrounder.applicable_ground_skills(
            skills=practice_skills,
            objects=self.objects(),
            true_atoms=SkillGrounder.all_possible_ground_atoms(
                objects=self.objects(), predicates=self.predicates()
            ),
        )
        self._pomdp_model = Tossing3DPracticeModel(
            seed=self.seed,
            practice_cost=self.pomdp_practice_cost,
            exploration_epsilon=self.exploration_epsilon,
            reset_cost=self.ask_for_reset_cube_bin_cost,
            hard_budget=self.pomdp_hard_budget,
            ground_skills=tuple(ground_skills),
        )
        available = {ground_skill.skill.name for ground_skill in ground_skills}
        missing = {PICK_SKILL, TOSS_SKILL, OPEN_GRIPPER_SKILL} - available
        if missing:
            raise ValueError(
                "Tossing3DPomdpMethod requires canonical Tossing3D skills; missing "
                f"{sorted(missing)} from {sorted(available)}"
            )

    @property
    def pomdp_state(self) -> Tossing3DBeliefState:
        return self._pomdp_state

    def get_practice_policy(self, *, task: Task) -> Policy:
        """Start a fresh session budget without resetting the learned skill state."""
        self._pomdp_state = self._pomdp_state.model_copy(update={"accumulated_cost": 0.0})
        return super().get_practice_policy(task=task)

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
        elif name in {PICK_SKILL, OPEN_GRIPPER_SKILL}:
            self._pomdp_state = self._pomdp_model.observe_robot_skill(
                state=self._pomdp_state, skill_name=name, success=success
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
        """Advance inferred learning curves at the session boundary.

        For parameter-free skills this is a forecast, not a controller update.
        Subsequent outcomes reweight improving versus stationary hypotheses.
        """
        self.observe_environment_reset(state=self.env.get_current_state())
        super().end_cycle()
        self._pomdp_state = self._pomdp_state.after_refit()

    def select_skill_to_practice(self, *, true_atoms: frozenset[GroundAtom]) -> list[GroundSkill]:
        _, action = solve_belief_space_expectimax(
            environment_state=Tossing3DSearchState(state=self._pomdp_state, true_atoms=true_atoms),
            belief_state=self._pomdp_state,
            summed_cost=self._pomdp_state.accumulated_cost,
            horizon=self.pomdp_horizon,
            model=self._pomdp_model,
            num_samples=self.pomdp_num_samples,
        )
        if action == STOP_ACTION:
            return [STOP_SKILL]
        assert isinstance(action, Tossing3DAction)

        self.record_practice_target(name=action.ground_skill.skill.name, field="scored")
        return [action.ground_skill]
