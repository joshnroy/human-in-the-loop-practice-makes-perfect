from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.types import (
    GroundSkill,
    LabeledAction,
    Policy,
    Rollout,
    SetupCommand,
    Skill,
)
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State
from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Predicate, Task
from hitl_pmp.planning.grounding import SkillGrounder


class RandomSkillsMethod(Method):
    """ "Random Skills" (Figure 4 of the paper): no planning anywhere, not even
    at evaluation time -- ported from predicators' RandomNSRTsExplorer. Each
    step, uniformly samples one currently-applicable GroundSkill (via
    SkillGrounder) and executes it with its own base sampler's params. Never
    pursues a task's actual goal, so it essentially never solves anything --
    matching the paper's own near-0% Random Skills curve, the same flat
    baseline MAPLE-Q also produces (for a different reason: MAPLE-Q *does*
    try to plan/act toward the goal, it just fails to learn to from so few
    online transitions).

    A static-method container, never instantiated, same as every other
    business-logic class in this project. ClassVars are configured once per
    domain (see environments/lightswitch/cli.py's eventual wiring) --
    reset_state() reseeds rng before each (method, seed) run."""

    env: ClassVar[type[Environment]]
    tasks: ClassVar[type[Tasks]]
    predicates: ClassVar[tuple[Predicate, ...]]
    skills: ClassVar[tuple[Skill, ...]]
    objects: ClassVar[tuple[Object, ...]]
    compute_action: ClassVar[Callable[..., Action]]
    sample_params: ClassVar[Callable[..., np.ndarray]]
    rng: ClassVar[np.random.Generator]

    @staticmethod
    def reset_state(*, seed: int) -> None:
        RandomSkillsMethod.rng = np.random.default_rng(seed)

    @staticmethod
    def reset_environment(*, start_state: State) -> bool:
        """No irreversible actions exist in Light Switch and the base PMP paper
        has no human-in-the-loop layer at all -- this reproduction never
        needs a real "self-navigate without help" recovery, so a direct
        environment set stands in for it."""
        RandomSkillsMethod.env.set_state(state=start_state)
        return True

    @staticmethod
    def get_task_policy(*, task: Task) -> Policy:
        del task  # never consulted -- this Method has no goal-directed behavior at all
        return lambda state: RandomSkillsMethod._act(state=state)

    @staticmethod
    def _act(*, state: State) -> LabeledAction:
        true_atoms = SkillGrounder.abstract_state(
            state=state,
            objects=RandomSkillsMethod.objects,
            predicates=RandomSkillsMethod.predicates,
        )
        applicable = SkillGrounder.applicable_ground_skills(
            skills=RandomSkillsMethod.skills,
            objects=RandomSkillsMethod.objects,
            true_atoms=true_atoms,
        )
        # In Light Switch, MoveRobot is applicable from every valid robot position
        # (even a boundary cell has one neighbor), so `applicable` is never empty --
        # not a general guarantee for an arbitrary domain's operators.
        ground_skill = applicable[RandomSkillsMethod.rng.integers(len(applicable))]
        params = RandomSkillsMethod.sample_params(
            ground_skill=ground_skill, rng=RandomSkillsMethod.rng
        )
        action = RandomSkillsMethod.compute_action(
            ground_skill=ground_skill, params=params, state=state
        )
        objects_desc = ", ".join(obj.name for obj in ground_skill.objects)
        return LabeledAction(action=action, label=f"{ground_skill.skill.name}({objects_desc})")

    @staticmethod
    def generate_train_task(*, tbd_inputs: Any) -> Task:
        del tbd_inputs
        return RandomSkillsMethod.tasks.sample_train_task()

    @staticmethod
    def execute_setup_command(*, setup_command: SetupCommand) -> None:
        """Unreachable in this reproduction: PMP-family methods never use a
        HumanOracle (Light Switch has no irreversible action), so nothing ever
        constructs a SetupCommand targeting this Method."""
        raise NotImplementedError(
            "RandomSkillsMethod.execute_setup_command is unreachable in this reproduction"
        )

    @staticmethod
    def execute_skill(*, skill: GroundSkill) -> Rollout:
        state_before = RandomSkillsMethod.env.get_current_state()
        params = RandomSkillsMethod.sample_params(ground_skill=skill, rng=RandomSkillsMethod.rng)
        action = RandomSkillsMethod.compute_action(
            ground_skill=skill, params=params, state=state_before
        )
        state_after = RandomSkillsMethod.env.take_action(action=action)
        return Rollout(states=[state_before, state_after], actions=[action])

    @staticmethod
    def improve_skill_parameters(*, skill: GroundSkill, rollout: Rollout) -> None:
        """A no-op: Random Skills never learns anything from its own
        executions (no competence model, no wrapped sampler) -- matching
        predicators' RandomNSRTsExplorer, which does no sampler learning."""
        del skill, rollout
