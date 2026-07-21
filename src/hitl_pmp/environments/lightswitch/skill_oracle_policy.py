from typing import Any

import numpy as np

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Policy, Rollout, SetupCommand
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task

from .environment import LightSwitchEnvironment
from .skills import LightSwitchSkills


class SkillOraclePolicy:
    """Cheats with privileged ground-truth state -- routes every action through
    skills.py's lifted -> grounded -> compute_action pipeline (unlike
    ActionOraclePolicy, which acts directly in raw action space). Domain-agnostic
    at the get_labeled_action entrypoint: it dispatches on Problem.env (the
    currently-wired Environment -- see core/README.md's Problem facade section)
    to whichever domain-specific get_<domain>_labeled_action implements that
    domain's own oracle logic. Only Light Switch exists so far, so there's only
    one branch -- a second domain would add its own get_<domain>_labeled_action
    plus one more branch here, not a second SkillOraclePolicy-like class. A
    static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def get_labeled_action(*, state: State) -> LabeledAction:
        if Problem.env is LightSwitchEnvironment:
            return SkillOraclePolicy.get_lightswitch_labeled_action(state=state)
        raise NotImplementedError(
            f"SkillOraclePolicy has no oracle logic for Problem.env={Problem.env!r} yet."
        )

    @staticmethod
    def get_lightswitch_labeled_action(*, state: State) -> LabeledAction:
        """MoveRobot to the light's cell, then TurnOnLight by exactly the
        remaining gap -- same two-action solve as ActionOraclePolicy; the point
        of this method is to exercise Skill/GroundSkill end-to-end with a real
        caller, not to behave differently."""
        env = LightSwitchEnvironment
        robot, light = env.robot, env.light
        robot_x = state.get(obj=robot, feature_name="x")
        light_x = state.get(obj=light, feature_name="x")

        cells = env.get_cells()
        light_cell = cells[-1]  # matches build_initial_state's light placement

        if abs(robot_x - light_x) >= env.same_position_tolerance:
            # MoveRobot's compute_action only reads the target cell (and robot's own
            # position) -- current_cell is part of the skill's signature purely for
            # fidelity with predicators' MoveRobot(robot, current_cell, target_cell),
            # so any nearest-cell estimate is fine here.
            current_cell = min(
                cells, key=lambda cell: abs(state.get(obj=cell, feature_name="x") - robot_x)
            )
            ground_skill = GroundSkill(
                skill=LightSwitchSkills.MOVE_ROBOT, objects=(robot, current_cell, light_cell)
            )
            params: np.ndarray = np.zeros(0)
        else:
            level = state.get(obj=light, feature_name="level")
            target = state.get(obj=light, feature_name="target")
            ground_skill = GroundSkill(
                skill=LightSwitchSkills.TURN_ON_LIGHT, objects=(robot, light_cell, light)
            )
            params = np.array([target - level])

        action = LightSwitchSkills.compute_action(
            ground_skill=ground_skill, params=params, state=state
        )
        objects_desc = ", ".join(obj.name for obj in ground_skill.objects)
        label = f"{ground_skill.skill.name}({objects_desc})"
        if params.size > 0:
            # Show every entry in params, not just params[0] -- params has no name
            # of its own at the GroundSkill/compute_action level (it's a bare
            # ndarray, deliberately skill-agnostic), so this must stay generic over
            # param_dim rather than assuming a single scalar, or a future skill with
            # param_dim > 1 would silently lose everything past the first entry.
            rounded_params = [round(float(p), 2) for p in params]
            label += f", params={rounded_params}"
        return LabeledAction(action=action, label=label)


class SkillOracleMethod(Method):
    """Wraps SkillOraclePolicy as a core.Method, so it runs through the same
    practice_loop.py:PracticeLoop harness as every other Method -- see
    ActionOracleMethod (action_oracle_policy.py) for the identical reasoning;
    this differs only in which oracle policy get_task_policy wraps. A
    static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def reset_environment(*, start_state: State) -> bool:
        """No irreversible actions exist in Light Switch and the base PMP paper
        has no human-in-the-loop layer at all -- this reproduction never needs a
        real "self-navigate without help" recovery, so a direct environment set
        stands in for it (matches RandomSkillsMethod's own reasoning)."""
        LightSwitchEnvironment.set_state(state=start_state)
        return True

    @staticmethod
    def get_task_policy(*, task: Task) -> Policy:
        del task  # never consulted -- this oracle always drives toward the
        # light using privileged state, regardless of which task it's handed
        return lambda state: SkillOraclePolicy.get_labeled_action(state=state)

    @staticmethod
    def generate_train_task(*, tbd_inputs: Any) -> Task:
        raise NotImplementedError(
            "SkillOracleMethod.generate_train_task is unreachable: this oracle never practices."
        )

    @staticmethod
    def execute_setup_command(*, setup_command: SetupCommand) -> None:
        raise NotImplementedError(
            "SkillOracleMethod.execute_setup_command is unreachable: "
            "no HumanOracle is ever used in this reproduction."
        )

    @staticmethod
    def execute_skill(*, skill: GroundSkill) -> Rollout:
        raise NotImplementedError(
            "SkillOracleMethod.execute_skill is unreachable: this oracle "
            "computes its own ground skill choice directly, it never practices one."
        )

    @staticmethod
    def improve_skill_parameters(*, skill: GroundSkill, rollout: Rollout) -> None:
        raise NotImplementedError(
            "SkillOracleMethod.improve_skill_parameters is unreachable: this oracle never learns."
        )
