from typing import ClassVar

import numpy as np

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Action, State

from .environment import LightSwitchEnvironment
from .predicates import ADJACENT, LIGHT_IN_CELL, LIGHT_OFF, LIGHT_ON, ROBOT_IN_CELL


class LightSwitchSkills:
    """Lifted skill templates for Light Switch, ported from predicators'
    ground_truth_models/grid_row/{options,nsrts}.py -- preconditions/add_effects/
    delete_effects mirror nsrts.py's four NSRTs exactly, so planning/ can
    task-plan over these the same way predicators task-plans over NSRTs (via Fast
    Downward). A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    _robot: ClassVar[Variable] = Variable(name="robot", type=LightSwitchEnvironment.robot_type)
    _current_cell: ClassVar[Variable] = Variable(
        name="current_cell", type=LightSwitchEnvironment.cell_type
    )
    _target_cell: ClassVar[Variable] = Variable(
        name="target_cell", type=LightSwitchEnvironment.cell_type
    )
    _light: ClassVar[Variable] = Variable(name="light", type=LightSwitchEnvironment.light_type)
    _cell1: ClassVar[Variable] = Variable(name="cell1", type=LightSwitchEnvironment.cell_type)
    _cell2: ClassVar[Variable] = Variable(name="cell2", type=LightSwitchEnvironment.cell_type)
    _cell3: ClassVar[Variable] = Variable(name="cell3", type=LightSwitchEnvironment.cell_type)

    MOVE_ROBOT: ClassVar[Skill] = Skill(
        name="MoveRobot",
        parameters=(_robot, _current_cell, _target_cell),
        preconditions=frozenset({
            LiftedAtom(predicate=ADJACENT, variables=(_current_cell, _target_cell)),
            LiftedAtom(predicate=ROBOT_IN_CELL, variables=(_robot, _current_cell)),
        }),
        add_effects=frozenset({
            LiftedAtom(predicate=ROBOT_IN_CELL, variables=(_robot, _target_cell))
        }),
        delete_effects=frozenset({
            LiftedAtom(predicate=ROBOT_IN_CELL, variables=(_robot, _current_cell))
        }),
        param_dim=0,
    )
    TURN_ON_LIGHT: ClassVar[Skill] = Skill(
        name="TurnOnLight",
        parameters=(_robot, _current_cell, _light),
        preconditions=frozenset({
            LiftedAtom(predicate=LIGHT_IN_CELL, variables=(_light, _current_cell)),
            LiftedAtom(predicate=ROBOT_IN_CELL, variables=(_robot, _current_cell)),
            LiftedAtom(predicate=LIGHT_OFF, variables=(_light,)),
        }),
        add_effects=frozenset({LiftedAtom(predicate=LIGHT_ON, variables=(_light,))}),
        delete_effects=frozenset({LiftedAtom(predicate=LIGHT_OFF, variables=(_light,))}),
        param_dim=1,
    )
    TURN_OFF_LIGHT: ClassVar[Skill] = Skill(
        name="TurnOffLight",
        parameters=(_robot, _current_cell, _light),
        preconditions=frozenset({
            LiftedAtom(predicate=LIGHT_IN_CELL, variables=(_light, _current_cell)),
            LiftedAtom(predicate=ROBOT_IN_CELL, variables=(_robot, _current_cell)),
            LiftedAtom(predicate=LIGHT_ON, variables=(_light,)),
        }),
        add_effects=frozenset({LiftedAtom(predicate=LIGHT_OFF, variables=(_light,))}),
        delete_effects=frozenset({LiftedAtom(predicate=LIGHT_ON, variables=(_light,))}),
        param_dim=1,
    )
    # theta is ignored by compute_action below (matches predicators' own hardcoded
    # no-op) -- param_dim=1 anyway, purely to mirror the reference implementation's
    # signature; nothing here ever reads the sampled value. Its add_effect
    # (RobotInCell(robot, cell3)) is symbolically claimed by the NSRT but never
    # actually achieved by the option, exactly like predicators' own JumpToLight --
    # this is the "impossible skill" a competence-tracking Method must learn to
    # stop practicing.
    JUMP_TO_LIGHT: ClassVar[Skill] = Skill(
        name="JumpToLight",
        parameters=(_robot, _cell1, _cell2, _cell3, _light),
        preconditions=frozenset({
            LiftedAtom(predicate=ROBOT_IN_CELL, variables=(_robot, _cell1)),
            LiftedAtom(predicate=ADJACENT, variables=(_cell1, _cell2)),
            LiftedAtom(predicate=ADJACENT, variables=(_cell2, _cell3)),
            LiftedAtom(predicate=LIGHT_IN_CELL, variables=(_light, _cell3)),
        }),
        add_effects=frozenset({LiftedAtom(predicate=ROBOT_IN_CELL, variables=(_robot, _cell3))}),
        delete_effects=frozenset({LiftedAtom(predicate=ROBOT_IN_CELL, variables=(_robot, _cell1))}),
        param_dim=1,
    )

    @staticmethod
    def sample_params(*, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        """Uniform(-1, 1) per continuous dim, matching predicators' light_sampler --
        every param_dim>0 skill here shares this one sampling distribution,
        regardless of what the sampled value ends up meaning to compute_action."""
        return rng.uniform(-1.0, 1.0, size=ground_skill.skill.param_dim)

    @staticmethod
    def compute_action(*, ground_skill: GroundSkill, params: np.ndarray, state: State) -> Action:
        """The lifted "option policy" layer: reads state (and, for the toggle
        skills, the sampled param) to produce one raw [dx, dlight] action --
        mirrors predicators' _create_move_robot_policy/_toggle_light_policy/
        _jump_to_light_policy."""
        skills = LightSwitchSkills
        skill = ground_skill.skill

        if skill == skills.MOVE_ROBOT:
            robot, _current_cell, target_cell = ground_skill.objects
            dx = state.get(obj=target_cell, feature_name="x") - state.get(
                obj=robot, feature_name="x"
            )
            return np.array([dx, 0.0])

        if skill in (skills.TURN_ON_LIGHT, skills.TURN_OFF_LIGHT):
            dlight = float(params[0])
            return np.array([0.0, dlight])

        if skill == skills.JUMP_TO_LIGHT:
            # The "impossible" skill: predicators hardcodes a no-op regardless of
            # state or params, so a Method can never actually complete a task via
            # this skill's own policy.
            return np.array([0.0, 0.0])

        raise ValueError(f"Unknown skill: {skill.name}")
