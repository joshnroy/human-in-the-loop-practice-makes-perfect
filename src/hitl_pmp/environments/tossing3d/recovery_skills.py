"""Same-side operators: EES plans retrieval from the observed landing state.

This model is opt-in with the same-side scene. It does not change EES's target
selection, invoke a human reset, or prescribe a pick/throw loop.
"""

from typing import ClassVar

import numpy as np

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.core.problem.tasks.types import Predicate
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    HAND_EMPTY,
    HOLDING,
    IN_BIN,
    ON_GROUND,
    REACHABLE,
    Tossing3DAtoms,
)
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills

ON_BIN_RIM = Predicate(
    name="OnBinRim",
    types=(Tossing3DEnvironment.cube_type, Tossing3DEnvironment.bin_type),
    holds=lambda state, objects: Tossing3DAtoms.holds(
        state=state, name="OnBinRim", objects=objects
    ),
)

ON_FLOOR = Predicate(
    name="OnFloor",
    types=(Tossing3DEnvironment.cube_type, Tossing3DEnvironment.bin_type),
    holds=lambda state, objects: (
        ON_GROUND.holds(state, (objects[0],)) and not IN_BIN.holds(state, objects)
    ),
)
CLOSED_EMPTY = Predicate(
    name="ClosedEmpty",
    types=(Tossing3DEnvironment.robot_type, Tossing3DEnvironment.cube_type),
    holds=lambda state, objects: (
        not HAND_EMPTY.holds(state, (objects[0],)) and not HOLDING.holds(state, objects)
    ),
)


class SameSideSkills:
    """Operators for floor and bin recovery, with unchanged toss parameter bounds."""

    _robot: ClassVar[Variable] = Variable(name="robot", type=Tossing3DEnvironment.robot_type)
    _cube: ClassVar[Variable] = Variable(name="cube", type=Tossing3DEnvironment.cube_type)
    _bin: ClassVar[Variable] = Variable(name="bin", type=Tossing3DEnvironment.bin_type)
    _barrier: ClassVar[Variable] = Variable(name="barrier", type=Tossing3DEnvironment.barrier_type)
    _empty: ClassVar[LiftedAtom] = LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,))
    _held: ClassVar[LiftedAtom] = LiftedAtom(predicate=HOLDING, variables=(_robot, _cube))
    _floor: ClassVar[LiftedAtom] = LiftedAtom(predicate=ON_FLOOR, variables=(_cube, _bin))
    _inside: ClassVar[LiftedAtom] = LiftedAtom(predicate=IN_BIN, variables=(_cube, _bin))
    _reachable: ClassVar[LiftedAtom] = LiftedAtom(predicate=REACHABLE, variables=(_cube, _barrier))
    _closed: ClassVar[LiftedAtom] = LiftedAtom(predicate=CLOSED_EMPTY, variables=(_robot, _cube))

    _rim: ClassVar[LiftedAtom] = LiftedAtom(predicate=ON_BIN_RIM, variables=(_cube, _bin))

    PICK_RIM: ClassVar[Skill] = Skill(
        name="PickCubeFromRim",
        parameters=(_robot, _cube, _bin, _barrier),
        preconditions=frozenset({_empty, _rim, _reachable}),
        add_effects=frozenset({_held}),
        delete_effects=frozenset({_empty, _rim}),
        param_dim=0,
    )

    PICK_FLOOR: ClassVar[Skill] = Skill(
        name="PickCubeFromFloor",
        parameters=(_robot, _cube, _bin, _barrier),
        preconditions=frozenset({_empty, _floor, _reachable}),
        add_effects=frozenset({_held}),
        delete_effects=frozenset({_empty, _floor}),
        param_dim=0,
    )
    PICK_BIN: ClassVar[Skill] = Skill(
        name="PickCubeFromBin",
        parameters=(_robot, _cube, _bin, _barrier),
        preconditions=frozenset({_empty, _inside, _reachable}),
        add_effects=frozenset({_held}),
        delete_effects=frozenset({_empty, _inside}),
        param_dim=0,
    )
    TOSS: ClassVar[Skill] = Skill(
        name="MoveToTossLocationAndToss",
        parameters=(_robot, _bin, _cube, _barrier),
        preconditions=frozenset({_held, _reachable}),
        add_effects=frozenset({_empty, _inside, _reachable}),
        delete_effects=frozenset({_held, _floor}),
        param_dim=4,
    )
    OPEN: ClassVar[Skill] = Skill(
        name="OpenGripper",
        parameters=(_robot, _cube),
        preconditions=frozenset({_closed}),
        add_effects=frozenset({_empty}),
        delete_effects=frozenset({_closed}),
        param_dim=0,
    )

    @staticmethod
    def skills() -> tuple[Skill, ...]:
        return (
            SameSideSkills.PICK_FLOOR,
            SameSideSkills.PICK_BIN,
            SameSideSkills.PICK_RIM,
            SameSideSkills.TOSS,
            SameSideSkills.OPEN,
        )

    @staticmethod
    def sample_params(*, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        if ground_skill.skill == SameSideSkills.TOSS:
            original = GroundSkill(
                skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS, objects=ground_skill.objects
            )
            return Tossing3DSkills.sample_params(ground_skill=original, rng=rng)
        if ground_skill.skill in SameSideSkills.skills():
            return np.zeros(0)
        raise ValueError(f"Unknown skill: {ground_skill.skill.name}")

    @staticmethod
    def compute_action(*, ground_skill: GroundSkill, params: np.ndarray, state: State) -> Action:
        if ground_skill.skill == SameSideSkills.TOSS:
            original = GroundSkill(
                skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS, objects=ground_skill.objects
            )
            return Tossing3DSkills.compute_action(ground_skill=original, params=params, state=state)
        ids = {
            SameSideSkills.PICK_FLOOR: Tossing3DEnvironment.pick_cube_id,
            SameSideSkills.PICK_BIN: Tossing3DEnvironment.pick_cube_from_bin_id,
            SameSideSkills.PICK_RIM: Tossing3DEnvironment.pick_cube_from_bin_id,
            SameSideSkills.OPEN: Tossing3DEnvironment.open_gripper_id,
        }
        if ground_skill.skill not in ids:
            raise ValueError(f"Unknown skill: {ground_skill.skill.name}")
        return np.array([ids[ground_skill.skill], 0, 0, 0, 0], dtype=float)
