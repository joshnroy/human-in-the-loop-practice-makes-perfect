from typing import ClassVar

import numpy as np

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Action, State

from .environment import Tossing3DEnvironment
from .predicates import AT_THROW_POSE, HAND_EMPTY, HOLDING, IN_GOAL_REGION, REACHABLE


class Tossing3DSkills:
    """Lifted skill templates for Tossing3D. Preconditions/add_effects/delete_effects
    are LiftedAtoms over each skill's own Variables, in the same shape as Tossing Room's
    and predicators' NSRTs, so EES can task-plan over them with Fast Downward. A
    static-method container, never instantiated, same as every other business-logic
    class in this project.

    The models are kept exactly as strong as the underlying KINDER controllers, no
    more -- Tossing Room's log records what a too-permissive precondition costs (Fast
    Downward emitted plans that picked up in the wrong room, a silent no-op, solving
    1/10 tasks). Two places where that bites here:

    * `Pick` requires `Reachable`. Without it the planner will happily propose
      retrieving a cube from beyond the immovable barrier and tossing it again, which
      the base can never do -- and that plan looks *cheap*, so it would be preferred
      over admitting the task is lost.
    * `Pick` deletes `AtThrowPose`. KINDER's `pick_shelf` drives the base to the cube
      before grasping, so picking genuinely un-does having moved to the throw pose;
      leaving it out lets the planner emit Move, Pick, Toss and skip a real Move.
    """

    _robot: ClassVar[Variable] = Variable(name="robot", type=Tossing3DEnvironment.robot_type)
    _cube: ClassVar[Variable] = Variable(name="cube", type=Tossing3DEnvironment.cube_type)
    _bin: ClassVar[Variable] = Variable(name="bin", type=Tossing3DEnvironment.bin_type)
    _barrier: ClassVar[Variable] = Variable(name="barrier", type=Tossing3DEnvironment.barrier_type)
    _region: ClassVar[Variable] = Variable(name="region", type=Tossing3DEnvironment.region_type)

    PICK: ClassVar[Skill] = Skill(
        name="Pick",
        parameters=(_robot, _cube, _barrier, _bin),
        preconditions=frozenset({
            LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,)),
            LiftedAtom(predicate=REACHABLE, variables=(_cube, _barrier)),
        }),
        add_effects=frozenset({LiftedAtom(predicate=HOLDING, variables=(_robot, _cube))}),
        delete_effects=frozenset({
            LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,)),
            # pick_shelf navigates the base to the cube, so a pick really does leave
            # the throw pose.
            LiftedAtom(predicate=AT_THROW_POSE, variables=(_robot, _bin)),
        }),
        # KINDER's own PickShelfController.sample_parameters draws exactly these two:
        # the base standoff distance from the cube and the base rotation offset.
        param_dim=2,
    )
    MOVE_TO_THROW_POSE: ClassVar[Skill] = Skill(
        name="MoveToThrowPose",
        parameters=(_robot, _bin),
        preconditions=frozenset(),
        add_effects=frozenset({LiftedAtom(predicate=AT_THROW_POSE, variables=(_robot, _bin))}),
        delete_effects=frozenset(),
        param_dim=0,
    )
    TOSS: ClassVar[Skill] = Skill(
        name="Toss",
        parameters=(_robot, _cube, _bin, _region, _barrier),
        preconditions=frozenset({
            LiftedAtom(predicate=HOLDING, variables=(_robot, _cube)),
            LiftedAtom(predicate=AT_THROW_POSE, variables=(_robot, _bin)),
        }),
        add_effects=frozenset({
            LiftedAtom(predicate=IN_GOAL_REGION, variables=(_cube, _region)),
            LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,)),
        }),
        # The cube leaves the hand whatever happens, and it crosses the barrier
        # whatever happens -- a toss is what makes the cube unreachable, whether or
        # not it lands in the region. Declaring the Reachable delete is what makes the
        # planner's model of a *failed* toss honest rather than optimistic.
        delete_effects=frozenset({
            LiftedAtom(predicate=HOLDING, variables=(_robot, _cube)),
            LiftedAtom(predicate=REACHABLE, variables=(_cube, _barrier)),
        }),
        param_dim=1,
    )

    @staticmethod
    def sample_params(
        *, ground_skill: GroundSkill, rng: np.random.Generator, env: Tossing3DEnvironment
    ) -> np.ndarray:
        """The uniform prior a learned sampler starts from and improves on.

        `Pick` draws over KINDER's own `MOVE_TO_TARGET_DISTANCE_BOUNDS`/
        `MOVE_TO_TARGET_ROT_BOUNDS`, i.e. the same range `PickShelfController` itself
        randomizes; `Toss` draws its swing dial over a band deliberately wider than
        the one that reaches the goal region, so an unpracticed policy does not
        already succeed most of the time.
        """
        skill = ground_skill.skill
        if skill == Tossing3DSkills.PICK:
            return np.array([
                rng.uniform(env.pick_distance_low, env.pick_distance_high),
                rng.uniform(env.pick_rot_low, env.pick_rot_high),
            ])
        if skill == Tossing3DSkills.TOSS:
            return np.array([rng.uniform(env.swing_low, env.swing_high)])
        return np.empty(0)

    @staticmethod
    def compute_action(*, ground_skill: GroundSkill, params: np.ndarray, state: State) -> Action:
        """The lifted "option policy" layer: turn a chosen (ground skill, params) into
        one raw `[skill_id, param0, param1]` action. Dispatches by Skill value equality
        (not identity), so a Method that reconstructs an equal-content Skill still
        works."""
        del state  # every parameter this domain needs is in `params` already
        skills = Tossing3DSkills
        env = Tossing3DEnvironment
        skill = ground_skill.skill

        if skill == skills.PICK:
            return np.array([float(env.SKILL_PICK), float(params[0]), float(params[1])])
        if skill == skills.MOVE_TO_THROW_POSE:
            return np.array([float(env.SKILL_MOVE_TO_THROW_POSE), 0.0, 0.0])
        if skill == skills.TOSS:
            return np.array([float(env.SKILL_TOSS), float(params[0]), 0.0])

        raise ValueError(f"Unknown skill: {skill.name}")
