import numpy as np

from hitl_pmp.core.method.types import GroundSkill, LabeledAction
from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.core.problem.tasks.types import Goal

from .environment import TossingRoomEnvironment
from .predicates import BIN_EMPTY, ITEM_IN_BIN
from .skills import TossingRoomSkills


class SkillOraclePolicy:
    """Cheats with privileged ground-truth state AND the task goal -- routes every
    action through skills.py's lifted -> grounded -> compute_action pipeline. Unlike
    Light Switch's oracle, this one needs the goal: from state alone it cannot tell a
    throw-recycling task from a throw-trash task (both start with empty bins, the
    robot at start, an empty hand), so which item/bin/room to head for is read off the
    goal's own atoms. A static-method container, never instantiated, same as every
    other business-logic class in this project.

    Forward-only by construction: recycling sits LEFT of start (down across the
    reversible-leftward ledge), trash/button sit RIGHT of start, and every solve is
    pick -> step toward the target room -> act. The oracle therefore never issues the
    one blocked rightward ledge step, so it never needs human help to recover."""

    @staticmethod
    def get_labeled_action(
        *, state: State, env: TossingRoomEnvironment, goal: Goal
    ) -> LabeledAction:
        rooms = env.get_rooms()
        robot_room = int(round(state.get(obj=env.robot, feature_name="room")))

        if any(atom.predicate == BIN_EMPTY for atom in goal.atoms):
            ground_skill, params = SkillOraclePolicy._empty_step(
                env=env, rooms=rooms, robot_room=robot_room
            )
        else:
            item = SkillOraclePolicy._goal_item(goal=goal)
            ground_skill, params = SkillOraclePolicy._throw_step(
                state=state, env=env, rooms=rooms, robot_room=robot_room, item=item
            )
        return SkillOraclePolicy._to_labeled_action(
            ground_skill=ground_skill, params=params, state=state
        )

    @staticmethod
    def _goal_item(*, goal: Goal) -> Object:
        """The item named by the goal's ItemInBin atom (objects = (item, bin))."""
        item_in_bin = next(atom for atom in goal.atoms if atom.predicate == ITEM_IN_BIN)
        return item_in_bin.objects[0]

    @staticmethod
    def _empty_step(
        *, env: TossingRoomEnvironment, rooms: tuple[Object, ...], robot_room: int
    ) -> tuple[GroundSkill, np.ndarray]:
        if robot_room != env.button_room:
            return SkillOraclePolicy._move_toward(
                rooms=rooms, robot_room=robot_room, target_room=env.button_room
            )
        ground_skill = GroundSkill(
            skill=TossingRoomSkills.PRESS,
            objects=(env.robot, env.button, rooms[robot_room], env.recycling_bin, env.trash_bin),
        )
        return ground_skill, np.zeros(0)

    @staticmethod
    def _throw_step(
        *,
        state: State,
        env: TossingRoomEnvironment,
        rooms: tuple[Object, ...],
        robot_room: int,
        item: Object,
    ) -> tuple[GroundSkill, np.ndarray]:
        kind = int(round(state.get(obj=item, feature_name="kind")))
        holding = int(round(state.get(obj=env.robot, feature_name="holding")))
        bin_obj = env.bin_for_kind(kind=kind)
        bin_room = env.bin_room_for_kind(kind=kind)

        if holding != kind:
            # Pick the item up from the pile at start (where the robot begins).
            ground_skill = GroundSkill(
                skill=TossingRoomSkills.PICKUP,
                # ?bin is the item's own bin: Pickup's model deletes ItemInBin for the
                # item it replaces, and BinAcceptsItem pins the binding.
                objects=(env.robot, item, rooms[robot_room], env.pile, bin_obj),
            )
            return ground_skill, np.zeros(0)
        if robot_room != bin_room:
            return SkillOraclePolicy._move_toward(
                rooms=rooms, robot_room=robot_room, target_room=bin_room
            )
        # In the bin's room, holding the item: throw with force set to the known target.
        target = state.get(obj=item, feature_name="target_force")
        ground_skill = GroundSkill(
            skill=TossingRoomSkills.THROW, objects=(env.robot, item, bin_obj, rooms[robot_room])
        )
        return ground_skill, np.array([target])

    @staticmethod
    def _move_toward(
        *, rooms: tuple[Object, ...], robot_room: int, target_room: int
    ) -> tuple[GroundSkill, np.ndarray]:
        step = 1 if target_room > robot_room else -1
        to_room = robot_room + step
        ground_skill = GroundSkill(
            skill=TossingRoomSkills.MOVE_ROOM,
            objects=(TossingRoomEnvironment.robot, rooms[robot_room], rooms[to_room]),
        )
        return ground_skill, np.zeros(0)

    @staticmethod
    def _to_labeled_action(
        *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> LabeledAction:
        action = TossingRoomSkills.compute_action(
            ground_skill=ground_skill, params=params, state=state
        )
        objects_desc = ", ".join(obj.name for obj in ground_skill.objects)
        label = f"{ground_skill.skill.name}({objects_desc})"
        if params.size > 0:
            rounded_params = [round(float(p), 2) for p in params]
            label += f", params={rounded_params}"
        return LabeledAction(action=action, label=label)
