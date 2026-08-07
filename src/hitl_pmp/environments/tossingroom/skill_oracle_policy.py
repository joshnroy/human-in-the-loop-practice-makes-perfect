import numpy as np

from hitl_pmp.core.method.types import GroundSkill, LabeledAction
from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.core.problem.tasks.types import Goal

from .environment import TossingRoomEnvironment
from .predicates import RECYCLING_BIN_EMPTY, RECYCLING_IN_BIN, TRASH_BIN_EMPTY
from .skills import TossingRoomSkills


class SkillOraclePolicy:
    """Cheats with privileged ground-truth state AND the task goal -- routes every
    action through skills.py's lifted -> grounded -> compute_action pipeline.

    Goal-DEPENDENT, like Tossing Room's oracle and unlike Light Switch's: from state
    alone it cannot tell a throw-recycling task from a throw-trash task (both start with
    empty bins, the robot at start, an empty hand). Here the goal additionally selects
    *which lifted skill* to use, not just which objects to bind -- `TrashInBin` in the
    goal routes to `PickupTrash`/`ThrowTrash`, `RecyclingInBin` to
    `PickupRecycling`/`ThrowRecycling`. That is the one behavioural difference from
    Tossing Room's oracle, and it is forced by the split.

    Forward-only by construction: the recycling bin and its button sit LEFT of start
    (down across the reversible-leftward ledge), the trash bin and its button sit RIGHT
    of start, and every throw solve is pick -> step toward the target room -> throw. The
    oracle therefore never issues the one blocked rightward ledge step, so it never needs
    human help to recover. For the EMPTY family that forward-only walk is no longer
    automatic -- both buttons must be pressed, rightmost first -- see `_empty_step`.

    A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    @staticmethod
    def get_labeled_action(
        *, state: State, env: TossingRoomEnvironment, goal: Goal
    ) -> LabeledAction:
        rooms = env.get_rooms()
        robot_room = int(round(state.get(obj=env.robot, feature_name="room")))

        if any(atom.predicate in (TRASH_BIN_EMPTY, RECYCLING_BIN_EMPTY) for atom in goal.atoms):
            ground_skill, params = SkillOraclePolicy._empty_step(
                state=state, env=env, rooms=rooms, robot_room=robot_room, goal=goal
            )
        else:
            ground_skill, params = SkillOraclePolicy._throw_step(
                state=state,
                env=env,
                rooms=rooms,
                robot_room=robot_room,
                recycling=SkillOraclePolicy._goal_is_recycling(goal=goal),
            )
        return SkillOraclePolicy._to_labeled_action(
            ground_skill=ground_skill, params=params, state=state
        )

    @staticmethod
    def _goal_is_recycling(*, goal: Goal) -> bool:
        """Which throw family this goal names. The in-bin predicate is split per kind,
        so the goal's own predicate identifies the family -- there is no need to inspect
        the bound objects the way Tossing Room's `_goal_item` had to."""
        return any(atom.predicate == RECYCLING_IN_BIN for atom in goal.atoms)

    @staticmethod
    def _empty_step(
        *,
        state: State,
        env: TossingRoomEnvironment,
        rooms: tuple[Object, ...],
        robot_room: int,
        goal: Goal,
    ) -> tuple[GroundSkill, np.ndarray]:
        """Empty each still-full bin named by the goal, via that bin's own button.

        The ORDER is the whole difficulty. Each bin has its own button beside it, and the
        one-way ledge only ever permits a LEFTWARD (down-index) crossing, so any button
        the robot leaves to its right is gone for good. Visiting the still-full bins in
        DESCENDING room index is therefore the one order that can work -- on the default
        layout that is the trash button (room 6) first, then the recycling one (room 1)
        behind the ledge. Doing it the other way strands the robot with the trash bin
        still full, which is exactly the irreversibility this domain is about."""
        goal_bins = [
            atom.objects[0]
            for atom in goal.atoms
            if atom.predicate in (TRASH_BIN_EMPTY, RECYCLING_BIN_EMPTY)
        ]
        full = [
            bin_obj
            for bin_obj in goal_bins
            if int(round(state.get(obj=bin_obj, feature_name="count"))) > 0
        ]
        # `full` is only empty when the goal is already satisfied, which run_task_episode
        # checks before ever calling the policy -- falling back to goal_bins keeps this a
        # total Policy rather than letting that invariant become an IndexError.
        bin_obj = min(
            full or goal_bins,
            key=lambda candidate: -int(round(state.get(obj=candidate, feature_name="room"))),
        )
        kind = int(round(state.get(obj=bin_obj, feature_name="kind")))
        button = env.button_for_kind(kind=kind)
        button_room = int(round(state.get(obj=button, feature_name="room")))
        if robot_room != button_room:
            return SkillOraclePolicy._move_toward(
                rooms=rooms, robot_room=robot_room, target_room=button_room
            )
        skill = (
            TossingRoomSkills.PRESS_RECYCLING
            if kind == env.RECYCLING_KIND
            else TossingRoomSkills.PRESS_TRASH
        )
        ground_skill = GroundSkill(
            skill=skill,
            objects=(
                env.robot,
                button,
                rooms[robot_room],
                bin_obj,
                env.item_for_kind(kind=kind),
            ),
        )
        return ground_skill, np.zeros(0)

    @staticmethod
    def _throw_step(
        *,
        state: State,
        env: TossingRoomEnvironment,
        rooms: tuple[Object, ...],
        robot_room: int,
        recycling: bool,
    ) -> tuple[GroundSkill, np.ndarray]:
        item = env.recycling if recycling else env.trash
        bin_obj = env.recycling_bin if recycling else env.trash_bin
        bin_room = env.recycling_bin_room if recycling else env.trash_bin_room
        pickup = TossingRoomSkills.PICKUP_RECYCLING if recycling else TossingRoomSkills.PICKUP_TRASH
        throw = TossingRoomSkills.THROW_RECYCLING if recycling else TossingRoomSkills.THROW_TRASH

        kind = int(round(state.get(obj=item, feature_name="kind")))
        holding = int(round(state.get(obj=env.robot, feature_name="holding")))

        if holding != kind:
            # Pick the item up from the pile at start (where the robot begins).
            ground_skill = GroundSkill(
                skill=pickup, objects=(env.robot, item, rooms[robot_room], env.pile)
            )
            return ground_skill, np.zeros(0)
        if robot_room != bin_room:
            return SkillOraclePolicy._move_toward(
                rooms=rooms, robot_room=robot_room, target_room=bin_room
            )
        # In the bin's room, holding the item: throw with the force the environment's own
        # relation says this (distance, weight) pair needs. The oracle is privileged
        # twice over -- it reads the two causes out of the State like anyone could, AND it
        # knows the coefficients, which neither throw sampler does.
        required = env.required_force(
            throw_distance=float(state.get(obj=bin_obj, feature_name="throw_distance")),
            item_weight=float(state.get(obj=item, feature_name="weight")),
        )
        ground_skill = GroundSkill(
            skill=throw, objects=(env.robot, item, bin_obj, rooms[robot_room])
        )
        return ground_skill, np.array([required])

    @staticmethod
    def _move_toward(
        *, rooms: tuple[Object, ...], robot_room: int, target_room: int
    ) -> tuple[GroundSkill, np.ndarray]:
        step = 1 if target_room > robot_room else -1
        to_room = robot_room + step
        ground_skill = GroundSkill(
            skill=TossingRoomSkills.MOVE_ROOM,
            objects=(
                TossingRoomEnvironment.robot,
                rooms[robot_room],
                rooms[to_room],
            ),
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
