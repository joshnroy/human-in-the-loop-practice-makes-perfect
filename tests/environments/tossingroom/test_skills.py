import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossingroom.environment import TossingRoomEnvironment
from hitl_pmp.environments.tossingroom.predicates import (
    BIN_ACCEPTS_ITEM,
    CAN_MOVE_ROOM,
    HAND_EMPTY,
    HOLDING,
    ITEM_IN_BIN,
    PILE_IN_ROOM,
    ROBOT_IN_ROOM,
)
from hitl_pmp.environments.tossingroom.skill_provider import TossingRoomSkillProvider
from hitl_pmp.environments.tossingroom.skills import TossingRoomSkills
from hitl_pmp.environments.tossingroom.tasks import TossingRoomTasks
from hitl_pmp.planning.grounding import SkillGrounder

_ENV = TossingRoomEnvironment()
_ROBOT = TossingRoomEnvironment.robot
_RECYCLING = TossingRoomEnvironment.recycling
_RECYCLING_BIN = TossingRoomEnvironment.recycling_bin


def _state():
    return _ENV.build_initial_state(trash_target_force=0.5, recycling_target_force=0.5)


def test_pickup_declares_its_parameters_and_effects() -> None:
    skill = TossingRoomSkills.PICKUP
    assert skill.name == "Pickup"
    assert skill.param_dim == 0
    robot, item, room, pile, bin_var = skill.parameters
    assert skill.preconditions == frozenset({
        LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(robot, room)),
        # The pile precondition is what stops the planner scheduling a pickup in
        # a room the dynamics refuse to pick up in -- see
        # TestPickupIsRestrictedToThePileRoom.
        LiftedAtom(predicate=PILE_IN_ROOM, variables=(pile, room)),
        LiftedAtom(predicate=HAND_EMPTY, variables=(robot,)),
        # Pins ?bin to the item's own bin, so the delete effect below costs one
        # grounding per item rather than widening what Pickup is applicable to.
        LiftedAtom(predicate=BIN_ACCEPTS_ITEM, variables=(item, bin_var)),
    })
    assert skill.add_effects == frozenset({LiftedAtom(predicate=HOLDING, variables=(robot, item))})
    # _apply_pickup clears the fetched item's in_bin flag -- a fresh item off the pile
    # is not in any bin -- and the model has to say so, or a planner believes an item
    # survives being replaced.
    assert skill.delete_effects == frozenset({
        LiftedAtom(predicate=HAND_EMPTY, variables=(robot,)),
        LiftedAtom(predicate=ITEM_IN_BIN, variables=(item, bin_var)),
    })


def test_move_room_requires_a_traversable_step() -> None:
    skill = TossingRoomSkills.MOVE_ROOM
    assert skill.name == "MoveRoom"
    assert skill.param_dim == 0
    robot, from_room, to_room = skill.parameters
    assert skill.preconditions == frozenset({
        LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(robot, from_room)),
        # CanMoveRoom rather than Adjacent: adjacency is symmetric, but _apply_move
        # refuses the rightward step across the one-way ledge, so a symmetric
        # precondition let the planner schedule a move the dynamics drop.
        LiftedAtom(predicate=CAN_MOVE_ROOM, variables=(from_room, to_room)),
    })
    assert skill.add_effects == frozenset({
        LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(robot, to_room))
    })


def test_throw_declares_one_continuous_force_param() -> None:
    skill = TossingRoomSkills.THROW
    assert skill.name == "Throw"
    assert skill.param_dim == 1
    robot, item, _bin, _room = skill.parameters
    assert LiftedAtom(predicate=HOLDING, variables=(robot, item)) in skill.preconditions
    assert LiftedAtom(predicate=ITEM_IN_BIN, variables=(item, _bin)) in skill.add_effects


def test_press_adds_bin_empty_for_both_bins() -> None:
    skill = TossingRoomSkills.PRESS
    assert skill.name == "Press"
    assert skill.param_dim == 0
    assert len(skill.add_effects) == 2  # BinEmpty(recycling_bin) and BinEmpty(trash_bin)


def test_sample_params_is_empty_for_zero_dim_skills() -> None:
    ground_skill = GroundSkill(
        skill=TossingRoomSkills.MOVE_ROOM,
        objects=(_ROBOT, _ENV.get_rooms()[3], _ENV.get_rooms()[2]),
    )
    params = TossingRoomSkills.sample_params(
        ground_skill=ground_skill, rng=np.random.default_rng(0)
    )
    assert params.shape == (0,)


def test_sample_params_for_throw_is_a_single_value_in_unit_interval() -> None:
    ground_skill = GroundSkill(
        skill=TossingRoomSkills.THROW,
        objects=(_ROBOT, _RECYCLING, _RECYCLING_BIN, _ENV.get_rooms()[_ENV.recycling_bin_room]),
    )
    rng = np.random.default_rng(0)
    for _ in range(50):
        params = TossingRoomSkills.sample_params(ground_skill=ground_skill, rng=rng)
        assert params.shape == (1,)
        assert 0.0 <= params[0] <= 1.0


def test_compute_action_for_pickup_encodes_the_item_kind() -> None:
    state = _state()
    ground_skill = GroundSkill(
        skill=TossingRoomSkills.PICKUP,
        objects=(_ROBOT, _RECYCLING, _ENV.get_rooms()[3], _ENV.pile, _RECYCLING_BIN),
    )
    action = TossingRoomSkills.compute_action(
        ground_skill=ground_skill, params=np.zeros(0), state=state
    )
    assert action.tolist() == [
        float(TossingRoomEnvironment.SKILL_PICKUP),
        float(TossingRoomEnvironment.RECYCLING_KIND),
        0.0,
    ]


def test_compute_action_for_move_room_encodes_the_destination_index() -> None:
    state = _state()
    rooms = _ENV.get_rooms()
    ground_skill = GroundSkill(
        skill=TossingRoomSkills.MOVE_ROOM, objects=(_ROBOT, rooms[3], rooms[2])
    )
    action = TossingRoomSkills.compute_action(
        ground_skill=ground_skill, params=np.zeros(0), state=state
    )
    assert action.tolist() == [float(TossingRoomEnvironment.SKILL_MOVE_ROOM), 2.0, 0.0]


def test_compute_action_for_throw_uses_the_sampled_force() -> None:
    state = _state()
    ground_skill = GroundSkill(
        skill=TossingRoomSkills.THROW,
        objects=(_ROBOT, _RECYCLING, _RECYCLING_BIN, _ENV.get_rooms()[_ENV.recycling_bin_room]),
    )
    action = TossingRoomSkills.compute_action(
        ground_skill=ground_skill, params=np.array([0.42]), state=state
    )
    assert action.tolist() == [
        float(TossingRoomEnvironment.SKILL_THROW),
        float(TossingRoomEnvironment.RECYCLING_KIND),
        0.42,
    ]


def test_compute_action_dispatches_by_value_not_identity() -> None:
    state = _state()
    rooms = _ENV.get_rooms()
    move = TossingRoomSkills.MOVE_ROOM
    reconstructed = Skill(
        name=move.name,
        parameters=move.parameters,
        preconditions=move.preconditions,
        add_effects=move.add_effects,
        delete_effects=move.delete_effects,
        param_dim=move.param_dim,
    )
    assert reconstructed is not TossingRoomSkills.MOVE_ROOM
    assert reconstructed == TossingRoomSkills.MOVE_ROOM
    ground_skill = GroundSkill(skill=reconstructed, objects=(_ROBOT, rooms[3], rooms[2]))
    action = TossingRoomSkills.compute_action(
        ground_skill=ground_skill, params=np.zeros(0), state=state
    )
    assert action.tolist() == [float(TossingRoomEnvironment.SKILL_MOVE_ROOM), 2.0, 0.0]


def test_compute_action_rejects_an_unknown_skill() -> None:
    unknown = Skill(
        name="Unknown",
        parameters=(),
        preconditions=frozenset(),
        add_effects=frozenset(),
        delete_effects=frozenset(),
        param_dim=0,
    )
    ground_skill = GroundSkill(skill=unknown, objects=())
    with pytest.raises(ValueError, match="Unknown skill"):
        TossingRoomSkills.compute_action(
            ground_skill=ground_skill, params=np.zeros(0), state=_state()
        )


def test_move_room_ground_skill_grounds_preconditions() -> None:
    rooms = _ENV.get_rooms()
    ground_skill = GroundSkill(
        skill=TossingRoomSkills.MOVE_ROOM, objects=(_ROBOT, rooms[3], rooms[2])
    )
    assert ground_skill.add_effects == frozenset({
        GroundAtom(predicate=ROBOT_IN_ROOM, objects=(_ROBOT, rooms[2]))
    })


class TestPickupIsRestrictedToThePileRoom:
    """Pickup's lifted model must not permit what the dynamics deny.

    `TossingRoomEnvironment._apply_pickup` only acts when `robot_room == start_room`
    (there is a single limitless pile there), but Pickup's preconditions used to say
    only "robot is in some room + hand empty". Because pickup-early and pickup-late use
    the identical multiset of skills, the two orderings are *exactly cost-tied*, so Fast
    Downward broke the tie arbitrarily and routinely emitted plans that walk away from
    the pile and then pick up in the bin room -- a silent no-op. Measured on the default
    task distribution before the fix: 9 of 10 plans contained an unexecutable Pickup and
    only 1 of 10 tasks was solved (the one Press-only goal).

    This is the same defect class as the Ball-Ring `ignore_effects` bug: an
    over-permissive symbolic model producing plans that look valid and cannot execute.
    """

    @staticmethod
    def test_pickup_is_not_applicable_outside_the_pile_room() -> None:
        env = TossingRoomEnvironment()
        tasks = TossingRoomTasks(env=env, seed=0)
        provider = TossingRoomSkillProvider(env=env)
        state = tasks.sample_test_task().initial_state
        other_room = (env.start_room + 1) % env.num_rooms
        state.set(obj=env.robot, feature_name="room", feature_val=float(other_room))
        state.set(obj=env.robot, feature_name="holding", feature_val=0.0)

        atoms = SkillGrounder.abstract_state(
            state=state, objects=provider.objects(), predicates=provider.predicates()
        )
        applicable = SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
        )
        pickups = [g for g in applicable if g.skill.name == "Pickup"]
        assert not pickups, (
            f"Pickup is applicable in room {other_room}, but the pile is in "
            f"room {env.start_room}; executing it is a silent no-op."
        )

    @staticmethod
    def test_pickup_is_applicable_in_the_pile_room() -> None:
        """The complement, so the fix cannot be 'make Pickup never applicable'."""
        env = TossingRoomEnvironment()
        tasks = TossingRoomTasks(env=env, seed=0)
        provider = TossingRoomSkillProvider(env=env)
        state = tasks.sample_test_task().initial_state
        state.set(obj=env.robot, feature_name="room", feature_val=float(env.start_room))
        state.set(obj=env.robot, feature_name="holding", feature_val=0.0)

        atoms = SkillGrounder.abstract_state(
            state=state, objects=provider.objects(), predicates=provider.predicates()
        )
        applicable = SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
        )
        assert [g for g in applicable if g.skill.name == "Pickup"]


class TestOperatorsMatchTheDynamics:
    """Two more places the lifted models permitted what the environment denies, found
    by the cross-domain operator-fidelity walk. Same defect class as Pickup: the
    planner emits a plan that looks valid and executes as a silent no-op."""

    @staticmethod
    def _applicable(*, env, provider, state):
        atoms = SkillGrounder.abstract_state(
            state=state, objects=provider.objects(), predicates=provider.predicates()
        )
        return SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
        )

    @staticmethod
    def test_throw_requires_a_bin_that_accepts_the_held_item() -> None:
        """`_apply_throw` routes by the HELD item's kind and ignores the bound bin
        entirely, so Throw(trash -> recycling_bin) can never succeed at any force. The
        model bound ?bin to any bin in the room."""
        env = TossingRoomEnvironment()
        provider = TossingRoomSkillProvider(env=env)
        state = TossingRoomTasks(env=env, seed=0).sample_test_task().initial_state
        # hold trash, stand in the recycling bin's room
        state.set(obj=env.robot, feature_name="holding", feature_val=float(env.TRASH_KIND))
        state.set(obj=env.robot, feature_name="room", feature_val=float(env.recycling_bin_room))

        mismatched = [
            g
            for g in TestOperatorsMatchTheDynamics._applicable(
                env=env, provider=provider, state=state
            )
            if g.skill.name == "Throw" and g.objects[2] is env.recycling_bin
        ]
        assert not mismatched, "Throw is applicable with a bin that cannot accept the held item"

    @staticmethod
    def test_move_room_cannot_cross_the_ledge_rightward() -> None:
        """`Adjacent` is symmetric, but `_apply_move` blocks stepping RIGHT across the
        one-way ledge. Unlike Pickup, this divergence was never acknowledged."""
        env = TossingRoomEnvironment()
        provider = TossingRoomSkillProvider(env=env)
        state = TossingRoomTasks(env=env, seed=0).sample_test_task().initial_state
        state.set(obj=env.robot, feature_name="room", feature_val=float(env.blocked_right_from))

        rooms = env.get_rooms()
        blocked = [
            g
            for g in TestOperatorsMatchTheDynamics._applicable(
                env=env, provider=provider, state=state
            )
            if g.skill.name == "MoveRoom"
            and g.objects[1] == rooms[env.blocked_right_from]
            and g.objects[2] == rooms[env.blocked_right_from + 1]
        ]
        assert not blocked, "MoveRoom is applicable across the one-way ledge rightward"

    @staticmethod
    def test_move_room_can_still_cross_the_ledge_leftward() -> None:
        """The complement: only the rightward step is blocked."""
        env = TossingRoomEnvironment()
        provider = TossingRoomSkillProvider(env=env)
        state = TossingRoomTasks(env=env, seed=0).sample_test_task().initial_state
        state.set(obj=env.robot, feature_name="room", feature_val=float(env.blocked_right_from + 1))
        rooms = env.get_rooms()
        assert [
            g
            for g in TestOperatorsMatchTheDynamics._applicable(
                env=env, provider=provider, state=state
            )
            if g.skill.name == "MoveRoom" and g.objects[2] == rooms[env.blocked_right_from]
        ]


class TestOnlyALandedThrowSatisfiesTheThrowAddEffects:
    """`Throw`'s add effect is `ItemInBin`, and EES scores a skill attempt by
    `ground_skill.add_effects <= abstract_state(next_state)` (`ees_method.py`'s
    `observe_pending`/`observe_outcome`) -- that boolean is both the competence
    observation and the sampler's training label.

    A `Throw` ALWAYS releases the item, landed or not. So an `ItemInBin` that means
    only "this bin holds at least one item of that kind" scores *every* throw into an
    already-non-empty bin a success at any force whatsoever. Two mechanisms make a bin
    non-empty, and both are exercised below: an EMPTY-goal task prefills both bins
    (`tasks.py`, `initial_count_low=1`), and within one practice period the first
    landed throw of a kind makes every later throw of that kind free.

    The damage is asymmetric, which is why this corrupts competence *ranking* and not
    just level: only the trash bin sits on the reachable side of the one-way ledge, so
    only trash throws can be repeated within a period.
    """

    @staticmethod
    def _atoms(*, env: TossingRoomEnvironment, provider: TossingRoomSkillProvider):
        return SkillGrounder.abstract_state(
            state=env.get_current_state(),
            objects=provider.objects(),
            predicates=provider.predicates(),
        )

    @staticmethod
    def _walk(*, env: TossingRoomEnvironment, from_room: int, to_room: int) -> None:
        step = 1 if to_room > from_room else -1
        for room in range(from_room + step, to_room + step, step):
            env.take_action(
                action=np.array([float(TossingRoomEnvironment.SKILL_MOVE_ROOM), float(room), 0.0])
            )

    @staticmethod
    def _robot_room(*, env: TossingRoomEnvironment) -> int:
        state = env.get_current_state()
        return int(round(state.get(obj=TossingRoomEnvironment.robot, feature_name="room")))

    @staticmethod
    def _bin_count(*, env: TossingRoomEnvironment, bin_obj) -> float:
        return env.get_current_state().get(obj=bin_obj, feature_name="count")

    @staticmethod
    def _fetch_and_carry(*, env: TossingRoomEnvironment, item, bin_room: int) -> None:
        """Pick the item up from the pile at the start room and walk it to its bin's
        room, entirely through the real dynamics -- no privileged `set_state`, so the
        `Holding`/room facts the throw is scored against are ones the agent could
        genuinely have produced."""
        cls = TestOnlyALandedThrowSatisfiesTheThrowAddEffects
        kind = int(round(env.get_current_state().get(obj=item, feature_name="kind")))
        env.take_action(
            action=np.array([float(TossingRoomEnvironment.SKILL_PICKUP), float(kind), 0.0])
        )
        cls._walk(env=env, from_room=cls._robot_room(env=env), to_room=bin_room)

    @staticmethod
    def _score_throw(
        *,
        env: TossingRoomEnvironment,
        provider: TossingRoomSkillProvider,
        item,
        bin_obj,
        force: float,
    ) -> bool:
        """Execute one `Throw` exactly the way `EESMethod` does and return the success
        boolean it would record."""
        rooms = env.get_rooms()
        robot_room = TestOnlyALandedThrowSatisfiesTheThrowAddEffects._robot_room(env=env)
        ground_skill = GroundSkill(
            skill=TossingRoomSkills.THROW,
            objects=(TossingRoomEnvironment.robot, item, bin_obj, rooms[robot_room]),
        )
        env.take_action(
            action=provider.compute_action(
                ground_skill=ground_skill,
                params=np.array([force]),
                state=env.get_current_state(),
            )
        )
        return ground_skill.add_effects <= TestOnlyALandedThrowSatisfiesTheThrowAddEffects._atoms(
            env=env, provider=provider
        )

    @staticmethod
    def test_a_landed_throw_is_scored_a_success() -> None:
        """The complement, so the fix cannot be 'never score a throw a success'."""
        env = TossingRoomEnvironment()
        provider = TossingRoomSkillProvider(env=env)
        env.set_state(
            state=env.build_initial_state(trash_target_force=0.62, recycling_target_force=0.5)
        )
        cls = TestOnlyALandedThrowSatisfiesTheThrowAddEffects
        cls._fetch_and_carry(
            env=env, item=TossingRoomEnvironment.trash, bin_room=env.trash_bin_room
        )
        before = cls._bin_count(env=env, bin_obj=TossingRoomEnvironment.trash_bin)
        assert before == 0.0, "non-vacuity: this case must start from an EMPTY bin"
        scored = cls._score_throw(
            env=env,
            provider=provider,
            item=TossingRoomEnvironment.trash,
            bin_obj=TossingRoomEnvironment.trash_bin,
            force=0.62,
        )
        after = cls._bin_count(env=env, bin_obj=TossingRoomEnvironment.trash_bin)
        assert after == before + 1.0, "non-vacuity: this throw must genuinely have LANDED"
        assert scored is True

    @staticmethod
    def test_a_missed_throw_into_a_prefilled_bin_is_not_scored_a_success() -> None:
        """Mechanism 1: an EMPTY-goal task prefills both bins with 1..3 items, so every
        throw taken in the same period lands in an already-non-empty bin."""
        env = TossingRoomEnvironment()
        provider = TossingRoomSkillProvider(env=env)
        env.set_state(
            state=env.build_initial_state(
                trash_target_force=0.62, recycling_target_force=0.5, trash_count=2
            )
        )
        cls = TestOnlyALandedThrowSatisfiesTheThrowAddEffects
        cls._fetch_and_carry(
            env=env, item=TossingRoomEnvironment.trash, bin_room=env.trash_bin_room
        )
        before = cls._bin_count(env=env, bin_obj=TossingRoomEnvironment.trash_bin)
        assert before >= 1.0, "non-vacuity: this case must start from a NON-EMPTY bin"
        # target 0.62, throw_tolerance 0.1 -> a force of 0.05 misses by a mile.
        scored = cls._score_throw(
            env=env,
            provider=provider,
            item=TossingRoomEnvironment.trash,
            bin_obj=TossingRoomEnvironment.trash_bin,
            force=0.05,
        )
        after = cls._bin_count(env=env, bin_obj=TossingRoomEnvironment.trash_bin)
        assert after == before, "non-vacuity: this throw must genuinely have MISSED"
        assert scored is False, (
            "a throw that missed was scored a success purely because the bin was "
            "already non-empty -- this is the competence/sampler-label corruption"
        )

    @staticmethod
    def test_a_missed_throw_after_an_earlier_landed_one_is_not_scored_a_success() -> None:
        """Mechanism 2: within one practice period there is no reset, so the first
        landed trash throw leaves the trash bin non-empty and every later trash throw
        in that period scores free. Only trash can do this -- the recycling bin sits
        behind the one-way ledge, so a period never gets a second recycling attempt --
        which is exactly why the inflation is asymmetric between the two skills."""
        env = TossingRoomEnvironment()
        provider = TossingRoomSkillProvider(env=env)
        env.set_state(
            state=env.build_initial_state(trash_target_force=0.62, recycling_target_force=0.5)
        )
        cls = TestOnlyALandedThrowSatisfiesTheThrowAddEffects
        cls._fetch_and_carry(
            env=env, item=TossingRoomEnvironment.trash, bin_room=env.trash_bin_room
        )
        assert cls._score_throw(
            env=env,
            provider=provider,
            item=TossingRoomEnvironment.trash,
            bin_obj=TossingRoomEnvironment.trash_bin,
            force=0.62,
        ), "non-vacuity: the first throw of this period must genuinely have landed"

        # Walk back for a fresh item and throw again, badly.
        cls._walk(env=env, from_room=env.trash_bin_room, to_room=env.start_room)
        cls._fetch_and_carry(
            env=env, item=TossingRoomEnvironment.trash, bin_room=env.trash_bin_room
        )
        before = cls._bin_count(env=env, bin_obj=TossingRoomEnvironment.trash_bin)
        assert before >= 1.0, "non-vacuity: the earlier throw must have left the bin non-empty"
        scored = cls._score_throw(
            env=env,
            provider=provider,
            item=TossingRoomEnvironment.trash,
            bin_obj=TossingRoomEnvironment.trash_bin,
            force=0.05,
        )
        after = cls._bin_count(env=env, bin_obj=TossingRoomEnvironment.trash_bin)
        assert after == before, "non-vacuity: the second throw must genuinely have MISSED"
        assert scored is False, (
            "the second throw of the period missed but was scored a success, because "
            "the first one had already made the bin non-empty"
        )
