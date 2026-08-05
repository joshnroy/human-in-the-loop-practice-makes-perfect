import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossingroom.environment import TossingRoomEnvironment
from hitl_pmp.environments.tossingroom.predicates import (
    BIN_ACCEPTS_ITEM,
    BIN_EMPTY,
    BUTTON_FOR_BIN,
    BUTTON_IN_ROOM,
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
    robot, item, room, pile = skill.parameters
    assert skill.preconditions == frozenset({
        LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(robot, room)),
        # The pile precondition is what stops the planner scheduling a pickup in
        # a room the dynamics refuse to pick up in -- see
        # TestPickupIsRestrictedToThePileRoom.
        LiftedAtom(predicate=PILE_IN_ROOM, variables=(pile, room)),
        LiftedAtom(predicate=HAND_EMPTY, variables=(robot,)),
    })
    assert skill.add_effects == frozenset({LiftedAtom(predicate=HOLDING, variables=(robot, item))})
    assert skill.delete_effects == frozenset({LiftedAtom(predicate=HAND_EMPTY, variables=(robot,))})


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


def test_throw_requires_an_empty_bin_and_fills_it() -> None:
    """Bins hold at most one item and `_apply_throw` refuses a throw at a full one, so
    the model must say so -- and a landed throw must give up `BinEmpty`, or the planner
    believes it can throw a second item into the same bin."""
    skill = TossingRoomSkills.THROW
    _robot, _item, bin_var, _room = skill.parameters
    assert LiftedAtom(predicate=BIN_EMPTY, variables=(bin_var,)) in skill.preconditions
    assert LiftedAtom(predicate=BIN_EMPTY, variables=(bin_var,)) in skill.delete_effects


def test_press_empties_exactly_its_own_buttons_bin() -> None:
    """One button per bin, beside it. `ButtonForBin` is what pins ?bin to the pressed
    button, and `BinAcceptsItem` pins ?item to that bin -- which makes the `ItemInBin`
    delete an ordinary per-bin delete effect rather than the blanket `ignore_effects`
    the both-bins button needed."""
    skill = TossingRoomSkills.PRESS
    assert skill.name == "Press"
    assert skill.param_dim == 0
    robot, button, room, bin_var, item = skill.parameters
    assert skill.preconditions == frozenset({
        LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(robot, room)),
        LiftedAtom(predicate=BUTTON_IN_ROOM, variables=(button, room)),
        LiftedAtom(predicate=BUTTON_FOR_BIN, variables=(button, bin_var)),
        LiftedAtom(predicate=BIN_ACCEPTS_ITEM, variables=(item, bin_var)),
    })
    assert skill.add_effects == frozenset({LiftedAtom(predicate=BIN_EMPTY, variables=(bin_var,))})
    assert skill.delete_effects == frozenset({
        LiftedAtom(predicate=ITEM_IN_BIN, variables=(item, bin_var))
    })
    assert skill.ignore_effects == frozenset()


def test_compute_action_for_press_encodes_the_buttons_kind() -> None:
    """`Press`'s `arg` used to be unused; it now names which button is pressed."""
    state = _state()
    ground_skill = GroundSkill(
        skill=TossingRoomSkills.PRESS,
        objects=(
            _ROBOT,
            _ENV.recycling_button,
            _ENV.get_rooms()[_ENV.recycling_bin_room],
            _RECYCLING_BIN,
            _RECYCLING,
        ),
    )
    action = TossingRoomSkills.compute_action(
        ground_skill=ground_skill, params=np.zeros(0), state=state
    )
    assert action.tolist() == [
        float(TossingRoomEnvironment.SKILL_PRESS),
        float(TossingRoomEnvironment.RECYCLING_KIND),
        0.0,
    ]


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
        objects=(_ROBOT, _RECYCLING, _ENV.get_rooms()[3], _ENV.pile),
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


class TestAThrowIsScoredASuccessOnlyWhenItLands:
    """The defect this redesign exists to close.

    EES scores a skill execution by `add_effects <= atoms(next_state)`
    (`ees_method.py`, `observe_pending`), and that verdict feeds both the per-ground-
    skill competence model and the label on the sampler's training row. `Throw`'s
    add-effect is `ItemInBin(item, bin)`, which reads the BIN's count -- a property of
    the bin, not of the throw. Since a throw always releases the item whether or not it
    lands, any throw into an already-occupied bin was scored a success at any force
    (measured: 109/154 of recorded trash-`Throw` successes were misses).

    Capacity 1 plus `Throw`'s `BinEmpty` precondition removes the state that made that
    possible: at throw time the count is provably 0, so `ItemInBin` goes false -> true
    exactly once per throw, and the scored verdict equals the landed outcome."""

    @staticmethod
    def _applicable_throws(*, env, provider, state):
        atoms = SkillGrounder.abstract_state(
            state=state, objects=provider.objects(), predicates=provider.predicates()
        )
        return [
            g
            for g in SkillGrounder.applicable_ground_skills(
                skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
            )
            if g.skill.name == "Throw"
        ]

    @staticmethod
    def _at_the_recycling_bin(*, env, count: int):
        state = env.build_initial_state(
            trash_target_force=0.5, recycling_target_force=0.5, recycling_count=count
        )
        state.set(obj=env.robot, feature_name="room", feature_val=float(env.recycling_bin_room))
        state.set(obj=env.robot, feature_name="holding", feature_val=float(env.RECYCLING_KIND))
        return state

    @staticmethod
    def _throw(*, env, provider, state, force: float) -> tuple[bool, bool]:
        """Returns (what EES would score, whether the item actually landed)."""
        ground_skill = GroundSkill(
            skill=TossingRoomSkills.THROW,
            objects=(
                env.robot,
                env.recycling,
                env.recycling_bin,
                env.get_rooms()[env.recycling_bin_room],
            ),
        )
        before = int(round(state.get(obj=env.recycling_bin, feature_name="count")))
        env.set_state(state=state.model_copy(deep=True))
        next_state = env.take_action(
            action=provider.compute_action(
                ground_skill=ground_skill, params=np.array([force]), state=state
            )
        )
        scored = ground_skill.add_effects <= SkillGrounder.abstract_state(
            state=next_state, objects=provider.objects(), predicates=provider.predicates()
        )
        after = int(round(next_state.get(obj=env.recycling_bin, feature_name="count")))
        return scored, after > before

    @staticmethod
    def test_a_landed_throw_is_scored_a_success() -> None:
        env = TossingRoomEnvironment()
        provider = TossingRoomSkillProvider(env=env)
        state = TestAThrowIsScoredASuccessOnlyWhenItLands._at_the_recycling_bin(env=env, count=0)
        scored, landed = TestAThrowIsScoredASuccessOnlyWhenItLands._throw(
            env=env, provider=provider, state=state, force=0.5
        )
        assert (scored, landed) == (True, True)

    @staticmethod
    def test_a_missed_throw_is_scored_a_failure() -> None:
        env = TossingRoomEnvironment()
        provider = TossingRoomSkillProvider(env=env)
        state = TestAThrowIsScoredASuccessOnlyWhenItLands._at_the_recycling_bin(env=env, count=0)
        scored, landed = TestAThrowIsScoredASuccessOnlyWhenItLands._throw(
            env=env, provider=provider, state=state, force=0.95
        )
        assert (scored, landed) == (False, False)

    @staticmethod
    def test_throw_is_not_applicable_at_a_bin_that_already_holds_an_item() -> None:
        """The state that made the vacuous success possible is now unreachable by any
        planner or practice policy: `Throw` requires `BinEmpty`, so it is never selected
        at an occupied bin, and the dynamics refuse it there anyway."""
        env = TossingRoomEnvironment()
        provider = TossingRoomSkillProvider(env=env)
        state = TestAThrowIsScoredASuccessOnlyWhenItLands._at_the_recycling_bin(env=env, count=1)
        assert not TestAThrowIsScoredASuccessOnlyWhenItLands._applicable_throws(
            env=env, provider=provider, state=state
        )

    @staticmethod
    def test_throw_is_applicable_at_an_empty_bin() -> None:
        """The complement, so the precondition cannot be 'Throw is never applicable'."""
        env = TossingRoomEnvironment()
        provider = TossingRoomSkillProvider(env=env)
        state = TestAThrowIsScoredASuccessOnlyWhenItLands._at_the_recycling_bin(env=env, count=0)
        assert TestAThrowIsScoredASuccessOnlyWhenItLands._applicable_throws(
            env=env, provider=provider, state=state
        )


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
