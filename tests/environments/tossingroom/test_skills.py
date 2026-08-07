"""The defining property of this domain: `Throw` is TWO lifted skills, not one.

Tossing Room has a single `Throw(robot, item, bin, room)` whose `item` variable ranges
over both kinds, so one `LearnedSkillSampler` (keyed by skill *name* in
`EesMethod.sampler`) sees trash and recycling experience together. Here the two are
`ThrowTrash` and `ThrowRecycling`, each with its own item and bin *types*, so nothing
can bind a recycling object into the trash throw or vice versa.

The type split is what enforces it. `SkillGrounder._applicable_groundings` binds a
parameter only to objects whose `type` matches exactly, and `PddlWriter` emits the same
flat typing to Fast Downward, so "each skill binds its own item and bin" is a fact about
the representation rather than a precondition a future edit could drop.
"""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossingroom.environment import (
    TossingRoomEnvironment,
)
from hitl_pmp.environments.tossingroom.predicates import (
    CAN_MOVE_ROOM,
    HAND_EMPTY,
    HOLDING_RECYCLING,
    HOLDING_TRASH,
    PILE_IN_ROOM,
    RECYCLING_BIN_EMPTY,
    RECYCLING_BIN_IN_ROOM,
    RECYCLING_BUTTON_IN_ROOM,
    RECYCLING_IN_BIN,
    ROBOT_IN_ROOM,
    TRASH_BIN_EMPTY,
    TRASH_BIN_IN_ROOM,
    TRASH_BUTTON_IN_ROOM,
    TRASH_IN_BIN,
)
from hitl_pmp.environments.tossingroom.skill_provider import (
    TossingRoomSkillProvider,
)
from hitl_pmp.environments.tossingroom.skills import (
    TossingRoomSkills,
)
from hitl_pmp.environments.tossingroom.tasks import (
    TossingRoomGoalType,
    TossingRoomTasks,
)
from hitl_pmp.planning.grounding import SkillGrounder

_ENV = TossingRoomEnvironment()
_ROBOT = TossingRoomEnvironment.robot
_TRASH = TossingRoomEnvironment.trash
_RECYCLING = TossingRoomEnvironment.recycling
_TRASH_BIN = TossingRoomEnvironment.trash_bin
_RECYCLING_BIN = TossingRoomEnvironment.recycling_bin


def _state():
    return _ENV.build_initial_state(weight_seed=0)


class TestThrowIsTwoSkills:
    """The whole point of the domain. Every assertion here is about the *lifted* layer,
    which is what `EesMethod.sampler` keys on."""

    @staticmethod
    def test_there_are_two_differently_named_throw_skills() -> None:
        names = [skill.name for skill in TossingRoomSkillProvider(env=_ENV).skills()]
        assert "ThrowTrash" in names
        assert "ThrowRecycling" in names
        assert names.count("ThrowTrash") == 1
        assert names.count("ThrowRecycling") == 1

    @staticmethod
    def test_no_shared_throw_skill_survives() -> None:
        """A single `Throw` is exactly what the DEFAULT arm exists not to have: one name
        is one sampler, so a shared name would restore the transfer being removed. It is
        available deliberately, and only deliberately, behind `--unsplit-skills` -- see
        `test_unsplit_skills.py`; what this asserts is that it cannot arrive by accident.
        """
        names = {skill.name for skill in TossingRoomSkillProvider(env=_ENV).skills()}
        assert "Throw" not in names
        assert not hasattr(TossingRoomSkills, "THROW")

    @staticmethod
    def test_the_two_throw_skills_are_distinct_objects_with_distinct_content() -> None:
        trash, recycling = (
            TossingRoomSkills.THROW_TRASH,
            TossingRoomSkills.THROW_RECYCLING,
        )
        assert trash is not recycling
        assert trash != recycling
        assert trash.preconditions != recycling.preconditions

    @staticmethod
    def test_each_throw_binds_its_own_item_and_bin_type() -> None:
        """No `item` variable ranging over both kinds: the trash throw's item parameter
        is typed `trash` and the recycling throw's is typed `recycling`, so grounding
        cannot cross them."""
        _robot, trash_item, trash_bin, _room = TossingRoomSkills.THROW_TRASH.parameters
        assert trash_item.type == TossingRoomEnvironment.trash_type
        assert trash_bin.type == TossingRoomEnvironment.trash_bin_type

        _robot, rec_item, rec_bin, _room = TossingRoomSkills.THROW_RECYCLING.parameters
        assert rec_item.type == TossingRoomEnvironment.recycling_type
        assert rec_bin.type == TossingRoomEnvironment.recycling_bin_type

        assert TossingRoomEnvironment.trash_type != TossingRoomEnvironment.recycling_type
        assert TossingRoomEnvironment.trash_bin_type != TossingRoomEnvironment.recycling_bin_type

    @staticmethod
    def test_a_cross_kind_grounding_is_rejected_outright() -> None:
        """Not merely inapplicable -- unconstructible. `GroundSkill` validates object
        types against the skill's parameters, so `ThrowTrash(robot, recycling, ...)`
        raises rather than silently becoming a skill that can never succeed."""
        with pytest.raises(ValueError, match="type"):
            GroundSkill(
                skill=TossingRoomSkills.THROW_TRASH,
                objects=(
                    _ROBOT,
                    _RECYCLING,
                    _TRASH_BIN,
                    _ENV.get_rooms()[_ENV.trash_bin_room],
                ),
            )

    @staticmethod
    def test_both_throws_have_the_same_parameter_shape() -> None:
        """Same architecture, different weights: the sampler input row is
        `[1.0] + concat(state[obj] for obj in ground_skill.objects) + params`, so the two
        samplers must see identically shaped rows or the comparison would confound a
        different network with different experience."""
        trash, recycling = (
            TossingRoomSkills.THROW_TRASH,
            TossingRoomSkills.THROW_RECYCLING,
        )
        assert trash.param_dim == recycling.param_dim == 1
        assert [parameter.type.dim for parameter in trash.parameters] == [
            parameter.type.dim for parameter in recycling.parameters
        ]

    @staticmethod
    def test_throw_trash_declares_its_own_preconditions_and_effects() -> None:
        skill = TossingRoomSkills.THROW_TRASH
        robot, item, bin_var, room = skill.parameters
        assert skill.preconditions == frozenset({
            LiftedAtom(predicate=HOLDING_TRASH, variables=(robot, item)),
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(robot, room)),
            LiftedAtom(predicate=TRASH_BIN_IN_ROOM, variables=(bin_var, room)),
            # Capacity 1: _apply_throw refuses a throw at a full bin, so the model must
            # say so or it would claim more than the dynamics allow.
            LiftedAtom(predicate=TRASH_BIN_EMPTY, variables=(bin_var,)),
        })
        assert skill.add_effects == frozenset({
            LiftedAtom(predicate=TRASH_IN_BIN, variables=(item, bin_var)),
            LiftedAtom(predicate=HAND_EMPTY, variables=(robot,)),
        })
        assert skill.delete_effects == frozenset({
            LiftedAtom(predicate=HOLDING_TRASH, variables=(robot, item)),
            # A landed throw fills the bin -- without this the planner would believe it
            # can throw a second item into the same one.
            LiftedAtom(predicate=TRASH_BIN_EMPTY, variables=(bin_var,)),
        })

    @staticmethod
    def test_throw_recycling_declares_its_own_preconditions_and_effects() -> None:
        skill = TossingRoomSkills.THROW_RECYCLING
        robot, item, bin_var, room = skill.parameters
        assert skill.preconditions == frozenset({
            LiftedAtom(predicate=HOLDING_RECYCLING, variables=(robot, item)),
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(robot, room)),
            LiftedAtom(predicate=RECYCLING_BIN_IN_ROOM, variables=(bin_var, room)),
            LiftedAtom(predicate=RECYCLING_BIN_EMPTY, variables=(bin_var,)),
        })
        assert skill.add_effects == frozenset({
            LiftedAtom(predicate=RECYCLING_IN_BIN, variables=(item, bin_var)),
            LiftedAtom(predicate=HAND_EMPTY, variables=(robot,)),
        })
        assert skill.delete_effects == frozenset({
            LiftedAtom(predicate=HOLDING_RECYCLING, variables=(robot, item)),
            LiftedAtom(predicate=RECYCLING_BIN_EMPTY, variables=(bin_var,)),
        })

    @staticmethod
    def test_no_bin_accepts_item_precondition_is_needed_any_more() -> None:
        """Tossing Room needed `BinAcceptsItem` because `Throw` could bind a mismatched
        bin, and its capacity-1 redesign additionally uses it to pin `Press`'s `?item`.
        Both jobs are done by the types here -- `PressTrash`'s item parameter is typed
        `trash`, one object -- so the predicate stays dropped rather than kept as a
        tautology, and this test is what would notice it creeping back.

        `ButtonForBin`, which that same redesign added to tie each button to the one bin
        it empties, is dropped for exactly the same reason: the button types are split,
        so `PressTrash` can only bind `trash_button`.

        Both come back under `--unsplit-skills`, where the shared types stop separating
        the kinds and they are live constraints again -- asserted in
        `test_unsplit_skills.py`. This test is about the DEFAULT arm's predicate set."""
        predicate_names = {
            predicate.name for predicate in TossingRoomSkillProvider(env=_ENV).predicates()
        }
        assert "BinAcceptsItem" not in predicate_names
        assert "ButtonForBin" not in predicate_names


class TestPickupSplitsWithTheItemTypes:
    """Splitting the item types forces `Pickup` to split too -- a `trash` object cannot
    bind a parameter typed `recycling`. Both halves are `param_dim=0`, so neither gets a
    sampler and the split is representational only; it changes nothing about learning."""

    @staticmethod
    def test_there_are_two_pickups_and_neither_has_continuous_parameters() -> None:
        assert TossingRoomSkills.PICKUP_TRASH.name == "PickupTrash"
        assert TossingRoomSkills.PICKUP_RECYCLING.name == "PickupRecycling"
        assert TossingRoomSkills.PICKUP_TRASH.param_dim == 0
        assert TossingRoomSkills.PICKUP_RECYCLING.param_dim == 0

    @staticmethod
    def test_pickup_trash_keeps_the_pile_room_precondition() -> None:
        skill = TossingRoomSkills.PICKUP_TRASH
        robot, item, room, pile = skill.parameters
        assert skill.preconditions == frozenset({
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(robot, room)),
            LiftedAtom(predicate=PILE_IN_ROOM, variables=(pile, room)),
            LiftedAtom(predicate=HAND_EMPTY, variables=(robot,)),
        })
        assert skill.add_effects == frozenset({
            LiftedAtom(predicate=HOLDING_TRASH, variables=(robot, item))
        })


def test_move_room_requires_a_traversable_step() -> None:
    skill = TossingRoomSkills.MOVE_ROOM
    assert skill.name == "MoveRoom"
    assert skill.param_dim == 0
    robot, from_room, to_room = skill.parameters
    assert skill.preconditions == frozenset({
        LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(robot, from_room)),
        LiftedAtom(predicate=CAN_MOVE_ROOM, variables=(from_room, to_room)),
    })
    assert skill.add_effects == frozenset({
        LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(robot, to_room))
    })


class TestPressSplitsWithTheBinAndButtonTypes:
    """Each bin has its own button beside it, and a press empties only that bin. Tossing
    Room expresses that as one lifted `Press` whose `?bin`/`?item` are pinned by
    `ButtonForBin`/`BinAcceptsItem`; here the bin and button types are split, so it is
    two lifted skills with one grounding each -- the same relationship the throws have.
    Both are `param_dim=0`, so neither gets a sampler and learning is untouched."""

    @staticmethod
    def test_there_are_two_presses_and_no_shared_one() -> None:
        names = {skill.name for skill in TossingRoomSkillProvider(env=_ENV).skills()}
        assert "PressTrash" in names
        assert "PressRecycling" in names
        assert "Press" not in names
        assert not hasattr(TossingRoomSkills, "PRESS")

    @staticmethod
    def test_press_trash_empties_only_the_trash_bin() -> None:
        skill = TossingRoomSkills.PRESS_TRASH
        robot, button, room, bin_var, item = skill.parameters
        assert skill.param_dim == 0
        assert skill.preconditions == frozenset({
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(robot, room)),
            LiftedAtom(predicate=TRASH_BUTTON_IN_ROOM, variables=(button, room)),
        })
        assert skill.add_effects == frozenset({
            LiftedAtom(predicate=TRASH_BIN_EMPTY, variables=(bin_var,))
        })
        assert skill.delete_effects == frozenset({
            LiftedAtom(predicate=TRASH_IN_BIN, variables=(item, bin_var))
        })

    @staticmethod
    def test_press_recycling_empties_only_the_recycling_bin() -> None:
        skill = TossingRoomSkills.PRESS_RECYCLING
        robot, button, room, bin_var, item = skill.parameters
        assert skill.preconditions == frozenset({
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(robot, room)),
            LiftedAtom(predicate=RECYCLING_BUTTON_IN_ROOM, variables=(button, room)),
        })
        assert skill.add_effects == frozenset({
            LiftedAtom(predicate=RECYCLING_BIN_EMPTY, variables=(bin_var,))
        })
        assert skill.delete_effects == frozenset({
            LiftedAtom(predicate=RECYCLING_IN_BIN, variables=(item, bin_var))
        })

    @staticmethod
    def test_neither_press_needs_ignore_effects_any_more() -> None:
        """The blanket `ignore_effects={TrashInBin, RecyclingInBin}` existed because one
        button emptied BOTH bins -- a universal delete no per-item `delete_effect` could
        express. One button per bin makes the delete ordinary, and keeping the blanket
        would now be strictly weaker than the truth."""
        assert TossingRoomSkills.PRESS_TRASH.ignore_effects == frozenset()
        assert TossingRoomSkills.PRESS_RECYCLING.ignore_effects == frozenset()

    @staticmethod
    def test_a_cross_kind_press_grounding_is_rejected_outright() -> None:
        """The `ButtonForBin` analogue, at the strength the type split buys: pressing the
        recycling button cannot even be *constructed* as a trash press."""
        with pytest.raises(ValueError, match="type"):
            GroundSkill(
                skill=TossingRoomSkills.PRESS_TRASH,
                objects=(
                    _ROBOT,
                    TossingRoomEnvironment.recycling_button,
                    _ENV.get_rooms()[_ENV.trash_bin_room],
                    _TRASH_BIN,
                    _TRASH,
                ),
            )


def test_sample_params_is_empty_for_zero_dim_skills() -> None:
    ground_skill = GroundSkill(
        skill=TossingRoomSkills.MOVE_ROOM,
        objects=(_ROBOT, _ENV.get_rooms()[3], _ENV.get_rooms()[2]),
    )
    params = TossingRoomSkills.sample_params(
        ground_skill=ground_skill, rng=np.random.default_rng(0)
    )
    assert params.shape == (0,)


@pytest.mark.parametrize("skill_name", ["THROW_TRASH", "THROW_RECYCLING"])
def test_sample_params_for_each_throw_is_a_single_value_in_unit_interval(
    *, skill_name: str
) -> None:
    skill = getattr(TossingRoomSkills, skill_name)
    item = _TRASH if skill_name == "THROW_TRASH" else _RECYCLING
    bin_obj = _TRASH_BIN if skill_name == "THROW_TRASH" else _RECYCLING_BIN
    room_index = _ENV.trash_bin_room if skill_name == "THROW_TRASH" else _ENV.recycling_bin_room
    ground_skill = GroundSkill(
        skill=skill, objects=(_ROBOT, item, bin_obj, _ENV.get_rooms()[room_index])
    )
    rng = np.random.default_rng(0)
    for _ in range(50):
        params = TossingRoomSkills.sample_params(ground_skill=ground_skill, rng=rng)
        assert params.shape == (1,)
        assert 0.0 <= params[0] <= 1.0


def test_compute_action_for_pickup_trash_encodes_the_trash_kind() -> None:
    ground_skill = GroundSkill(
        skill=TossingRoomSkills.PICKUP_TRASH,
        objects=(_ROBOT, _TRASH, _ENV.get_rooms()[3], _ENV.pile),
    )
    action = TossingRoomSkills.compute_action(
        ground_skill=ground_skill, params=np.zeros(0), state=_state()
    )
    assert action.tolist() == [
        float(TossingRoomEnvironment.SKILL_PICKUP),
        float(TossingRoomEnvironment.TRASH_KIND),
        0.0,
    ]


def test_compute_action_for_pickup_recycling_encodes_the_recycling_kind() -> None:
    ground_skill = GroundSkill(
        skill=TossingRoomSkills.PICKUP_RECYCLING,
        objects=(_ROBOT, _RECYCLING, _ENV.get_rooms()[3], _ENV.pile),
    )
    action = TossingRoomSkills.compute_action(
        ground_skill=ground_skill, params=np.zeros(0), state=_state()
    )
    assert action.tolist() == [
        float(TossingRoomEnvironment.SKILL_PICKUP),
        float(TossingRoomEnvironment.RECYCLING_KIND),
        0.0,
    ]


def test_compute_action_for_move_room_encodes_the_destination_index() -> None:
    rooms = _ENV.get_rooms()
    ground_skill = GroundSkill(
        skill=TossingRoomSkills.MOVE_ROOM, objects=(_ROBOT, rooms[3], rooms[2])
    )
    action = TossingRoomSkills.compute_action(
        ground_skill=ground_skill, params=np.zeros(0), state=_state()
    )
    assert action.tolist() == [
        float(TossingRoomEnvironment.SKILL_MOVE_ROOM),
        2.0,
        0.0,
    ]


def test_compute_action_for_each_throw_uses_the_sampled_force_and_its_own_kind() -> None:
    state = _state()
    rooms = _ENV.get_rooms()
    trash_throw = GroundSkill(
        skill=TossingRoomSkills.THROW_TRASH,
        objects=(_ROBOT, _TRASH, _TRASH_BIN, rooms[_ENV.trash_bin_room]),
    )
    recycling_throw = GroundSkill(
        skill=TossingRoomSkills.THROW_RECYCLING,
        objects=(_ROBOT, _RECYCLING, _RECYCLING_BIN, rooms[_ENV.recycling_bin_room]),
    )
    assert TossingRoomSkills.compute_action(
        ground_skill=trash_throw, params=np.array([0.42]), state=state
    ).tolist() == [
        float(TossingRoomEnvironment.SKILL_THROW),
        float(TossingRoomEnvironment.TRASH_KIND),
        0.42,
    ]
    assert TossingRoomSkills.compute_action(
        ground_skill=recycling_throw, params=np.array([0.42]), state=state
    ).tolist() == [
        float(TossingRoomEnvironment.SKILL_THROW),
        float(TossingRoomEnvironment.RECYCLING_KIND),
        0.42,
    ]


def test_compute_action_dispatches_by_value_not_identity() -> None:
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
    assert reconstructed is not move
    assert reconstructed == move
    ground_skill = GroundSkill(skill=reconstructed, objects=(_ROBOT, rooms[3], rooms[2]))
    action = TossingRoomSkills.compute_action(
        ground_skill=ground_skill, params=np.zeros(0), state=_state()
    )
    assert action.tolist() == [
        float(TossingRoomEnvironment.SKILL_MOVE_ROOM),
        2.0,
        0.0,
    ]


def test_compute_action_rejects_an_unknown_skill() -> None:
    unknown = Skill(
        name="Unknown",
        parameters=(),
        preconditions=frozenset(),
        add_effects=frozenset(),
        delete_effects=frozenset(),
        param_dim=0,
    )
    with pytest.raises(ValueError, match="Unknown skill"):
        TossingRoomSkills.compute_action(
            ground_skill=GroundSkill(skill=unknown, objects=()), params=np.zeros(0), state=_state()
        )


def test_move_room_ground_skill_grounds_preconditions() -> None:
    rooms = _ENV.get_rooms()
    ground_skill = GroundSkill(
        skill=TossingRoomSkills.MOVE_ROOM, objects=(_ROBOT, rooms[3], rooms[2])
    )
    assert ground_skill.add_effects == frozenset({
        GroundAtom(predicate=ROBOT_IN_ROOM, objects=(_ROBOT, rooms[2]))
    })


class TestGroundingCannotCrossTheTwoKinds:
    """The grounder's view, which is what a planner and a practicing Method both see."""

    @staticmethod
    def _throw_task_state():
        """A THROW task's initial state, so both bins start EMPTY. That matters now that
        each throw carries its bin's empty precondition: an EMPTY task prefills both bins
        and no throw is applicable at all in it, which is the subject of its own test
        below rather than a confound for these."""
        env = TossingRoomEnvironment()
        return (
            TossingRoomTasks(env=env, seed=0, forced_goal_type=TossingRoomGoalType.TRASH)
            .sample_test_task()
            .initial_state
        )

    @staticmethod
    def _applicable(*, state):
        env = TossingRoomEnvironment()
        provider = TossingRoomSkillProvider(env=env)
        atoms = SkillGrounder.abstract_state(
            state=state, objects=provider.objects(), predicates=provider.predicates()
        )
        return SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
        )

    @staticmethod
    def test_holding_trash_in_the_trash_bin_room_enables_only_throw_trash() -> None:
        env = TossingRoomEnvironment()
        state = TestGroundingCannotCrossTheTwoKinds._throw_task_state()
        state.set(obj=env.robot, feature_name="holding", feature_val=float(env.TRASH_KIND))
        state.set(obj=env.robot, feature_name="room", feature_val=float(env.trash_bin_room))
        throws = [
            ground.skill.name
            for ground in TestGroundingCannotCrossTheTwoKinds._applicable(state=state)
            if ground.skill.name.startswith("Throw")
        ]
        assert throws == ["ThrowTrash"]

    @staticmethod
    def test_holding_recycling_in_the_recycling_bin_room_enables_only_throw_recycling() -> None:
        env = TossingRoomEnvironment()
        state = TestGroundingCannotCrossTheTwoKinds._throw_task_state()
        state.set(obj=env.robot, feature_name="holding", feature_val=float(env.RECYCLING_KIND))
        state.set(obj=env.robot, feature_name="room", feature_val=float(env.recycling_bin_room))
        throws = [
            ground.skill.name
            for ground in TestGroundingCannotCrossTheTwoKinds._applicable(state=state)
            if ground.skill.name.startswith("Throw")
        ]
        assert throws == ["ThrowRecycling"]

    @staticmethod
    def test_a_full_bin_makes_its_own_throw_inapplicable() -> None:
        """Capacity 1, at the symbolic layer: `_apply_throw` refuses a throw at a full
        bin, so the operator must be inapplicable there too -- otherwise the model claims
        more than the dynamics allow, which is exactly what the cross-domain fidelity
        walk exists to catch."""
        env = TossingRoomEnvironment()
        state = TestGroundingCannotCrossTheTwoKinds._throw_task_state()
        state.set(obj=env.robot, feature_name="holding", feature_val=float(env.TRASH_KIND))
        state.set(obj=env.robot, feature_name="room", feature_val=float(env.trash_bin_room))
        state.set(obj=env.trash_bin, feature_name="count", feature_val=1.0)
        assert not [
            ground
            for ground in TestGroundingCannotCrossTheTwoKinds._applicable(state=state)
            if ground.skill.name.startswith("Throw")
        ]

    @staticmethod
    def test_holding_trash_in_the_recycling_bin_room_enables_no_throw_at_all() -> None:
        """The complement of the two above: standing at the wrong bin is not a throw the
        sampler could waste experience on, it is no throw at all."""
        env = TossingRoomEnvironment()
        state = TestGroundingCannotCrossTheTwoKinds._throw_task_state()
        state.set(obj=env.robot, feature_name="holding", feature_val=float(env.TRASH_KIND))
        state.set(obj=env.robot, feature_name="room", feature_val=float(env.recycling_bin_room))
        assert not [
            ground
            for ground in TestGroundingCannotCrossTheTwoKinds._applicable(state=state)
            if ground.skill.name.startswith("Throw")
        ]


class TestPickupIsRestrictedToThePileRoom:
    """Ported verbatim from Tossing Room, where an over-permissive `Pickup` let Fast
    Downward emit plans that walk past the pile and pick up in the bin room -- a silent
    no-op that solved 1/10 tasks. Both halves of the split Pickup must keep the guard."""

    @staticmethod
    def test_no_pickup_is_applicable_outside_the_pile_room() -> None:
        env = TossingRoomEnvironment()
        state = TestGroundingCannotCrossTheTwoKinds._throw_task_state()
        other_room = (env.start_room + 1) % env.num_rooms
        state.set(obj=env.robot, feature_name="room", feature_val=float(other_room))
        state.set(obj=env.robot, feature_name="holding", feature_val=0.0)
        assert not [
            ground
            for ground in TestGroundingCannotCrossTheTwoKinds._applicable(state=state)
            if ground.skill.name.startswith("Pickup")
        ]

    @staticmethod
    def test_both_pickups_are_applicable_in_the_pile_room() -> None:
        env = TossingRoomEnvironment()
        state = TestGroundingCannotCrossTheTwoKinds._throw_task_state()
        state.set(obj=env.robot, feature_name="room", feature_val=float(env.start_room))
        state.set(obj=env.robot, feature_name="holding", feature_val=0.0)
        names = {
            ground.skill.name
            for ground in TestGroundingCannotCrossTheTwoKinds._applicable(state=state)
            if ground.skill.name.startswith("Pickup")
        }
        assert names == {"PickupTrash", "PickupRecycling"}


def test_move_room_cannot_cross_the_ledge_rightward() -> None:
    env = TossingRoomEnvironment()
    state = TossingRoomTasks(env=env, seed=0).sample_test_task().initial_state
    state.set(obj=env.robot, feature_name="room", feature_val=float(env.blocked_right_from))
    rooms = env.get_rooms()
    assert not [
        ground
        for ground in TestGroundingCannotCrossTheTwoKinds._applicable(state=state)
        if ground.skill.name == "MoveRoom"
        and ground.objects[1] == rooms[env.blocked_right_from]
        and ground.objects[2] == rooms[env.blocked_right_from + 1]
    ]


def test_move_room_can_still_cross_the_ledge_leftward() -> None:
    env = TossingRoomEnvironment()
    state = TossingRoomTasks(env=env, seed=0).sample_test_task().initial_state
    state.set(obj=env.robot, feature_name="room", feature_val=float(env.blocked_right_from + 1))
    rooms = env.get_rooms()
    assert [
        ground
        for ground in TestGroundingCannotCrossTheTwoKinds._applicable(state=state)
        if ground.skill.name == "MoveRoom" and ground.objects[2] == rooms[env.blocked_right_from]
    ]
