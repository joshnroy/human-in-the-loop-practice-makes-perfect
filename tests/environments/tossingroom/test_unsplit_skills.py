"""`--unsplit-skills`: one shared lifted `Throw` whose `item` ranges over both kinds.

The default symbolic layer splits the throw into `ThrowTrash`/`ThrowRecycling` so that
`EesMethod.sampler` -- which keys its `LearnedSkillSampler` dict by skill *name* -- gives
each kind its own classifier. This flag selects the other arm of that comparison: one
`Throw`, so both kinds' training rows pool into one classifier and trash experience can
transfer to recycling.

**The flag necessarily reaches past `Throw`.** `Type` has no hierarchy and
`SkillGrounder._applicable_groundings`/`GroundSkill` both match on exact type equality, so
an `item` variable that ranges over both kinds requires both item objects to carry ONE
type -- and the same argument applies to the bin the throw names, since a single lifted
throw has a single `?bin` parameter to bind. Sharing those types makes the per-kind
`Pickup` and `Press` unexpressible (a `PickupTrash` typed over the shared item would
ground with recycling too), and it brings back `BinAcceptsItem`/`ButtonForBin`, which stop
being tautologies the moment the types stop separating the kinds. Every assertion about
that widening is in `TestTheFlagCollapsesThreeSkillsNotOne` below.

The raw dynamics are untouched: `take_action` routes on the `kind` FEATURE, never on a
type, so both arms drive exactly the same world.
"""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.environments.tossingroom.environment import TossingRoomEnvironment
from hitl_pmp.environments.tossingroom.predicates import (
    BIN_ACCEPTS_ITEM,
    BIN_EMPTY,
    BIN_IN_ROOM,
    BUTTON_FOR_BIN,
    BUTTON_IN_ROOM,
    HAND_EMPTY,
    HOLDING,
    ITEM_IN_BIN,
    PILE_IN_ROOM,
    ROBOT_IN_ROOM,
)
from hitl_pmp.environments.tossingroom.problem import TossingRoomProblem
from hitl_pmp.environments.tossingroom.skill_provider import (
    TossingRoomOracle,
    TossingRoomSkillProvider,
)
from hitl_pmp.environments.tossingroom.skills import (
    TossingRoomSkills,
    TossingRoomUnsplitSkills,
)
from hitl_pmp.environments.tossingroom.tasks import TossingRoomGoalType, TossingRoomTasks
from hitl_pmp.methods.oracle.skill_oracle_method import SkillOracleMethod
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.practice_loop import PracticeLoop


def _unsplit_env() -> TossingRoomEnvironment:
    return TossingRoomEnvironment(unsplit_skills=True)


def _trash(*, env: TossingRoomEnvironment):
    return env.item_for_kind(kind=env.TRASH_KIND)


def _recycling(*, env: TossingRoomEnvironment):
    return env.item_for_kind(kind=env.RECYCLING_KIND)


class TestTheDefaultIsUnchanged:
    """The flag is off unless asked for, and off means the split layer verbatim."""

    @staticmethod
    def test_the_field_defaults_to_false() -> None:
        assert TossingRoomEnvironment().unsplit_skills is False
        assert TossingRoomEnvironment.model_fields["unsplit_skills"].default is False

    @staticmethod
    def test_the_default_skill_set_is_still_the_seven_split_skills() -> None:
        env = TossingRoomEnvironment()
        names = [skill.name for skill in TossingRoomSkillProvider(env=env).skills()]
        assert names == [
            "PickupTrash",
            "PickupRecycling",
            "MoveRoom",
            "ThrowTrash",
            "ThrowRecycling",
            "PressTrash",
            "PressRecycling",
        ]

    @staticmethod
    def test_the_default_objects_still_carry_the_split_types() -> None:
        env = TossingRoomEnvironment()
        assert _trash(env=env) is TossingRoomEnvironment.trash
        assert env.bin_for_kind(kind=env.TRASH_KIND) is TossingRoomEnvironment.trash_bin
        assert env.button_for_kind(kind=env.TRASH_KIND) is TossingRoomEnvironment.trash_button

    @staticmethod
    def test_the_default_state_is_keyed_by_the_split_typed_objects() -> None:
        """The byte-identity guarantee in miniature: a default run's `State` must be the
        same mapping it was before this flag existed, object identity included."""
        default_state = TossingRoomEnvironment().build_initial_state(weight_seed=0)
        assert TossingRoomEnvironment.trash in default_state.data
        assert TossingRoomEnvironment.trash_bin in default_state.data
        unsplit_state = _unsplit_env().build_initial_state(weight_seed=0)
        assert TossingRoomEnvironment.trash not in unsplit_state.data
        assert TossingRoomEnvironment.unsplit_trash in unsplit_state.data


class TestTheSharedThrow:
    """The flag's headline: one lifted `Throw`, and its `item` binds either kind."""

    @staticmethod
    def test_there_is_exactly_one_throw_and_it_is_named_throw() -> None:
        env = _unsplit_env()
        names = [skill.name for skill in TossingRoomSkillProvider(env=env).skills()]
        assert names.count("Throw") == 1
        assert "ThrowTrash" not in names
        assert "ThrowRecycling" not in names

    @staticmethod
    def test_its_item_and_bin_variables_are_the_shared_types() -> None:
        _robot, item, bin_var, _room = TossingRoomUnsplitSkills.THROW.parameters
        assert item.type == TossingRoomEnvironment.item_type
        assert bin_var.type == TossingRoomEnvironment.bin_type

    @staticmethod
    def test_the_same_item_variable_binds_both_kinds() -> None:
        """The property the whole flag exists for, asserted at the layer that decides it:
        both item objects carry the shared type, so ONE `?item` ranges over both."""
        env = _unsplit_env()
        _robot, item, _bin_var, _room = TossingRoomUnsplitSkills.THROW.parameters
        assert _trash(env=env).type == item.type
        assert _recycling(env=env).type == item.type
        rooms = env.get_rooms()
        for kind in (env.TRASH_KIND, env.RECYCLING_KIND):
            GroundSkill(
                skill=TossingRoomUnsplitSkills.THROW,
                objects=(
                    env.robot,
                    env.item_for_kind(kind=kind),
                    env.bin_for_kind(kind=kind),
                    rooms[env.bin_room_for_kind(kind=kind)],
                ),
            )

    @staticmethod
    def test_a_cross_kind_grounding_is_constructible_so_a_predicate_must_exclude_it() -> None:
        """The cost of the shared binding, stated rather than hidden. With one item type
        and one bin type, `Throw(trash -> recycling_bin)` is a well-typed grounding that
        `_apply_throw` (which routes purely by the HELD item's kind) could never satisfy.
        `BinAcceptsItem` is what rules it out, exactly as it did before the split -- so
        the predicate is a live constraint here, not the tautology it is by default."""
        env = _unsplit_env()
        rooms = env.get_rooms()
        crossed = GroundSkill(
            skill=TossingRoomUnsplitSkills.THROW,
            objects=(
                env.robot,
                _trash(env=env),
                env.bin_for_kind(kind=env.RECYCLING_KIND),
                rooms[env.recycling_bin_room],
            ),
        )
        state = env.build_initial_state(weight_seed=0)
        assert not BIN_ACCEPTS_ITEM.holds(state, (crossed.objects[1], crossed.objects[2]))
        assert BIN_ACCEPTS_ITEM.holds(
            state, (_trash(env=env), env.bin_for_kind(kind=env.TRASH_KIND))
        )

    @staticmethod
    def test_throw_declares_its_preconditions_and_effects() -> None:
        skill = TossingRoomUnsplitSkills.THROW
        robot, item, bin_var, room = skill.parameters
        assert skill.param_dim == 1
        assert skill.preconditions == frozenset({
            LiftedAtom(predicate=HOLDING, variables=(robot, item)),
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(robot, room)),
            LiftedAtom(predicate=BIN_IN_ROOM, variables=(bin_var, room)),
            LiftedAtom(predicate=BIN_ACCEPTS_ITEM, variables=(item, bin_var)),
            LiftedAtom(predicate=BIN_EMPTY, variables=(bin_var,)),
        })
        assert skill.add_effects == frozenset({
            LiftedAtom(predicate=ITEM_IN_BIN, variables=(item, bin_var)),
            LiftedAtom(predicate=HAND_EMPTY, variables=(robot,)),
        })
        assert skill.delete_effects == frozenset({
            LiftedAtom(predicate=HOLDING, variables=(robot, item)),
            LiftedAtom(predicate=BIN_EMPTY, variables=(bin_var,)),
        })

    @staticmethod
    def test_the_sampler_input_row_keeps_its_width() -> None:
        """Same architecture, different pooling: the shared `Throw` must present the same
        row width as either split throw, or "one sampler versus two" would be confounded
        with "a differently shaped network"."""
        split = [p.type.dim for p in TossingRoomSkills.THROW_TRASH.parameters]
        shared = [p.type.dim for p in TossingRoomUnsplitSkills.THROW.parameters]
        assert split == shared


class TestTheFlagCollapsesThreeSkillsNotOne:
    """`Pickup` and `Press` come along, because the shared types the throw needs make
    their per-kind versions unexpressible. Both are `param_dim=0`, so neither has a
    sampler and nothing about *learning* changes -- but the skill set really is four
    lifted skills where the default has seven, and that is the flag's whole footprint."""

    @staticmethod
    def test_the_skill_set_is_the_four_unsplit_skills() -> None:
        env = _unsplit_env()
        names = [skill.name for skill in TossingRoomSkillProvider(env=env).skills()]
        assert names == ["Pickup", "MoveRoom", "Throw", "Press"]

    @staticmethod
    def test_neither_pickup_nor_press_has_continuous_parameters() -> None:
        assert TossingRoomUnsplitSkills.PICKUP.param_dim == 0
        assert TossingRoomUnsplitSkills.PRESS.param_dim == 0

    @staticmethod
    def test_pickup_keeps_the_pile_room_precondition() -> None:
        skill = TossingRoomUnsplitSkills.PICKUP
        robot, item, room, pile = skill.parameters
        assert skill.preconditions == frozenset({
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(robot, room)),
            LiftedAtom(predicate=PILE_IN_ROOM, variables=(pile, room)),
            LiftedAtom(predicate=HAND_EMPTY, variables=(robot,)),
        })
        assert skill.add_effects == frozenset({
            LiftedAtom(predicate=HOLDING, variables=(robot, item))
        })

    @staticmethod
    def test_press_is_pinned_to_one_bin_and_one_item_by_predicates() -> None:
        """What the split enforced with types, the unsplit layer has to assert: without
        `ButtonForBin` a press would bind the other bin, and without `BinAcceptsItem` the
        in-bin atom it deletes would name the wrong item."""
        skill = TossingRoomUnsplitSkills.PRESS
        robot, button, room, bin_var, item = skill.parameters
        assert skill.preconditions == frozenset({
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(robot, room)),
            LiftedAtom(predicate=BUTTON_IN_ROOM, variables=(button, room)),
            LiftedAtom(predicate=BUTTON_FOR_BIN, variables=(button, bin_var)),
            LiftedAtom(predicate=BIN_ACCEPTS_ITEM, variables=(item, bin_var)),
        })
        assert skill.add_effects == frozenset({
            LiftedAtom(predicate=BIN_EMPTY, variables=(bin_var,))
        })
        assert skill.delete_effects == frozenset({
            LiftedAtom(predicate=ITEM_IN_BIN, variables=(item, bin_var))
        })
        assert skill.ignore_effects == frozenset()

    @staticmethod
    def test_the_two_dropped_predicates_come_back_and_only_under_the_flag() -> None:
        unsplit = {p.name for p in TossingRoomSkillProvider(env=_unsplit_env()).predicates()}
        assert {"BinAcceptsItem", "ButtonForBin"} <= unsplit
        assert not {"HoldingTrash", "TrashInBin", "TrashBinEmpty"} & unsplit

        split = {
            p.name for p in TossingRoomSkillProvider(env=TossingRoomEnvironment()).predicates()
        }
        assert not {"BinAcceptsItem", "ButtonForBin"} & split

    @staticmethod
    def test_the_shared_types_replace_the_split_ones() -> None:
        types = TossingRoomSkillProvider(env=_unsplit_env()).types()
        assert TossingRoomEnvironment.item_type in types
        assert TossingRoomEnvironment.bin_type in types
        assert TossingRoomEnvironment.button_type in types
        assert TossingRoomEnvironment.trash_type not in types
        assert TossingRoomEnvironment.trash_bin_type not in types


class TestTheRawWorldIsUnchanged:
    """`take_action` routes on the `kind` feature, never on a type, so both arms of the
    flag drive one world. Asserted on the encoded actions rather than argued."""

    @staticmethod
    def test_the_shared_throw_encodes_the_action_its_split_counterpart_does() -> None:
        split_env = TossingRoomEnvironment()
        unsplit_env = _unsplit_env()
        params = np.array([0.42])
        for kind, split_skill in (
            (split_env.TRASH_KIND, TossingRoomSkills.THROW_TRASH),
            (split_env.RECYCLING_KIND, TossingRoomSkills.THROW_RECYCLING),
        ):
            split_action = TossingRoomSkills.compute_action(
                ground_skill=GroundSkill(
                    skill=split_skill,
                    objects=(
                        split_env.robot,
                        split_env.item_for_kind(kind=kind),
                        split_env.bin_for_kind(kind=kind),
                        split_env.get_rooms()[split_env.bin_room_for_kind(kind=kind)],
                    ),
                ),
                params=params,
                state=split_env.build_initial_state(weight_seed=0),
            )
            unsplit_action = TossingRoomUnsplitSkills.compute_action(
                ground_skill=GroundSkill(
                    skill=TossingRoomUnsplitSkills.THROW,
                    objects=(
                        unsplit_env.robot,
                        unsplit_env.item_for_kind(kind=kind),
                        unsplit_env.bin_for_kind(kind=kind),
                        unsplit_env.get_rooms()[unsplit_env.bin_room_for_kind(kind=kind)],
                    ),
                ),
                params=params,
                state=unsplit_env.build_initial_state(weight_seed=0),
            )
            assert split_action.tolist() == unsplit_action.tolist()

    @staticmethod
    def test_both_arms_build_numerically_identical_initial_states() -> None:
        """Same features, same values, same object *names* -- only the types differ."""
        split_state = TossingRoomEnvironment().build_initial_state(weight_seed=7)
        unsplit_state = _unsplit_env().build_initial_state(weight_seed=7)
        split_by_name = {obj.name: features for obj, features in split_state.data.items()}
        unsplit_by_name = {obj.name: features for obj, features in unsplit_state.data.items()}
        assert set(split_by_name) == set(unsplit_by_name)
        for name, features in split_by_name.items():
            assert features.tolist() == unsplit_by_name[name].tolist()

    @staticmethod
    def test_the_evaluation_horizon_is_unchanged() -> None:
        """The horizon is a layout fact, so collapsing the symbolic layer must not move
        it -- otherwise the two arms would be scored over different episode lengths."""
        for env in (TossingRoomEnvironment(), _unsplit_env()):
            problem = TossingRoomProblem(env=env, tasks=TossingRoomTasks(env=env, seed=0))
            assert problem.max_episode_steps() == 12


@pytest.mark.parametrize("goal_type", list(TossingRoomGoalType))
def test_the_oracle_still_solves_every_goal_family_under_the_flag(
    *, goal_type: TossingRoomGoalType
) -> None:
    env = _unsplit_env()
    tasks = TossingRoomTasks(env=env, seed=0, forced_goal_type=goal_type)
    problem = TossingRoomProblem(env=env, tasks=tasks)
    method = SkillOracleMethod(env=env, oracle=TossingRoomOracle(env=env))
    for sample in (tasks.sample_train_task, tasks.sample_test_task):
        for _ in range(10):
            task = sample()
            solved, _, _ = problem.run_task_episode(
                task=task, policy=method.get_task_policy(task=task)
            )
            assert solved is True


def test_ees_plans_and_practices_through_real_fast_downward_under_the_flag() -> None:
    """The wiring check the split layer already has: the unsplit PDDL has to be something
    Fast Downward accepts, and the one `Throw` has to actually be practiced. Deliberately
    short (4 cycles x 60 steps) -- this is not a measurement, and nothing here compares
    the two arms."""
    env = _unsplit_env()
    tasks = TossingRoomTasks(env=env, seed=0, num_test_tasks=4)
    problem = TossingRoomProblem(env=env, tasks=tasks)
    method = EesMethod(
        env=env,
        skill_provider=TossingRoomSkillProvider(env=env),
        seed=0,
        sampler_max_train_iters=100,
    )
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=Metrics(),
        num_cycles=4,
        max_steps_per_interaction=60,
        num_test_tasks=4,
    )
    assert method.sampler(skill_name="Throw", param_dim=1).num_observations > 0
    # ...and no per-kind sampler exists at all, which is the arm's defining property.
    assert method.sampler(skill_name="ThrowTrash", param_dim=1).num_observations == 0
    assert method.sampler(skill_name="ThrowRecycling", param_dim=1).num_observations == 0
