"""Do the two throw skills really get two independent samplers?

The experiment this domain exists for reads a per-skill learning curve for `ThrowTrash`
and `ThrowRecycling` off one EES run and claims the two learn at different rates. That
claim is worthless unless the two samplers are genuinely separate: same architecture,
different weights, no shared training data. `EesMethod.sampler` keys `_samplers` by
`skill_name`, so separation *should* follow from the two lifted skills having different
names -- but "should follow from" is exactly the kind of assumption an experiment must
not rest on, so it is asserted here directly, through the real `EesMethod`.

**The control is the point.** `TestTheControl` runs the identical body against the same
domain under `--unsplit-skills`, whose single `Throw` name makes both kinds land in one
sampler with pooled data. It fails if the separation assertions were vacuously true of
any domain, and it also documents the transfer channel the default split removes: with
one `Throw` a trash throw's row trains the very classifier a recycling throw is later
scored by.

That control lived here before, against the *original* unsplit Tossing Room fork, and was
lost when the fork was retired. It is restored rather than reinvented: the flag makes a
shared lifted `Throw` expressible on this domain again, so the control is once more a
statement about two symbolic layers over one world instead of about two domains.

The item weights are set explicitly below rather than left at `canonical_item_weight`.
Both items start at the same placeholder value (nothing has been picked up yet), so the
by-content assertions would be vacuous on an untouched initial state -- they would be
looking for values no row could contain either way.

Nothing here runs Fast Downward: `execute_ground_skill` + `observe_sampler_outcome` is
the whole path from "a skill was practiced" to "a sampler was trained", so driving those
two directly exercises the real mechanism without a planner in the loop.
"""

import numpy as np

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossingroom.environment import (
    TossingRoomEnvironment,
)
from hitl_pmp.environments.tossingroom.skill_provider import (
    TossingRoomSkillProvider,
)
from hitl_pmp.environments.tossingroom.skills import (
    TossingRoomSkills,
    TossingRoomUnsplitSkills,
)
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

# The two kinds' weights, set into every state below. `weight` is one of the two
# observable CAUSES of the required throw force and sits in each throw's own sampler input
# row, so distinct values make a leaked row identifiable by CONTENT rather than only by
# count. Both items would otherwise carry the same placeholder `canonical_item_weight`.
_TRASH_WEIGHT = 0.7
_RECYCLING_WEIGHT = 0.3


def _split_method() -> tuple[EesMethod, TossingRoomEnvironment]:
    env = TossingRoomEnvironment()
    method = EesMethod(env=env, skill_provider=TossingRoomSkillProvider(env=env), seed=0)
    return method, env


def _state_with_distinct_weights(*, env: TossingRoomEnvironment):
    state = env.build_initial_state(weight_seed=0)
    state.set(
        obj=env.item_for_kind(kind=env.TRASH_KIND),
        feature_name="weight",
        feature_val=_TRASH_WEIGHT,
    )
    state.set(
        obj=env.item_for_kind(kind=env.RECYCLING_KIND),
        feature_name="weight",
        feature_val=_RECYCLING_WEIGHT,
    )
    return state


def _split_throws(*, env: TossingRoomEnvironment) -> tuple[GroundSkill, GroundSkill]:
    rooms = env.get_rooms()
    trash = GroundSkill(
        skill=TossingRoomSkills.THROW_TRASH,
        objects=(env.robot, env.trash, env.trash_bin, rooms[env.trash_bin_room]),
    )
    recycling = GroundSkill(
        skill=TossingRoomSkills.THROW_RECYCLING,
        objects=(env.robot, env.recycling, env.recycling_bin, rooms[env.recycling_bin_room]),
    )
    return trash, recycling


def _practice(*, method: EesMethod, ground_skill: GroundSkill, state, success: bool) -> None:
    """One full practice execution: pick a parameter through the skill's own sampler,
    then record the outcome exactly as `_EesEpisode.observe_pending` does."""
    _labeled, record = method.execute_ground_skill(
        ground_skill=ground_skill, state=state, explore=True
    )
    assert record is not None, "a param_dim>0 skill practiced with explore=True must record"
    method.observe_sampler_outcome(
        skill_name=record.skill_name,
        param_dim=record.param_dim,
        sampler_input=record.sampler_input,
        success=success,
    )


class TestTheTwoThrowSamplersAreSeparate:
    @staticmethod
    def test_they_are_different_sampler_objects() -> None:
        method, _env = _split_method()
        trash_sampler = method.sampler(skill_name="ThrowTrash", param_dim=1)
        recycling_sampler = method.sampler(skill_name="ThrowRecycling", param_dim=1)
        assert trash_sampler is not recycling_sampler
        assert trash_sampler.skill_name == "ThrowTrash"
        assert recycling_sampler.skill_name == "ThrowRecycling"

    @staticmethod
    def test_practising_one_leaves_the_other_with_no_data() -> None:
        method, env = _split_method()
        state = _state_with_distinct_weights(env=env)
        trash_throw, recycling_throw = _split_throws(env=env)

        for _ in range(12):
            _practice(method=method, ground_skill=trash_throw, state=state, success=True)

        assert method.sampler(skill_name="ThrowTrash", param_dim=1).num_observations == 12
        assert method.sampler(skill_name="ThrowRecycling", param_dim=1).num_observations == 0

    @staticmethod
    def test_each_sampler_holds_exactly_its_own_rows() -> None:
        """Stronger than the counts: the recycling sampler must not contain any row
        recorded for a trash throw, matched by content rather than by count."""
        method, env = _split_method()
        state = _state_with_distinct_weights(env=env)
        trash_throw, recycling_throw = _split_throws(env=env)

        for _ in range(8):
            _practice(method=method, ground_skill=trash_throw, state=state, success=True)
        for _ in range(3):
            _practice(method=method, ground_skill=recycling_throw, state=state, success=False)

        trash_rows = method.sampler(skill_name="ThrowTrash", param_dim=1).observed_inputs()
        recycling_rows = method.sampler(skill_name="ThrowRecycling", param_dim=1).observed_inputs()
        assert len(trash_rows) == 8
        assert len(recycling_rows) == 3
        # The item's weight feature -- one of the two observable CAUSES of the required
        # throw force -- differs between the two kinds in this state, so a leaked row is
        # identifiable by content, not just by count. Non-vacuity first: each sampler
        # really does hold its own kind's weight, so the two absences below are absences
        # of something that would otherwise be there.
        assert [row for row in trash_rows if _TRASH_WEIGHT in row]
        assert [row for row in recycling_rows if _RECYCLING_WEIGHT in row]
        assert not [row for row in recycling_rows if _TRASH_WEIGHT in row]
        assert not [row for row in trash_rows if _RECYCLING_WEIGHT in row]

    @staticmethod
    def test_a_trained_trash_sampler_leaves_the_recycling_sampler_untrained() -> None:
        """The transfer question in its operational form: after fitting, does the
        recycling sampler score anything at all? An unfitted `LearnedSkillSampler`
        returns a flat 0.5 for every candidate, i.e. it is still choosing uniformly."""
        method, env = _split_method()
        state = _state_with_distinct_weights(env=env)
        trash_throw, _recycling_throw = _split_throws(env=env)
        rng = np.random.default_rng(0)
        for _ in range(20):
            params = np.array([rng.uniform()])
            row = method.sampler_input_row(ground_skill=trash_throw, state=state, params=params)
            method.observe_sampler_outcome(
                skill_name="ThrowTrash",
                param_dim=1,
                sampler_input=row,
                success=bool(abs(params[0] - 0.7) < 0.1),
            )
        method.fit_samplers()

        assert method.sampler(skill_name="ThrowTrash", param_dim=1).is_fitted
        assert not method.sampler(skill_name="ThrowRecycling", param_dim=1).is_fitted


class TestTheControl:
    """Run the same body on the *unsplit* symbolic layer (`--unsplit-skills`). If these
    pass, the assertions above are measuring something real rather than restating a truth
    about `EesMethod`."""

    @staticmethod
    def test_the_shared_throw_pools_both_kinds_into_one_sampler() -> None:
        env = TossingRoomEnvironment(unsplit_skills=True)
        method = EesMethod(env=env, skill_provider=TossingRoomSkillProvider(env=env), seed=0)
        rooms = env.get_rooms()
        state = _state_with_distinct_weights(env=env)
        trash_throw = GroundSkill(
            skill=TossingRoomUnsplitSkills.THROW,
            objects=(
                env.robot,
                env.item_for_kind(kind=env.TRASH_KIND),
                env.bin_for_kind(kind=env.TRASH_KIND),
                rooms[env.trash_bin_room],
            ),
        )
        recycling_throw = GroundSkill(
            skill=TossingRoomUnsplitSkills.THROW,
            objects=(
                env.robot,
                env.item_for_kind(kind=env.RECYCLING_KIND),
                env.bin_for_kind(kind=env.RECYCLING_KIND),
                rooms[env.recycling_bin_room],
            ),
        )
        for _ in range(8):
            _practice(method=method, ground_skill=trash_throw, state=state, success=True)
        for _ in range(3):
            _practice(method=method, ground_skill=recycling_throw, state=state, success=False)

        # One sampler, both kinds' rows pooled into it -- the transfer channel the default
        # split removes.
        sampler = method.sampler(skill_name="Throw", param_dim=1)
        assert sampler.num_observations == 11
        rows = sampler.observed_inputs()
        assert [row for row in rows if _TRASH_WEIGHT in row]
        assert [row for row in rows if _RECYCLING_WEIGHT in row]
        # ...and the per-kind names the split arm uses are not in play at all here.
        assert method.sampler(skill_name="ThrowTrash", param_dim=1).num_observations == 0
        assert method.sampler(skill_name="ThrowRecycling", param_dim=1).num_observations == 0

    @staticmethod
    def test_one_kinds_practice_trains_the_classifier_the_other_is_scored_by() -> None:
        """The transfer question in the same operational form the split arm answers `no`
        to: after fitting on trash rows alone, is the recycling throw's sampler fitted?
        Under one shared `Throw` it is the same object, so it is."""
        env = TossingRoomEnvironment(unsplit_skills=True)
        method = EesMethod(env=env, skill_provider=TossingRoomSkillProvider(env=env), seed=0)
        rooms = env.get_rooms()
        state = _state_with_distinct_weights(env=env)
        trash_throw = GroundSkill(
            skill=TossingRoomUnsplitSkills.THROW,
            objects=(
                env.robot,
                env.item_for_kind(kind=env.TRASH_KIND),
                env.bin_for_kind(kind=env.TRASH_KIND),
                rooms[env.trash_bin_room],
            ),
        )
        rng = np.random.default_rng(0)
        for _ in range(20):
            params = np.array([rng.uniform()])
            row = method.sampler_input_row(ground_skill=trash_throw, state=state, params=params)
            method.observe_sampler_outcome(
                skill_name="Throw",
                param_dim=1,
                sampler_input=row,
                success=bool(abs(params[0] - 0.7) < 0.1),
            )
        method.fit_samplers()
        assert method.sampler(skill_name="Throw", param_dim=1).is_fitted
