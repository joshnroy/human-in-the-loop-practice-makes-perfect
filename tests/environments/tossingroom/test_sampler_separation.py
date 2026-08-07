"""Do the two throw skills really get two independent samplers?

The experiment this domain exists for reads a per-skill learning curve for `ThrowTrash`
and `ThrowRecycling` off one EES run and claims the two learn at different rates. That
claim is worthless unless the two samplers are genuinely separate: same architecture,
different weights, no shared training data. `EesMethod.sampler` keys `_samplers` by
`skill_name`, so separation *should* follow from the two lifted skills having different
names -- but "should follow from" is exactly the kind of assumption an experiment must
not rest on, so it is asserted here directly, through the real `EesMethod`.

**The control is currently missing, and that is a known gap.** This file used to run the
identical body against the *original* Tossing Room fork -- a different domain that
happened to share this one's present name -- whose single `Throw` name made both kinds
land in one sampler with pooled data. That was a negative control: it fails if the
separation assertions were vacuously true of any domain. The fork has been retired, so
the control went with it. It is restorable, and should be restored, once a shared lifted
`Throw` is expressible on this domain again; until then the assertions below are weaker
than they read.

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
)
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod


def _split_method() -> tuple[EesMethod, TossingRoomEnvironment]:
    env = TossingRoomEnvironment()
    method = EesMethod(env=env, skill_provider=TossingRoomSkillProvider(env=env), seed=0)
    return method, env


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
        state = env.build_initial_state(
            weight_seed=0,
        )
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
        state = env.build_initial_state(
            weight_seed=0,
        )
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
        # identifiable by content, not just by count.
        assert not [row for row in recycling_rows if 0.7 in row]
        assert not [row for row in trash_rows if 0.3 in row]

    @staticmethod
    def test_a_trained_trash_sampler_leaves_the_recycling_sampler_untrained() -> None:
        """The transfer question in its operational form: after fitting, does the
        recycling sampler score anything at all? An unfitted `LearnedSkillSampler`
        returns a flat 0.5 for every candidate, i.e. it is still choosing uniformly."""
        method, env = _split_method()
        state = env.build_initial_state(
            weight_seed=0,
        )
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
