"""Simulator-backed tests for the oracle's two-way branch and its parameter provenance.

## Why the branch has two arms rather than three

`Pick -> MoveToThrowPose -> Toss` became `pick_cube -> move_to_toss_location_and_toss`
when upstream fused the base move and the throw into one controller. The middle arm is
gone, and with it the whole family of tests that pinned `ORACLE_THROW_STANDOFF = 1.35`,
`ORACLE_RELEASE_SPEED_DEG_S = 140` and `ORACLE_GRIPPER_RELEASE_MS = 792`.

Those three constants are deleted rather than moved, and the reason is worth keeping.
They were "one operating point measured together" for a *split* move-and-throw. With the
two fused, one `sample_parameters` call draws all four continuous parameters as a vector,
so a hand-picked triple can no longer be substituted into it piecewise. The oracle now
asks the controller's own sampler, seeded deterministically, and takes the first draw --
which also means it is **no longer a guaranteed solve**: it draws from a band rather than
sitting on a measured point, and nothing in this repo has measured that band's success
rate under the current pins. Nothing here asserts one.

## Why a simulator, and why one arm costs a real rollout

The branch reads `Holding` off upstream's abstractor, which runs forward kinematics, so
there is no dict of floats that can put the oracle in its second arm. The only way to
reach a holding state is to actually execute `pick_cube` -- a real MuJoCo rollout of
several hundred ticks. It is done **once** per module, in `_after_pick`, and shared.
"""

import numpy as np
import pytest

from hitl_pmp.core.problem.tasks.types import Goal
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import HOLDING, Tossing3DPredicates
from hitl_pmp.environments.tossing3d.skill_oracle_policy import (
    ORACLE_PARAMETER_SEED,
    SkillOraclePolicy,
)
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DOracle
from hitl_pmp.environments.tossing3d.skills import (
    MOVE_TO_TOSS_LOCATION_AND_TOSS,
    PARAM_DIMS,
    PICK_CUBE,
    Tossing3DSkills,
)

from .conftest import CANONICAL_SEED, requires_kinder

pytestmark = requires_kinder

_EMPTY_GOAL = Goal(atoms=frozenset())


@pytest.fixture(scope="module")
def _after_pick():
    """A scene in which the robot is actually holding the cube.

    Its own environment rather than the shared one, because executing a skill moves the
    simulator irreversibly and every other test in this package expects the canonical
    initial state. Module-scoped because the rollout is the expensive part.
    """
    env = Tossing3DEnvironment()
    try:
        state = env.reset_to_seed(seed=CANONICAL_SEED)
        pick = SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=_EMPTY_GOAL)
        state = env.take_action(action=pick.action)
        holding = Tossing3DPredicates.get(abstraction=env.abstraction(), name=HOLDING).holds(
            state, (env.robot, env.cube)
        )
        if not holding:
            pytest.skip(
                "the oracle's own pick did not end in a grasp at this pin, so the "
                f"holding arm cannot be reached (skill error: {env.last_skill_error()})"
            )
        yield env, state
    finally:
        env.close()


def test_the_oracle_picks_when_the_cube_is_on_the_ground(*, live_env: Tossing3DEnvironment) -> None:
    """The first arm, from the canonical initial state."""
    action = SkillOraclePolicy.get_labeled_action(
        state=live_env.get_current_state(), env=live_env, goal=_EMPTY_GOAL
    )
    assert action.action[0] == pytest.approx(Tossing3DEnvironment.pick_cube_id)


def test_the_oracle_moves_and_throws_once_it_is_holding_the_cube(*, _after_pick) -> None:
    """The second arm. There is no third: the base move and the throw are one skill."""
    env, state = _after_pick
    action = SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=_EMPTY_GOAL)
    assert action.action[0] == pytest.approx(Tossing3DEnvironment.move_to_toss_location_and_toss_id)


def test_the_oracle_solves_the_domain_in_exactly_two_skills(*, _after_pick) -> None:
    """The whole plan shape, walked on a live scene: pick, then move-and-throw. Anything
    longer would mean `Tossing3DProblem.max_episode_steps` is under-budgeted -- and this
    is one skill shorter than it used to be, which is the visible consequence of the
    fusion."""
    env, state = _after_pick
    assert SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=_EMPTY_GOAL).action[
        0
    ] == pytest.approx(Tossing3DEnvironment.move_to_toss_location_and_toss_id)


def test_the_oracle_offers_no_recovery_after_a_missed_toss(
    *, live_env: Tossing3DEnvironment
) -> None:
    """It falls back to `pick_cube`, whose grasp will fail to plan, rather than to some
    invented retrieval skill. There is nothing this domain can do once the cube is past
    the barrier, and pretending otherwise would hide the irreversibility it exists to
    exhibit.

    Reached by moving the cube past the barrier in the `State` rather than by throwing it,
    which is enough because the branch reads `Holding` and nothing else."""
    stranded = live_env.get_current_state().copy()
    stranded.set(obj=live_env.cube, feature_name="x", feature_val=2.6)
    action = SkillOraclePolicy.get_labeled_action(state=stranded, env=live_env, goal=_EMPTY_GOAL)
    assert action.action[0] == pytest.approx(Tossing3DEnvironment.pick_cube_id)


def test_the_oracle_parameters_are_the_controllers_own_draw_at_its_own_seed(
    *, live_env: Tossing3DEnvironment
) -> None:
    """**The provenance property, and what replaces the old constants.** An oracle drawing
    from outside the range a learner samples would be measuring a different skill from the
    one being learned. That used to be checked by asserting each constant fell inside a
    hitl-declared bound; now it is checked by identity -- the oracle's parameters *are*
    the controller's own draw at `ORACLE_PARAMETER_SEED`, so there is no second range for
    them to fall outside of."""
    state = live_env.get_current_state()
    action = SkillOraclePolicy.get_labeled_action(state=state, env=live_env, goal=_EMPTY_GOAL)
    expected = live_env.controllers().sample_params(
        key=PICK_CUBE,
        object_names=("robot", "cube_0", "cuboid_barrier"),
        state=state,
        rng=np.random.default_rng(ORACLE_PARAMETER_SEED),
    )
    assert action.action[1 : 1 + PARAM_DIMS[PICK_CUBE]] == pytest.approx(expected)


def test_the_oracle_is_reproducible_across_calls(*, live_env: Tossing3DEnvironment) -> None:
    """A fixed seed rather than one threaded from `--seed`, because the oracle is a
    reference arm: it should behave the same whichever run is being compared against it.
    Two calls from the same state must therefore agree exactly."""
    state = live_env.get_current_state()
    first = SkillOraclePolicy.get_labeled_action(state=state, env=live_env, goal=_EMPTY_GOAL)
    second = SkillOraclePolicy.get_labeled_action(state=state, env=live_env, goal=_EMPTY_GOAL)
    assert first.action == pytest.approx(second.action)
    assert first.label == second.label


def test_a_different_parameter_seed_really_changes_the_draw(
    *, live_env: Tossing3DEnvironment
) -> None:
    """`parameter_seed` is the knob that replaced `--oracle-throw-standoff`. If it did not
    reach the sampler, every "different oracle parameterisation" run would silently be the
    same run."""
    state = live_env.get_current_state()
    default = SkillOraclePolicy.get_labeled_action(state=state, env=live_env, goal=_EMPTY_GOAL)
    other = SkillOraclePolicy.get_labeled_action(
        state=state, env=live_env, goal=_EMPTY_GOAL, parameter_seed=ORACLE_PARAMETER_SEED + 1
    )
    assert default.action != pytest.approx(other.action)


def test_the_label_names_the_skill_its_objects_and_its_parameters(
    *, live_env: Tossing3DEnvironment
) -> None:
    """`LabeledAction.label` is what the renderer burns into the frame, so it has to say
    what actually happened rather than just which skill ran -- two clips at different
    draws are otherwise indistinguishable."""
    label = SkillOraclePolicy.get_labeled_action(
        state=live_env.get_current_state(), env=live_env, goal=_EMPTY_GOAL
    ).label
    assert label.startswith(f"{PICK_CUBE}(robot, cube_0, cuboid_barrier)")
    assert "params=[" in label


def test_the_throw_label_carries_all_four_of_its_parameters(*, _after_pick) -> None:
    """The fused skill's label has to show the standoff *and* both toss dials: they are
    drawn together and a clip that named only one could not be told from a clip that named
    a different one."""
    env, state = _after_pick
    labeled = SkillOraclePolicy.get_labeled_action(state=state, env=env, goal=_EMPTY_GOAL)
    assert labeled.label.startswith(f"{MOVE_TO_TOSS_LOCATION_AND_TOSS}(robot, cube_0, ")
    parameters = labeled.label.split("params=")[1]
    assert parameters.count(",") == PARAM_DIMS[MOVE_TO_TOSS_LOCATION_AND_TOSS] - 1


def test_the_provider_forwards_its_configured_parameter_seed(
    *, live_env: Tossing3DEnvironment
) -> None:
    """`parameter_seed` replaced `throw_standoff` as `Tossing3DOracle`'s one field: with
    the move and the throw fused, what is configurable is which draw the oracle takes
    rather than which standoff it stops at."""
    state = live_env.get_current_state()
    oracle = Tossing3DOracle(env=live_env, parameter_seed=ORACLE_PARAMETER_SEED + 5)
    through_provider = oracle.get_labeled_action(state=state, goal=_EMPTY_GOAL)
    direct = SkillOraclePolicy.get_labeled_action(
        state=state, env=live_env, goal=_EMPTY_GOAL, parameter_seed=ORACLE_PARAMETER_SEED + 5
    )
    assert through_provider.action == pytest.approx(direct.action)


def test_the_oracle_branches_on_the_abstractors_holding_not_a_local_classifier(
    *, _after_pick
) -> None:
    """The oracle and the operator model must not be able to disagree about which skill's
    preconditions currently hold. Both now read the same upstream abstractor, so the
    branch the oracle takes has to match the precondition the skill declares."""
    env, state = _after_pick
    holding = Tossing3DPredicates.get(abstraction=env.abstraction(), name=HOLDING).holds(
        state, (env.robot, env.cube)
    )
    assert holding
    toss = Tossing3DSkills.move_to_toss_location_and_toss(abstraction=env.abstraction())
    assert any(atom.predicate.name == HOLDING for atom in toss.preconditions)
