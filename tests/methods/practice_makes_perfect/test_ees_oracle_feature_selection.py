"""EES's use of a domain's oracle feature selection for the learned sampler --
predicators' `active_sampler_learning_feature_selection = "oracle"`, which Ball-Ring
needs to reproduce the paper's Figure 4 cup-placement curve. EES stays
domain-agnostic: it reaches oracle features only through
`SkillProvider.oracle_sampler_input`, falling back to the `"all"` layout otherwise.
Light Switch defines no oracle features, so it must be entirely unaffected.
"""

import numpy as np

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.ballring.environment import BallRingEnvironment
from hitl_pmp.environments.ballring.skill_provider import BallRingSkillProvider
from hitl_pmp.environments.ballring.skills import BallRingSkills
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.environments.lightswitch.skills import LightSwitchSkills
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.methods.practice_makes_perfect.wrapped_sampler import LearnedSkillSampler

BE = BallRingEnvironment


def _ballring_method_state():
    env = BE()
    method = EesMethod(env=env, skill_provider=BallRingSkillProvider(env=env), seed=0)
    state = env.sample_initial_state(rng=np.random.default_rng(0))
    return method, env, state


def _place_cup(*, env) -> GroundSkill:
    return GroundSkill(
        skill=BallRingSkills.PLACE_CUP_WITHOUT_BALL_ON_TABLE,
        objects=(env.robot, env.ball, env.cup, env.target_table()),
    )


def test_ees_builds_the_oracle_row_for_the_cup_placement_skill() -> None:
    method, env, state = _ballring_method_state()
    place_cup = _place_cup(env=env)
    params = np.array([0.4, 0.7])
    row = method.sampler_input_row(ground_skill=place_cup, state=state, params=params)
    # Exactly the curated oracle vector (bias + 7 table features + 2 placement coords).
    assert len(row) == 10
    assert row == BallRingSkills.oracle_sampler_input(
        ground_skill=place_cup, state=state, params=params
    )


def test_ees_falls_back_to_all_features_for_place_ball_on_table() -> None:
    method, env, state = _ballring_method_state()
    place_ball = GroundSkill(
        skill=BallRingSkills.PLACE_BALL_ON_TABLE,
        objects=(env.robot, env.ball, env.cup, env.target_table()),
    )
    params = np.array([0.4, 0.7])
    row = method.sampler_input_row(ground_skill=place_ball, state=state, params=params)
    expected = LearnedSkillSampler.build_sampler_input(
        state_features=method.state_features(ground_skill=place_ball, state=state), params=params
    )
    assert row == expected
    # bias + concat(robot 2, ball 4, cup 4, table 7) + params 2.
    assert len(row) == 1 + (2 + 4 + 4 + 7) + 2


def test_light_switch_is_unaffected_and_still_uses_all_features() -> None:
    env = LightSwitchEnvironment(grid_size=4)
    method = EesMethod(env=env, skill_provider=LightSwitchSkillProvider(env=env), seed=0)
    state = env.build_initial_state(light_level=0.2, light_target=0.8)
    cells = env.get_cells()
    turn_on = GroundSkill(
        skill=LightSwitchSkills.TURN_ON_LIGHT, objects=(env.robot, cells[-1], env.light)
    )
    params = np.array([0.5])
    # No oracle features defined for Light Switch -> the hook declines.
    assert (
        LightSwitchSkillProvider(env=env).oracle_sampler_input(
            ground_skill=turn_on, state=state, params=params
        )
        is None
    )
    row = method.sampler_input_row(ground_skill=turn_on, state=state, params=params)
    expected = LearnedSkillSampler.build_sampler_input(
        state_features=method.state_features(ground_skill=turn_on, state=state), params=params
    )
    assert row == expected


def test_recorded_training_row_matches_the_action_the_skill_actually_commands() -> None:
    """The train/score-consistency crux: the classifier input stored for a practice
    attempt uses the *converted* placement coordinates, which must equal the (x, y)
    the realized action commands -- otherwise the sampler would be trained on inputs
    it will never be scored on."""
    method, env, state = _ballring_method_state()
    place_cup = _place_cup(env=env)
    labeled, record = method.execute_ground_skill(ground_skill=place_cup, state=state, explore=True)
    assert record is not None
    assert record.param_dim == 2
    assert len(record.sampler_input) == 10
    # The stored row's last two entries are the placement coords; the action commands
    # a place-onto-table at exactly those coords ([1, 3, 0, x, y]).
    assert record.sampler_input[-2] == float(labeled.action[3])
    assert record.sampler_input[-1] == float(labeled.action[4])


def test_the_input_row_is_state_dependent_so_it_must_be_snapshotted_at_decision_time() -> None:
    """Why EES stores the decision-time row instead of rebuilding it at observe time:
    the oracle row reads the table's geometry, which the environment can move (a
    fallen object, a later placement). Rebuilding after the state changed would train
    the classifier on a different row than the one that was scored."""
    method, env, state = _ballring_method_state()
    place_cup = _place_cup(env=env)
    params = np.array([0.4, 0.7])
    row_before = method.sampler_input_row(ground_skill=place_cup, state=state, params=params)

    moved = state.model_copy(deep=True)
    moved.set(obj=env.target_table(), feature_name="x", feature_val=0.123)
    row_after = method.sampler_input_row(ground_skill=place_cup, state=moved, params=params)

    assert row_before != row_after  # so the decision-time snapshot genuinely matters
