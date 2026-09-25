"""The one toss both layouts share: [standoff, speed, release], controller direction.

Offline. The direction selector is replaced by a recording stub wherever a state
without simulator geometry is used, so these tests pin the plumbing -- what the
sampler draws, what reaches the controller, what the classifier sees -- rather than
the selection rule, which `test_toss_direction.py` covers.
"""

import math

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
from hitl_pmp.environments.tossing3d.recovery_skills import SameSideSkills
from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.environments.tossing3d.toss import Tossing3DToss
from hitl_pmp.environments.tossing3d.toss_direction import (
    NoFeasibleTossDirectionError,
    TossDirectionChoice,
    TossDirectionSelector,
)
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import (
    FAR_STAND_X_LIMIT_M,
    WIDE_TOSS_RELEASE_MS_BOUNDS,
    WIDE_TOSS_SPEED_BOUNDS,
    WIDE_TOSS_STANDOFF_BOUNDS,
    WideLongRangeTossProposal,
)

from .observations import state


def _toss(*, env: Tossing3DEnvironment, side=Tossing3DSides.opposite) -> GroundSkill:
    return GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier, side),
    )


@pytest.fixture
def fixed_direction(*, monkeypatch: pytest.MonkeyPatch):
    """Choose `direction_deg` for every standoff, recording the standoffs asked about."""

    def install(*, direction_deg: int) -> list[float]:
        asked: list[float] = []

        def select_for_state(*, state, standoff: float) -> TossDirectionChoice:
            del state
            asked.append(standoff)
            return TossDirectionChoice(
                direction_deg=direction_deg,
                rotation=math.radians(direction_deg),
                stand_xy=(0.0, 0.0),
                clearance_m=1.0,
            )

        monkeypatch.setattr(TossDirectionSelector, "select_for_state", select_for_state)
        return asked

    return install


def test_the_toss_has_three_learned_parameters() -> None:
    assert Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS.param_dim == 3
    assert Tossing3DToss.PARAM_BOUNDS == (
        WIDE_TOSS_STANDOFF_BOUNDS,
        WIDE_TOSS_SPEED_BOUNDS,
        WIDE_TOSS_RELEASE_MS_BOUNDS,
    )


def test_the_proposal_draws_standoff_speed_release_and_no_yaw() -> None:
    rng = np.random.default_rng(7)
    direct = np.random.default_rng(7)
    for _ in range(20):
        drawn = WideLongRangeTossProposal.sample(rng=rng)
        assert drawn.shape == (3,)
        assert tuple(drawn) == (
            float(direct.uniform(*WIDE_TOSS_STANDOFF_BOUNDS)),
            float(direct.uniform(*WIDE_TOSS_SPEED_BOUNDS)),
            float(direct.uniform(*WIDE_TOSS_RELEASE_MS_BOUNDS)),
        )
        assert WideLongRangeTossProposal.contains(params=drawn)


def test_compute_action_sends_the_chosen_direction_as_the_controller_rotation(
    *, fixed_direction
) -> None:
    asked = fixed_direction(direction_deg=270)
    env = Tossing3DEnvironment()
    action = Tossing3DSkillProvider(env=env).compute_action(
        ground_skill=_toss(env=env), params=np.array([1.31, 128.5, 733.0]), state=state(env=env)
    )
    assert list(action) == pytest.approx([
        Tossing3DEnvironment.move_to_toss_location_and_toss_id,
        1.31,
        1.5 * math.pi,
        128.5,
        733.0,
    ])
    assert asked == [1.31]


def test_the_classifier_row_is_bias_then_the_three_throw_parameters() -> None:
    env = Tossing3DEnvironment()
    params = np.array([1.3, 125.0, 760.0])
    for scene in (
        state(env=env, base_x=0.2, base_y=-0.4, base_rot=np.pi / 2, bin_x=1.7),
        state(env=env, base_x=-1.0, base_rot=0.0, bin_x=3.2),
    ):
        assert Tossing3DSkillProvider(env=env).hand_selected_feature_transform(
            ground_skill=_toss(env=env), state=scene, params=params
        ) == [1.0, 1.3, 125.0, 760.0]


def test_the_direction_is_reported_as_its_own_annotation(*, fixed_direction) -> None:
    fixed_direction(direction_deg=90)
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    toss = _toss(env=env)
    action = provider.compute_action(
        ground_skill=toss, params=np.array([1.5, 200.0, 600.0]), state=state(env=env)
    )
    assert provider.action_annotations(ground_skill=toss, action=action) == {
        "toss_direction_deg": 90.0
    }
    pick = GroundSkill(
        skill=Tossing3DSkills.PICK_CUBE,
        objects=(env.robot, env.cube, env.barrier, Tossing3DSides.robot, env.bin),
    )
    assert provider.action_annotations(ground_skill=pick, action=np.zeros(5)) == {}


def test_a_standoff_with_no_feasible_direction_raises_from_proposal_checking(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_direction(*, state, standoff: float) -> TossDirectionChoice:
        del state
        raise NoFeasibleTossDirectionError(
            standoff=standoff,
            bin_pose=(2.0, 0.0, 0.0),
            robot_pose=(0.0, 0.0, 0.0),
            reasons={0: "base_motion_plan_failed"},
        )

    monkeypatch.setattr(TossDirectionSelector, "select_for_state", no_direction)
    env = Tossing3DEnvironment()
    with pytest.raises(NoFeasibleTossDirectionError):
        Tossing3DSkillProvider(env=env).parameter_rejection_reason(
            ground_skill=_toss(env=env), params=np.array([1.5, 200.0, 600.0]), state=state(env=env)
        )


def test_ees_does_not_turn_a_no_direction_standoff_into_an_empty_pool(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The empty-pool replan catches `NoFeasibleParametersError`; this must escape."""
    from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

    def no_direction(*, state, standoff: float) -> TossDirectionChoice:
        del state
        raise NoFeasibleTossDirectionError(
            standoff=standoff, bin_pose=(2.0, 0.0, 0.0), robot_pose=(0.0, 0.0, 0.0), reasons={}
        )

    monkeypatch.setattr(TossDirectionSelector, "select_for_state", no_direction)
    env = Tossing3DEnvironment()
    method = EesMethod(env=env, skill_provider=Tossing3DSkillProvider(env=env), seed=0)
    with pytest.raises(NoFeasibleTossDirectionError):
        method.sample_parameter_candidates(
            ground_skill=_toss(env=env), state=state(env=env), explore=True
        )


def test_same_side_and_barrier_layouts_share_one_toss_implementation(*, fixed_direction) -> None:
    """Decision 4, as identity rather than as equal-looking copies: both layouts route
    the toss through the same `Tossing3DToss` callables, with the same bounds, the same
    sampled stream for the same rng, the same action and the same classifier row."""
    fixed_direction(direction_deg=180)
    barrier_env = Tossing3DEnvironment()
    same_env = Tossing3DEnvironment(layout=Tossing3DLayout.SAME_SIDE)
    barrier = Tossing3DSkillProvider(env=barrier_env)
    same_side = Tossing3DSkillProvider(env=same_env)
    barrier_toss = _toss(env=barrier_env, side=Tossing3DSides.opposite)
    same_toss = _toss(env=same_env, side=Tossing3DSides.robot)

    assert SameSideSkills.TOSS is Tossing3DToss
    assert same_toss.skill is barrier_toss.skill
    assert SameSideSkills.TOSS.PARAM_BOUNDS is Tossing3DToss.PARAM_BOUNDS

    streams = [np.random.default_rng(11) for _ in range(4)]
    for _ in range(25):
        drawn = [
            barrier.sample_params(ground_skill=barrier_toss, rng=streams[0]),
            same_side.sample_params(ground_skill=same_toss, rng=streams[1]),
            SameSideSkills.sample_params(ground_skill=same_toss, rng=streams[2]),
            Tossing3DToss.sample_params(rng=streams[3]),
        ]
        for other in drawn[1:]:
            assert np.array_equal(drawn[0], other)
        scene = state(env=barrier_env)
        same_scene = state(env=same_env)
        assert np.array_equal(
            barrier.compute_action(ground_skill=barrier_toss, params=drawn[0], state=scene),
            same_side.compute_action(ground_skill=same_toss, params=drawn[0], state=same_scene),
        )
        assert barrier.hand_selected_feature_transform(
            ground_skill=barrier_toss, state=scene, params=drawn[0]
        ) == same_side.hand_selected_feature_transform(
            ground_skill=same_toss, state=same_scene, params=drawn[0]
        )


def test_both_layouts_dispatch_through_the_same_toss_callables(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replace each shared callable with a sentinel: both layouts must reach it."""
    calls: list[tuple[str, str]] = []

    def recorder(*, name: str):
        def record(**kwargs):
            del kwargs
            calls.append((name, current[0]))
            return {
                "sample_params": np.zeros(3),
                "compute_action": np.zeros(5),
                "rejection_reason": None,
                "feature_row": [1.0],
            }[name]

        return record

    for name in ("sample_params", "compute_action", "rejection_reason", "feature_row"):
        monkeypatch.setattr(Tossing3DToss, name, recorder(name=name))
    current = [""]
    for layout in (Tossing3DLayout.BARRIER, Tossing3DLayout.SAME_SIDE):
        current[0] = layout.value
        env = Tossing3DEnvironment(layout=layout)
        provider = Tossing3DSkillProvider(env=env)
        toss = _toss(env=env)
        scene = state(env=env)
        provider.sample_params(ground_skill=toss, rng=np.random.default_rng(0))
        provider.compute_action(ground_skill=toss, params=np.zeros(3), state=scene)
        provider.parameter_rejection_reason(ground_skill=toss, params=np.zeros(3), state=scene)
        provider.hand_selected_feature_transform(ground_skill=toss, state=scene, params=np.zeros(3))
    for layout in (Tossing3DLayout.BARRIER, Tossing3DLayout.SAME_SIDE):
        assert [name for name, seen in calls if seen == layout.value] == [
            "sample_params",
            "compute_action",
            "rejection_reason",
            "feature_row",
        ]


def test_the_state_log_carries_three_params_and_the_direction_as_its_own_field() -> None:
    from hitl_pmp.environments.tossing3d.state_log import SkillEvent

    toss = np.array([
        Tossing3DEnvironment.move_to_toss_location_and_toss_id,
        1.4,
        math.pi,
        250.0,
        610.0,
    ])
    params, direction = Tossing3DEnvironment.skill_log_params(action=toss)
    assert params == (1.4, 250.0, 610.0)
    assert direction == pytest.approx(180.0)
    event = SkillEvent(
        name="MoveToTossLocationAndToss", objects=(), params=params, toss_direction_deg=direction
    )
    assert SkillEvent(**event.model_dump()).toss_direction_deg == pytest.approx(180.0)
    pick = np.array([Tossing3DEnvironment.pick_cube_id, 0.0, 0.0, 0.0, 0.0])
    assert Tossing3DEnvironment.skill_log_params(action=pick) == ((0.0, 0.0, 0.0, 0.0), None)


@pytest.mark.parametrize(
    ("bin_x", "base_x", "expected_low"),
    [
        (3.2, 0.0, 3.2 - FAR_STAND_X_LIMIT_M),  # far, floor lifted by the stand line
        (3.42, 0.0, 3.42 - FAR_STAND_X_LIMIT_M),  # farthest receiver
        (1.6, 0.0, 1.25),  # far but near the barrier: the controller floor binds
        (-0.35, 0.0, 1.25),  # robot side: the controller floor binds
    ],
)
def test_the_standoff_band_is_a_function_of_the_bin_pose(
    *, bin_x: float, base_x: float, expected_low: float
) -> None:
    env = Tossing3DEnvironment()
    low, high = Tossing3DToss.standoff_bounds(
        ground_skill=_toss(env=env), state=state(env=env, bin_x=bin_x, base_x=base_x)
    )
    assert low == pytest.approx(expected_low)
    assert high == WIDE_TOSS_STANDOFF_BOUNDS[1]


def test_a_far_bin_draws_its_standoff_uniformly_inside_its_band_without_rejection() -> None:
    """Same three uniforms consumed per draw as the unbanded proposal; the standoff is
    drawn over the per-bin band directly, so nothing is ever redrawn."""
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    scene = state(env=env, bin_x=3.2)
    rng = np.random.default_rng(3)
    direct = np.random.default_rng(3)
    low = 3.2 - FAR_STAND_X_LIMIT_M
    standoffs = []
    for _ in range(200):
        drawn = provider.sample_params_at_state(ground_skill=_toss(env=env), rng=rng, state=scene)
        assert tuple(drawn) == (
            float(direct.uniform(low, WIDE_TOSS_STANDOFF_BOUNDS[1])),
            float(direct.uniform(*WIDE_TOSS_SPEED_BOUNDS)),
            float(direct.uniform(*WIDE_TOSS_RELEASE_MS_BOUNDS)),
        )
        standoffs.append(float(drawn[0]))
    assert min(standoffs) >= low
    assert max(standoffs) - min(standoffs) > 0.9 * (WIDE_TOSS_STANDOFF_BOUNDS[1] - low)


def test_ees_draws_toss_candidates_at_the_decision_state(*, fixed_direction) -> None:
    from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

    fixed_direction(direction_deg=0)
    env = Tossing3DEnvironment()
    method = EesMethod(
        env=env, skill_provider=Tossing3DSkillProvider(env=env), seed=0, num_candidates=20
    )
    candidates = method.sample_parameter_candidates(
        ground_skill=_toss(env=env), state=state(env=env, bin_x=3.3), explore=True
    )
    assert len(candidates) == 20
    assert min(float(candidate[0]) for candidate in candidates) >= 3.3 - FAR_STAND_X_LIMIT_M


def test_random_skills_draws_toss_parameters_at_the_decision_state(*, fixed_direction) -> None:
    from hitl_pmp.methods.practice_makes_perfect.random_skills_method import (
        RandomSkillsMethod,
    )

    fixed_direction(direction_deg=0)
    env = Tossing3DEnvironment()
    method = RandomSkillsMethod(env=env, skill_provider=Tossing3DSkillProvider(env=env), seed=0)
    from .observations import HOLDING_ATOMS

    scene = state(env=env, bin_x=3.3, abstract_atoms=HOLDING_ATOMS, cube_z=0.4)
    for _ in range(20):
        labeled, ground_skill = method.choose_ground_skill(state=scene)
        if ground_skill is not None and ground_skill.skill.param_dim == 3:
            assert float(labeled.action[1]) >= 3.3 - FAR_STAND_X_LIMIT_M
