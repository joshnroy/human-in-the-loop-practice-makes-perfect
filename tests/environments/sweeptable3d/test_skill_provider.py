"""Provider contracts, independent sweep samples and oracle geometry without MuJoCo."""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.core.problem.tasks.types import Goal
from hitl_pmp.environments.sweeptable3d.objects import SweepTable3DObjects as O
from hitl_pmp.environments.sweeptable3d.predicates import SweepTable3DPredicates
from hitl_pmp.environments.sweeptable3d.skill_provider import (
    SweepTable3DActions,
    SweepTable3DOracle,
    SweepTable3DSkillProvider,
)
from hitl_pmp.environments.sweeptable3d.skills import NOOP_ID, SKILL_IDS
from hitl_pmp.environments.sweeptable3d.skills import SweepTable3DSkills as S

from .test_symbolic import SymbolicScene


class SymbolicEnvironment(Environment):
    def take_action(self, *, action: Action) -> State:
        raise AssertionError("symbolic provider must not step the environment")

    def get_valid_actions(self) -> list[Action]:
        return []

    def noop_action(self) -> Action:
        return np.array([NOOP_ID, 0.0, 0.0, 0.0])

    def hard_reset(self) -> None:
        raise AssertionError("symbolic provider must not reset the environment")


def _sweep():
    return GroundSkill(skill=S.SWEEP_TABLE, objects=(O.robot, O.wiper, O.box, *O.cubes))


def _oracle_action(*, state):
    return SweepTable3DOracle(env=SymbolicEnvironment()).get_labeled_action(
        state=state, goal=Goal(atoms=SweepTable3DPredicates.goal_atoms())
    )


def test_provider_accepts_core_environment_and_exposes_domain_interfaces():
    provider = SweepTable3DSkillProvider(env=SymbolicEnvironment())
    assert provider.skills() == S.robot_skills()
    assert provider.types() == O.all_types()
    assert provider.objects() == O.all_objects()
    assert provider.predicates() == SweepTable3DPredicates.all()
    assert tuple(obj for obj in _sweep().objects if obj.type == O.cube_type) == O.cubes
    assert O.box in _sweep().objects


def test_all_eight_robot_ids_are_unique_and_noop_is_separate():
    assert {skill.name for skill in S.robot_skills()} == set(SKILL_IDS)
    assert set(SKILL_IDS.values()) == set(range(8))
    assert NOOP_ID == -1


def test_sweep_encodes_all_ball_grounding_with_angle_and_speed():
    provider = SweepTable3DSkillProvider(env=SymbolicEnvironment())
    action = provider.compute_action(
        ground_skill=_sweep(), params=np.array([-25.0, 0.15]), state=State(data={})
    )
    assert action.tolist() == [SKILL_IDS["SweepTable"], 0.0, -25.0, 0.15]


@pytest.mark.parametrize(
    ("skill", "objects"),
    [
        (S.PICK_WIPER, (O.robot, O.wiper)),
        (S.RETURN_WIPER, (O.robot, O.wiper)),
        (S.PICK_CUBE_FROM_TABLE, (O.robot, O.cubes[3])),
        (S.PICK_CUBE_FROM_FLOOR, (O.robot, O.cubes[3])),
        (S.PLACE_CUBE_IN_BOX, (O.robot, O.cubes[3], O.box)),
        (S.RETURN_CUBE_TO_TABLE_FROM_BOX, (O.robot, O.cubes[3], O.box, O.wiper, *O.balls)),
        (S.RETURN_CUBE_TO_TABLE_FROM_FLOOR, (O.robot, O.cubes[3], O.wiper, *O.balls)),
    ],
)
def test_discrete_actions_encode_their_cube_and_have_no_parameters(*, skill, objects):
    ground = GroundSkill(skill=skill, objects=objects)
    provider = SweepTable3DSkillProvider(env=SymbolicEnvironment())
    params = provider.sample_params(ground_skill=ground, rng=np.random.default_rng(0))
    assert params.shape == (0,)
    action = SweepTable3DActions.encode(ground_skill=ground, params=params)
    assert action.tolist() == [
        SKILL_IDS[skill.name],
        3.0 if O.cubes[3] in objects else 0.0,
        0.0,
        0.0,
    ]


def test_sweep_samples_independent_angle_and_speed_across_the_full_bounds():
    provider = SweepTable3DSkillProvider(env=SymbolicEnvironment())
    rng = np.random.default_rng(10)
    expected_rng = np.random.default_rng(10)
    draws = np.array([provider.sample_params(ground_skill=_sweep(), rng=rng) for _ in range(300)])
    expected = np.array([
        [expected_rng.uniform(-45.0, 45.0), expected_rng.uniform(0.05, 0.30)] for _ in range(300)
    ])
    np.testing.assert_array_equal(draws, expected)
    for column, (low, high) in enumerate(S.SWEEP_BOUNDS):
        assert low <= draws[:, column].min() < (low + high) / 2
        assert (low + high) / 2 < draws[:, column].max() <= high


def test_human_reset_interface_preserves_shelf_pricing_and_destinations():
    provider = SweepTable3DSkillProvider(env=SymbolicEnvironment(), human_reset_practice_cost=7.0)
    resets = provider.human_cube_bin_reset_skills()
    assert [reset.skill.name for reset in resets] == ["reset_cubes", "reset_wiper"]
    assert [reset.skill.practice_cost for reset in resets] == [7.0, 7.0]
    assert resets[0].objects == (O.robot, O.box, O.wiper, *O.cubes)
    assert resets[1].objects == (O.robot, O.wiper)
    assert provider.human_cube_bin_reset_skill() == resets[0]
    assert provider.movables_reset_skills() == resets
    for reset in resets:
        assert provider.is_movables_reset_skill(ground_skill=reset)
        assert provider.movables_reset_destination(ground_skill=reset) == reset.skill.name
        assert provider.human_skill_name(ground_skill=reset) == reset.skill.name
    assert S.RESET_CUBES.practice_cost == 1.0


def test_human_reset_can_be_disabled():
    provider = SweepTable3DSkillProvider(env=SymbolicEnvironment(), offer_human_reset=False)
    assert provider.human_cube_bin_reset_skills() == ()
    assert provider.human_cube_bin_reset_skill() is None


@pytest.mark.parametrize("cost", [-1.0, float("nan"), float("inf")])
def test_human_reset_cost_must_be_finite_and_nonnegative(*, cost):
    with pytest.raises(ValueError):
        SweepTable3DSkillProvider(env=SymbolicEnvironment(), human_reset_practice_cost=cost)


def test_oracle_picks_wiper_when_every_cube_is_on_table():
    result = _oracle_action(state=SymbolicScene.state())
    assert result.label == "PickWiper"
    assert result.action[0] == SKILL_IDS["PickWiper"]


@pytest.mark.parametrize("bin_y", [-0.12, 0.0, 0.12])
def test_oracle_aims_toward_tray_with_bounded_angle_and_nominal_speed(*, bin_y):
    state = SymbolicScene.state(holding=O.wiper, bin_y=bin_y, cube_y=0.03)
    result = _oracle_action(state=state)
    assert result.label == "SweepTable"
    assert result.action[0] == SKILL_IDS["SweepTable"]
    angle, speed = result.action[2:]
    expected = np.clip(np.rad2deg(np.arctan2(bin_y - 0.03, 1.50 - state.table_edge_x)), -45, 45)
    assert angle == pytest.approx(expected)
    assert speed == 0.15


def test_oracle_uses_state_table_geometry_when_available():
    state = SymbolicScene.state(
        holding=O.wiper,
        bin_y=0.10,
        cube_y=0.04,
        table_center_y=0.08,
        table_edge_x=1.30,
        sweep_start_x=1.70,
    )
    action = _oracle_action(state=state).action
    assert action[2] == pytest.approx(np.rad2deg(np.arctan2(0.10 - 0.04, 1.50 - 1.30)))
    assert action[3] == 0.15


def test_oracle_params_change_with_bin_position_and_remain_bounded():
    negative = _oracle_action(state=SymbolicScene.state(holding=O.wiper, bin_y=-0.10)).action
    positive = _oracle_action(state=SymbolicScene.state(holding=O.wiper, bin_y=0.10)).action
    assert negative[2] < 0.0 < positive[2]
    assert not np.array_equal(negative[2:], positive[2:])
    extreme = _oracle_action(state=SymbolicScene.state(holding=O.wiper, bin_y=3.0)).action
    for value, (low, high) in zip(extreme[2:], S.SWEEP_BOUNDS, strict=True):
        assert low <= value <= high


def test_oracle_does_not_offer_a_fallback_for_partial_capture():
    result = _oracle_action(
        state=SymbolicScene.state(on_table=O.cubes[1:], in_box=O.cubes[:1], holding=O.wiper)
    )
    assert result.label == "no-op"
    assert result.action[0] == NOOP_ID


def test_oracle_does_not_deliver_remaining_balls_individually():
    result = _oracle_action(state=SymbolicScene.state(on_table=O.cubes[2:], in_box=O.cubes[:2]))
    assert result.label == "no-op"
    assert result.action[0] == NOOP_ID


def test_oracle_places_a_held_cube_in_bin():
    result = _oracle_action(state=SymbolicScene.state(holding=O.cubes[3]))
    assert result.label == "PlaceCubeInBox"
    assert result.action[:2].tolist() == [SKILL_IDS["PlaceCubeInBox"], 3.0]


@pytest.mark.parametrize("holding", [None, O.wiper, O.cubes[2]])
def test_oracle_never_recovers_ground_failure(*, holding):
    result = _oracle_action(
        state=SymbolicScene.state(
            on_table=O.cubes[1:], on_floor=O.cubes[:1], holding=holding, active=False
        )
    )
    assert result.label == "no-op"
    assert result.action.tolist() == [NOOP_ID, 0.0, 0.0, 0.0]


def test_oracle_stops_after_all_cubes_are_delivered():
    result = _oracle_action(state=SymbolicScene.state(on_table=(), in_box=O.cubes))
    assert result.label == "no-op"
