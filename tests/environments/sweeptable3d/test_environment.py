"""Runtime wiring without creating a simulator or changing an object pose."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.controllers import ControllerFailure
from hitl_pmp.environments.sweeptable3d.controllers import TableSweep
from hitl_pmp.environments.sweeptable3d.environment import SweepTable3DEnvironment
from hitl_pmp.environments.sweeptable3d.objects import SweepTable3DObjects as O
from hitl_pmp.environments.sweeptable3d.skills import SKILL_IDS


@pytest.mark.parametrize("yaw", [-75, -30, 0, 30, 75])
def test_travel_yaw_is_relative_to_the_outward_table_perpendicular(*, yaw):
    direction = TableSweep.direction(yaw=yaw)
    assert np.linalg.norm(direction) == pytest.approx(1)
    assert direction[0] < 0
    assert direction[2] == 0
    assert np.rad2deg(np.arctan2(direction[1], -direction[0])) == pytest.approx(yaw)


@pytest.mark.parametrize("yaw", [-75, -30, 0, 30, 75])
def test_stroke_clears_the_entire_blade_beyond_the_edge(*, yaw):
    wiper = SimpleNamespace(
        shelf=SimpleNamespace(front_x=1.373),
        model=SimpleNamespace(geom_size=np.array([[0.01, 0.09, 0]])),
        head_geom=0,
        radius=0.01,
    )
    sweep = TableSweep(wiper=wiper, center_y=0)
    length = sweep.stroke_length(yaw=yaw)
    direction = sweep.direction(yaw=yaw)
    end_x = sweep.START_X + length * direction[0]
    assert end_x + 0.09 * abs(direction[1]) + 0.01 == pytest.approx(1.373 - 0.015)


def test_runtime_passes_both_parameters_unchanged_instead_of_autoaiming(*, monkeypatch):
    controllers = Mock()
    monkeypatch.setattr(SweepTable3DEnvironment, "controllers", lambda self: controllers)
    monkeypatch.setattr(
        SweepTable3DEnvironment, "backend", lambda self: SimpleNamespace(task_failed=lambda: False)
    )
    env = SweepTable3DEnvironment()
    env._dispatch(skill_id=SKILL_IDS["SweepTable"], values=np.array([1, 0, -34.5, 0.123]))
    controllers.sweep_table.assert_called_once_with(angle=-34.5, speed=0.123)


@pytest.mark.parametrize("name", ["PickWiper", "SweepTable", "PlaceCubeInBox"])
def test_ground_failure_blocks_further_delivery_skills(*, monkeypatch, name):
    monkeypatch.setattr(SweepTable3DEnvironment, "controllers", lambda self: Mock())
    monkeypatch.setattr(
        SweepTable3DEnvironment, "backend", lambda self: SimpleNamespace(task_failed=lambda: True)
    )
    with pytest.raises(ControllerFailure, match="task failed"):
        SweepTable3DEnvironment()._dispatch(skill_id=SKILL_IDS[name], values=np.zeros(4))


def test_failed_task_still_allows_robot_cube_reset(*, monkeypatch):
    controllers = Mock()
    monkeypatch.setattr(SweepTable3DEnvironment, "controllers", lambda self: controllers)
    monkeypatch.setattr(
        SweepTable3DEnvironment, "backend", lambda self: SimpleNamespace(task_failed=lambda: True)
    )
    SweepTable3DEnvironment()._dispatch(
        skill_id=SKILL_IDS["ReturnCubeToTableFromFloor"], values=np.array([7, 2, 0, 0])
    )
    controllers.return_cube_to_table.assert_called_once_with(cube="cube_2")


def test_observation_uses_table_object_types_and_live_bin_features(*, monkeypatch):
    features = {
        obj.name: np.arange(len(obj.type.feature_names), dtype=float) for obj in O.all_objects()
    }
    atoms = frozenset({("TaskActive", ())})
    backend = SimpleNamespace(features=lambda: features, abstract_atoms=lambda: atoms)
    monkeypatch.setattr(SweepTable3DEnvironment, "backend", lambda self: backend)
    env = SweepTable3DEnvironment()
    state = env._observed_state(seed=23, steps_taken=4)
    assert state.abstract_atoms == atoms
    assert np.array_equal(state.data[O.box], features["box_0"])
    env.current_state = state
    assert env._bookkeeping() == (23, 4)


@pytest.mark.parametrize("offset,yaw", [(np.nan, 0), (0, np.inf), (0.41, 0), (0, 76)])
def test_out_of_bounds_strokes_fail_before_any_motion(*, offset, yaw):
    wiper = SimpleNamespace(shelf=Mock())
    sweep = TableSweep(wiper=wiper, center_y=0)
    from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

    with pytest.raises(SkillFailed, match="bounds"):
        sweep.run(offset=offset, yaw=yaw)
    assert not wiper.shelf.mock_calls
