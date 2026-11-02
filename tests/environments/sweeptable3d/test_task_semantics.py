"""Production sweep semantics without constructing any physics model.

TDD evidence: the initial version of this file produced 14/14 failures with
``scripts/with_env.sh pytest -q tests/environments/sweeptable3d/test_task_semantics.py``.
Later focused RED runs caught board-footprint classification (6 failures), controller
error/restoration handling (2), preplaced balls (1), and passive tool-return rearming (2).
Each was followed by a passing focused run. The initial integration run passed 288/288:
``scripts/with_env.sh pytest -q tests/environments/sweeptable3d
--ignore=tests/environments/sweeptable3d/test_live.py``.
No simulator models were constructed for these runs.
"""

from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import PrivateAttr

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.sweepoffshelf3d.controllers import ControllerFailure
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed
from hitl_pmp.environments.sweeptable3d.backend import SweepTable3DBackend
from hitl_pmp.environments.sweeptable3d.environment import SweepTable3DEnvironment
from hitl_pmp.environments.sweeptable3d.objects import SweepTable3DObjects as O
from hitl_pmp.environments.sweeptable3d.predicates import SweepTable3DPredicates
from hitl_pmp.environments.sweeptable3d.skills import SKILL_IDS
from hitl_pmp.environments.sweeptable3d.skills import SweepTable3DSkills as S


class SemanticBackend(SweepTable3DBackend):
    _atoms: set[Any] = PrivateAttr(default_factory=set)
    _captured: bool = PrivateAttr(default=False)

    def _physical_atoms(self):
        return frozenset(self._atoms)

    def _capture_settled(self):
        return self._captured

    def observe_ground_contacts(self):
        pass

    def _cube_placements(self, *, names):
        return dict.fromkeys(names)

    def _commit(self, *, placements):
        self._atoms = {atom for atom in self._atoms if atom[0] not in {"InBox", "OnFloor"}}
        self._atoms.update(("OnTable", (name,)) for name in placements)


class Scene:
    @staticmethod
    def ready(*, holding=True):
        backend = SemanticBackend()
        backend._atoms = {("OnTable", (ball.name,)) for ball in O.cubes}
        if holding:
            backend._atoms.add(("HoldingWiper", (O.robot.name, O.wiper.name)))
        else:
            Scene.home(backend=backend)
        return backend

    @staticmethod
    def home(*, backend):
        backend._atoms.discard(("HoldingWiper", (O.robot.name, O.wiper.name)))
        backend._atoms.update({("HandEmpty", (O.robot.name,)), ("WiperHome", (O.wiper.name,))})


def test_ball_schema_fixed_bin_and_angle_speed_bounds():
    assert O.ball_type.name == "table_ball"
    assert O.balls == O.cubes and len(O.balls) == 5
    assert S.SWEEP_BOUNDS == ((-45.0, 45.0), (0.05, 0.30))
    assert SweepTable3DBackend().fixed_bin_offset == -0.18
    assert SweepTable3DEnvironment().fixed_bin_offset == -0.18


@pytest.mark.parametrize(
    "captured,completed,ground",
    [(False, True, False), (True, False, False), (True, True, True), (True, True, False)],
)
def test_only_completed_settled_capture_can_win_and_floor_wins(*, captured, completed, ground):
    backend = Scene.ready()
    backend.begin_sweep()
    assert not backend.task_succeeded()
    backend._captured = captured
    backend._ground_failed = ground
    backend.finish_sweep(completed=completed)
    success = captured and completed and not ground
    assert backend.task_succeeded() is success
    assert backend.task_failed() is not success
    assert (("SweepSucceeded", ()) in backend.abstract_atoms()) is success
    assert ("TaskActive", ()) not in backend.abstract_atoms()
    with pytest.raises(SkillFailed):
        backend.begin_sweep()


def test_pick_place_capture_cannot_earn_sweep_goal():
    backend = Scene.ready(holding=False)
    backend._atoms = {("InBox", (ball.name, O.box.name)) for ball in O.cubes}
    backend._captured = True
    assert not backend.task_succeeded()
    assert ("SweepSucceeded", ()) not in backend.abstract_atoms()
    goals = SweepTable3DPredicates.goal_atoms()
    assert any(atom.predicate.name == "SweepSucceeded" for atom in goals)
    place = GroundSkill(skill=S.PLACE_CUBE_IN_BOX, objects=(O.robot, O.cubes[0], O.box))
    assert not goals <= place.add_effects | {
        atom for atom in goals if atom.predicate.name == "InBox"
    }


def test_tool_return_and_noop_recovery_cannot_rearm_all_balls_stayed_table():
    backend = Scene.ready()
    backend.begin_sweep()
    backend.finish_sweep(completed=True)
    backend.begin_robot_reset()
    Scene.home(backend=backend)
    backend.finish_robot_reset()
    assert backend.task_failed()
    assert ("TaskActive", ()) not in backend.abstract_atoms()
    with pytest.raises(SkillFailed):
        backend.begin_sweep()
    assert backend.reset_cubes() == list(backend.cube_names)
    assert not backend.task_failed()
    assert ("TaskActive", ()) in backend.abstract_atoms()


@pytest.mark.parametrize("hand_empty,home", [(False, False), (True, False), (False, True)])
def test_restored_balls_alone_never_clear_attempt(*, hand_empty, home):
    backend = Scene.ready()
    backend.begin_sweep()
    backend.finish_sweep(completed=False)
    backend._atoms = {("OnTable", (ball.name,)) for ball in O.cubes}
    if hand_empty:
        backend._atoms.add(("HandEmpty", (O.robot.name,)))
    if home:
        backend._atoms.add(("WiperHome", (O.wiper.name,)))
    backend.reset_cubes()
    assert backend.task_failed()


def test_full_robot_restoration_then_tool_home_rearms_practice():
    backend = Scene.ready()
    backend.begin_sweep()
    backend.finish_sweep(completed=False)
    backend._atoms.discard(("OnTable", (O.cubes[0].name,)))
    backend.begin_robot_reset(restores_balls=True)
    backend._atoms.add(("OnTable", (O.cubes[0].name,)))
    backend.finish_robot_reset()
    assert backend.task_failed()
    backend.begin_robot_reset()
    Scene.home(backend=backend)
    backend.finish_robot_reset()
    assert not backend.task_failed()
    backend.begin_sweep()


def test_failed_human_reset_cannot_rearm_even_if_scene_looks_restored():
    backend = Scene.ready(holding=False)
    backend.begin_sweep()
    backend.finish_sweep(completed=False)
    backend.reset_cubes_success_probability = 0
    assert backend.reset_cubes() == []
    assert backend.task_failed()


def test_environment_passes_angle_speed_without_duplicating_controller_attempt(*, monkeypatch):
    backend = Scene.ready()
    controllers = Mock()
    monkeypatch.setattr(SweepTable3DEnvironment, "backend", lambda self: backend)
    monkeypatch.setattr(SweepTable3DEnvironment, "controllers", lambda self: controllers)
    env = SweepTable3DEnvironment()
    env._dispatch(skill_id=SKILL_IDS["SweepTable"], values=np.array([1, 0, -34.5, 0.123]))
    controllers.sweep_table.assert_called_once_with(angle=-34.5, speed=0.123)
    assert not backend._sweep_attempted


def test_environment_reports_controller_parameter_rejection_as_skill_failure(*, monkeypatch):
    backend = Scene.ready()
    controllers = Mock()
    controllers.sweep_table.side_effect = SkillFailed("invalid parameters")
    monkeypatch.setattr(SweepTable3DEnvironment, "backend", lambda self: backend)
    monkeypatch.setattr(SweepTable3DEnvironment, "controllers", lambda self: controllers)
    with pytest.raises(ControllerFailure, match="parameters"):
        SweepTable3DEnvironment()._dispatch(
            skill_id=SKILL_IDS["SweepTable"], values=np.array([1, 0, 90, 0.1])
        )


def test_human_ball_restoration_invalidates_prior_success_before_tool_return():
    backend = Scene.ready()
    backend.begin_sweep()
    backend._captured = True
    backend.finish_sweep(completed=True)
    assert backend.task_succeeded()
    backend.reset_cubes()
    assert not backend.task_succeeded()
    assert backend.task_failed()


def test_preplaced_balls_cannot_count_toward_a_new_sweep():
    backend = Scene.ready()
    backend._atoms.remove(("OnTable", (O.balls[0].name,)))
    backend._atoms.add(("InBox", (O.balls[0].name, O.box.name)))
    backend._captured = True
    with pytest.raises(SkillFailed, match="all five"):
        backend.begin_sweep()
    assert not backend.task_succeeded()


def test_late_floor_contact_overrides_prior_sweep_success():
    backend = Scene.ready()
    backend.begin_sweep()
    backend._captured = True
    backend.finish_sweep(completed=True)
    assert backend.task_succeeded()
    backend._ground_failed = True
    assert backend.task_failed()
    assert not backend.task_succeeded()
    assert ("SweepSucceeded", ()) not in backend.abstract_atoms()


def test_passive_ball_return_during_tool_return_is_not_explicit_restoration():
    backend = Scene.ready()
    backend.begin_sweep()
    backend.finish_sweep(completed=False)
    backend._atoms.remove(("OnTable", (O.balls[0].name,)))
    backend.begin_robot_reset()
    backend._atoms.add(("OnTable", (O.balls[0].name,)))
    Scene.home(backend=backend)
    backend.finish_robot_reset()
    assert backend.task_failed()
    assert ("TaskActive", ()) not in backend.abstract_atoms()
