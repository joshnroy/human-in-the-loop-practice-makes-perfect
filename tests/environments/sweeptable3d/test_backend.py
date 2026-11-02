"""Reset collision checks, independent RNG and physical terminal latching, without MuJoCo."""

import copy
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import PrivateAttr
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweepoffshelf3d.backend import SweepOffShelf3DBackend
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed
from hitl_pmp.environments.sweeptable3d import backend as backend_module
from hitl_pmp.environments.sweeptable3d import wiper as wiper_module
from hitl_pmp.environments.sweeptable3d.backend import GroundContactFailure, SweepTable3DBackend
from hitl_pmp.environments.sweeptable3d.balls import BallScene
from hitl_pmp.environments.sweeptable3d.wiper import BODY_REST, TableWiper

from .test_wiper import FakeShelf


class MockBackend(SweepTable3DBackend):
    _scene: Any = PrivateAttr()

    def env(self):
        return self._scene.env

    def core(self):
        return self._scene.core

    def model(self):
        return self._scene.model

    def data(self):
        return self._scene.data

    def sim(self):
        return self._scene.sim


class MockScene:
    def __init__(self):
        names = [
            "world",
            *SweepTable3DBackend.cube_names,
            "box_0",
            "cupboard_1",
            "robot_base_link",
            "wiper_0",
            "moving_floor",
        ]
        bodies = {name: SimpleNamespace(id=index, name=name) for index, name in enumerate(names)}
        self.positions = {name: np.array([1.45, 0.0, 0.3105]) for name in names[1:6]}
        self.positions["box_0"] = np.array([1.247, 0.0, 0.1])
        self.positions["wiper_0"] = np.array([1.414, -0.42, 0.31])
        self.quaternion = np.array([1.0, 0, 0, 0])
        self.objects = {
            name: SimpleNamespace(
                get_pose=Mock(
                    side_effect=lambda name=name: (
                        self.positions[name].copy(),
                        self.quaternion.copy(),
                    )
                )
            )
            for name in self.positions
        }
        geom_names = [
            "floor",
            *["" for _ in range(5)],
            "box_0_floor",
            "cupboard_floor",
            "robot_floor",
            "",
            "",
            "floor_visual",
            "floor_moving",
            "floor_tiles",
            "bin_plane",
        ]
        self.model = SimpleNamespace(
            ngeom=len(geom_names),
            geom_bodyid=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 0, 10, 0, 6]),
            geom_type=np.array([0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 6, 6, 6, 0]),
            body_parentid=np.zeros(len(names), dtype=int),
            body_weldid=np.array([0, 1, 2, 3, 4, 5, 6, 0, 8, 9, 10]),
            geom_contype=np.array([1] * 11 + [0, 1, 1, 1]),
            geom_conaffinity=np.array([1] * 11 + [0, 1, 1, 1]),
            geom=Mock(side_effect=lambda geom: SimpleNamespace(name=geom_names[geom])),
            body=Mock(
                side_effect=lambda body: (
                    bodies[body] if isinstance(body, str) else bodies[names[body]]
                )
            ),
        )
        self.data = SimpleNamespace(
            contact=[],
            ncon=0,
            qpos=np.zeros(42),
            geom_xpos=np.zeros((len(geom_names), 3)),
        )
        self.data.geom_xpos[7, 1] = 0.12
        self.core = SimpleNamespace(
            get_object=Mock(side_effect=self.objects.__getitem__),
            _objects_dict=self.objects,
            shelf_top_z=Mock(return_value=0.30),
            shelf_front_x=Mock(return_value=1.373),
            _shelf_geoms=Mock(return_value=[7, 7]),
        )
        self.sim = SimpleNamespace(step=Mock(), forward=Mock())
        self.env = SimpleNamespace(
            step=Mock(side_effect=self.control_step), reset=Mock(), close=Mock()
        )
        self.forward_calls = []
        self.block_first = False
        self.block_all = False
        self.block_geom = 1
        self.scratch_data = None
        self.commits = []
        self.atoms = {
            ("HandEmpty", ("robot",)),
            ("WiperHome", ("wiper_0",)),
            *(("OnTable", (name,)) for name in SweepTable3DBackend.cube_names),
        }

    def control_step(self, *args, **kwargs):
        for _ in range(3):
            self.sim.step()
        return "control-result"

    def scratch(self, *, names):
        data = SimpleNamespace(qpos=self.data.qpos.copy(), contact=[], ncon=0)
        addresses = {name: index * 7 for index, name in enumerate(SweepTable3DBackend.cube_names)}
        addresses["wiper_0"] = 35
        for name in names:
            data.qpos[addresses[name] + 2] = 100
        self.scratch_data = data
        return self.model, data, addresses, {0, 7}

    def forward(self, *args):
        model, data = args
        assert model is self.model
        self.forward_calls.append(data.qpos.copy())
        blocked = self.block_all or (self.block_first and len(self.forward_calls) == 1)
        data.contact = (
            [SimpleNamespace(geom1=self.block_geom, geom2=8, dist=-0.001)] if blocked else []
        )
        data.ncon = len(data.contact)

    def commit(self, *, placements):
        self.commits.append(copy.deepcopy(placements))
        for name, (position, _) in placements.items():
            self.positions[name] = position.copy()

    def contacts(self, *, geom1, geom2, distance=0.0):
        self.data.contact = [SimpleNamespace(geom1=geom1, geom2=geom2, dist=distance)]
        self.data.ncon = 1


@pytest.fixture
def scene_backend(*, monkeypatch):
    scene = MockScene()
    backend = MockBackend(scene_bg=False)
    backend._scene = scene
    monkeypatch.setattr(SweepOffShelf3DBackend, "reset", Mock())
    monkeypatch.setattr(BallScene, "configure", Mock())
    monkeypatch.setattr(MockBackend, "_scratch", lambda self, *, names: scene.scratch(names=names))
    monkeypatch.setattr(
        MockBackend, "_commit", lambda self, *, placements: scene.commit(placements=placements)
    )
    monkeypatch.setattr(MockBackend, "_physical_atoms", lambda self: frozenset(scene.atoms))
    monkeypatch.setitem(
        sys.modules, "mujoco", SimpleNamespace(mj_forward=Mock(side_effect=scene.forward))
    )
    shelf = object()
    monkeypatch.setattr(backend_module, "ShelfSkills", Mock(return_value=shelf))
    wiper = Mock()
    pose = np.eye(4)
    pose[:3, 3] = scene.positions["wiper_0"]
    wiper.rest_pose.return_value = pose
    wiper.at_home.return_value = True
    wiper.initialize_rest.side_effect = lambda: backend.env().step(np.zeros(11))
    factory = Mock(return_value=wiper)
    monkeypatch.setattr(wiper_module, "TableWiper", factory)
    backend._observe_steps(env=scene.env)
    backend.reset(seed=17)
    scene.commits.clear()
    scene.forward_calls.clear()
    return scene, backend, wiper, factory, shelf


def test_reset_initializes_live_home_check_and_preserves_setup_monitor_boundary(*, scene_backend):
    scene, backend, wiper, factory, shelf = scene_backend
    assert backend.wiper_at_home(front=0, wiper_board=0)
    factory.assert_called_once_with(shelf=shelf)
    wiper.initialize_rest.assert_called_once_with()
    assert backend._monitor_ground
    assert not backend.task_failed()
    assert backend._original_sim_step.call_count == 3
    BallScene.configure.assert_called_once_with(core=scene.core)


def test_compact_sampling_rejects_robot_collision_in_scratch_only(*, scene_backend):
    scene, backend, *_ = scene_backend
    scene.positions.update({name: np.array([3.0, 3.0, 0.01]) for name in backend.cube_names})
    before = scene.data.qpos.copy()
    scene.block_first = True
    placements = backend._cube_placements(names=["cube_0"])
    assert len(scene.forward_calls) == 2
    assert not np.array_equal(placements["cube_0"][0], scene.forward_calls[0][:3])
    np.testing.assert_array_equal(scene.data.qpos, before)
    assert not scene.commits


def test_rolling_contacts_enable_after_rest_setup_like_investigation(*, scene_backend):
    _, backend, wiper, *_ = scene_backend
    wiper.initialize_rest.reset_mock()

    def configure(**kwargs):
        assert wiper.initialize_rest.call_count == 1

    BallScene.configure.side_effect = configure
    backend.reset(seed=0)


def test_exhausted_collision_search_keeps_live_objects_and_terminal_failure(*, scene_backend):
    scene, backend, *_ = scene_backend
    scene.block_all = True
    backend._ground_failed = True
    backend._ground_cubes = {"cube_0"}
    before = copy.deepcopy(scene.positions)
    with pytest.raises(RuntimeError, match="reset not applied"):
        backend.reset_cubes()
    assert not scene.commits
    for name in before:
        np.testing.assert_array_equal(scene.positions[name], before[name])
    assert backend.task_failed()
    assert backend.ground_cubes() == ("cube_0",)


def test_robot_reset_xy_is_compact_separated_and_never_committed(*, scene_backend):
    scene, backend, *_ = scene_backend
    before = copy.deepcopy(scene.positions)
    target = backend.robot_reset_xy(cube="cube_0")
    assert isinstance(target, tuple)
    assert 1.438 <= target[0] <= 1.555
    assert 0.05 <= target[1] <= 0.19
    for name in backend.cube_names[1:]:
        assert np.linalg.norm(np.array(target) - before[name][:2]) >= 0.035
    for name in before:
        np.testing.assert_array_equal(scene.positions[name], before[name])
    assert not scene.commits


def test_reset_streams_repeat_and_bin_does_not_depend_on_cube_draws(*, scene_backend):
    scene, backend, *_ = scene_backend
    backend.reset(seed=9)
    first_bin = scene.positions["box_0"].copy()
    assert first_bin[1] == pytest.approx(backend.table_center_y() - 0.18)
    first_cubes = np.stack([scene.positions[name].copy() for name in backend.cube_names])
    backend._cube_rng.random(100)
    backend.reset_cubes()
    np.testing.assert_array_equal(scene.positions["box_0"], first_bin)
    backend.reset(seed=9)
    np.testing.assert_array_equal(scene.positions["box_0"], first_bin)
    np.testing.assert_array_equal(
        np.stack([scene.positions[name] for name in backend.cube_names]), first_cubes
    )
    backend.reset(seed=10)
    np.testing.assert_array_equal(scene.positions["box_0"], first_bin)
    backend.fixed_bin_offset = -0.3
    backend.reset(seed=9)
    assert scene.positions["box_0"][1] == pytest.approx(-0.18)
    np.testing.assert_array_equal(
        np.stack([scene.positions[name] for name in backend.cube_names]), first_cubes
    )


def test_cube_layout_is_compact_and_pairwise_clear_after_repeated_resets(*, scene_backend):
    scene, backend, *_ = scene_backend
    bin_pose = scene.positions["box_0"].copy()
    for _ in range(4):
        assert backend.reset_cubes() == list(backend.cube_names)
        positions = np.stack([scene.positions[name] for name in backend.cube_names])
        assert np.all((positions[:, 0] >= 1.438) & (positions[:, 0] <= 1.555))
        assert np.all((positions[:, 1] >= 0.05) & (positions[:, 1] <= 0.19))
        distances = np.linalg.norm(positions[:, None, :2] - positions[None, :, :2], axis=-1)
        assert np.all(distances[np.triu_indices(5, k=1)] >= 0.035)
        np.testing.assert_array_equal(scene.positions["box_0"], bin_pose)


def test_only_physical_static_floor_geometries_are_ground(*, scene_backend):
    _, backend, *_ = scene_backend
    assert backend._ground_geoms == frozenset({0, 13})


@pytest.mark.parametrize("support", [6, 7, 8, 11, 12, 14])
def test_bin_bottom_table_robot_and_visuals_do_not_latch_floor_failure(*, scene_backend, support):
    scene, backend, *_ = scene_backend
    scene.contacts(geom1=1, geom2=support, distance=-0.001)
    backend.observe_ground_contacts()
    assert not backend.task_failed()
    assert backend.ground_cubes() == ()


@pytest.mark.parametrize("distance", [0.000001, 0.01, float("nan")])
def test_separated_contact_candidates_are_not_ground_touches(*, scene_backend, distance):
    scene, backend, *_ = scene_backend
    scene.contacts(geom1=1, geom2=0, distance=distance)
    backend.observe_ground_contacts()
    assert not backend.task_failed()


@pytest.mark.parametrize("reverse", [False, True])
def test_ground_touch_is_sticky_even_after_cube_is_airborne(*, scene_backend, reverse):
    scene, backend, *_ = scene_backend
    pair = (0, 1) if reverse else (1, 0)
    scene.contacts(geom1=pair[0], geom2=pair[1], distance=-0.0001)
    backend.observe_ground_contacts()
    assert backend.task_failed()
    scene.data.contact, scene.data.ncon = [], 0
    backend.observe_ground_contacts()
    assert backend.task_failed()
    assert backend.ground_cubes() == ("cube_0",)
    assert ("TaskActive", ()) not in backend.abstract_atoms()


def test_airborne_cube_is_off_table_without_being_on_floor_or_terminal(*, scene_backend):
    scene, backend, *_ = scene_backend
    scene.atoms = set()
    atoms = backend.abstract_atoms()
    assert ("OnFloor", ("cube_0",)) not in atoms
    assert ("AnyCubeOffTable", ()) in atoms
    assert ("TaskActive", ()) in atoms


def test_substep_bounce_latches_before_control_callbacks_and_raises_after_them(*, scene_backend):
    scene, backend, *_ = scene_backend
    events = []

    def physics_tick():
        events.append("physics")
        if events.count("physics") == 1:
            scene.contacts(geom1=1, geom2=0, distance=-0.0001)
        else:
            scene.data.contact, scene.data.ncon = [], 0

    backend._detach_physics_observer()
    scene.sim.step = physics_tick
    backend._observe_physics_steps()
    backend.set_step_observers(
        before=lambda: events.append("before"), after=lambda: events.append("after")
    )
    backend.set_tick_observer(observer=lambda: events.append(("tick", backend.task_failed())))
    with pytest.raises(GroundContactFailure):
        scene.env.step(np.zeros(11))
    assert scene.data.ncon == 0
    assert events == ["before", "physics", "physics", "physics", ("tick", True), "after"]
    assert backend.ground_cubes() == ("cube_0",)


def test_physics_observer_install_is_idempotent_and_preserves_existing_observer(*, scene_backend):
    scene, backend, *_ = scene_backend
    original_observed = scene.sim.step
    backend._observe_physics_steps()
    assert scene.sim.step is original_observed
    downstream = Mock(wraps=original_observed)
    scene.sim.step = downstream
    backend.close()
    assert scene.sim.step is downstream


@pytest.mark.parametrize("count", [0, 4, 5])
def test_robot_reset_clears_latch_only_after_all_five_are_on_table(*, scene_backend, count):
    scene, backend, *_ = scene_backend
    backend._ground_failed = True
    backend._ground_cubes = {"cube_0"}
    scene.atoms = set()
    backend.begin_robot_reset(restores_balls=True)
    assert scene.env.step(np.zeros(11)) == "control-result"
    scene.atoms = {
        ("HandEmpty", ("robot",)),
        ("WiperHome", ("wiper_0",)),
        *(("OnTable", (name,)) for name in backend.cube_names[:count]),
    }
    backend.finish_robot_reset()
    assert backend.task_failed() is (count != 5)
    assert not backend._resetting
    if count == 5:
        assert backend.ground_cubes() == ()


def test_unpaired_finish_and_failed_human_reset_preserve_failure(*, scene_backend):
    _, backend, *_ = scene_backend
    backend._ground_failed = True
    backend._ground_cubes = {"cube_0"}
    backend.finish_robot_reset()
    assert backend.task_failed()
    backend.reset_cubes_success_probability = 0
    assert backend.reset_cubes() == []
    assert backend.task_failed()


def test_wiper_human_reset_preserves_terminal_latch_and_outer_reset_scope(*, scene_backend):
    _, backend, wiper, *_ = scene_backend
    backend._ground_failed = True
    backend._ground_cubes = {"cube_0"}
    backend.begin_robot_reset()
    backend.reset_wiper()
    assert backend._resetting
    assert backend.task_failed()
    assert wiper.initialize_rest.call_count == 2


def test_wiper_reset_rejects_robot_collision_before_writing_pose(*, scene_backend):
    scene, backend, wiper, *_ = scene_backend
    scene.block_all, scene.block_geom = True, 9
    with pytest.raises(SkillFailed, match="collision-free rest placement"):
        backend.reset_wiper()
    assert not backend._resetting
    assert wiper.initialize_rest.call_count == 1


def test_native_rest_edge_contact_accepts_overhanging_handle_below_table():
    shelf = FakeShelf()
    wiper = TableWiper(shelf=shelf)
    pose = wiper.rest_pose()
    pose[:3, :3] = Rotation.from_euler("y", -4.681, degrees=True).as_matrix() @ BODY_REST
    pose[2, 3] = 0
    pose[2, 3] = shelf.board[1] - wiper.capsule_bottom(pose=pose)
    shelf.set_pose(pose=pose)
    shelf.support()
    assert wiper.handle_corners()[:, 2].min() < shelf.board[1] - 0.003
    assert wiper.handle_table_bottom() >= shelf.board[1] - 0.003
    assert wiper.at_home()


def test_native_rest_fix_still_rejects_handle_penetrating_table_footprint():
    shelf = FakeShelf()
    wiper = TableWiper(shelf=shelf)
    pose = wiper.rest_pose()
    pose[2, 3] -= 0.006
    shelf.set_pose(pose=pose)
    shelf.support()
    assert not wiper.at_home()


@pytest.mark.parametrize(
    "count,settled,held", [(4, True, False), (5, False, False), (5, True, True), (5, True, False)]
)
def test_capture_requires_every_ball_in_shallow_tray_unheld_and_settled(
    *, scene_backend, monkeypatch, count, settled, held
):
    scene, backend, *_ = scene_backend
    names = backend.ball_names[:count]
    inside = Mock(side_effect=lambda **kwargs: kwargs["name"] in names)
    stopped = Mock(return_value=settled)
    monkeypatch.setattr(BallScene, "in_tray", inside)
    monkeypatch.setattr(BallScene, "settled", stopped)
    monkeypatch.setattr(MockBackend, "held", lambda self, *, name: held and name == "cube_0")
    assert backend._capture_settled() is (count == 5 and settled and not held)
    if count == 5 and not held:
        stopped.assert_called_once_with(core=scene.core, names=backend.ball_names)


def test_table_abstraction_uses_shallow_tray_and_only_actual_floor_contact(
    *, scene_backend, monkeypatch
):
    scene, backend, *_ = scene_backend
    monkeypatch.setattr(MockBackend, "held", lambda self, *, name: False)
    monkeypatch.setattr(
        MockBackend, "_on_table", lambda self, *, name: name == "cube_2", raising=False
    )
    monkeypatch.setattr(
        BallScene, "in_tray", lambda **kwargs: kwargs["name"] in {"cube_0", "cube_1"}
    )
    scene.contacts(geom1=1, geom2=0)
    atoms = SweepTable3DBackend._physical_atoms(backend)
    assert ("OnFloor", ("cube_0",)) in atoms
    assert ("InBox", ("cube_0", "box_0")) not in atoms
    assert ("InBox", ("cube_1", "box_0")) in atoms
    assert ("OnTable", ("cube_2",)) in atoms
    assert not any(objects == ("cube_3",) for _, objects in atoms)
    assert ("HandEmpty", ("robot",)) in atoms


@pytest.mark.parametrize(
    "position,on_table",
    [
        ([0.0, 0.0, 0.01685], True),
        ([0.2, 0.0, 0.01685], False),
        ([0.0, 0.6, 0.01685], False),
        ([0.0, 0.0, 0.05], False),
        ([float("nan"), 0.0, 0.01685], False),
    ],
)
def test_table_restoration_requires_actual_board_footprint(*, scene_backend, position, on_table):
    scene, backend, *_ = scene_backend
    scene.model.geom_size = np.zeros((scene.model.ngeom, 3))
    scene.model.geom_size[7] = [0.127, 0.55, 0.00635]
    scene.data.geom_xmat = np.tile(np.eye(3).reshape(1, 9), (scene.model.ngeom, 1))
    scene.positions["cube_0"] = scene.data.geom_xpos[7] + position
    assert backend._on_table(name="cube_0") is on_table


@pytest.mark.parametrize("held_ball", SweepTable3DBackend.ball_names)
def test_human_tool_reset_rejects_held_ball_before_any_motion(
    *, scene_backend, monkeypatch, held_ball
):
    scene, backend, wiper, factory, *_ = scene_backend
    monkeypatch.setattr(MockBackend, "held", lambda self, *, name: name == held_ball)
    backend.grip = 1.0
    backend._ground_failed = True
    initial_steps = backend._original_sim_step.call_count
    with pytest.raises(SkillFailed, match="holding a ball"):
        backend.reset_wiper()
    assert backend.grip == 1.0
    assert backend.task_failed()
    assert not backend._resetting
    assert not scene.commits
    assert backend._original_sim_step.call_count == initial_steps
    assert wiper.initialize_rest.call_count == factory.call_count == 1
