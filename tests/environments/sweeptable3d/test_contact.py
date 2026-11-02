"""Contact-force and stroke-safety contracts without a physics simulator."""

from itertools import chain, repeat
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed
from hitl_pmp.environments.sweeptable3d import contact as contact_module
from hitl_pmp.environments.sweeptable3d.contact import ContactTableSweep, TableContactLoad


class ContactRig:
    """Supply explicit contact-frame forces and capsule poses."""

    def __init__(self):
        self.model = SimpleNamespace(
            opt=SimpleNamespace(timestep=0.01), geom_size=np.array([[0.01, 0.1, 0]])
        )
        self.data = SimpleNamespace(
            geom_xpos=np.array([[1.4, 0, 0.31]]),
            geom_xmat=np.zeros((1, 9)),
            contact=[],
            ncon=0,
        )
        self.forces = []
        self.shelf = SimpleNamespace(
            data=self.data,
            model=self.model,
            front_x=1.0,
            cupboard_geoms={1, 2},
            mujoco=SimpleNamespace(
                mj_contactForce=Mock(side_effect=lambda *args: self.read_force(args=args))
            ),
        )
        self.wiper = SimpleNamespace(
            shelf=self.shelf, model=self.model, data=self.data, head_geom=0, radius=0.01, top=0.3
        )
        self.load = TableContactLoad(wiper=self.wiper)
        self.load.servo = SimpleNamespace(normal_force_n=0.0)
        self.orient()

    def orient(self, *, yaw: float = 0, tilt: float = 0):
        rotation = (
            Rotation.from_euler("z", yaw, degrees=True).as_matrix()
            @ Rotation.from_euler("x", tilt - 90, degrees=True).as_matrix()
        )
        self.data.geom_xmat[0] = rotation.reshape(9)

    def read_force(self, *, args):
        args[3][:] = self.forces[args[2]]

    def add_contact(self, *, force: float, head_first: bool = False, geoms=None):
        frame = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        if head_first:
            frame[[0, 2]] *= -1
        pair = (0, 1) if head_first else (1, 0)
        if geoms is not None:
            pair = geoms
        self.data.contact.append(SimpleNamespace(geom1=pair[0], geom2=pair[1], frame=frame.ravel()))
        self.forces.append(np.array([force, 7.0, -11.0, 0, 0, 0]))
        self.data.ncon += 1

    def update(self, *, count: int = 1):
        for _ in range(count):
            self.load.update()


@pytest.mark.parametrize("head_first", [False, True])
def test_support_is_upward_for_both_geom_orders(*, head_first):
    rig = ContactRig()
    rig.add_contact(force=5, head_first=head_first)
    rig.update()
    assert 0 < rig.load.support_force_n < 5
    rig.update(count=100)
    assert rig.load.support_force_n == pytest.approx(5)
    assert rig.load.max_support_force_n == pytest.approx(5)
    assert rig.load.servo.normal_force_n < rig.load.TARGET_FORCE_N


def test_support_sums_only_live_head_table_contacts():
    rig = ContactRig()
    rig.add_contact(force=2, head_first=True)
    rig.add_contact(force=3)
    rig.add_contact(force=100, geoms=(0, 9))
    rig.add_contact(force=100, geoms=(1, 9))
    rig.add_contact(force=100)
    rig.data.ncon -= 1
    rig.update(count=100)
    assert rig.load.support_force_n == pytest.approx(5)
    assert {call.args[2] for call in rig.shelf.mujoco.mj_contactForce.call_args_list} == {0, 1}


def test_force_saturates_without_windup_and_recovers_from_excess_load():
    rig = ContactRig()
    rig.update(count=500)
    assert rig.load.servo.normal_force_n == rig.load.MAX_FORCE_N
    integral = rig.load.integral_n
    rig.update(count=100)
    assert rig.load.integral_n == integral
    assert rig.load.max_commanded_force_n == rig.load.MAX_FORCE_N
    rig.add_contact(force=100)
    rig.update(count=500)
    assert rig.load.servo.normal_force_n == 0
    integral = rig.load.integral_n
    rig.update(count=100)
    assert rig.load.integral_n == integral
    assert abs(integral) <= rig.load.MAX_FORCE_N
    rig.forces[0][:] = 0
    rig.update(count=500)
    assert 0 < rig.load.servo.normal_force_n <= rig.load.MAX_FORCE_N


@pytest.mark.parametrize("initial_integral,support", [(12.0, 3.0), (-12.0, 0.0)])
def test_integral_can_unwind_back_out_of_saturation(*, initial_integral, support):
    rig = ContactRig()
    rig.load.integral_n = initial_integral
    rig.load.support_force_n = support
    rig.add_contact(force=support)
    rig.update(count=250)
    assert abs(rig.load.integral_n) < abs(initial_integral)
    assert 0 < rig.load.servo.normal_force_n < rig.load.MAX_FORCE_N


@pytest.mark.parametrize(
    "yaw,crossing_center_x", [(0, 1.009), (30, 1.059), (-30, 1.059), (90, 1.109)]
)
def test_leading_capsule_edge_crossing_unloads_before_center_crosses(*, yaw, crossing_center_x):
    rig = ContactRig()
    rig.orient(yaw=yaw)
    rig.update(count=500)
    assert rig.load.servo.normal_force_n > 0
    assert rig.load.integral_n > 0
    rig.data.geom_xpos[0, 0] = crossing_center_x
    assert crossing_center_x > rig.shelf.front_x
    rig.update()
    assert rig.load.servo.normal_force_n == 0
    assert rig.load.integral_n == 0
    rig.update(count=5)
    assert rig.load.servo.normal_force_n == 0


def test_force_tapers_on_approach_to_edge_without_stored_integral():
    far = ContactRig()
    near = ContactRig()
    far.orient(yaw=30)
    near.orient(yaw=30)
    far.data.geom_xpos[0, 0] = 1.09
    near.data.geom_xpos[0, 0] = 1.07
    far.update()
    near.update()
    assert 0 < near.load.servo.normal_force_n < far.load.servo.normal_force_n


def test_edge_taper_also_unloads_accumulated_integral_force():
    rig = ContactRig()
    rig.update(count=500)
    rig.data.geom_xpos[0, 0] = 1.0101
    rig.update(count=100)
    assert 0 <= rig.load.servo.normal_force_n <= 0.06 + 1e-12


@pytest.mark.parametrize("yaw,center_x", [(0, 1.02), (30, 1.07), (-30, 1.07)])
def test_prolonged_edge_dwell_cannot_integrate_past_tapered_force_limit(*, yaw, center_x):
    rig = ContactRig()
    rig.orient(yaw=yaw)
    rig.data.geom_xpos[0, 0] = center_x
    for _ in range(1000):
        rig.update()
        assert 0 <= rig.load.servo.normal_force_n <= 6.0 + 1e-12
    assert rig.load.servo.normal_force_n == pytest.approx(6.0)
    integral = rig.load.integral_n
    rig.update(count=100)
    assert rig.load.integral_n == integral
    assert rig.load.max_commanded_force_n == pytest.approx(6.0)


@pytest.mark.parametrize("yaw", [-27, 27])
def test_revision3_unloads_at_stall_geometry_despite_inboard_head_center(*, yaw):
    """Characterize seed0/bin-.18/speed.10's final head position, not a fix."""
    rig = ContactRig()
    rig.model.geom_size[0, 1] = 0.09
    rig.shelf.front_x = 1.373
    rig.wiper.top = 0.2794
    rig.data.geom_xpos[0] = [1.413642077964539, -0.09417513445138292, 0.2941724570499715]
    rig.orient(yaw=yaw)
    rig.load.integral_n = 6.0
    head_center_margin = rig.data.geom_xpos[0, 0] - rig.shelf.front_x
    assert head_center_margin > 0.04
    assert rig.load.maximum_clearance() < ContactTableSweep.MAX_CLEARANCE
    rig.update(count=100)
    assert rig.load.support_force_n == 0
    assert rig.load.integral_n == 0
    assert rig.load.servo.normal_force_n == 0


def test_revision3_cutoff_ignores_remaining_head_table_support():
    """Leading-edge release is geometric, even if the rest of the head is supported."""
    rig = ContactRig()
    rig.orient(yaw=27)
    rig.model.geom_size[0, 1] = 0.09
    rig.shelf.front_x = 1.373
    rig.data.geom_xpos[0, 0] = 1.413642077964539
    rig.add_contact(force=2.0)
    rig.load.integral_n = 6.0
    rig.update()
    assert rig.load.servo.normal_force_n == 0
    assert rig.load.integral_n == 0
    rig.shelf.mujoco.mj_contactForce.assert_not_called()


def test_disabled_controller_releases_force_without_reading_contacts():
    rig = ContactRig()
    rig.update(count=100)
    assert rig.load.servo.normal_force_n > 0
    rig.add_contact(force=100)
    rig.load.enabled = False
    rig.update(count=5)
    assert rig.load.servo.normal_force_n == 0
    rig.shelf.mujoco.mj_contactForce.assert_not_called()


@pytest.mark.parametrize("tilt", [-30, 30])
def test_clearance_checks_high_endpoint_even_when_low_endpoint_touches(*, tilt):
    rig = ContactRig()
    rig.orient(tilt=tilt)
    rig.data.geom_xpos[0, 2] = 0.36
    assert rig.load.maximum_clearance() == pytest.approx(0.1)
    assert rig.load.maximum_clearance() > ContactTableSweep.MAX_CLEARANCE


def test_horizontal_clearance_is_independent_of_yaw():
    rig = ContactRig()
    rig.data.geom_xpos[0, 2] = 0.312
    for yaw in [-60, 0, 45]:
        rig.orient(yaw=yaw)
        assert rig.load.maximum_clearance() == pytest.approx(0.002)


class PullRig:
    """Record requested motion; optionally track it exactly, without dynamics."""

    def __init__(self, *, monkeypatch, track: bool = True):
        self.track = track
        self.position = np.array([1.6, 0.0, 0.5])
        self.waypoint = self.position.copy()
        self.pose = np.eye(4)
        self.contact = SimpleNamespace(
            TARGET_FORCE_N=TableContactLoad.TARGET_FORCE_N,
            MAX_FORCE_N=TableContactLoad.MAX_FORCE_N,
            support_force_n=1.0,
            maximum_clearance=Mock(return_value=0),
            update=Mock(),
            enabled=True,
            max_clearance_m=0.0,
            max_support_force_n=1.0,
            max_commanded_force_n=2.0,
        )
        self.servo = MagicMock(servo_updates=0, clipped_torque_count=0, normal_force_n=2.0)
        self.servo.__enter__.return_value = self.servo
        self.servo.__exit__.return_value = False
        monkeypatch.setattr(contact_module, "TableContactLoad", lambda **kwargs: self.contact)
        monkeypatch.setattr(contact_module, "SweepServo", lambda **kwargs: self.servo)
        self.shelf = SimpleNamespace(
            data=SimpleNamespace(qpos=np.zeros(10), site_xpos=self.position.reshape(1, 3).copy()),
            ba=np.arange(3),
            qa=np.arange(3, 10),
            site=0,
            in_hand=np.eye(4),
            ik=Mock(side_effect=lambda **kwargs: self.solve(**kwargs)),
            step=Mock(side_effect=lambda **kwargs: self.step(**kwargs)),
            hold=Mock(side_effect=lambda **kwargs: self.check_released()),
        )
        self.wiper = SimpleNamespace(
            shelf=self.shelf,
            wiper_pose=Mock(side_effect=lambda: self.pose.copy()),
            require_held=Mock(),
            line=Mock(side_effect=lambda **kwargs: self.check_released()),
        )
        self.stroke = ContactTableSweep(wiper=self.wiper, center_y=0.0)
        self.stroke.stroke_length = Mock(return_value=0.05)

    def solve(self, *, position, seed, **kwargs):
        self.waypoint = position.copy()
        return seed.copy(), 0.0

    def step(self, *, base, **kwargs):
        if self.track:
            self.shelf.data.qpos[self.shelf.ba] = base
            self.shelf.data.site_xpos[0] = self.waypoint

    def check_released(self):
        assert not self.contact.enabled
        assert self.servo.normal_force_n == 0

    def pull(self):
        self.stroke.pull(position=self.position, rotation=np.eye(3), yaw=0)


@pytest.mark.parametrize("unsafe", ["clearance", "orientation"])
def test_guard_holds_progress_and_fails_boundedly_when_pose_never_recovers(*, monkeypatch, unsafe):
    rig = PullRig(monkeypatch=monkeypatch)
    if unsafe == "clearance":
        rig.contact.maximum_clearance.side_effect = chain([0.0], repeat(0.01))
    else:
        rig.pose[:3, :3] = Rotation.from_euler("z", 12, degrees=True).as_matrix()
    with pytest.raises(SkillFailed, match="contact guard"):
        rig.pull()
    assert rig.stroke.metrics["guard_paused_steps"] == 50
    assert 0 < rig.shelf.ik.call_count < 50
    for call in rig.shelf.ik.call_args_list:
        np.testing.assert_allclose(call.kwargs["position"], rig.position)
        np.testing.assert_allclose(call.kwargs["base"], np.zeros(3))
    rig.wiper.line.assert_not_called()
    rig.servo.__exit__.assert_called_once()


def test_guard_recovers_then_releases_force_before_settling_and_lifting(*, monkeypatch):
    rig = PullRig(monkeypatch=monkeypatch)
    rig.contact.maximum_clearance.side_effect = chain([0.0, 0.01, 0.01], repeat(0.0))
    rig.pull()
    assert rig.stroke.metrics["guard_paused_steps"] == 2
    assert rig.stroke.metrics["controller_revision"] == 3
    assert rig.stroke.metrics["clearance_limit_m"] == rig.stroke.MAX_CLEARANCE
    assert rig.stroke.metrics["target_support_force_n"] == rig.contact.TARGET_FORCE_N
    assert rig.stroke.metrics["downforce_limit_n"] == rig.contact.MAX_FORCE_N
    for call in rig.shelf.ik.call_args_list[:2]:
        np.testing.assert_allclose(call.kwargs["position"], rig.position)
    assert rig.shelf.ik.call_args.kwargs["position"][0] < rig.position[0]
    rig.wiper.line.assert_called_once()
    assert rig.shelf.hold.call_count == 2
    rig.servo.__exit__.assert_called_once()


def test_tracking_lag_limits_advance_and_reports_stall(*, monkeypatch):
    rig = PullRig(monkeypatch=monkeypatch, track=False)
    with pytest.raises(SkillFailed, match="stalled"):
        rig.pull()
    requested_progress = [
        rig.position[0] - call.kwargs["position"][0] for call in rig.shelf.ik.call_args_list
    ]
    limit = rig.stroke.TRACK_TOLERANCE + rig.stroke.SPEED * 0.1
    assert 0 < max(requested_progress) <= limit + 1e-12
    assert requested_progress[-1] == requested_progress[-2]
    assert rig.stroke.metrics["guard_paused_steps"] == 0
    rig.wiper.line.assert_not_called()
    rig.servo.__exit__.assert_called_once()


@pytest.mark.parametrize("clearance,allowed", [(0.0042, True), (0.005, True), (0.0051, False)])
def test_five_millimeter_guard_boundary_after_seating(*, monkeypatch, clearance, allowed):
    rig = PullRig(monkeypatch=monkeypatch)
    rig.contact.maximum_clearance.side_effect = chain([0.0], repeat(clearance))
    if allowed:
        rig.pull()
        assert rig.stroke.metrics["guard_paused_steps"] == 0
        rig.wiper.line.assert_called_once()
    else:
        with pytest.raises(SkillFailed, match="contact guard"):
            rig.pull()
        assert rig.stroke.metrics["guard_paused_steps"] == 50
        rig.wiper.line.assert_not_called()
    assert rig.stroke.metrics["controller_revision"] == 3
    assert rig.stroke.metrics["clearance_limit_m"] == pytest.approx(0.005)


def test_guard_recovery_resets_consecutive_pause_budget(*, monkeypatch):
    rig = PullRig(monkeypatch=monkeypatch)
    rig.contact.maximum_clearance.side_effect = chain(
        [0.0], repeat(0.01, 30), [0.0], repeat(0.01, 30), repeat(0.0)
    )
    rig.pull()
    assert rig.stroke.metrics["guard_paused_steps"] == 60
    rig.wiper.line.assert_called_once()
    rig.servo.__exit__.assert_called_once()


def test_repeated_brief_clearance_recoveries_exhaust_total_stroke_budget(*, monkeypatch):
    """Threshold chatter can stall without ever reaching 50 consecutive pauses."""
    rig = PullRig(monkeypatch=monkeypatch, track=False)
    rig.stroke.SPEED = 0.1
    rig.stroke.stroke_length.return_value = 0.329
    rig.contact.maximum_clearance.side_effect = chain(
        [0.0], chain.from_iterable(repeat([0.0049, 0.0051]))
    )
    with pytest.raises(SkillFailed, match="stalled at 0.000/0.329 m"):
        rig.pull()
    assert rig.shelf.ik.call_count == 198
    assert rig.stroke.metrics["guard_paused_steps"] == 99
    assert rig.stroke.metrics["clearance_limit_m"] == pytest.approx(0.005)
    for index, call in enumerate(rig.shelf.ik.call_args_list):
        expected = rig.position + ([-0.01, 0, 0] if index % 2 == 0 else [0, 0, 0])
        np.testing.assert_allclose(call.kwargs["position"], expected)
    rig.wiper.line.assert_not_called()
    rig.shelf.hold.assert_not_called()
    rig.servo.__exit__.assert_called_once()


@pytest.mark.parametrize("joint_target", [np.nan, 0.7])
def test_invalid_ik_stops_before_commanding_arm_and_preserves_metrics(*, monkeypatch, joint_target):
    rig = PullRig(monkeypatch=monkeypatch)
    rig.shelf.ik.side_effect = None
    rig.shelf.ik.return_value = (np.full(7, joint_target), 0.0)
    with pytest.raises(SkillFailed, match="IK lost its continuous branch"):
        rig.pull()
    labels = [call.kwargs["label"] for call in rig.shelf.step.call_args_list]
    assert labels == ["sweep_contact_seat"]
    assert rig.stroke.metrics["controller_revision"] == 3
    rig.wiper.line.assert_not_called()
    rig.servo.__exit__.assert_called_once()


def test_guard_discards_unreached_waypoint_and_holds_actual_base(*, monkeypatch):
    rig = PullRig(monkeypatch=monkeypatch, track=False)
    rig.shelf.data.qpos[rig.shelf.ba] = [0.6, 0.2, 0.05]
    rig.contact.maximum_clearance.side_effect = chain([0.0, 0.0], repeat(0.01))

    def partial_progress(*, label, **kwargs):
        if label == "sweep_pull":
            rig.shelf.data.site_xpos[0] = rig.position + [-0.001, 0, 0]

    rig.shelf.step.side_effect = partial_progress
    with pytest.raises(SkillFailed, match="contact guard"):
        rig.pull()
    first, guarded = rig.shelf.ik.call_args_list[:2]
    assert first.kwargs["position"][0] < guarded.kwargs["position"][0] < rig.position[0]
    np.testing.assert_allclose(guarded.kwargs["position"], rig.position + [-0.001, 0, 0])
    np.testing.assert_allclose(guarded.kwargs["base"], [0.6, 0.2, 0.05])


@pytest.mark.parametrize("missing", ["support", "clearance"])
def test_seating_requires_support_and_clearance_before_any_progress(*, monkeypatch, missing):
    rig = PullRig(monkeypatch=monkeypatch)
    if missing == "support":
        rig.contact.support_force_n = 0
    else:
        rig.contact.maximum_clearance.return_value = 0.01
    with pytest.raises(SkillFailed, match="could not establish supported blade contact"):
        rig.pull()
    assert rig.shelf.step.call_count == 20
    rig.shelf.ik.assert_not_called()
    rig.wiper.line.assert_not_called()
    rig.servo.__exit__.assert_called_once()
