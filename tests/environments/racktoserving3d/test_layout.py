"""The one rack layout distribution every dumpling placement draws from."""

import itertools
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.layout import RackLayout
from hitl_pmp.environments.racktoserving3d.semantics import SupportRegion

RACK = SupportRegion(x_min=1.3725, x_max=1.6275, y_min=-0.1875, y_max=0.1875, top_z=0.925)


def _layout(*, seed, count=5, others=()):
    return RackLayout.sample(rng=np.random.default_rng(seed), rack=RACK, count=count, others=others)


@pytest.mark.parametrize("seed", range(50))
def test_every_layout_keeps_margins_spacing_and_yaw(*, seed):
    layout = _layout(seed=seed)
    assert len(layout) == 5
    for x, y, yaw in layout:
        assert RACK.x_min + 0.03 <= x <= RACK.x_max - 0.03
        assert RACK.y_min + 0.03 <= y <= RACK.y_max - 0.03
        assert -math.pi / 4 <= yaw <= math.pi / 4
    for (ax, ay, _), (bx, by, _) in itertools.combinations(layout, 2):
        assert math.hypot(ax - bx, ay - by) >= 0.061


def test_a_seed_fully_determines_the_layout_and_seeds_differ():
    assert _layout(seed=7) == _layout(seed=7)
    assert _layout(seed=7) != _layout(seed=8)


def test_layouts_spread_over_the_whole_rack_not_one_fixed_pattern():
    points = np.array([point for seed in range(200) for point in _layout(seed=seed)])
    assert points[:, 0].min() < RACK.x_min + 0.04 and points[:, 0].max() > RACK.x_max - 0.04
    assert points[:, 1].min() < RACK.y_min + 0.04 and points[:, 1].max() > RACK.y_max - 0.04
    assert points[:, 2].min() < -math.pi / 4 + 0.05 and points[:, 2].max() > math.pi / 4 - 0.05


@pytest.mark.parametrize("seed", range(20))
def test_one_put_back_is_conditioned_on_the_others_current_positions(*, seed):
    others = [(x, y) for x, y, _ in _layout(seed=seed, count=4)]
    ((x, y, yaw),) = _layout(seed=seed + 1000, count=1, others=others)
    assert RACK.x_min + 0.03 <= x <= RACK.x_max - 0.03
    assert RACK.y_min + 0.03 <= y <= RACK.y_max - 0.03
    assert -math.pi / 4 <= yaw <= math.pi / 4
    assert all(math.hypot(x - ox, y - oy) >= 0.061 for ox, oy in others)


def test_a_rack_with_no_room_left_raises_instead_of_looping():
    with pytest.raises(RuntimeError, match="no rack layout"):
        _layout(seed=0, count=40)


def test_yaw_quaternion_is_about_world_z_in_mujoco_order():
    np.testing.assert_allclose(RackLayout.quaternion(yaw=0.0), [1, 0, 0, 0])
    np.testing.assert_allclose(
        RackLayout.quaternion(yaw=math.pi / 2), [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]
    )


@pytest.mark.parametrize("seed", [0, 12345])
def test_every_episode_reset_starts_from_its_seeds_draw_of_the_layout(*, monkeypatch, seed):
    from hitl_pmp.environments.racktoserving3d.backend import RackServingBackend
    from hitl_pmp.environments.sweepoffshelf3d.backend import SweepOffShelf3DBackend

    monkeypatch.setattr(SweepOffShelf3DBackend, "reset", lambda self, *, seed: None)
    monkeypatch.setattr(
        RackServingBackend, "model", lambda self: SimpleNamespace(geom_condim=np.zeros(1))
    )
    monkeypatch.setattr(RackServingBackend, "data", lambda self: SimpleNamespace(time=0.0))
    monkeypatch.setattr(RackServingBackend, "core", lambda self: SimpleNamespace())
    calls = []

    def placements(self, *, names, start_layout="scattered"):  # noqa: PLR0917 -- a stand-in
        calls.append((list(names), start_layout, self._rng.bit_generator.state))
        return {"sentinel": names}

    committed = []
    monkeypatch.setattr(RackServingBackend, "food_placements", placements)
    monkeypatch.setattr(
        RackServingBackend, "_commit", lambda self, *, placements: committed.append(placements)
    )
    names = list(RackServingBackend.cube_names)
    # A scattered-only config draws no stack/scattered coin: the seed's own first draw.
    RackServingBackend(config=RackServingConfig(gathered_fraction=0.0)).reset(seed=seed)
    assert calls == [(names, "scattered", np.random.default_rng(seed).bit_generator.state)]
    assert committed == [{"sentinel": names}]
    # The default mixture draws its coin first, then places from the same generator.
    calls.clear()
    RackServingBackend().reset(seed=seed)
    rng = np.random.default_rng(seed)
    kind = "stack" if rng.uniform() < RackServingConfig().gathered_fraction else "scattered"
    assert calls == [(names, kind, rng.bit_generator.state)]


class _Scratch:
    """A stand-in collision copy: qpos only, so placements are read back exactly."""

    def __init__(self, *, names):
        self.names = list(names)
        self.addresses = {name: 7 * index for index, name in enumerate(self.names)}
        self.model = SimpleNamespace(body=lambda name: SimpleNamespace(id=self.names.index(name)))
        self.data = SimpleNamespace(qpos=np.zeros(7 * len(self.names)))


@pytest.fixture
def layout_backend(*, monkeypatch):
    from hitl_pmp.environments.racktoserving3d.backend import RackServingBackend

    names = list(RackServingBackend.cube_names)
    scratch = _Scratch(names=names)
    centres = {name: np.array([0.5, 0.5, 0.94]) for name in names}
    blocked: list[bool] = []
    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(mj_forward=lambda *args: None))
    monkeypatch.setattr(
        RackServingBackend,
        "_scratch",
        lambda self, *, names: (scratch.model, scratch.data, scratch.addresses, set()),
    )
    monkeypatch.setattr(
        RackServingBackend,
        "_blocked",
        staticmethod(lambda **kwargs: blocked.pop(0) if blocked else False),
    )
    monkeypatch.setattr(RackServingBackend, "body_geoms", lambda self, *, name: set())
    monkeypatch.setattr(RackServingBackend, "support", lambda self, *, name: RACK)
    monkeypatch.setattr(
        RackServingBackend,
        "corners",
        lambda self, *, name: np.array([centres[name] - 0.01, centres[name] + 0.01]),
    )
    backend = RackServingBackend()
    backend._rng = np.random.default_rng(3)
    return SimpleNamespace(backend=backend, centres=centres, blocked=blocked, names=names)


def _expected(*, seed, count, others=()):
    return RackLayout.sample(rng=np.random.default_rng(seed), rack=RACK, count=count, others=others)


def test_human_batch_return_draws_the_shared_layout(*, layout_backend):
    backend = layout_backend.backend
    placements = backend.food_placements(names=layout_backend.names)
    expected = _expected(seed=3, count=5)
    height = RACK.top_z + backend.config.dumpling_height / 2
    for name, (x, y, yaw) in zip(layout_backend.names, expected, strict=True):
        position, quaternion = placements[name]
        np.testing.assert_allclose(position[:2], [x, y])
        assert position[2] == pytest.approx(height, abs=0.003)
        np.testing.assert_allclose(quaternion, RackLayout.quaternion(yaw=yaw))


def test_a_blocked_batch_return_redraws_the_whole_layout(*, layout_backend):
    layout_backend.blocked.extend([False, False, True])
    placements = layout_backend.backend.food_placements(names=layout_backend.names)
    rng = np.random.default_rng(3)
    RackLayout.sample(rng=rng, rack=RACK, count=5)
    second = RackLayout.sample(rng=rng, rack=RACK, count=5)
    for name, (x, y, _) in zip(layout_backend.names, second, strict=True):
        np.testing.assert_allclose(placements[name][0][:2], [x, y])


def test_returning_some_food_conditions_on_the_rest(*, layout_backend):
    remaining = {"cube_0": (1.45, -0.1), "cube_1": (1.55, 0.1)}
    for name, (x, y) in remaining.items():
        layout_backend.centres[name] = np.array([x, y, 0.94])
    moving = [name for name in layout_backend.names if name not in remaining]
    placements = layout_backend.backend.food_placements(names=moving)
    expected = _expected(seed=3, count=3, others=list(remaining.values()))
    assert set(placements) == set(moving)
    for name, (x, y, _) in zip(moving, expected, strict=True):
        np.testing.assert_allclose(placements[name][0][:2], [x, y])


def test_the_put_back_draws_from_the_layout_conditioned_on_the_others(*, layout_backend):
    remaining = [(1.45, -0.1), (1.55, 0.1), (1.5, 0.0), (1.42, 0.12)]
    for name, (x, y) in zip(layout_backend.names[1:], remaining, strict=True):
        layout_backend.centres[name] = np.array([x, y, 0.94])
    draws = layout_backend.backend.rack_put_back_draws(name="cube_0", count=4)
    rng = np.random.default_rng(3)
    for x, y in draws:
        ((ex, ey, _),) = RackLayout.sample(rng=rng, rack=RACK, count=1, others=remaining)
        assert (x, y) == pytest.approx((ex, ey))
        assert all(math.hypot(x - ox, y - oy) >= 0.05 for ox, oy in remaining)
