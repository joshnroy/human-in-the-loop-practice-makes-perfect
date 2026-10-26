"""Regression checks for the legacy fixed-bin opt-out and upstream scene sources.

The default backend now makes the bin movable for PourCubeOutOfBin. Explicit
movable_bin=False retains the earlier static-bin dynamics, including placement and
reset behavior. These tests keep that opt-out working; test_pour_bin exercises the
new default under actual contact dynamics.
"""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.bin_on_ground import BinOnGroundGeometry
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
from hitl_pmp.environments.tossing3d.sides import BIN_RESET_REGION_BY_SIDE, Tossing3DSide

_KINDER = importlib.util.find_spec("kinder") is not None
needs_kinder = pytest.mark.skipif(not _KINDER, reason="KINDER is not installed")
_DRAGGED_FIXTURE = Path(__file__).parent / "fixtures" / "pick_dragged_bin_state.json"
_POSE = ("x", "y", "z", "qw", "qx", "qy", "qz")


def _bin_options(*, layout: Tossing3DLayout) -> dict:
    path = layout.task_config_path()
    if path is None:
        import kinder

        path = (
            Path(kinder.__path__[0])
            / "envs"
            / "dynamic3d"
            / "tasks"
            / "Tossing3D"
            / "Tossing3D-o1.json"
        )
    return json.loads(path.read_text())["objects"]["bin"]["bin_0"]


@needs_kinder
def test_same_side_bin_is_the_upstream_fixed_bin() -> None:
    upstream = _bin_options(layout=Tossing3DLayout.BARRIER)
    same_side = _bin_options(layout=Tossing3DLayout.SAME_SIDE)
    assert upstream.get("fixed") is True
    for key in ("fixed", "length", "width", "height", "wall_thickness"):
        assert same_side.get(key) == upstream[key], key


@needs_kinder
@pytest.mark.parametrize("layout", [Tossing3DLayout.BARRIER, Tossing3DLayout.SAME_SIDE])
def test_the_compiled_bin_has_no_joint(*, layout: Tossing3DLayout) -> None:
    import mujoco

    from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend

    backend = KinderBackend(
        movable_bin=False, task_config_path=layout.task_config_path(), scene_bg=False
    )
    try:
        backend.reset(seed=125)
        model = backend._object_centric()._robot_env.sim.model.mj_model  # noqa: SLF001
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bin_0")
        assert model.body_subtreemass[body] == pytest.approx(1.0)
        assert model.body_subtreemass[model.body("cube_0").id] == pytest.approx(0.1)
        assert model.body_jntnum[body] == 0
        assert model.body_mocapid[body] >= 0
    finally:
        backend.close()


@needs_kinder
def test_a_pick_from_inside_the_bin_does_not_move_it() -> None:
    """A practice pick of a cube lying inside the bin (astar-sampler seed 0, 2026-10-15)
    dragged the free 0.1 kg bin 17 cm. Replayed, the fixed bin must not move at all."""
    from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend

    backend = KinderBackend(movable_bin=False, scene_bg=False)
    try:
        backend.reset(seed=125)
        plain = json.loads(_DRAGGED_FIXTURE.read_text())
        backend.restore(snapshot=backend.plain_to_snapshot(plain=plain, dtype=np.float64))
        before = backend.snapshot_to_plain(snapshot=backend.live_state())["bin_0"]
        backend.run_pick_cube()
        after = backend.snapshot_to_plain(snapshot=backend.live_state())["bin_0"]
    finally:
        backend.close()
    np.testing.assert_allclose(after[:7], before[:7], atol=1e-9)


@needs_kinder
@pytest.mark.parametrize("side", list(Tossing3DSide))
def test_a_human_reset_still_relocates_the_fixed_bin(*, side: Tossing3DSide) -> None:
    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment

    env = Tossing3DEnvironment()
    try:
        env.hard_reset()
        before = env.backend().snapshot_to_plain(snapshot=env.backend().live_state())["bin_0"]
        assert env.reset_movables(destination=side.value)
        after = env.backend().snapshot_to_plain(snapshot=env.backend().live_state())["bin_0"]
    finally:
        env.close()
    x_min, y_min, x_max, y_max = BIN_RESET_REGION_BY_SIDE[side].ranges[0]
    assert x_min - 1e-6 <= after[0] <= x_max + 1e-6
    assert y_min - 1e-6 <= after[1] <= y_max + 1e-6
    assert math.hypot(after[0] - before[0], after[1] - before[1]) > 0.0
    assert BinOnGroundGeometry.holds(bin_=dict(zip(_POSE, after[:7], strict=True)))


@needs_kinder
def test_a_sampled_start_places_the_fixed_bin_per_seed() -> None:
    """The full reset to a freshly sampled start re-places the bin from the seed."""
    from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend

    backend = KinderBackend(movable_bin=False, scene_bg=False)
    try:
        poses = []
        for seed in (0, 1, 2):
            backend.reset(seed=seed)
            poses.append(backend.snapshot_to_plain(snapshot=backend.live_state())["bin_0"][:2])
    finally:
        backend.close()
    assert len({tuple(np.round(p, 6)) for p in poses}) == 3


@needs_kinder
def test_same_side_observes_static_colliders_like_upstream() -> None:
    """Upstream made the static-collider observation opt-in per task (kindergarden
    a01546a) and opted Tossing3D in. The same-side copy must opt in too, or bumping the
    pin would silently drop the room walls and barrier from that layout's state."""
    import kinder

    path = (
        Path(kinder.__path__[0])
        / "envs"
        / "dynamic3d"
        / "tasks"
        / "Tossing3D"
        / "Tossing3D-o1.json"
    )
    upstream = json.loads(path.read_text())
    same_side_path = Tossing3DLayout.SAME_SIDE.task_config_path()
    assert same_side_path is not None
    same_side = json.loads(same_side_path.read_text())
    assert (
        same_side.get("observe_static_colliders")
        == upstream.get("observe_static_colliders")
        is True
    )
