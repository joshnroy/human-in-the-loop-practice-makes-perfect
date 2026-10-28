"""Native KINDER regressions for table initialization and irreversible floor failure."""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.environments.sweeptable3d.environment import SweepTable3DEnvironment

pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KINDER"),
    pytest.mark.xdist_group("sweeptable3d_test_live"),
]


@pytest.fixture(name="env", scope="module")
def _env():
    env = SweepTable3DEnvironment(scene_bg=False, fixed_bin_offset=0.15)
    try:
        yield env
    finally:
        env.close()


def test_scene_uses_capsule_and_compact_cube_region_with_live_shifted_bin(*, env):
    env.reset_to_seed(seed=0)
    backend = env.backend()
    features = backend.features()
    assert (features["box_0"][1] + features["box_0"][3]) / 2 == pytest.approx(0.15, abs=1e-4)
    shelf = env.controllers()._skills
    assert int(backend.model().geom_type[shelf.head_geom]) == 3
    assert all(("OnTable", (f"cube_{index}",)) in backend.abstract_atoms() for index in range(5))
    assert all(abs(features[f"cube_{index}"][1]) <= 0.071 for index in range(5))
    assert not backend.task_failed()
    assert ("WiperHome", ("wiper_0",)) in backend.abstract_atoms()


def test_cube_resets_do_not_resample_the_bin_and_seed_rebuild_is_reproducible(*, env):
    env.reset_to_seed(seed=13)
    backend = env.backend()
    before = backend.features()
    env.reset_movables(destination="reset_cubes")
    reset = backend.features()
    assert np.allclose(reset["box_0"], before["box_0"], atol=1e-4)
    assert not np.allclose(reset["cube_0"], before["cube_0"])
    env.reset_to_seed(seed=13)
    rebuilt = backend.features()
    for name in ["box_0", *(f"cube_{index}" for index in range(5))]:
        assert np.allclose(rebuilt[name], before[name], atol=1e-6)


def test_airborne_is_not_ground_but_real_floor_contact_latches_until_reset(*, env):
    env.reset_to_seed(seed=0)
    backend = env.backend()
    backend._commit(
        placements={
            "cube_0": (np.array([0.5, 0.65, 0.20]), np.array([1.0, 0.0, 0.0, 0.0])),
        }
    )
    backend.observe_ground_contacts()
    assert not backend.task_failed()
    for _ in range(1000):
        backend.sim().step()
        if backend.task_failed():
            break
    assert backend.task_failed()
    assert backend.ground_cubes() == ("cube_0",)
    assert ("TaskActive", ()) not in backend.abstract_atoms()
    backend._commit(
        placements={
            "cube_0": (np.array([0.5, 0.65, 0.30]), np.array([1.0, 0.0, 0.0, 0.0])),
        }
    )
    backend.observe_ground_contacts()
    assert backend.task_failed()
    env.reset_movables(destination="reset_cubes")
    assert not backend.task_failed()
    assert ("TaskActive", ()) in backend.abstract_atoms()
