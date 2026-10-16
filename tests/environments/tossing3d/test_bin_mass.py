"""The bin is heavy enough that the robot cannot move it, in both layouts.

Upstream's `Tossing3D-o1.json` used to leave `bin_0` at the `Bin` class default of
0.1 kg -- the same as the cube -- so a cube striking the rim tipped it (102/4366 tosses
in six 2026-10-15 practice runs) and a pick of a cube lying inside it could drag it
17 cm. Josh's decision (2026-10-15): make the sim bin one the robot cannot move, and
make the real one match. The mass was chosen by replaying logged impacts and picks
against a sweep of masses; see the kindergarden change for the numbers.

`scenes/same_side.json` is this package's own copy of the task, so it has to carry the
same bin or the two layouts would be measuring different objects.
"""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.bin_on_ground import BinOnGroundGeometry
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout

_KINDER = importlib.util.find_spec("kinder") is not None
_DRAGGED_FIXTURE = Path(__file__).parent / "fixtures" / "pick_dragged_bin_state.json"


def _upstream_bin_options() -> dict:
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


def _same_side_bin_options() -> dict:
    path = Tossing3DLayout.SAME_SIDE.task_config_path()
    assert path is not None
    return json.loads(path.read_text())["objects"]["bin"]["bin_0"]


@pytest.mark.skipif(not _KINDER, reason="KINDER is not installed")
def test_same_side_bin_has_the_upstream_mass_and_geometry() -> None:
    upstream = _upstream_bin_options()
    same_side = _same_side_bin_options()
    assert "mass" in upstream, "upstream's bin must declare its mass, not take a default"
    for key in ("mass", "length", "width", "height", "wall_thickness"):
        assert same_side.get(key) == upstream[key], key


@pytest.mark.skipif(not _KINDER, reason="KINDER is not installed")
def test_a_pick_from_inside_the_bin_no_longer_drags_it() -> None:
    """A practice pick of a cube lying inside the bin (astar-sampler seed 0, 2026-10-15)
    dragged the 0.1 kg bin 17 cm. Replayed against the installed scene, the bin must
    stay put and upright."""
    from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend

    backend = KinderBackend(scene_bg=False)
    try:
        backend.reset(seed=125)
        plain = json.loads(_DRAGGED_FIXTURE.read_text())
        backend.restore(snapshot=backend.plain_to_snapshot(plain=plain, dtype=np.float64))
        before = backend.snapshot_to_plain(snapshot=backend.live_state())["bin_0"]
        backend.run_pick_cube()
        after = backend.snapshot_to_plain(snapshot=backend.live_state())["bin_0"]
    finally:
        backend.close()
    assert math.hypot(after[0] - before[0], after[1] - before[1]) < 0.005
    names = ("x", "y", "z", "qw", "qx", "qy", "qz")
    assert BinOnGroundGeometry.holds(bin_=dict(zip(names, after[:7], strict=True)))
