"""The overlay's projection, checked against a saved ceiling frame.

`fixtures/ceiling/calibration.json` is what `ceiling_helper.py --calibration-only`
exported from prpl-tidybot's ceiling calibration (`load_ceiling_cameras(['AD8A29AE',
'5A5C29AE'])`) on 2026-10-21. `frame_18-12-46_crop.png` is rows 140..600, columns
300..1160 of the stitched frame the cube localizer saved at 18:12:46 on 2026-10-20
(`raw/2026-10-20/18-10-36/localization_failures/00_18-12-46/frame_00.png`), in grey:
the FAR bin and the tape, both in the first camera. The bin's pose is the skill
server's observation just before (`skill_calls.jsonl` call 2): (0.2269, 1.3700), yaw
0.0164.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from analysis.real_dashboard.camera import CeilingCalibration, Overlay
from analysis.real_dashboard.dashboard import RealRunDashboard
from analysis.real_dashboard.room import BIN_SIZE_M

CEILING = Path(__file__).parent / "fixtures" / "ceiling"
CROP_ORIGIN = (300, 140)  # (column, row) of the crop's top-left in the stitched frame
OBSERVED_BIN = (0.226875, 1.3700000000000028, 0.01636246173744693)
RIM_HEIGHT = 0.21

pytestmark = pytest.mark.xdist_group("test_real_dashboard_camera")


@pytest.fixture(scope="module")
def calibration():
    return CeilingCalibration.load(path=CEILING / "calibration.json")


@pytest.fixture(scope="module")
def gradient():
    grey = np.asarray(Image.open(CEILING / "frame_18-12-46_crop.png"), dtype=np.float64)
    rows, cols = np.gradient(grey)
    return np.hypot(rows, cols)


def _rectangle(*, x, y, yaw, z):
    c, s = math.cos(yaw), math.sin(yaw)
    half_l, half_w = BIN_SIZE_M[0] / 2, BIN_SIZE_M[1] / 2
    corners = [(half_l, half_w), (-half_l, half_w), (-half_l, -half_w), (half_l, -half_w)]
    return [(x + u * c - v * s, y + u * s + v * c, z) for u, v in corners]


def _edge_offset(*, gradient, start, end, search=20):
    """The shift (px) square to the segment `start`-`end` (stitched pixels) at which
    the image's edge strength along it peaks, searched over +-`search` px."""
    a = np.array(start) - CROP_ORIGIN
    b = np.array(end) - CROP_ORIGIN
    direction = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-direction[1], direction[0]])
    # The middle 60% of the edge: corners are where two edges blur together.
    samples = [a + (b - a) * t for t in np.linspace(0.2, 0.8, 40)]
    best, best_strength = 0, -1.0
    for shift in range(-search, search + 1):
        strength = 0.0
        for point in samples:
            col, row = np.round(point + shift * normal).astype(int)
            strength += gradient[row, col]
        if strength > best_strength:
            best, best_strength = shift, strength
    return best


def test_the_projection_matches_prpl_tidybots_own(*, calibration):
    # `CeilingCamera.project` (cv.projectPoints, no distortion) gave these on 2026-10-21.
    rim = _rectangle(x=OBSERVED_BIN[0], y=OBSERVED_BIN[1], yaw=OBSERVED_BIN[2], z=RIM_HEIGHT)
    seen = calibration.project(points=rim)
    assert [camera for camera, _ in seen[0]] == [0]
    pixels = [seen[corner][0][1] for corner in range(4)]
    expected = [(788.2, 167.5), (648.6, 170.6), (651.9, 275.1), (791.5, 271.3)]
    for pixel, want in zip(pixels, expected, strict=True):
        assert pixel == pytest.approx(want, abs=0.1)


def test_a_point_off_both_cameras_is_not_drawn(*, calibration):
    assert calibration.project(points=[(30.0, 30.0, 0.0)]) == [[]]


def test_the_observed_bin_rim_lands_on_the_bin_within_a_few_pixels(*, calibration, gradient):
    rim = _rectangle(x=OBSERVED_BIN[0], y=OBSERVED_BIN[1], yaw=OBSERVED_BIN[2], z=RIM_HEIGHT)
    pixels = [calibration.project(points=[corner])[0][0][1] for corner in rim]
    for index in range(4):
        offset = _edge_offset(gradient=gradient, start=pixels[index], end=pixels[(index + 1) % 4])
        # Measured 2026-10-21: -3, 4, 5, -1 px.
        assert abs(offset) <= 6, f"rim edge {index} is {offset} px off the bin's edge"


def test_the_far_spot_outline_lands_on_the_bin_placed_there(*, calibration, gradient):
    """The spot is where the person is asked to put the bin, and this bin was put
    ~3 cm from it: observed at (0.227, 1.370) against the spot's (0.24, 1.40). So the
    spot's outline sits on the bin to within that placement, not within a few pixels;
    measured 2026-10-21, its edges are 14, 8, 17 and 6 px off (the observed bin's are
    3, 4, 5 and 1)."""
    dashboard = RealRunDashboard(run_dir=CEILING)
    far = next(spot for spot in dashboard.refresh().room.bin_spots if spot.name == "FAR")
    rim = _rectangle(x=far.x, y=far.y, yaw=far.yaw, z=RIM_HEIGHT)
    pixels = [calibration.project(points=[corner])[0][0][1] for corner in rim]
    for index in range(4):
        offset = _edge_offset(gradient=gradient, start=pixels[index], end=pixels[(index + 1) % 4])
        assert abs(offset) <= 18, f"spot edge {index} is {offset} px off the bin's edge"


def test_the_tape_edge_lands_on_the_tape_at_floor_height(*, calibration, gradient):
    # The robot-side edge, y = 0.349 + 0.0267 x, between map x -0.6 and 0.6. The tape's
    # far edge is ~16 px across it, so the search stays inside +-10 px.
    ends = [(x, 0.349 + 0.0267 * x, 0.0) for x in (-0.6, 0.6)]
    pixels = [calibration.project(points=[end])[0][0][1] for end in ends]
    offset = _edge_offset(gradient=gradient, start=pixels[0], end=pixels[1], search=10)
    assert abs(offset) <= 5


def test_the_overlay_names_every_layer_and_keeps_shapes_in_frame(*, calibration):
    state = RealRunDashboard(run_dir=CEILING).refresh()
    overlay = Overlay.build(calibration=calibration, room=state.room)
    kinds = {shape.kind for shape in overlay.shapes}
    assert {"tape", "spot"} <= kinds
    for shape in overlay.shapes:
        for u, v in shape.points:
            assert 0 <= u <= overlay.width
            assert 0 <= v <= overlay.height
