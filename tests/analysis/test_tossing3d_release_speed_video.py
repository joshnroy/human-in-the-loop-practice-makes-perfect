"""Unit tests for the release-speed video composer's geometry and captions.

No video is encoded here and no simulator is started. What is worth pinning is the world
-> pixel map the arc panel draws through, and what the status bar *says*, because both are
claims about the measurement and both would look plausible on screen while being wrong.
"""

import numpy as np
import pytest

from analysis.tossing3d_release_speed_video import (
    PanelTransform,
    TossPhase,
    ballistic_tail,
    legend_rows,
    phase_at,
    playback_indices,
    status_fields,
)


def _synthetic_clip(*, cut_short: bool) -> dict[str, object]:
    """One throw's recorded frames, from a parabola whose crossing is known in closed form.

    `cut_short` intercepts the cube in mid-air, which is what the bin's far wall does to a
    flat throw -- the case the dotted extrapolation exists for.
    """
    g, z0, vz, x0, vx = 9.81, 0.9, 3.0, 1.4, 2.6
    impact_t = (vz + np.sqrt(vz**2 - 2 * g * (0.025 - z0))) / g
    land_t = 0.45 * impact_t if cut_short else impact_t
    times = np.arange(0.0, land_t + 1e-9, 0.005)
    return {
        "frame_times": (times + 5.0).tolist(),
        "frame_cube_x": (x0 + vx * times).tolist(),
        "frame_cube_z": (z0 + vz * times - 0.5 * g * times**2).tolist(),
        "release_t": 5.0,
        "land_t": 5.0 + land_t,
        "ballistic_impact_t": 5.0 + impact_t,
        "ballistic_impact_x": x0 + vx * impact_t,
        "commanded_speed_deg": 105.0,
        "commanded_release_ms": 720.0,
        "cube_x_final": x0 + vx * impact_t + 0.01,
        "base_x_before_toss": 0.65,
    }


def _transform() -> PanelTransform:
    return PanelTransform.fit(
        x_min=0.4, x_max=2.9, z_min=0.0, z_max=1.2, width=640, height=480, margin=0
    )


def test_panel_transform_is_isotropic_so_an_arc_is_not_stretched() -> None:
    """One metre must be the same number of pixels along x as along z.

    A panel whose whole subject is "how far, how high" cannot use different scales on its
    two axes: the parabola's shape would be an artefact of the aspect ratio, and a reader
    comparing the height of a lob against the length of a flat throw would be reading the
    figure's geometry rather than the throw's.
    """
    t = _transform()
    assert t.metres_per_pixel_x == pytest.approx(t.metres_per_pixel_z, rel=1e-12)


def test_panel_transform_puts_z_zero_at_the_bottom_and_x_min_at_the_left() -> None:
    t = _transform()
    left, bottom = t.to_pixel(x=t.x_min, z=0.0)
    assert left == pytest.approx(0.0, abs=0.5)
    assert bottom == pytest.approx(t.height, abs=0.5)


def test_panel_transform_is_monotone_the_right_way_round_on_both_axes() -> None:
    """z grows *up* the world and *down* the image; getting that sign wrong draws every
    parabola as a valley and would be easy to mistake for a physics bug."""
    t = _transform()
    x_near, _ = t.to_pixel(x=1.0, z=0.0)
    x_far, _ = t.to_pixel(x=2.0, z=0.0)
    _, z_low = t.to_pixel(x=1.0, z=0.1)
    _, z_high = t.to_pixel(x=1.0, z=0.9)
    assert x_far > x_near
    assert z_high < z_low


def test_panel_transform_widens_x_rather_than_squashing_z_when_the_arc_is_tall() -> None:
    """The requested z span is never compressed to fit; the x span grows instead.

    Enforcing isotropy has to give somewhere. Clipping the top off a lob would silently
    hide the very throws whose height is the point, so the spare room is taken sideways.
    """
    tall = PanelTransform.fit(
        x_min=0.0, x_max=1.0, z_min=0.0, z_max=4.0, width=640, height=480, margin=0
    )
    assert tall.z_max >= 4.0
    assert tall.x_max - tall.x_min > 1.0
    assert tall.metres_per_pixel_x == pytest.approx(tall.metres_per_pixel_z, rel=1e-12)


def test_playback_indices_brackets_the_flight_and_strides() -> None:
    times = np.arange(0.0, 4.0, 0.005)
    indices = playback_indices(
        times=times, release_t=1.0, land_t=1.5, pre_seconds=0.4, post_seconds=0.3, stride=2
    )
    assert times[indices[0]] == pytest.approx(0.6, abs=0.006)
    assert times[indices[-1]] == pytest.approx(1.8, abs=0.006)
    gaps = {
        round(float(times[b] - times[a]), 6) for a, b in zip(indices[:-1], indices[1:], strict=True)
    }
    assert gaps == {0.01}


def test_playback_indices_clamps_to_the_recording_rather_than_running_off_it() -> None:
    times = np.arange(0.0, 1.0, 0.005)
    indices = playback_indices(
        times=times, release_t=0.1, land_t=0.9, pre_seconds=5.0, post_seconds=5.0, stride=1
    )
    assert indices[0] == 0
    assert indices[-1] == len(times) - 1


def test_phase_at_names_the_four_stretches_in_order() -> None:
    assert phase_at(t=0.5, release_t=1.0, land_t=1.5) is TossPhase.SWING
    assert phase_at(t=1.0, release_t=1.0, land_t=1.5) is TossPhase.FLIGHT
    assert phase_at(t=1.2, release_t=1.0, land_t=1.5) is TossPhase.FLIGHT
    assert phase_at(t=1.6, release_t=1.0, land_t=1.5) is TossPhase.LANDED


def test_status_fields_withholds_the_range_until_the_cube_has_landed() -> None:
    """The distance is the video's punchline and must not be readable before the throw.

    A bar that shows the final range while the cube is still in the air answers the
    question the footage is being watched to answer, and worse, invites the reader to
    believe the number was somehow observed at that instant.
    """
    before = dict(
        status_fields(
            speed=105.0,
            release_ms=720.0,
            index=1,
            total=5,
            seed=0,
            standoff=1.35,
            base_x=0.65,
            phase=TossPhase.FLIGHT,
            cube_x=1.2,
            impact_x=1.7904,
            resting_x=1.7990,
            solved=False,
        )
    )
    after = dict(
        status_fields(
            speed=105.0,
            release_ms=720.0,
            index=1,
            total=5,
            seed=0,
            standoff=1.35,
            base_x=0.65,
            phase=TossPhase.LANDED,
            cube_x=1.7990,
            impact_x=1.7904,
            resting_x=1.7990,
            solved=False,
        )
    )
    assert "1.140" not in "".join(before.values())
    assert "1.140" in after["BALLISTIC"]
    assert after["SPEED"] == "105 deg/s"
    assert after["RELEASE"] == "720 ms"
    assert after["THROW"] == "2/5"


def test_ballistic_tail_reaches_the_reported_crossing_when_the_throw_was_intercepted() -> None:
    """The dotted continuation must land exactly on the marker it explains.

    It is drawn as the fitted parabola rather than a chord, so a tail that stopped short
    of -- or ran past -- the reported ground crossing would be a visible contradiction
    between two things the panel asserts about the same throw.
    """
    clip = _synthetic_clip(cut_short=True)
    xs, zs = ballistic_tail(clip=clip)

    assert len(xs) > 2
    assert xs[-1] == pytest.approx(clip["ballistic_impact_x"], abs=1e-6)
    assert zs[-1] == pytest.approx(0.025, abs=1e-6)
    # It begins where the footage stopped, not at the release.
    assert xs[0] == pytest.approx(clip["frame_cube_x"][-1], abs=1e-3)


def test_ballistic_tail_is_empty_when_the_cube_reached_open_floor_on_its_own() -> None:
    """Nothing to extrapolate, so nothing is drawn: a stub there would claim the
    measurement went further than the footage did."""
    assert ballistic_tail(clip=_synthetic_clip(cut_short=False)) == ([], [])


def test_legend_rows_stay_dim_until_their_throw_has_actually_been_shown() -> None:
    clips = [_synthetic_clip(cut_short=False), _synthetic_clip(cut_short=True)]
    rows = legend_rows(clips=clips, colors=[(1, 2, 3), (4, 5, 6)], measured=1)
    assert [measured for _, _, measured in rows] == [True, False]
    assert "not thrown yet" in rows[1][1]
    # Labelled by both dials, and reporting the distance *from the base* rather than the
    # world x -- the same quantity the surface figures and the statistics are computed on,
    # so a reader can check one artifact against the other.
    thrown = clips[0]["ballistic_impact_x"] - clips[0]["base_x_before_toss"]
    assert f"{thrown:.3f}" in rows[0][1]
    assert "105 deg/s @  720 ms" in rows[0][1]


def test_status_fields_reports_the_throw_distance_relative_to_the_base() -> None:
    """ "How far the cube goes" is a distance, and a world x is not one on its own.

    Both dials are shown, and both distances are reported from the base -- the ballistic
    crossing, which is the primary criterion, and the resting position beside it. A world x
    in either slot would silently carry wherever `MoveToThrowPose` parked the robot.
    """
    fields = dict(
        status_fields(
            speed=140.0,
            release_ms=1400.0,
            index=8,
            total=9,
            seed=0,
            standoff=1.35,
            base_x=0.650361,
            phase=TossPhase.LANDED,
            cube_x=2.5498,
            impact_x=2.5404,
            resting_x=2.5498,
            solved=False,
        )
    )
    assert "1.890" in fields["BALLISTIC"]
    assert "1.899" in fields["RESTING"]
    assert fields["RELEASE"] == "1400 ms"
    assert fields["THROW"] == "9/9"
    assert fields["SCORES"] == "no"
