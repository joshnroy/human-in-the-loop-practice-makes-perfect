"""The kitchen judges "settled" by where the dumplings are, not how fast they spin."""

import numpy as np

from hitl_pmp.environments.pouring3d.kitchen.settling import PayloadRestWindow

TICK = 0.1


def _filled(*, track, seconds: float = 3.0) -> tuple[PayloadRestWindow, float, np.ndarray]:
    window = PayloadRestWindow()
    times = np.arange(round(seconds / TICK) + 1) * TICK + 60.0
    for t in times:
        window.record(time=float(t), positions=track(t=float(t)))
    return window, float(times[-1]), track(t=float(times[-1]))


def test_rocking_in_place_at_high_angular_speed_is_settled():
    # The diagnosed Pour2 dumpling: +-1 mm about one contact point, ~2 rad/s.
    rest = np.array([[0.9, 0.85, 1.02], [1.5, 0.3, 0.8], [1.52, 0.31, 0.8]])

    def track(*, t: float) -> np.ndarray:
        p = rest.copy()
        p[0] += 0.001 * np.sin(2 * np.pi * 3.0 * t) * np.array([1.0, 0.0, 0.3])
        return p

    window, now, current = _filled(track=track)
    assert window.settled(time=now, positions=current)


def test_a_six_millimetre_slide_inside_the_window_is_not_settled():
    rest = np.zeros((3, 3))

    def track(*, t: float) -> np.ndarray:
        p = rest.copy()
        p[1, 0] = 0.006 * float(np.clip((t - 61.5) / 0.5, 0.0, 1.0))
        return p

    window, now, current = _filled(track=track)
    assert not window.settled(time=now, positions=current)
    assert window.resting(time=now, positions=current).tolist() == [True, False, True]


def test_a_slide_that_ended_before_the_window_is_settled():
    def track(*, t: float) -> np.ndarray:
        p = np.zeros((3, 3))
        p[1, 0] = 0.006 * float(np.clip((t - 60.0) / 0.5, 0.0, 1.0))
        return p

    window, now, current = _filled(track=track, seconds=3.0)
    assert window.settled(time=now, positions=current)


def test_too_short_or_empty_history_is_not_settled():
    still = np.zeros((3, 3))
    assert not PayloadRestWindow().settled(time=60.0, positions=still)
    window, now, current = _filled(track=lambda *, t: still, seconds=1.9)
    assert not window.settled(time=now, positions=current)


def test_a_teleport_since_the_last_sample_is_not_settled():
    still = np.zeros((3, 3))
    window, now, _ = _filled(track=lambda *, t: still)
    moved = still.copy()
    moved[2, 2] = 0.05
    assert not window.settled(time=now, positions=moved)


def test_time_running_backwards_discards_the_history():
    still = np.zeros((3, 3))
    window, now, _ = _filled(track=lambda *, t: still)
    assert not window.settled(time=now - 1.0, positions=still)
    window.record(time=now - 10.0, positions=still)
    assert not window.settled(time=now - 10.0, positions=still)
