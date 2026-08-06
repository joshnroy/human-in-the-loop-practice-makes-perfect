"""Tests for the CPU-vs-GPU sampler benchmark plot/report.

The two things worth pinning are the ones a reader takes on trust from the figure:

1. **The crossover bracket**, which is the transferable answer. Reporting one where
   none exists -- or missing one that does -- is the whole result being wrong.
2. **The `cpu1` relabel.** A file produced with an explicit `set_num_threads` is a
   *different arm*, not another sample of the default CPU arm. Merging the two would
   average a 0.3 s measurement with an 18 s one and hide the effect entirely.
"""

import json

from analysis.practice_makes_perfect.sampler_device_bench import SamplerDeviceBenchPlot


def _row(*, device, n, seconds):
    return {
        "device": device,
        "n": n,
        "dim": 12,
        "max_train_iters": 1000,
        "median_seconds": seconds,
        "min_seconds": seconds,
        "max_seconds": seconds,
        "reps": 3,
        "samples": [seconds],
    }


def test_crossover_is_reported_as_the_bracketing_interval():
    """CPU wins at 64, GPU wins at 128, so the crossover is bracketed by those two --
    not interpolated to a spurious exact n."""
    rows = [
        _row(device="cpu", n=64, seconds=0.3),
        _row(device="cuda", n=64, seconds=0.7),
        _row(device="cpu", n=128, seconds=18.0),
        _row(device="cuda", n=128, seconds=0.7),
    ]
    assert (
        SamplerDeviceBenchPlot.crossover(rows=rows, slow="cpu", fast="cuda")
        == "crossover between n = 64 and n = 128"
    )


def test_no_crossover_is_said_in_full_when_one_arm_wins_everywhere():
    rows = [
        _row(device="cpu", n=8, seconds=0.3),
        _row(device="cuda", n=8, seconds=0.7),
        _row(device="cpu", n=64, seconds=0.3),
        _row(device="cuda", n=64, seconds=0.7),
    ]
    message = SamplerDeviceBenchPlot.crossover(rows=rows, slow="cpu", fast="cuda")
    assert message == "no crossover: cpu is faster at every measured n (max n = 64)"


def test_no_crossover_the_other_way_round():
    rows = [
        _row(device="cpu", n=8, seconds=9.0),
        _row(device="cuda", n=8, seconds=0.7),
    ]
    message = SamplerDeviceBenchPlot.crossover(rows=rows, slow="cpu", fast="cuda")
    assert message == "no crossover: cuda is faster at every measured n (min n = 8)"


def test_an_n_measured_on_only_one_device_is_ignored_rather_than_compared():
    """An unpaired n cannot support a comparison; including it would silently invent
    one against a missing value."""
    rows = [
        _row(device="cpu", n=8, seconds=0.3),
        _row(device="cuda", n=8, seconds=0.7),
        _row(device="cpu", n=4096, seconds=99.0),
    ]
    message = SamplerDeviceBenchPlot.crossover(rows=rows, slow="cpu", fast="cuda")
    assert message == "no crossover: cpu is faster at every measured n (max n = 8)"


def test_an_explicit_thread_setting_becomes_its_own_arm(*, tmp_path):
    """Merging a --threads run into the default CPU arm would average 0.3 s with 18 s
    and erase the effect the figure exists to show."""
    default_path = tmp_path / "default.json"
    default_path.write_text(
        json.dumps({
            "called_set_num_threads": False,
            "rows": [_row(device="cpu", n=128, seconds=18.0)],
        })
    )
    pinned_path = tmp_path / "pinned.json"
    pinned_path.write_text(
        json.dumps({
            "called_set_num_threads": True,
            "rows": [_row(device="cpu", n=128, seconds=0.35)],
        })
    )
    rows = SamplerDeviceBenchPlot.load(paths=[default_path, pinned_path])
    by_device = {row["device"]: row["median_seconds"] for row in rows}
    assert by_device == {"cpu": 18.0, "cpu1": 0.35}


def test_the_figure_is_written(*, tmp_path):
    rows = [
        _row(device="cpu", n=64, seconds=0.3),
        _row(device="cuda", n=64, seconds=0.7),
        _row(device="cpu", n=128, seconds=18.0),
        _row(device="cuda", n=128, seconds=0.7),
    ]
    out_path = tmp_path / "figure.png"
    SamplerDeviceBenchPlot.plot(rows=rows, out_path=out_path, title="test")
    assert out_path.stat().st_size > 0
