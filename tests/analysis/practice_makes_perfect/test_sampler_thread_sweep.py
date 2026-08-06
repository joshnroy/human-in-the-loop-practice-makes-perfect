"""Tests for the thread-sweep figure.

The one thing here that is wrong in a way no one would notice from the picture: the
shipped default records `threads = -1`, meaning "`set_num_threads` was never called".
Plotting that on the numeric thread axis would put it at x = -1, or -- worse, if it were
coerced -- silently merge it with a real setting. It is a different condition, and the
measured numbers make that concrete: at n = 16 the default costs 0.873 s on the same box
where an explicit `set_num_threads(24)` costs 120.426 s.
"""

from analysis.practice_makes_perfect.sampler_thread_sweep import SamplerThreadSweep


def _row(*, threads, seconds, load=40.0):
    return {
        "threads": threads,
        "median_seconds": seconds,
        "min_seconds": seconds,
        "max_seconds": seconds,
        "load_average_1min": load,
        "n": 16,
        "dim": 12,
        "max_train_iters": 1000,
    }


def test_the_shipped_default_is_split_out_from_the_explicit_settings():
    rows = [
        _row(threads=-1, seconds=0.873),
        _row(threads=24, seconds=120.426),
        _row(threads=1, seconds=0.338),
    ]
    explicit, default = SamplerThreadSweep.split(rows=rows)
    assert [row["threads"] for row in explicit] == [1, 24]
    assert [row["median_seconds"] for row in default] == [0.873]


def test_explicit_settings_come_back_in_thread_order_not_file_order():
    """The x axis is the thread count; a file written in sweep order would otherwise
    draw the line doubling back on itself."""
    rows = [_row(threads=n, seconds=float(n)) for n in (8, 1, 24, 2, 16, 4)]
    explicit, _ = SamplerThreadSweep.split(rows=rows)
    assert [row["threads"] for row in explicit] == [1, 2, 4, 8, 16, 24]


def test_a_sweep_with_no_default_row_still_splits():
    rows = [_row(threads=1, seconds=0.338)]
    explicit, default = SamplerThreadSweep.split(rows=rows)
    assert len(explicit) == 1
    assert default == []


def test_the_summary_names_the_default_rather_than_a_thread_count():
    rows = [_row(threads=1, seconds=0.338), _row(threads=-1, seconds=0.873)]
    assert SamplerThreadSweep.summary(rows=rows) == "1t 0.338s; default 0.873s"


def test_the_figure_is_written(*, tmp_path):
    rows = [
        _row(threads=1, seconds=0.338, load=43.6),
        _row(threads=24, seconds=120.426, load=67.7),
        _row(threads=-1, seconds=0.873, load=66.2),
    ]
    out_path = tmp_path / "figure.png"
    SamplerThreadSweep.plot(rows=rows, out_path=out_path, title="test")
    assert out_path.stat().st_size > 0
