"""Both committed pickup-weight run trees follow `<results-root>/<method>/<seed>/`.

`scripts/run_sweep.py`'s `SweepRun` construction defines that layout
(`output_dir = results_root / method / str(seed)`), and it is what every `analysis/`
reader globs for. The two 2026-08-07 trees disagreed with each other: the two-way-ledge
set had the `<method>` level and the reset-free set did not, so
`PickupWeightStranding.read_arm` -- which reads `root/ees/<seed>/stats.json`, and whose
`--help` says so -- worked on one committed tree and could not be pointed at the other at
all.

Pinned as a test rather than fixed once, because the failure mode is silent to everything
except a reader who tries the documented command: nothing imports these paths at
collection time, and a sweep writes wherever `--results-root` says.
"""

from pathlib import Path

import pytest

from analysis.practice_makes_perfect.pickup_weight_stranding import PickupWeightStranding

_LOGS = Path(__file__).resolve().parents[3] / "docs/experiment-logs"
_TREES = (
    "2026-08-07-pickup-weight-reset-free-runs",
    "2026-08-07-pickup-weight-two-way-ledge-runs",
)
_ARMS = ("never", "scheduled")
_SEEDS = tuple(range(10))


@pytest.mark.parametrize("tree", _TREES)
@pytest.mark.parametrize("arm", _ARMS)
def test_every_run_sits_at_method_then_seed(*, tree: str, arm: str) -> None:
    """The canonical layout, asserted on the committed bytes rather than on a sweep."""
    root = _LOGS / tree / arm
    missing = [seed for seed in _SEEDS if not (root / "ees" / str(seed) / "stats.json").is_file()]
    assert missing == [], f"{tree}/{arm}: no ees/<seed>/stats.json for seeds {missing}"


@pytest.mark.parametrize("tree", _TREES)
@pytest.mark.parametrize("arm", _ARMS)
def test_no_seed_directory_hangs_directly_off_the_arm(*, tree: str, arm: str) -> None:
    """The old reset-free layout, pinned out. A tree carrying both would let a reader
    point an analysis script at the wrong half and get a silently different answer."""
    stranded = [seed for seed in _SEEDS if (_LOGS / tree / arm / str(seed)).exists()]
    assert stranded == [], f"{tree}/{arm}: seed dirs hang directly off the arm: {stranded}"


@pytest.mark.parametrize("tree", _TREES)
@pytest.mark.parametrize("arm", _ARMS)
def test_the_stranding_reader_can_be_pointed_at_the_committed_tree(*, tree: str, arm: str) -> None:
    """The consequence the layout exists for: `read_arm` is the documented entrypoint, so
    it has to work against what is committed, not only against a scratch sweep."""
    records = PickupWeightStranding.read_arm(root=_LOGS / tree / arm, seeds=list(_SEEDS))
    assert [record.seed for record in records] == list(_SEEDS)
