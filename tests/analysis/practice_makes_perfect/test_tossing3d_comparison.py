"""Pins the counts behind the Tossing3D log against the committed aggregate.

`docs/experiment-logs/2026-08-04-tossing3d-ees.md` reports every success rate as a
count (`7/10`, `67/100`). Those counts came out of the runs' own `stats.json` --
`Metrics.record_evaluation` writes `(transitions, num_solved, num_total)` triples, so
the numerator and the denominator were both recorded and neither is reconstructed by
multiplying a percentage back out. The triples are committed as
`2026-08-04-tossing3d-arms.json` precisely so that claim stays checkable after the
results directory is gone, and these tests are what check it: if the committed counts
ever stopped reproducing the published percentages, the log and its data would have
drifted apart silently, which is the failure this file exists to make loud.

The expected percentages below are quoted from the log, not computed from the JSON and
recorded -- otherwise the test would agree with any data it was handed.
"""

import json
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.tossing3d_comparison import Tossing3DComparison
from hitl_pmp.core.metrics.metrics import Metrics

ARMS_JSON = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "experiment-logs"
    / "2026-08-04-tossing3d-arms.json"
)

# Quoted from the log's "Per-seed end-of-training" lines, in seed order 0..9.
PUBLISHED_FINAL_PERCENTAGES = {
    "ees": [70, 60, 90, 80, 40, 50, 50, 40, 100, 90],
    "random-skills": [30, 20, 30, 10, 20, 20, 20, 30, 10, 20],
}
# Quoted from the log's arm table: pre-practice and end-of-training, pooled over seeds.
PUBLISHED_POOLED = {
    "ees": {"first": (33, 100), "final": (67, 100)},
    "random-skills": {"first": (14, 100), "final": (21, 100)},
}


@pytest.fixture(name="arms")
def _arms() -> dict[str, dict[str, list[list[int]]]]:
    return json.loads(ARMS_JSON.read_text())


@pytest.mark.parametrize("arm", ["ees", "random-skills"])
def test_committed_counts_reproduce_the_published_per_seed_percentages(
    *, arm: str, arms: dict[str, dict[str, list[list[int]]]]
) -> None:
    """The conversion's whole claim: dividing the recorded numerator by the recorded
    denominator gives back exactly what was published as a percentage. A mismatch would
    mean the published rate was wrong, which matters more than the formatting."""
    seeds = sorted(arms[arm], key=int)
    assert len(seeds) == 10
    finals = [arms[arm][seed][-1] for seed in seeds]
    assert [100.0 * solved / total for _, solved, total in finals] == pytest.approx(
        PUBLISHED_FINAL_PERCENTAGES[arm]
    )


@pytest.mark.parametrize("arm", ["ees", "random-skills"])
def test_every_seed_evaluates_the_same_ten_tasks_at_every_checkpoint(
    *, arm: str, arms: dict[str, dict[str, list[list[int]]]]
) -> None:
    """The pooled `x/100` in the log is only a real count if the denominators are
    uniform -- otherwise pooling and averaging the per-seed rates disagree and one of
    the two is being passed off as the other."""
    for curve in arms[arm].values():
        assert {total for _, _, total in curve} == {10}
        assert len(curve) == 11


@pytest.mark.parametrize("arm", ["ees", "random-skills"])
def test_pooled_endpoint_counts_match_the_arm_table(
    *, arm: str, arms: dict[str, dict[str, list[list[int]]]]
) -> None:
    """33/100 -> 67/100 for EES, 14/100 -> 21/100 for the floor, summed rather than
    averaged."""
    curves = list(arms[arm].values())
    first = (sum(c[0][1] for c in curves), sum(c[0][2] for c in curves))
    final = (sum(c[-1][1] for c in curves), sum(c[-1][2] for c in curves))
    assert first == PUBLISHED_POOLED[arm]["first"]
    assert final == PUBLISHED_POOLED[arm]["final"]


def test_per_seed_counts_reads_the_recorded_triples_not_a_rounded_rate(*, tmp_path: Path) -> None:
    """7/30 is 23.333...%, so a reader handed only the percentage cannot recover the 7.
    `per_seed_counts` must return the integers `Metrics` recorded, at whatever
    denominator that run used."""
    run_dir = tmp_path / "ees" / "3"
    run_dir.mkdir(parents=True)
    metrics = Metrics()
    metrics.record_evaluation(num_online_transitions=0, num_solved=7, num_total=30)
    metrics.record_evaluation(num_online_transitions=100, num_solved=30, num_total=30)
    (run_dir / "stats.json").write_text(metrics.model_dump_json())

    counts = Tossing3DComparison.per_seed_counts(root=tmp_path, method="ees")
    assert counts == {3: {0: (7, 30), 100: (30, 30)}}
    assert Tossing3DComparison.pooled(counts=counts, transitions=0) == (7, 30)
