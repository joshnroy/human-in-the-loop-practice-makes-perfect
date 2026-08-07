"""Integrity tests for the fourth cell's committed runs: the pickup-weight fork under
`--two-way-ledge`.

These do not test any analysis code. They assert facts about the **committed data**, and
they exist because the fourth cell of a 2x2 is only interpretable if it differs from the
banked cells in exactly the one intended way. Both were pre-registered in
`docs/experiment-logs/2026-08-07-pickup-weight-two-way-ledge.md` before the sweep ran.

The comparison is against the *resolved* argparse namespace recorded in
`config_snapshot.json`, not against the command line that produced it -- so a default
that drifted between the two sweeps is caught as well as a flag someone forgot to pass.
"""

import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_BANKED = _REPO / "docs/experiment-logs/2026-08-07-pickup-weight-reset-free-runs"
_NEW = _REPO / "docs/experiment-logs/2026-08-07-pickup-weight-two-way-ledge-runs"


@pytest.mark.parametrize("policy", ["scheduled", "never"])
def test_new_cell_matches_the_banked_cells_settings_key_by_key(*, policy: str) -> None:
    """Exactly two keys may differ from PR #122's banked cell:

    * `two_way_ledge` -- the intervention. It is absent entirely from the banked
      snapshot, because the flag did not exist on this fork when those runs were made.
    * `output_dir` -- the path the run was written to, which no result depends on.

    Everything else must be identical, including every value that was defaulted rather
    than passed."""
    banked = json.loads((_BANKED / policy / "0" / "config_snapshot.json").read_text())["args"]
    new = json.loads((_NEW / policy / "ees" / "0" / "config_snapshot.json").read_text())["args"]
    differing = {key for key in set(banked) | set(new) if banked.get(key) != new.get(key)}
    assert differing == {"two_way_ledge", "output_dir"}
    assert new["two_way_ledge"] == "True"
    assert banked["practice_reset_policy"] == new["practice_reset_policy"] == policy


@pytest.mark.parametrize("policy", ["scheduled", "never"])
def test_new_cell_reset_counts_are_the_manipulation_check(*, policy: str) -> None:
    """The reset policy has to show up as *measured* resets -- 10 per `scheduled` run and
    0 per `never` run -- rather than as a flag in a log line. A `never` arm that reset
    anyway would be measuring nothing, and the whole cell would be uninterpretable."""
    expected = 10 if policy == "scheduled" else 0
    for seed in range(10):
        stats = json.loads((_NEW / policy / "ees" / str(seed) / "stats.json").read_text())
        assert stats["num_practice_resets"] == expected, f"seed {seed}"
