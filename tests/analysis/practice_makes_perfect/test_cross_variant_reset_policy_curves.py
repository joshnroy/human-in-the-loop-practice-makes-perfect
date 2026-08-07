import json
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.cross_variant_reset_policy_curves import (
    CrossVariantResetPolicyCurves,
    PanelKey,
)


def _stats(*, finals: list[int]) -> dict:
    """A stats.json whose evaluation curve ends at `finals[-1]` solved of 30."""
    return {
        "evaluations": [[150 * i, solved, 30] for i, solved in enumerate(finals)],
        "breakdowns": [],
        "num_practice_resets": 10,
        "planning_failures_per_cycle": [],
        "planning_attempts_per_cycle": [],
        "practice_outcomes_per_cycle": [],
        "task_name": "t",
    }


def _write_arm(*, root: Path, finals_per_seed: list[list[int]], nested: bool) -> Path:
    for seed, finals in enumerate(finals_per_seed):
        run = root / "ees" / str(seed) if nested else root / str(seed)
        run.mkdir(parents=True)
        (run / "stats.json").write_text(json.dumps(_stats(finals=finals)))
    return root


def test_load_arm_reads_both_committed_directory_layouts(*, tmp_path: Path) -> None:
    """The two banked experiments were committed under different layouts -- #125's runs
    sit at `<arm>/ees/<seed>/` and #122's at `<arm>/<seed>/`. A loader that understood
    only one of them would silently read a 0-seed arm for the other."""
    flat = _write_arm(root=tmp_path / "flat", finals_per_seed=[[1, 5], [1, 7]], nested=False)
    nested = _write_arm(root=tmp_path / "nest", finals_per_seed=[[1, 5], [1, 7]], nested=True)
    assert CrossVariantResetPolicyCurves.load_arm(results_root=flat, num_seeds=2) == (
        CrossVariantResetPolicyCurves.load_arm(results_root=nested, num_seeds=2)
    )


def test_load_arm_raises_on_a_missing_seed(*, tmp_path: Path) -> None:
    """A reader that skipped one silently would report a 1-seed result as a 2-seed one."""
    root = _write_arm(root=tmp_path / "a", finals_per_seed=[[1, 5]], nested=False)
    with pytest.raises(FileNotFoundError, match="seed 1"):
        CrossVariantResetPolicyCurves.load_arm(results_root=root, num_seeds=2)


def test_final_scores_sum_to_the_arm_total_out_of_its_own_denominator(*, tmp_path: Path) -> None:
    """`x/300` is the sum of the per-seed finals over 10 seeds x 30 tasks. The
    denominator is carried, never assumed, so a short sweep reports `x/60` rather than
    silently claiming a 300-task denominator it does not have."""
    root = _write_arm(root=tmp_path / "a", finals_per_seed=[[0, 18], [0, 12]], nested=False)
    arm = CrossVariantResetPolicyCurves.load_arm(results_root=root, num_seeds=2)
    assert CrossVariantResetPolicyCurves.final_scores(arm=arm) == [18, 12]
    assert CrossVariantResetPolicyCurves.arm_total(arm=arm) == (30, 60)


def test_minimum_detectable_effect_uses_both_denominators() -> None:
    """MDE at 2.801585*sqrt(p_bar*(1-p_bar)*(1/n1+1/n2)) -- computed per comparison from
    its own two denominators, so a 300-vs-300 comparison and a 20-vs-20 one never share
    a single project-wide number."""
    wide = CrossVariantResetPolicyCurves.minimum_detectable_effect(
        successes=(150, 150), totals=(300, 300)
    )
    narrow = CrossVariantResetPolicyCurves.minimum_detectable_effect(
        successes=(10, 10), totals=(20, 20)
    )
    assert wide == pytest.approx(0.1143, abs=5e-4)
    assert narrow > wide


def test_bimodality_split_reports_counts_not_a_shape_claim(*, tmp_path: Path) -> None:
    """The one-way pickup-weight `never` arm is bimodal -- 4/10 seeds at 16-21 and 6/10
    at 5-7. The split is reported as the two counts and the gap between the clusters, so
    a reader sees the denominators rather than the word "bimodal"."""
    root = _write_arm(
        root=tmp_path / "a",
        finals_per_seed=[[0, v] for v in (18, 16, 5, 6, 7, 6, 21, 20, 7, 6)],
        nested=False,
    )
    arm = CrossVariantResetPolicyCurves.load_arm(results_root=root, num_seeds=10)
    split = CrossVariantResetPolicyCurves.mode_split(arm=arm)
    assert split.low_count == 6
    assert split.high_count == 4
    assert split.total == 10
    assert split.gap == 9  # 16 - 7


def test_panel_keys_are_the_full_two_by_two() -> None:
    """Four cells, derived from the (variant, ledge) pair rather than written out as
    four strings, so a missing cell is a KeyError rather than a quietly absent panel."""
    keys = CrossVariantResetPolicyCurves.panel_keys()
    assert len(keys) == 4
    assert PanelKey(variant="tossingroomsplitpickupweight", ledge="two-way") in keys
    assert {k.variant for k in keys} == {"tossingroomsplit", "tossingroomsplitpickupweight"}
    assert {k.ledge for k in keys} == {"one-way", "two-way"}


def test_render_writes_a_four_panel_figure(*, tmp_path: Path) -> None:
    arms = {}
    for key in CrossVariantResetPolicyCurves.panel_keys():
        for policy in ("scheduled", "never"):
            root = _write_arm(
                root=tmp_path / f"{key.variant}-{key.ledge}-{policy}",
                finals_per_seed=[[0, 5, 9], [0, 6, 11]],
                nested=False,
            )
            arms[(key, policy)] = CrossVariantResetPolicyCurves.load_arm(
                results_root=root, num_seeds=2
            )
    out = tmp_path / "fig.png"
    CrossVariantResetPolicyCurves.render(arms=arms, output=out)
    assert out.exists()
    assert out.stat().st_size > 5000


_REPO = Path(__file__).resolve().parents[3]
_BANKED = _REPO / "docs/experiment-logs/2026-08-07-pickup-weight-reset-free-runs"
_NEW = _REPO / "docs/experiment-logs/2026-08-07-pickup-weight-two-way-ledge-runs"


@pytest.mark.parametrize("policy", ["scheduled", "never"])
def test_new_cell_matches_the_banked_cells_settings_key_by_key(*, policy: str) -> None:
    """**The comparability assertion, pre-registered before the sweep ran.** The fourth
    cell of the 2x2 is only interpretable if it differs from PR #122's banked cells in
    exactly one intended way. So compare the two committed `config_snapshot.json`s
    key-by-key rather than trusting the command line that produced them, and allow only:

    * `two_way_ledge` -- the intervention (absent entirely in the banked snapshot,
      because the flag did not exist on this fork then).
    * `output_dir` -- the path the run was written to, which no result depends on.

    Everything else, including every value that was defaulted rather than passed, must be
    identical. A snapshot records the *resolved* namespace, so this catches a default
    that drifted as well as a flag someone forgot."""
    banked = json.loads((_BANKED / policy / "0" / "config_snapshot.json").read_text())["args"]
    new = json.loads((_NEW / policy / "ees" / "0" / "config_snapshot.json").read_text())["args"]
    differing = {k for k in set(banked) | set(new) if banked.get(k) != new.get(k)}
    assert differing == {"two_way_ledge", "output_dir"}
    assert new["two_way_ledge"] == "True"
    assert banked["practice_reset_policy"] == new["practice_reset_policy"] == policy


@pytest.mark.parametrize("policy", ["scheduled", "never"])
def test_new_cell_has_all_ten_seeds_and_the_expected_reset_counts(*, policy: str) -> None:
    """The manipulation check, on the committed data rather than on a log line: the
    reset policy has to show up as *measured* resets, 10 per scheduled run and 0 per
    never run. A cell whose `never` arm reset anyway would be measuring nothing."""
    expected = 10 if policy == "scheduled" else 0
    for seed in range(10):
        stats = json.loads((_NEW / policy / "ees" / str(seed) / "stats.json").read_text())
        assert stats["num_practice_resets"] == expected
