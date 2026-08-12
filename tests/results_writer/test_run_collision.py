"""Covers `RunNameCollisionCheck`: the setup-time guarantee of **one canonical run per
experiment**.

Three cases, and they are deliberately not the same error:

1. same name, **different** configs -- the namer forgot an axis of variation. A bug in
   our code, and `--re-run` does not silence it.
2. same name, **identical** configs -- a genuine re-run, which `--re-run` authorises.
3. no collision -- proceed.

The check itself is pure: it takes the runs that already exist as data, so every case
is tested without a network, a credential, or `wandb` installed. Fetching those runs
from a tracker is the backend's job and lives in `wandb_writer.py`.
"""

import pytest

from hitl_pmp.results_writer.run_collision import (
    DuplicateExperimentError,
    MissingVariationAxisError,
    RunNameCollisionCheck,
)
from hitl_pmp.results_writer.types import ExistingRun

NAME = "tossingroom-ees-oneway-never-ask-never-c100-seed3"

CONFIG = {
    "env": "tossingroom",
    "method": "ees",
    "seed": "3",
    "num_cycles": "100",
    "num_test_tasks": "30",
    "output_dir": "results/a/ees/3",
    "record_wandb": "True",
    "re_run": "False",
}


class Runs:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def existing(**overrides: str) -> ExistingRun:
        config = dict(CONFIG)
        config.update(overrides)
        return ExistingRun(
            name=NAME,
            identifier="ezy6q16y",
            url="https://wandb.ai/josh-princeton/hitl-pmp/runs/ezy6q16y",
            config=config,
        )


def test_no_existing_run_is_the_ordinary_case() -> None:
    RunNameCollisionCheck.check(name=NAME, config=CONFIG, existing=(), re_run=False)


def test_a_differing_config_reports_the_namer_as_missing_an_axis() -> None:
    """The whole point of comparing configs rather than names: two runs that are
    genuinely different experiments must never share one canonical name. The message
    names the differing field, because that field is the one-line fix."""
    with pytest.raises(MissingVariationAxisError) as raised:
        RunNameCollisionCheck.check(
            name=NAME, config=CONFIG, existing=(Runs.existing(num_test_tasks="10"),), re_run=False
        )
    message = str(raised.value)
    assert "num_test_tasks" in message
    assert "10" in message and "30" in message
    # Actionable at 2am: which run, and where the fix goes.
    assert "https://wandb.ai/josh-princeton/hitl-pmp/runs/ezy6q16y" in message
    assert "run_naming.py" in message


def test_re_run_does_not_silence_a_missing_axis() -> None:
    """`--re-run` authorises repeating **one identical experiment**. Silencing this
    case would knowingly write two different experiments into one canonical slot, which
    is precisely what the check exists to prevent."""
    with pytest.raises(MissingVariationAxisError):
        RunNameCollisionCheck.check(
            name=NAME, config=CONFIG, existing=(Runs.existing(num_test_tasks="10"),), re_run=True
        )


def test_an_identical_config_is_a_re_run_and_says_so() -> None:
    with pytest.raises(DuplicateExperimentError, match="--re-run"):
        RunNameCollisionCheck.check(
            name=NAME, config=CONFIG, existing=(Runs.existing(),), re_run=False
        )


def test_the_re_run_flag_authorises_exactly_that() -> None:
    RunNameCollisionCheck.check(name=NAME, config=CONFIG, existing=(Runs.existing(),), re_run=True)


def test_fields_that_cannot_change_the_result_are_not_compared() -> None:
    """The exclusion list is the pure observers plus the output location: a run
    recorded with different instrumentation, or written to a different directory, is
    the same experiment. Every one of those flags is separately asserted to leave
    `stats.json` byte-identical."""
    RunNameCollisionCheck.check(
        name=NAME,
        config=CONFIG,
        existing=(
            Runs.existing(
                output_dir="somewhere/else",
                record_wandb="False",
                re_run="True",
                num_render_checkpoints="4",
            ),
        ),
        re_run=True,
    )


def test_a_field_present_on_only_one_side_is_reported_rather_than_ignored() -> None:
    """Comparing over the intersection would make an added or removed flag invisible,
    which is the same silent-omission failure in a different place. The sentinel keeps
    "absent" distinguishable from any value a flag could take."""
    existing = Runs.existing()
    del existing.config["num_cycles"]
    with pytest.raises(MissingVariationAxisError) as raised:
        RunNameCollisionCheck.check(name=NAME, config=CONFIG, existing=(existing,), re_run=False)
    assert "num_cycles" in str(raised.value)


def test_a_missing_axis_outranks_a_duplicate_when_both_are_present() -> None:
    """A code bug is worth reporting ahead of an intentional-looking re-run: the
    duplicate is only meaningful once the namer is known to be complete."""
    with pytest.raises(MissingVariationAxisError):
        RunNameCollisionCheck.check(
            name=NAME,
            config=CONFIG,
            existing=(Runs.existing(), Runs.existing(num_test_tasks="10")),
            re_run=False,
        )


def test_a_differently_named_run_is_not_a_collision() -> None:
    """Only same-name runs are candidates; the tracker query filters on the name, and
    this pins that the checker agrees rather than comparing everything to everything."""
    other = Runs.existing(num_test_tasks="10")
    RunNameCollisionCheck.check(
        name="a-different-name", config=CONFIG, existing=(other,), re_run=False
    )
