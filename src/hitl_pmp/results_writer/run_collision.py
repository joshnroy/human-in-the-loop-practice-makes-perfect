"""`RunNameCollisionCheck`: **one canonical run per experiment**, enforced at setup.

## What it is actually for

Not "does a run with this name already exist" -- that would be a uniqueness check on a
string. The goal is that one experiment (one distinct configuration) has one canonical
run, so the check compares **configurations**, and the case it exists to catch is a bug
in our own code: two runs with *different* configs generating the *same* name means
`run_naming.RUN_NAME_FIELDS` is missing an axis of variation.

Three cases, three different meanings:

| existing run with this name | its config | verdict |
| --- | --- | --- |
| none | -- | proceed |
| one or more | **differs** | `MissingVariationAxisError` -- our namer is wrong |
| one or more | identical | `DuplicateExperimentError` -- a genuine re-run |

**`--re-run` authorises the third row only.** Silencing the second would knowingly write
two different experiments into one canonical slot, which is precisely what this check
exists to prevent, so `re_run` is not even consulted on that path. A missing axis also
outranks a duplicate when both are present: the duplicate is only meaningful once the
namer is known to be complete.

## What counts as "the same experiment"

Everything in the resolved namespace **except the flags that cannot change the result**:
the pure observers (`--record-*`, `--num-render-checkpoints`, `--record-full-loop`), the
output location (`--output-dir`), and `--re-run` itself. That complement is not a
judgement call -- each recording flag is separately asserted to leave `stats.json`
byte-identical, which is this repo's definition of "did not alter results", so a run
recorded with different instrumentation or written to a different directory *is* the
same experiment. Every other flag feeds the environment, the method or the seed, so a
difference in one is a different experiment by definition.

`ConfigSnapshot` deliberately keeps no such list -- it records `vars(args)` wholesale,
because its job is provenance ("under what conditions did this happen") and dropping a
field would lose evidence. This is the opposite job (equivalence), so it needs the
exclusions that `ConfigSnapshot` must not have. The two are consistent: the snapshot
keeps everything, and equality is defined over a stated subset of it.

Keys present on only one side are compared against a sentinel rather than skipped.
Comparing over the intersection would make an added or removed flag invisible, which is
the same silent-omission failure in a different place.

## Pure, and therefore testable

This module takes the runs that already exist as data and imports no tracker. Fetching
them is the backend's job (`wandb_writer.py`), so every case here -- including the exact
wording of both errors, which is as much the deliverable as the check -- is covered
without a network, a credential, or `wandb` installed.
"""

from typing import ClassVar

from hitl_pmp.results_writer.types import ExistingRun

_NAMER_SOURCE = "RUN_NAME_FIELDS in src/hitl_pmp/results_writer/run_naming.py"


class RunNameCollisionError(ValueError):
    """Base for the two ways a run name can already be taken. A `ValueError`, so it
    joins the other up-front `open_if_requested` failures rather than needing its own
    handling in the harness."""


class MissingVariationAxisError(RunNameCollisionError):
    """Two different experiments would share one canonical run name: a defect in the
    run namer, not in the run being launched."""


class DuplicateExperimentError(RunNameCollisionError):
    """This exact experiment has already been recorded."""


class RunNameCollisionCheck:
    """A static-method container, never instantiated, same as every other genuinely
    stateless business-logic class in this project."""

    # The flags that cannot change what a run produces. See the module docstring; the
    # membership rule is "is separately asserted to leave stats.json byte-identical".
    EXCLUDED_FROM_COMPARISON: ClassVar[frozenset[str]] = frozenset({
        "output_dir",
        "record_wandb",
        "record_sampler_draws",
        "record_skill_competence",
        "record_episode_traces",
        "record_full_loop",
        "num_render_checkpoints",
        "re_run",
    })

    # Distinguishable from any value a flag could take, so "the flag was not there" and
    # "the flag was there and was empty" never read the same in an error message.
    ABSENT: ClassVar[str] = "<absent>"

    @staticmethod
    def check(
        *,
        name: str,
        config: dict[str, str],
        existing: tuple[ExistingRun, ...],
        re_run: bool,
    ) -> None:
        """Raise if `name` is already taken; return silently otherwise.

        `existing` is whatever the tracker holds under this name -- normally empty.
        Passed in rather than fetched here so this stays pure; the caller decides
        whether fetching is even possible (it is not offline)."""
        same_name = [run for run in existing if run.name == name]
        for run in same_name:
            differences = RunNameCollisionCheck._differences(config=config, existing=run)
            if differences:
                raise MissingVariationAxisError(
                    RunNameCollisionCheck._missing_axis_message(
                        name=name, run=run, differences=differences
                    )
                )
        if same_name and not re_run:
            raise DuplicateExperimentError(
                RunNameCollisionCheck._duplicate_message(name=name, run=same_name[0])
            )

    @staticmethod
    def comparable(*, config: dict[str, str]) -> dict[str, str]:
        """`config` reduced to the fields that determine what the run produces."""
        return {
            key: value
            for key, value in config.items()
            if key not in RunNameCollisionCheck.EXCLUDED_FROM_COMPARISON
        }

    @staticmethod
    def _differences(
        *, config: dict[str, str], existing: ExistingRun
    ) -> dict[str, tuple[str, str]]:
        """Every field the two runs disagree on, as `{key: (mine, theirs)}`, over the
        **union** of their keys."""
        mine = RunNameCollisionCheck.comparable(config=config)
        theirs = RunNameCollisionCheck.comparable(config=existing.config)
        return {
            key: (
                mine.get(key, RunNameCollisionCheck.ABSENT),
                theirs.get(key, RunNameCollisionCheck.ABSENT),
            )
            for key in sorted(set(mine) | set(theirs))
            if mine.get(key, RunNameCollisionCheck.ABSENT)
            != theirs.get(key, RunNameCollisionCheck.ABSENT)
        }

    @staticmethod
    def _missing_axis_message(
        *, name: str, run: ExistingRun, differences: dict[str, tuple[str, str]]
    ) -> str:
        listed = "\n".join(
            f"    {key}: this run={mine!r}  existing run={theirs!r}"
            for key, (mine, theirs) in differences.items()
        )
        return (
            f"The run namer is missing an axis of variation. This run would be called "
            f"{name!r}, but {run.url} already has that name with a DIFFERENT "
            f"configuration, so two different experiments would share one canonical "
            f"run.\n"
            f"  {len(differences)} field(s) differ:\n"
            f"{listed}\n"
            f"  Fix: add the field(s) above to {_NAMER_SOURCE} so the name tells the "
            f"two runs apart. (If a field reads {RunNameCollisionCheck.ABSENT!r}, the "
            f"CLI may instead have gained or lost that flag since the existing run was "
            f"recorded.)\n"
            f"  --re-run does NOT silence this: it authorises repeating one identical "
            f"experiment, never merging two different ones."
        )

    @staticmethod
    def _duplicate_message(*, name: str, run: ExistingRun) -> str:
        return (
            f"This exact experiment has already been recorded as {name!r} "
            f"({run.url}). Its configuration is identical to this run's on every field "
            f"that can change the result, so recording it again would leave two "
            f"canonical runs for one experiment.\n"
            f"  Pass --re-run to record it anyway, or delete run {run.identifier} first."
        )
