"""`RunNameCollisionCheck`: **one canonical run per experiment**, enforced at setup.

Not a uniqueness check on a string: it compares **configurations**, and the case it exists
to catch is a bug in our own code -- two runs with different configs generating the same
name means `run_naming.RUN_NAME_FIELDS` is missing an axis of variation.

| existing run with this name | its config | verdict |
| --- | --- | --- |
| none | -- | proceed |
| one or more | **differs** | `MissingVariationAxisError` -- our namer is wrong |
| one or more | identical | `DuplicateExperimentError` -- a genuine re-run |

**`--re-run` authorises the third row only**; `re_run` is not even consulted on the second
path, since silencing it would knowingly write two different experiments into one canonical
slot. A missing axis outranks a duplicate when both are present.

**"The same experiment"** is everything in the resolved namespace except the flags that
cannot change the result: the pure observers (`--record-*`, `--num-render-checkpoints`,
`--record-full-loop`), `--output-dir`, and `--re-run` itself. Each recording flag is
separately asserted to leave `stats.json` byte-identical, which is this repo's definition of
"did not alter results". `ConfigSnapshot` deliberately keeps no such list -- its job is
provenance, so it records `vars(args)` wholesale; this is the opposite job (equivalence).
Keys present on only one side are compared against a sentinel rather than skipped, since
comparing over the intersection would make an added or removed flag invisible.

**Pure, and therefore testable**: this module takes the runs that already exist as data and
imports no tracker, so every case -- including the exact wording of both errors -- is
covered without a network, a credential, or `wandb` installed."""

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
