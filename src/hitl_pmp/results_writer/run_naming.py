"""`RunNamer`: the one place a run's name is built, for any tracker backend.

## The problem this exists for

`WandbResultsWriter` named runs `f"{method}-seed{seed}"`. An 80-run study therefore
showed as `ees-seed0` ... `ees-seed9` repeated eight times: **every run in the project
collided on name regardless of environment, reset policy, ledge type or cycle budget**,
and the flat run list could not tell a `never` arm from a `scheduled` one without
opening each run. That is not hypothetical -- the live project holds three separate runs
called `ees-seed0`, one on Light Switch (a 35-key config) and two on Tossing Room (48
keys).

## Curated fields, not a hash of the config

Appending a hash of the resolved config would be unique by construction and unreadable
by construction. The name is for the **run list**, which is read by a human, so the
fields are curated and the *collision check* (`run_collision.py`) is what makes that
curation safe: if two different experiments ever produce one name, setup fails and names
the field to add. Uniqueness by construction versus readability plus a loud check --
this project chose the second.

## The format

    <env>-<method>[-<domain variant>...]-<reset policy>[-ask-<help policy>][-c<cycles>]-seed<seed>

    tossingroom-ees-oneway-split-never-ask-never-c100-seed3
    lightswitch-skill-oracle-scheduled-seed7

Ordered **outermost axis first, seed last**, so an alphabetical run list groups a whole
arm's seeds together instead of interleaving arms -- the same reason
`scripts/run_sweep.py` lays results out as `<results-root>/<method>/<seed>/`. Tokens are
lowercase `[a-z0-9-]`, since a name ends up in a URL and in offline directory names.

## Which fields are in the table, and why those

`RUN_NAME_FIELDS` holds the axes this project's committed studies actually vary, counted
from the run commands in `docs/experiment-logs/`: `--num-cycles` (97 mentions),
`--practice-reset-policy` (47), `--two-way-ledge` (21) and `--ask-for-help` (13), plus
`--env`/`--method`/`--seed`, plus `--unsplit-skills` -- which is in the table because
the collision check demanded it. Replayed over the 121 runs the project already held,
this table minus that one field produced 81 names, 40 of which grouped runs differing
*only* in `unsplit_skills`. That is the intended loop in miniature: curate, let the
check find the omission, add the field.

Deliberately **not** in the table: `--num-test-tasks` and `--max-steps-per-interaction`,
which appear often but as a study's fixed budget rather than an axis within it, and the
long tail of domain constants (`--throw-tolerance`, `--distance-coefficient`, ...). Two
runs differing only in one of those get the same name -- and the collision check fires,
names the field, and points here. **That is the intended workflow, not a hole**: the
table grows when an experiment actually varies something new, rather than pre-emptively
carrying every flag and making every name unreadable.

## Failing loudly beats defaulting quietly

A field this table names must be present in the resolved namespace, or `name` raises. It
never falls back to `getattr(args, dest, <default>)`, which is the specific trap here: a
renamed flag would silently drop an axis from *every* name, and the absent-attribute
default for `num_cycles` happens to equal the literal `SkillOracleCli` passes today, so
it would look correct right up until a method computed its own cycle count. Fields that
are legitimately absent -- a method's or a domain's own flags -- are declared
`optional=True` one by one; see `RunNameField`.
"""

import argparse
import re

from hitl_pmp.results_writer.types import RunNameField

# Ordered: outermost axis first, seed last. See the module docstring for why these
# fields and not others -- and note that adding one is cheap and is what the collision
# check will tell you to do.
RUN_NAME_FIELDS: tuple[RunNameField, ...] = (
    RunNameField(dest="env"),
    RunNameField(dest="method"),
    # Tossing Room's own flags: absent on every other domain, and `env` -- already in
    # the name -- is what says so. `unsplit_skills` is in this table because the
    # collision check put it there: replayed over the 121 runs the project already
    # held, it was the *only* field 40 name-groups differed on, which is exactly the
    # "you forgot an axis" report this design exists to produce.
    RunNameField(dest="two_way_ledge", toggle=("twoway", "oneway"), optional=True),
    RunNameField(dest="unsplit_skills", toggle=("unsplit", "split"), optional=True),
    # Global, so never absent. Rendered bare ("never"/"scheduled"): its values are
    # self-describing and this is the axis most sweeps are built around.
    RunNameField(dest="practice_reset_policy"),
    # A method flag, absent on --method skill-oracle. Prefixed, because "never" is also
    # a reset policy value and two bare "never" tokens in one name would read as a
    # duplicate rather than as two different axes.
    RunNameField(dest="ask_for_help", prefix="ask-", optional=True),
    # A method flag, absent on --method skill-oracle. The literal cycle count rather
    # than a multiple of the default: expressing it as "1x"/"10x" would mean carrying a
    # copy of `--num-cycles`'s default here, which is exactly the kind of duplicated
    # constant that goes silently wrong when the original moves.
    RunNameField(dest="num_cycles", prefix="c", optional=True),
    RunNameField(dest="seed", prefix="seed"),
)

_UNSAFE_CHARACTERS = re.compile(r"[^a-z0-9]+")


class RunNamer:
    """Builds a tracker run name from a run's resolved configuration.

    A static-method container, never instantiated, same as every other genuinely
    stateless business-logic class in this project: naming carries nothing between
    calls.

    Backend-agnostic by construction -- it imports `argparse` and nothing else, so the
    next concrete `ResultsWriter` reuses it rather than reinventing a second convention
    that drifts from this one."""

    @staticmethod
    def name(*, args: argparse.Namespace) -> str:
        """This run's name, from the **resolved** argparse namespace.

        The resolved namespace is the input on purpose. It is already what
        `open_if_requested` receives, what `ConfigSnapshot` records and what the writer
        logs as the run config, so naming reads the same object the run is otherwise
        described by. The alternative -- a hand-built typed model of "the fields that
        matter" -- needs its own translation step from the namespace, and that
        translation is precisely where an axis of variation goes missing without
        anybody noticing. One source, one table, one place to add a field."""
        tokens = [
            token
            for field in RUN_NAME_FIELDS
            if (token := RunNamer._token(field=field, args=args)) is not None
        ]
        return "-".join(tokens)

    @staticmethod
    def _token(*, field: RunNameField, args: argparse.Namespace) -> str | None:
        """One field's contribution, or None when it is legitimately absent."""
        if not hasattr(args, field.dest):
            if field.optional:
                return None
            raise ValueError(
                f"the run namer needs --{field.dest.replace('_', '-')} to build a run "
                f"name, and this run's resolved configuration has no '{field.dest}'. "
                "Either the namespace is not a resolved one, or the flag was renamed "
                "and every run name has silently lost an axis of variation -- fix "
                "RUN_NAME_FIELDS in src/hitl_pmp/results_writer/run_naming.py rather "
                "than defaulting the value, which would put two different experiments "
                "under one name."
            )
        value = getattr(args, field.dest)
        if field.toggle is not None:
            when_true, when_false = field.toggle
            return when_true if value else when_false
        return f"{field.prefix}{RunNamer._slug(value=value)}"

    @staticmethod
    def _slug(*, value: object) -> str:
        """Lowercase `[a-z0-9-]`, so a name is safe in a URL and as a directory name.

        `str(value)` rather than any per-type handling: this project's enum flags
        already render as their own values (`str(PracticeResetPolicy.NEVER)` is
        `'never'`), which is the same assumption `ConfigSnapshot.args` makes."""
        return _UNSAFE_CHARACTERS.sub("-", str(value).lower()).strip("-")
