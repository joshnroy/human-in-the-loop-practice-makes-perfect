"""`RESULTS_WRITERS`: the static list of concrete `ResultsWriter`s every run is offered.

This is the whole wiring. `method_runner.py` iterates this tuple, asks each entry
whether this run asked for it, and drives whichever say yes -- so **adding an observer
is a class plus one line here**, with no change to the harness, no new flag plumbing
through `MethodRunner.run`'s signature, and no risk of a new output being wired into
one code path and forgotten in another.

## A tuple, not a dict

`cli.py`'s `ENVIRONMENTS`/`METHODS` are dicts because `--env`/`--method` select exactly
one entry *by name*. Nothing selects a results writer by name: every writer is offered
every run and each decides for itself, from its own flag, in its own
`open_if_requested`. So the collection needs no keys, and a tuple says that -- an
ordered, immutable declaration rather than a lookup table.

Order is the order writers are opened, and the order their hooks fire within a
checkpoint. Nothing today depends on it, and nothing should: a writer that needed to
run after another would be a dependency between observers, which is a different design
than a list. `RunProgressWriter` is listed first only because that is the order the
harness fired these two in before it joined the list, which keeps a checkpoint's side
effects in the sequence they have always happened in.

## Not every entry is opt-in

`RunProgressWriter` is **always on**: it has no flag and writes whenever there is an
`--output-dir` to write into. That is a property of the writer, not an exception the
registry makes for it -- `open_if_requested` asks each entry whether this run wants it,
and "always, when I can write at all" is one of the answers.

## What is deliberately not in this list

`stats.json`, `config_snapshot.json`, `timing.json`, `sampler_draws.jsonl` and
`episode_traces.jsonl` are all *not* `ResultsWriter`s, and three of them structurally
cannot be. `results_writer.py`'s module docstring and this folder's README give the
per-file reasoning; the short version is that `stats.json` is the run's product rather
than an observation of it and must not be movable by editing a list, and the others
fire at boundaries the harness does not own.
"""

from hitl_pmp.results_writer.results_writer import ResultsWriter
from hitl_pmp.results_writer.run_progress import RunProgressWriter
from hitl_pmp.results_writer.wandb_writer import WandbResultsWriter

RESULTS_WRITERS: tuple[type[ResultsWriter], ...] = (RunProgressWriter, WandbResultsWriter)
