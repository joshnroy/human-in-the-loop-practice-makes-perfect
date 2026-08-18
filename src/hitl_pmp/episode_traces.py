"""Per-step instrumentation: the full (state, action) trajectory of every evaluation episode,
not just its solved/unsolved outcome.

A sibling file rather than a field on `Metrics`, because `stats.json`'s byte-stability is
how this repo proves a change did not alter results. `TaskOutcome` records only whether a
test task was solved, never how many steps it took -- previously the only place a step count
could be recovered was a rendered `episode.mp4`'s frame count, which exists for one task per
checkpoint and only under `--output-dir`.

One JSON object per line, flushed once per *episode*: a record readable only after a clean
exit would be unavailable exactly when it is most wanted. Each record is one step of one
evaluation episode:

| field | meaning |
| --- | --- |
| `checkpoint` | eval sweep index, 0-based (0 is pre-practice) |
| `num_online_transitions` | `Metrics.evaluations[checkpoint][0]` -- the learning-curve x-axis |
| `task_index` | which fixed test task this is -- stable across checkpoints |
| `goal` | `Goal.describe()`'s rendering, same string `TaskOutcome.goal` carries |
| `step_index` | 0-based position of this action within the episode |
| `solved` | the whole episode's outcome -- same on every line, not a per-step check |
| `action_label` | `LabeledAction.label` -- which action/skill produced this step |
| `action` | the raw action vector, as a plain list of floats |
| `state` | the state *after* this action, flattened `"<object>.<feature>": float` |

`state` is deliberately the *whole* feature vector: which features matter is a per-domain
question this recorder does not try to answer.

**Never threaded into `Problem.run_task_episode`.** It reads back the `EpisodeTrace` that
every domain's `run_task_episode` already returns unconditionally, because a recorder carries
real per-run state (an open file handle) and a stateful recorder never crosses into `core` --
only plain data does. So it lives in `practice_loop.py`'s layer, alongside `sampler_draws.py`.

**Pure observer.** A run with recording on writes a byte-identical `stats.json` to one with
it off, asserted end-to-end through the real CLI in `tests/test_episode_traces.py`."""

import argparse
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, ConfigDict, PrivateAttr

from hitl_pmp.core.method.types import EpisodeTrace
from hitl_pmp.core.problem.environment.types import State

# The sibling `--output-dir` file this writes, named the same way `stats.json`,
# `timing.json`, `config_snapshot.json`, `sampler_draws.jsonl` and
# `competence_log.jsonl` are: after its content, not after the flag.
EPISODE_TRACES_FILENAME = "episode_traces.jsonl"


class EpisodeTraceRecorder(BaseModel):
    """Appends one `EpisodeStep` per step of every evaluation episode.

    Constructed only when `--record-episode-traces` is passed, so every run that
    does not ask for it is byte-identical to one from before this landed --
    including the absence of the file itself.

    A real pydantic instance rather than a static-method container, because it
    carries genuine per-run state (the open file handle), which is this project's
    stated dividing line between the two."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_path: Path

    _handle: TextIO | None = PrivateAttr(default=None)

    @staticmethod
    def open_if_requested(*, args: argparse.Namespace) -> "EpisodeTraceRecorder | None":
        """The recorder this run asked for, or None -- the one place the flag is
        interpreted, so no caller has to re-derive the path.

        Raises up front, before the run mutates anything, when the flag is passed
        without an `--output-dir` to write into -- matching
        `SamplerDrawRecorder.open_if_requested`'s own reasoning: a multi-checkpoint
        sweep that produced no instrumentation because of a missing second flag is
        the expensive way to find out."""
        if not getattr(args, "record_episode_traces", False):
            return None
        output_dir = getattr(args, "output_dir", None)
        if output_dir is None:
            raise ValueError(
                "--record-episode-traces needs --output-dir: the per-step record is "
                f"written as {EPISODE_TRACES_FILENAME} beside stats.json, and without "
                "an output directory there is nowhere to put it."
            )
        return EpisodeTraceRecorder(output_path=Path(output_dir) / EPISODE_TRACES_FILENAME)

    def record_episode(
        self,
        *,
        checkpoint: int,
        num_online_transitions: int,
        task_index: int,
        goal: str,
        solved: bool,
        trace: EpisodeTrace,
    ) -> None:
        """Write every step of one already-run evaluation episode.

        `trace.states[step_index + 1]` is the state *after* `trace.actions[step_index]`
        -- `trace.states[0]` is the episode's initial state, which carries no action of
        its own and so is not written as its own record (matching how `TaskOutcome`
        itself never records a "step 0, nothing happened yet" row)."""
        if not trace.actions:
            return
        handle = self._open()
        for step_index, action in enumerate(trace.actions):
            record = EpisodeStep(
                checkpoint=checkpoint,
                num_online_transitions=num_online_transitions,
                task_index=task_index,
                goal=goal,
                step_index=step_index,
                solved=solved,
                action_label=action.label,
                action=[float(value) for value in action.action],
                state=EpisodeTraceRecorder._flatten_state(state=trace.states[step_index + 1]),
            )
            handle.write(record.model_dump_json() + "\n")
        # Per episode, not per run: see the module docstring on why a file that only
        # parses after a clean exit is still useful at this granularity.
        handle.flush()

    @staticmethod
    def _flatten_state(*, state: State) -> dict[str, float]:
        """Every feature of every object in the state, keyed `"<object>.<feature>"` --
        the same flattening `SamplerDrawRecorder.read_features` uses, but over the
        whole state rather than one ground skill's bound objects, since a full
        trajectory has no fixed set of "the objects that matter"."""
        return {
            f"{obj.name}.{feature_name}": float(state.get(obj=obj, feature_name=feature_name))
            for obj in state.data
            for feature_name in obj.type.feature_names
        }

    def _open(self) -> TextIO:
        """Opened on first write rather than at construction, so a run whose
        episodes are all zero-step (already-satisfied tasks) leaves no stray empty
        file to misread as "recorded, and nothing happened"."""
        if self._handle is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.output_path.open("w", encoding="utf-8")
        return self._handle


class EpisodeStep(BaseModel):
    """One step of one evaluation episode.

    Frozen: a step is a record of something that already happened, so nothing should
    edit one after the fact. Field order is the order they are written in -- when
    (checkpoint, transitions), which episode (task, goal), which step, how it ended
    (solved), what was done (action), and finally the resulting state."""

    model_config = ConfigDict(frozen=True)

    checkpoint: int
    num_online_transitions: int
    task_index: int
    goal: str
    step_index: int
    solved: bool
    action_label: str
    action: list[float]
    state: dict[str, float]
