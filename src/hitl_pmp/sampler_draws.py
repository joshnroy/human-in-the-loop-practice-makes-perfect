"""Per-draw sampler instrumentation: what parameter the learned sampler actually
chose on every practice execution, and what the environment did with it.

## Why this is a sibling file and not a field on `Metrics`

`stats.json` is the serialized `Metrics`, and its **byte-stability is load-bearing**:
it is how this repo proves a change did not alter results. PR #146 used exactly that
property to show the Tossing Room rename was behaviour-neutral, and
`tests/scripts/test_reproducibility.py` rests on it for three domains. Anything
appended to `Metrics` that varies per run -- and a list of float parameters certainly
does -- breaks that check for every open PR at once.

`timing.json` and `config_snapshot.json` are separate files beside `stats.json` for
this same reason, and this is the third. The rule is worth stating once: **`stats.json`
holds only what must be byte-comparable; everything else observed about a run goes in a
sibling.**

## What it answers that integer counters cannot

`SkillPracticeTally` is eight counters. It can say a skill's informed draws succeeded
117/206 while its uniform draws succeeded 48/275 -- the central result of PR #133 --
but it cannot say *which standoff* those informed draws converged on, because the
parameter itself is never retained. "The sampler converges onto the standoff that
works" was therefore an inference from success rates, with no parameter trajectory
behind it. One line per draw is what makes that plottable.

## The format, and why JSONL

One JSON object per line, flushed as it is written. A 100-cycle Tossing3D run is
hours, so a record readable only after a clean exit would be unavailable exactly when
it is wanted -- during the run, and after a crash. Every flushed line stands alone, so
a killed run leaves a file that parses up to its last complete draw.

Each record is:

| field | meaning |
| --- | --- |
| `cycle` | learning cycle the draw happened in, 0-based |
| `skill` | lifted skill name |
| `consultation` | the `SamplerConsultation` pool -- `informed`, `epsilon_random`, `uninformative` |
| `success` | did the ground skill's add effects hold afterwards |
| `params` | the chosen continuous parameters, as drawn |
| `achieved` | post-action features of the ground skill's own objects, `"<object>.<feature>"` |

`consultation` is never `no_sampler`: a `param_dim == 0` skill never reaches a sampler,
so it produces no draw at all. That makes the file's row count equal to
`num_attempts - num_unparameterized_attempts` summed over `stats.json`'s per-cycle
tallies, which `tests/test_sampler_draws.py` asserts -- the two views of the same
events are kept honest against each other rather than allowed to drift.

`cycle` rather than online transitions, because a `Method` does not know the harness's
transition count. Join to `stats.json`'s `evaluations` (one entry per sweep, carrying
`num_online_transitions`) to put draws on the same x-axis as a learning curve;
`analysis/` owns that join.

## Pure observer

Nothing here draws randomness, and no call into it returns a value any caller branches
on -- the same contract `recording.LoopRecorder` holds for `--record-full-loop`. A run
with recording on takes exactly the actions it would have taken with it off and writes
a byte-identical `stats.json`, which is asserted end-to-end through the real CLI rather
than argued from inspection.
"""

import argparse
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, ConfigDict, PrivateAttr

from hitl_pmp.core.method.types import SamplerConsultation
from hitl_pmp.core.problem.environment.types import Object, State

# The sibling `--output-dir` file this writes, named the same way `stats.json`,
# `timing.json` and `config_snapshot.json` are: after its content, not after the flag.
SAMPLER_DRAWS_FILENAME = "sampler_draws.jsonl"


class SamplerDrawRecorder(BaseModel):
    """Appends one `SamplerDraw` per sampler-backed practice execution.

    Constructed only when `--record-sampler-draws` is passed, so every run that does
    not ask for it is byte-identical to one from before this landed -- including the
    absence of the file itself.

    A real pydantic instance rather than a static-method container, because it carries
    genuine per-run state (the open file handle and the cycle counter), which is this
    project's stated dividing line between the two.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_path: Path

    _handle: TextIO | None = PrivateAttr(default=None)
    _cycle: int = PrivateAttr(default=0)

    @staticmethod
    def open_if_requested(*, args: argparse.Namespace) -> "SamplerDrawRecorder | None":
        """The recorder this run asked for, or None -- the one place the flag is
        interpreted, so no caller has to re-derive the path.

        Raises up front, before the run mutates anything, when the flag is passed
        without an `--output-dir` to write into. That mirrors how `--record-full-loop`
        rejects a domain with no renderer rather than running to completion and
        silently writing nothing: a multi-hour run that produced no instrumentation
        because of a missing second flag is the expensive way to find out.
        """
        if not getattr(args, "record_sampler_draws", False):
            return None
        output_dir = getattr(args, "output_dir", None)
        if output_dir is None:
            raise ValueError(
                "--record-sampler-draws needs --output-dir: the per-draw record is "
                f"written as {SAMPLER_DRAWS_FILENAME} beside stats.json, and without "
                "an output directory there is nowhere to put it."
            )
        return SamplerDrawRecorder(output_path=Path(output_dir) / SAMPLER_DRAWS_FILENAME)

    def end_cycle(self) -> None:
        """Advance the cycle stamp. Called from `Method.end_cycle`, which
        `practice_loop.py` fires once per cycle before that cycle's evaluation sweep,
        so a draw's `cycle` is the cycle it was actually made in."""
        self._cycle += 1

    def record(
        self,
        *,
        skill_name: str,
        consultation: SamplerConsultation,
        success: bool,
        params: list[float],
        state: State,
        objects: tuple[Object, ...],
    ) -> None:
        """Write one draw. `state` is the state *after* the action, so `achieved`
        reports what the environment actually did rather than what was asked for --
        on Tossing3D that is the difference between the commanded standoff and the
        base pose the controller reached."""
        draw = SamplerDraw(
            cycle=self._cycle,
            skill=skill_name,
            consultation=consultation.value,
            success=success,
            params=params,
            achieved=SamplerDrawRecorder.read_features(state=state, objects=objects),
        )
        handle = self._open()
        handle.write(draw.model_dump_json() + "\n")
        # Per line, not per run: see the module docstring on why a file that only
        # parses after a clean exit is useless for a multi-hour run.
        handle.flush()

    @staticmethod
    def read_features(*, state: State, objects: tuple[Object, ...]) -> dict[str, float]:
        """Every feature of every object the ground skill binds, keyed
        `"<object>.<feature>"`.

        Deliberately the whole feature vector rather than a curated subset: which
        features matter is a per-domain question, and a recorder that knew the answer
        would need a branch per domain. On Tossing3D's `MoveToThrowPose(robot, cube,
        bin, goal_region)` this yields the base pose *and* the bin position, which is
        exactly what an achieved standoff is computed from.
        """
        return {
            f"{obj.name}.{feature_name}": float(state.get(obj=obj, feature_name=feature_name))
            for obj in objects
            for feature_name in obj.type.feature_names
        }

    def _open(self) -> TextIO:
        """Opened on first write rather than at construction, so a run that never
        consults a sampler leaves no stray empty file to misread as "recorded, and
        nothing happened"."""
        if self._handle is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.output_path.open("w", encoding="utf-8")
        return self._handle


class SamplerDraw(BaseModel):
    """One sampler-backed practice execution, start to finish.

    Frozen: a draw is a record of something that already happened, so nothing should
    edit one after the fact. Field order is the order they are written in, which is
    the order they read best in a `head` of the file -- when, what, how chosen, how it
    went, and finally the two vectors.
    """

    model_config = ConfigDict(frozen=True)

    cycle: int
    skill: str
    consultation: str
    success: bool
    params: list[float]
    achieved: dict[str, float]
