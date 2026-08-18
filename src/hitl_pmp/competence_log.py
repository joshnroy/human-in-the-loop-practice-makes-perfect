"""Per-checkpoint skill-competence instrumentation: what each ground skill's competence model
actually believed at every evaluation checkpoint.

A sibling file rather than a field on `Metrics`, because `stats.json`'s byte-stability is how
this repo proves a change did not alter results. `EesMethod` prices every plan as
`sum(-log(competence))`, but `Metrics.practice_outcomes_per_cycle` records only raw
success/attempt counts -- recovering the posterior mean from those means re-deriving the
window/recency/prior arithmetic the model already computed. This writes that number directly.

One JSON object per line, flushed as written: a record readable only after a clean exit would
be unavailable exactly when it is most wanted.

| field | meaning |
| --- | --- |
| `checkpoint` | index into `Metrics.evaluations`, 0-based (0 is the sweep before any practice) |
| `num_online_transitions` | `Metrics.evaluations[checkpoint][0]` -- the learning-curve x-axis |
| `skill` | lifted skill name |
| `objects` | the ground skill's bound object names, in parameter order |
| `competence` | `get_current_competence()`'s posterior mean at this checkpoint |

Keyed by *ground* skill (`skill` + `objects`), not lifted name alone: `EesMethod` estimates
competence per grounding, so two groundings of one lifted skill can carry different competence
and would collide under a lifted-only key.

**A ground skill absent from a checkpoint** has simply never been attempted --
`competence_model()` creates one lazily and `current_competences()` does not invent one, so
checkpoint 0 contributes no rows for a fresh `EesMethod` at all. Rows carrying a default value
would be a different and wrong claim.

**Pure observer.** A run with recording on writes a byte-identical `stats.json` to one with it
off, asserted end-to-end through the real CLI in `tests/test_competence_log.py`."""

import argparse
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, ConfigDict, PrivateAttr

from hitl_pmp.core.method.types import GroundSkill

# The sibling `--output-dir` file this writes, named the same way `stats.json`,
# `timing.json`, `config_snapshot.json` and `sampler_draws.jsonl` are: after its
# content, not after the flag.
COMPETENCE_LOG_FILENAME = "competence_log.jsonl"


class CompetenceLogRecorder(BaseModel):
    """Appends one `CompetenceLogRecord` per (checkpoint, ground skill) pair.

    Constructed only when `--record-skill-competence` is passed, so every run that
    does not ask for it is byte-identical to one from before this landed -- including
    the absence of the file itself. Lives at the top level (alongside
    `sampler_draws.py`) rather than under `methods/practice_makes_perfect/`, because
    it is read from `method_runner.py`, which sits *below* `hitl_pmp.methods` in the
    import-direction contract and so cannot import anything from there.

    A real pydantic instance rather than a static-method container, because it
    carries genuine per-run state (the open file handle), which is this project's
    stated dividing line between the two."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_path: Path

    _handle: TextIO | None = PrivateAttr(default=None)

    @staticmethod
    def open_if_requested(*, args: argparse.Namespace) -> "CompetenceLogRecorder | None":
        """The recorder this run asked for, or None -- the one place the flag is
        interpreted, so no caller has to re-derive the path.

        Raises when the flag is passed without an `--output-dir` to write into,
        matching `SamplerDrawRecorder.open_if_requested`'s own reasoning: a
        multi-hour run that produced no instrumentation because of a missing second
        flag is the expensive way to find out. Unlike that recorder, though, this one
        is built inside `method_runner.py`'s `run`, which is called only *after* the
        domain's own `run_method` has already constructed env/tasks/problem/method --
        so on a domain whose construction is itself costly (Tossing3D's MuJoCo scene,
        or its first-run MimicLabs asset download) this raises after that cost has
        already been paid, not before it. `SamplerDrawRecorder` avoids that because
        `EesCli.run` builds it earlier, before `env_cli.run_method` is even called."""
        if not getattr(args, "record_skill_competence", False):
            return None
        output_dir = getattr(args, "output_dir", None)
        if output_dir is None:
            raise ValueError(
                "--record-skill-competence needs --output-dir: the per-checkpoint "
                f"record is written as {COMPETENCE_LOG_FILENAME} beside stats.json, "
                "and without an output directory there is nowhere to put it."
            )
        return CompetenceLogRecorder(output_path=Path(output_dir) / COMPETENCE_LOG_FILENAME)

    def record_checkpoint(
        self,
        *,
        checkpoint: int,
        num_online_transitions: int,
        competences: dict[GroundSkill, float],
    ) -> None:
        """Write one line per ground skill `competences` reports. A Method that
        tracks nothing this checkpoint (an empty dict -- either because it tracks no
        competence model at all, or because nothing has been practiced yet) opens no
        file and writes no lines, matching `SamplerDrawRecorder._open`'s own
        lazy-open contract: a run that never has anything to record leaves no stray
        empty file to misread as "recorded, and nothing happened"."""
        if not competences:
            return
        handle = self._open()
        for ground_skill, competence in competences.items():
            record = CompetenceLogRecord(
                checkpoint=checkpoint,
                num_online_transitions=num_online_transitions,
                skill=ground_skill.skill.name,
                objects=tuple(obj.name for obj in ground_skill.objects),
                competence=competence,
            )
            handle.write(record.model_dump_json() + "\n")
        # Per checkpoint, not per run: see the module docstring on why a file that
        # only parses after a clean exit is useless for a long run.
        handle.flush()

    def _open(self) -> TextIO:
        if self._handle is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.output_path.open("w", encoding="utf-8")
        return self._handle


class CompetenceLogRecord(BaseModel):
    """One ground skill's tracked competence at one evaluation checkpoint.

    Frozen: a record is a record of something already computed, so nothing should
    edit one after the fact. Field order is the order they are written in -- when
    (checkpoint, transitions), what (skill, objects), then the value itself."""

    model_config = ConfigDict(frozen=True)

    checkpoint: int
    num_online_transitions: int
    skill: str
    objects: tuple[str, ...]
    competence: float
