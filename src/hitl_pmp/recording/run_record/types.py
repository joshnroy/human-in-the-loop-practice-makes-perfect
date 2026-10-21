"""The run recording format, version 1.

One `recording.jsonl` per process that executes actions: a step-protocol run writes one
for practice in its output directory and one per evaluation checkpoint in
`evaluations/NNNN/`. Each line is one JSON object with a `kind`:

- `header` (first line): schema version, environment, method, run seed, the
  environment's own scene-construction arguments, the feature names of every
  object (so a state vector can be read without the environment that wrote it), and
  which objects are fixed geometry.
- `segment`: a stretch that starts from a known state -- practice, or one evaluation
  task -- with its checkpoint, task index and task/scene seeds. Followed by a `tick`
  carrying that starting state.
- `cycle`: a practice learning session begins; carries the sampled task's seed.
- `action`: one executed robot skill or human skill, in execution order. Followed by
  its `tick`s, each carrying the action's `seq`.
- `tick`: one full object-centric state, `{object: [feature, ...]}`.
- `segment_end`: the stretch's outcome.

Playback reads only `tick` states: the recording is sufficient to redraw a run with no
physics, so a replay cannot diverge from what was executed.
"""

from enum import Enum
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: Final = 1
RECORDING_FILENAME: Final = "recording.jsonl"

PlainState = dict[str, list[float]]


class RecordingHeader(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["header"] = "header"
    schema_version: int = SCHEMA_VERSION
    environment: str
    method: str | None = None
    run_seed: int | None = None
    # Whatever the environment needs to rebuild an equivalent scene for drawing.
    scene: dict[str, Any] = Field(default_factory=dict)
    feature_names: dict[str, list[str]] = Field(default_factory=dict)
    # Fixed geometry: present in every state, never moved, and not restored by playback.
    static_objects: list[str] = Field(default_factory=list)


class Phase(str, Enum):
    PRACTICE = "practice"
    EVALUATION = "evaluation"


class SegmentStart(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["segment"] = "segment"
    phase: Phase
    checkpoint: int | None = None
    task_index: int | None = None
    task_seed: int | None = None
    scene_seed: int | None = None


class CycleStart(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["cycle"] = "cycle"
    cycle: int
    task_seed: int | None = None
    start_step: int | None = None


class ActionEntry(BaseModel):
    """One executed skill.

    `success` is whether the skill ran to completion without a controller error -- not
    whether it achieved anything, which only the task outcome says. `outcome` carries
    the reason when it did not (interruption, planner refusal). `goal_reached` marks an
    evaluation skill the goal latch stopped mid-execution because the task was solved:
    it did not complete, so `success` stays False. `start_step`/`end_step`
    are the counted practice (or evaluation) steps around it. A human skill also lists
    `moved_objects`: the resulting state of every object whose state it changed."""

    model_config = ConfigDict(extra="ignore")

    kind: Literal["action"] = "action"
    seq: int
    phase: Phase
    checkpoint: int | None = None
    task_index: int | None = None
    cycle: int | None = None
    actor: Literal["robot", "human"]
    skill: str
    objects: list[str] = Field(default_factory=list)
    params: list[float] = Field(default_factory=list)
    controller: dict[str, Any] = Field(default_factory=dict)
    label: str | None = None
    start_step: int | None = None
    end_step: int | None = None
    success: bool | None = None
    goal_reached: bool = False
    outcome: str | None = None
    cost: float | None = None
    moved_objects: PlainState | None = None
    num_ticks: int = 0


class TickEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["tick"] = "tick"
    # The action this tick belongs to; None for a segment's starting state.
    seq: int | None = None
    state: PlainState
    diagnostics: dict[str, Any] | None = None


class SegmentEnd(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["segment_end"] = "segment_end"
    phase: Phase
    checkpoint: int | None = None
    task_index: int | None = None
    success: bool | None = None
    outcome: str | None = None
    steps: int | None = None


RecordEntry = Annotated[
    SegmentStart | CycleStart | ActionEntry | TickEntry | SegmentEnd,
    Field(discriminator="kind"),
]
