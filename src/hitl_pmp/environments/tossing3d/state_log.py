"""Logs enough of a Tossing3D episode to re-render it later, without a live replay.

`Tossing3DRenderer` draws the live simulator, not the `core.State` it is handed (see
that module's docstring) -- a flat `State` is a lossy projection (four of the robot's
twenty-two features, ten of the cube's sixteen; see `KinderBackend.snapshot`), so it
cannot put MuJoCo back into an arbitrary mid-episode configuration. `KinderBackend.
snapshot`/`.restore` already solve that for one in-memory rewind (see their own
docstrings); this module is what lets the same round-trip happen from a *file*,
after the process that ran the episode is long gone -- log a plain, serializable
projection of every `ObjectCentricState` snapshot `drain_substep_states` collects,
then rebuild each one against a *fresh* scene at replay time and call `render()`.

One line of JSON per tick, streamed (`write` + `flush` per event, matching
`VideoStream.append`'s one-frame-at-a-time discipline) rather than buffered -- an
episode's states fit easily in memory (measured ~280 bytes each), but nothing here
should assume that stays true if this format is ever pointed at something coarser.

`SkillEvent` lines and `TickEvent` lines interleave in the order they happened, so a
reader can attribute every tick to the ground skill that produced it without a
separate index. The header's `seed`/`canonical_seed` are what a replay rebuilds the
scene from -- see `analysis/render_tossing3d_state_log.py`, this format's one reader.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from .layout import Tossing3DLayout


class SkillEvent(BaseModel):
    """One ground skill dispatch: its name, the objects it was bound to, and its
    sampled continuous parameters (empty for a param_dim=0 skill like PickCube)."""

    kind: str = "skill"
    name: str
    objects: tuple[str, ...]
    params: tuple[float, ...]


class TickEvent(BaseModel):
    """One physics tick's full object-centric state, as `KinderBackend.
    snapshot_to_plain` returns it -- `{object_name: [feature, ...]}`, in the object's
    own live feature order."""

    kind: str = "tick"
    state: dict[str, tuple[float, ...]]


class StateLogHeader(BaseModel):
    """What a replay needs to rebuild an equivalent scene before restoring ticks into
    it -- the construction args `Tossing3DCli.build_problem` itself takes, not a
    task's own atoms (a fresh `Tossing3DTasks` reproduces the same train-task stream
    from `seed` alone, so the header does not need to carry a task's initial state)."""

    layout: Tossing3DLayout = Tossing3DLayout.BARRIER
    variant: str
    scene_bg: bool
    canonical_seed: int
    seed: int
    test_env_seed_offset: int


class StateLogWriter(BaseModel):
    """Streams a `StateLogHeader` then a sequence of `SkillEvent`/`TickEvent` lines to
    one file, one JSON object per line. Not a `LoopRecorder`: this logs raw state for
    replay, not composed video frames for direct playback -- see the module docstring
    for why the two are different problems."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_path: Path
    header: StateLogHeader
    _file: Any = None

    def model_post_init(self, __context: object) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", encoding="utf-8")
        self._write(obj={"kind": "header", **self.header.model_dump()})

    def record_skill(
        self, *, name: str, objects: tuple[str, ...], params: tuple[float, ...]
    ) -> None:
        self._write(obj=SkillEvent(name=name, objects=objects, params=params).model_dump())

    def record_tick(self, *, state: dict[str, list[float]]) -> None:
        self._write(obj=TickEvent(state={k: tuple(v) for k, v in state.items()}).model_dump())

    def _write(self, *, obj: dict[str, Any]) -> None:
        self._file.write(json.dumps(obj) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class StateLogReader:
    """Reads a `StateLogWriter`'s file back into a header plus an ordered event list.
    A static-method container's worth of logic, but kept as a class since `replay`
    needs the header parsed before it can build the scene events restore into."""

    def __init__(self, *, path: Path) -> None:
        self.header: StateLogHeader | None = None
        self.events: list[SkillEvent | TickEvent] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                self._parse_line(line=line)

    def _parse_line(self, *, line: str) -> None:
        obj = json.loads(line)
        kind = obj.pop("kind")
        if kind == "header":
            self.header = StateLogHeader(**obj)
        elif kind == "skill":
            self.events.append(SkillEvent(**obj))
        elif kind == "tick":
            self.events.append(TickEvent(**obj))
        else:
            raise ValueError(f"unknown state-log line kind {kind!r}")
