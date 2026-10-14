"""Tossing3D's implementation of the run recording, and its replay renderer.

A recorded state is `KinderBackend.snapshot_to_plain` of the live KINDER
`ObjectCentricState` -- every object's full feature vector, which is what
`restore_plain_snapshot` needs to put a fresh scene back into it. This is the
information the Tossing3D state log carried (`state_log.py`): skill name, bound
objects, logged parameters, the controller-chosen toss direction, and per-tick states
with their abstraction diagnostics. `LegacyTossing3DStateLog` loads those older logs
into the same entries.
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import PrivateAttr

from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.recording.replay.renderer import ReplayRenderer
from hitl_pmp.recording.run_record.types import (
    ActionEntry,
    Phase,
    PlainState,
    RecordEntry,
    RecordingHeader,
    TickEntry,
)
from hitl_pmp.recording.run_record.writer import RunRecordWriter

from .cli import Tossing3DCli
from .environment import Tossing3DEnvironment
from .layout import Tossing3DLayout
from .problem import Tossing3DProblem
from .state_log import SkillEvent, StateLogHeader, StateLogReader, TickEvent


class Tossing3DRecording:
    environment = "tossing3d"

    @staticmethod
    def writer(
        *,
        env: Tossing3DEnvironment,
        args: argparse.Namespace,
        method: str | None,
        output_path: Path,
    ) -> RunRecordWriter:
        scene = StateLogHeader(
            layout=env.layout,
            variant=args.variant,
            scene_bg=args.scene_bg,
            canonical_seed=args.canonical_seed,
            seed=args.seed,
            test_env_seed_offset=args.test_env_seed_offset,
        )
        return RunRecordWriter(
            output_path=output_path,
            header=RecordingHeader(
                environment=Tossing3DRecording.environment,
                method=method,
                run_seed=args.seed,
                scene=scene.model_dump(mode="json"),
            ),
            state_provider=lambda: Tossing3DRecording.state(env=env),
            scene_description_provider=lambda: Tossing3DRecording.describe(env=env),
            task_seed_provider=lambda task: Tossing3DRecording.task_seed(env=env, task=task),
        )

    @staticmethod
    def state(*, env: Tossing3DEnvironment) -> PlainState:
        """The live state at MuJoCo's precision, from the same source as every tick the
        environment drains -- so a skill's last tick and the state it leaves are equal,
        and a segment's starting state or a human's result is recorded as precisely."""
        backend = env.backend()
        return backend.snapshot_to_plain(snapshot=backend.live_state())

    @staticmethod
    def describe(*, env: Tossing3DEnvironment) -> dict[str, Any]:
        """Feature names, and the fixed collision boxes as static objects: they belong to
        the compiled scene, never move, and playback does not restore them."""
        snapshot = env.backend().snapshot()
        return {
            "feature_names": {obj.name: list(snapshot.type_features[obj.type]) for obj in snapshot},
            "static_objects": sorted(
                obj.name for obj in snapshot if obj.name.startswith("collider:")
            ),
        }

    @staticmethod
    def task_seed(*, env: Tossing3DEnvironment, task: Task) -> int:
        return int(task.initial_state.get(obj=env.scene, feature_name="seed"))


class Tossing3DReplayRenderer(ReplayRenderer):
    """Rebuilds a scene from a recording header's `scene` and draws recorded states
    into it by restoring them -- no physics step is ever taken.

    `scene_bg` overrides the recorded value: presentation videos use the lab2 room
    whatever the run used."""

    scene: dict[str, Any]
    scene_bg: bool | None = True
    follow_robot: bool = True

    _problem: Tossing3DProblem | None = PrivateAttr(default=None)

    def model_post_init(self, __context: object) -> None:
        values = dict(self.scene)
        values["layout"] = Tossing3DLayout(values.get("layout", Tossing3DLayout.BARRIER))
        if self.scene_bg is not None:
            values["scene_bg"] = self.scene_bg
        problem = Tossing3DCli.build_problem(args=argparse.Namespace(**values))
        problem.hard_reset()
        problem.reset_to_task(task=problem.sample_train_task())
        self._problem = problem

    def fps(self) -> int:
        return self._env().backend().render_fps()

    def show(self, *, state: PlainState) -> np.ndarray:
        env = self._env()
        env.restore_plain_snapshot(plain=state, dtype=np.float64)
        return env.backend().render(follow_robot=self.follow_robot)

    def current_state(self) -> PlainState:
        """Read back from MuJoCo itself, not from the backend's cached copy of what was
        restored -- otherwise a round-trip check could not fail."""
        backend = self._env().backend()
        return backend.snapshot_to_plain(snapshot=backend.live_state())

    def close(self) -> None:
        if self._problem is not None:
            self._problem.env.close()

    def _env(self) -> Tossing3DEnvironment:
        assert self._problem is not None
        return self._problem.env


class LegacyTossing3DStateLog:
    """Loads a `tossing3d_state_log.jsonl` (the pre-recording format) as recording
    entries. Those logs carry no outcomes, counted steps or phases: every skill is a
    practice robot action of unknown outcome, and an evaluation interleaved into the
    same file is not distinguishable from practice."""

    @staticmethod
    def load(*, path: Path) -> tuple[RecordingHeader, Iterator[RecordEntry]]:
        with path.open(encoding="utf-8") as stream:
            first = json.loads(stream.readline())
        if first.get("kind") != "header":
            raise ValueError(f"{path} has no state-log header line")
        header = StateLogHeader.model_validate(first)
        return (
            RecordingHeader(
                environment=Tossing3DRecording.environment,
                run_seed=header.seed,
                scene=header.model_dump(mode="json"),
            ),
            LegacyTossing3DStateLog._entries(path=path),
        )

    @staticmethod
    def _entries(*, path: Path) -> Iterator[RecordEntry]:
        reader = StateLogReader(path=path)
        seq = -1
        action: ActionEntry | None = None
        ticks: list[TickEntry] = []
        for event in reader.events:
            if isinstance(event, SkillEvent):
                if action is not None:
                    yield action.model_copy(update={"num_ticks": len(ticks)})
                    yield from ticks
                elif ticks:
                    yield from ticks
                seq += 1
                action = ActionEntry(
                    seq=seq,
                    phase=Phase.PRACTICE,
                    actor="robot",
                    skill=event.name,
                    objects=list(event.objects),
                    params=list(event.params),
                    controller=(
                        {}
                        if event.toss_direction_deg is None
                        else {"toss_direction_deg": event.toss_direction_deg}
                    ),
                )
                ticks = []
            elif isinstance(event, TickEvent):
                ticks.append(
                    TickEntry(
                        seq=None if action is None else seq,
                        state={k: list(v) for k, v in event.state.items()},
                        diagnostics=event.abstraction_diagnostics,
                    )
                )
        if action is not None:
            yield action.model_copy(update={"num_ticks": len(ticks)})
        yield from ticks
