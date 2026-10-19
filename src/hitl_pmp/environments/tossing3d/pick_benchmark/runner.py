"""Run the pick from every benchmark state in MuJoCo, and keep enough of each run to
diagnose it without running it again.

The pick is pluggable (`PickImplementation`): the default is the current controller,
dispatched exactly as practice dispatches it (`Tossing3DEnvironment.take_action`), and
`--pick-impl module:Class` runs another implementation on the same states, so a new
pick controller is judged before/after on identical inputs.

Per case, under `<out>/runs/<case_id>/`: `run.json` (the dry-run verdict and its reason,
the dispatch's steps and error, the atoms before and after), `ticks.npz` (KINDER's
robot, cube and bin state at every control step, with upstream's holding evidence) and,
for every failure, `video.mp4` in the lab2 scene (at most `MAX_CONCURRENT_ENCODES` at
once across workers).
"""

import fcntl
import importlib
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend
from hitl_pmp.recording.run_record.sink import SkillTickSink

from .states import PickCapture, PickStateSet
from .types import PickResult, PickState
from .verdicts import PickVerdicts

VIDEO_FPS = 20
MAX_CONCURRENT_ENCODES = 2
FRAME_EVERY = 2
DEFAULT_IMPLEMENTATION = (
    "hitl_pmp.environments.tossing3d.pick_benchmark.runner:CurrentPickController"
)


class PickOutcome(BaseModel):
    steps: int
    terminated: bool
    error: str | None


class PickImplementation(Protocol):
    """A pick to benchmark: its own planning dry run, and one dispatch on the live env."""

    name: str

    def plan_failure(self, *, env: Tossing3DEnvironment) -> str | None: ...

    def pick(self, *, env: Tossing3DEnvironment) -> PickOutcome: ...


class CurrentPickController:
    """Upstream's `PickCubeController` as this domain dispatches it today."""

    name = "current PickCubeController (take_action skill 0)"

    def plan_failure(self, *, env: Tossing3DEnvironment) -> str | None:
        return env.backend().pick_cube_plan_failure()

    def pick(self, *, env: Tossing3DEnvironment) -> PickOutcome:
        env.take_action(action=np.array([float(env.pick_cube_id), 0, 0, 0, 0]))
        steps = int(sum(env.last_controller_steps()))
        error = env.last_skill_error()
        return PickOutcome(
            steps=steps,
            terminated=error is None and steps < KinderBackend.pick_step_limit,
            error=error,
        )


class Implementations:
    @staticmethod
    def load(*, spec: str) -> PickImplementation:
        module, _, attr = spec.partition(":")
        implementation: PickImplementation = getattr(importlib.import_module(module), attr)()
        return implementation


class TickLog(SkillTickSink):
    """Collects every control step the dispatched pick produced."""

    ticks: list[dict[str, list[float]]] = []
    holding: list[dict[str, Any]] = []

    def record_skill(
        self,
        *,
        name: str,
        objects: Sequence[str],
        params: Sequence[float],
        controller: Mapping[str, Any],
    ) -> None:
        del name, objects, params, controller

    def record_tick(
        self, *, state: Mapping[str, Sequence[float]], diagnostics: dict[str, Any] | None = None
    ) -> None:
        self.ticks.append({k: list(v) for k, v in state.items() if not k.startswith("collider:")})
        self.holding.append((diagnostics or {}).get("holding", {}))

    def save(self, *, path: Path) -> None:
        names = sorted(self.ticks[0]) if self.ticks else []
        arrays = {
            name: np.asarray([tick[name] for tick in self.ticks], dtype=np.float64)
            for name in names
        }
        np.savez_compressed(
            path,
            **arrays,
            holding=np.asarray([json.dumps(h, default=str) for h in self.holding]),
        )


class PickCaseRunner:
    @staticmethod
    def run(
        *,
        env: Tossing3DEnvironment,
        state: PickState,
        implementation: PickImplementation,
        out: Path,
        videos: bool = True,
    ) -> PickResult:
        case_id = f"{state.state_id}-{state.skill}"
        directory = out / "runs" / case_id
        directory.mkdir(parents=True, exist_ok=True)
        PickCapture.restore(env=env, state=state)
        started = time.monotonic()
        dry = implementation.plan_failure(env=env)
        dry_seconds = time.monotonic() - started
        first_frame = env.backend().render() if videos else None
        log = TickLog(ticks=[], holding=[])
        env.attach_state_log_writer(writer=log)
        env.set_substep_recording(enabled=videos)
        try:
            started = time.monotonic()
            outcome = implementation.pick(env=env)
            wall = time.monotonic() - started
            frames = env.drain_substep_frames() if videos else []
        finally:
            env.set_substep_recording(enabled=False)
            env._state_log_writer = None  # noqa: SLF001  (no public detach)
        after = PickCapture.atoms(env=env)
        holding = "Holding(robot,cube_0)" in after
        label = PickVerdicts.label(
            steps=outcome.steps, terminated=outcome.terminated, error=outcome.error, holding=holding
        )
        (directory / "run.json").write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "implementation": implementation.name,
                    "label": label,
                    "dry_run": {"plannable": dry is None, "reason": dry, "seconds": dry_seconds},
                    "dispatch": outcome.model_dump(),
                    "atoms_before": state.atoms,
                    "atoms_after": after,
                    "provenance": state.provenance,
                },
                indent=1,
            )
        )
        log.save(path=directory / "ticks.npz")
        video = None
        if label != "success" and videos:
            stills = [first_frame] if first_frame is not None else []
            video = PickCaseRunner.encode(
                frames=[*stills, *frames[::FRAME_EVERY]],
                path=directory / "video.mp4",
                lock_dir=out / "locks",
            )
        return PickResult(
            case_id=case_id,
            state_id=state.state_id,
            family=state.family,
            skill=state.skill,
            label=label,
            in_target=state.in_target,
            dry_run_plannable=dry is None,
            dry_run_reason=None if dry is None else dry[:200],
            pybullet="planned" if outcome.steps > 0 else "no_plan",
            controller_steps=outcome.steps,
            terminated=outcome.terminated,
            error=None if outcome.error is None else outcome.error[:300],
            wall_seconds=round(wall, 2),
            video=None if video is None else str(video.relative_to(out)),
        )

    @staticmethod
    def encode(*, frames: list[np.ndarray], path: Path, lock_dir: Path) -> Path | None:
        if not frames:
            return None
        import imageio

        lock_dir.mkdir(parents=True, exist_ok=True)
        while True:
            for slot in range(MAX_CONCURRENT_ENCODES):
                handle = (lock_dir / f"encode{slot}").open("w")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                try:
                    imageio.mimsave(
                        path,
                        [*frames, *([frames[-1]] * VIDEO_FPS)],
                        fps=VIDEO_FPS,
                        codec="libx264",
                        macro_block_size=8,
                    )
                    return path
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                    handle.close()
            time.sleep(1.0)


class PickShardWorker:
    """`python -m hitl_pmp.environments.tossing3d.pick_benchmark.runner <states> <out>
    <shard> <shards> <implementation> [--no-videos]`: every state whose index % shards
    == shard, skipping those already in a results file."""

    @staticmethod
    def done(*, out: Path) -> set[str]:
        return {r.case_id for r in PickShardWorker.results(out=out)}

    @staticmethod
    def results(*, out: Path) -> list[PickResult]:
        return [
            PickResult.model_validate_json(line)
            for path in sorted(out.glob("results-*.jsonl"))
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    @staticmethod
    def main(*, argv: list[str]) -> int:
        states_path, out = Path(argv[0]), Path(argv[1])
        shard, shards, spec = int(argv[2]), int(argv[3]), argv[4]
        videos = "--no-videos" not in argv
        implementation = Implementations.load(spec=spec)
        states = PickStateSet.load(path=states_path)
        done = PickShardWorker.done(out=out)
        env = Tossing3DEnvironment(scene_bg=True)
        try:
            for index, state in enumerate(states):
                case_id = f"{state.state_id}-{state.skill}"
                if index % shards != shard or case_id in done:
                    continue
                try:
                    result = PickCaseRunner.run(
                        env=env, state=state, implementation=implementation, out=out, videos=videos
                    )
                except Exception as exc:  # noqa: BLE001  (one broken case must not end the shard)
                    result = PickResult(
                        case_id=case_id,
                        state_id=state.state_id,
                        family=state.family,
                        skill=state.skill,
                        label="harness_error",
                        in_target=state.in_target,
                        error=f"{type(exc).__name__}: {exc}"[:300],
                    )
                with (out / f"results-{shard:02d}.jsonl").open("a") as stream:
                    stream.write(result.model_dump_json() + "\n")
                print(f"[{shard}] {case_id}: {result.label} ({result.wall_seconds}s)", flush=True)
        finally:
            env.close()
        return 0

    @staticmethod
    def launch(*, states_path: Path, out: Path, shards: int, spec: str, videos: bool) -> list[int]:
        out.mkdir(parents=True, exist_ok=True)
        workers = []
        for shard in range(shards):
            log = (out / f"worker-{shard:02d}.log").open("a")
            command = [
                sys.executable,
                "-m",
                "hitl_pmp.environments.tossing3d.pick_benchmark.runner",
                str(states_path),
                str(out),
                str(shard),
                str(shards),
                spec,
                *([] if videos else ["--no-videos"]),
            ]
            workers.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        return [worker.wait() for worker in workers]


if __name__ == "__main__":
    sys.exit(PickShardWorker.main(argv=sys.argv[1:]))
