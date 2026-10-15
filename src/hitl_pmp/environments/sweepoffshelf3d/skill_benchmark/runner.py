"""Run every benchmark case in MuJoCo with KINDER's own planners and controllers, and
keep enough of each run to diagnose it without running it again.

Per case, under `<out>/runs/<case_id>/`:

- `trace.json`: the attempt's `SkillTrace` (every candidate and why it was rejected),
  the label and PyBullet verdict (`verdicts.py`), and every plan a controller made in
  KINDER's PyBullet scene (base path, arm approach and retract, subsampled) with the
  MuJoCo re-check's verdict on it;
- `steps.npz`: the scene's whole `qpos` and the gripper command at every control step,
  with the step's label (which routine was driving) and the KINDER controller's phase;
- `video.mp4` for every failure: the run in the lab2 scene, at most
  `MAX_CONCURRENT_ENCODES` encoded at once across all workers.

Workers are separate processes (`ShardWorker`), each with its own scene, taking every
K-th case; each appends to its own results file, so a crashed worker loses only its
unfinished cases and a re-run resumes where it stopped.
"""

import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.skill_trace import SkillTraces

from .states import StateCapture, StateSet
from .types import BenchmarkCase, BenchmarkState, CaseResult
from .verdicts import Verdicts

FRAME_EVERY = 6
MAX_FRAMES = 900
VIDEO_FPS = 20
MAX_CONCURRENT_ENCODES = 2
PLAN_POINTS = 40
CONTROLLER_FLAGS = (
    "_navigated",
    "_pre_grasp",
    "_closed_gripper",
    "_lifted",
    "_pre_place",
    "_open_gripper",
    "_returned",
)


class StepLog:
    """A `scripted_skills.Recorder`: every control step either skill driver takes."""

    def __init__(self, *, env: SweepOffShelf3DEnvironment, frames: bool) -> None:
        self.env = env
        self.frames_on = frames
        self.qpos: list[np.ndarray] = []
        self.grip: list[float] = []
        self.labels: list[str] = []
        self.phases: list[str] = []
        self.marks: list[tuple[int, str]] = []
        self.frames: list[np.ndarray] = []
        self.frame_every = FRAME_EVERY
        self.controller: Any = None
        self.nq = int(env.backend().model().nq)
        if frames:
            # The start state, so a failure that never moved still has something to watch.
            self.frames.append(env.backend().env().render())

    def on_step(self, *, skills: Any, label: str) -> None:
        data = self.env.backend().data()
        self.qpos.append(np.asarray(data.qpos, dtype=np.float32).copy())
        self.grip.append(float(skills.grip))
        self.labels.append(label)
        self.phases.append(self.phase(skills=skills))
        if self.frames_on and len(self.qpos) % self.frame_every == 0:
            self.frames.append(self.env.backend().env().render())
            if len(self.frames) > MAX_FRAMES:
                self.frames = self.frames[::2]
                self.frame_every *= 2

    def mark(self, *, skills: Any, label: str) -> None:
        self.marks.append((len(self.qpos), label))

    def phase(self, *, skills: Any) -> str:
        controller = self.controller
        if controller is None:
            return str(getattr(skills, "phase", ""))
        flags = [f.strip("_") for f in CONTROLLER_FLAGS if getattr(controller, f, False)]
        return f"{type(controller).__name__}:{'+'.join(flags) or 'start'}"

    def save(self, *, path: Path) -> None:
        names = sorted(set(self.labels))
        index = {name: i for i, name in enumerate(names)}
        phases = sorted(set(self.phases))
        phase_index = {name: i for i, name in enumerate(phases)}
        np.savez_compressed(
            path,
            qpos=np.asarray(self.qpos, dtype=np.float32).reshape(len(self.qpos), self.nq),
            grip=np.asarray(self.grip, dtype=np.float32),
            label=np.asarray([index[label] for label in self.labels], dtype=np.int16),
            label_names=np.asarray(names),
            phase=np.asarray([phase_index[p] for p in self.phases], dtype=np.int16),
            phase_names=np.asarray(phases),
            marks=np.asarray([f"{step}:{label}" for step, label in self.marks]),
        )


class PlanLog:
    """Wraps `KinderSkills.plan_blocked` on one live instance: every plan a KINDER
    controller made in PyBullet, and what the MuJoCo re-check said about it."""

    def __init__(self, *, kinder: Any, steps: StepLog) -> None:
        self.kinder = kinder
        self.steps = steps
        self.plans: list[dict[str, Any]] = []
        original = kinder.plan_blocked

        def recorded(*, controller: Any) -> str | None:
            steps.controller = controller
            verdict = original(controller=controller)
            self.plans.append({
                "note_index": len(kinder.notes),
                "controller": type(controller).__name__,
                "base": PlanLog.base(plan=getattr(controller, "_current_base_motion_plan", None)),
                "approach": PlanLog.arm(plan=getattr(controller, "_current_arm_joint_plan", None)),
                "retract": PlanLog.arm(plan=getattr(controller, "_current_retract_plan", None)),
                "mujoco_check": verdict or "clear",
            })
            return verdict

        kinder.plan_blocked = recorded

    @staticmethod
    def subsample(*, points: list[Any]) -> list[Any]:
        if len(points) <= PLAN_POINTS:
            return points
        keep = np.linspace(0, len(points) - 1, PLAN_POINTS).round().astype(int)
        return [points[i] for i in keep]

    @staticmethod
    def base(*, plan: Any) -> dict[str, Any] | None:
        if not plan:
            return None
        points = [
            [round(float(p.x), 3), round(float(p.y), 3), round(float(p.theta()), 3)] for p in plan
        ]
        return {"length": len(points), "points": PlanLog.subsample(points=points)}

    @staticmethod
    def arm(*, plan: Any) -> dict[str, Any] | None:
        if not plan:
            return None
        points = [[round(float(v), 3) for v in list(q)[:7]] for q in plan]
        return {"length": len(points), "points": PlanLog.subsample(points=points)}


class CaseRunner:
    """One case on one live environment."""

    @staticmethod
    def run(
        *,
        env: SweepOffShelf3DEnvironment,
        state: BenchmarkState,
        case: BenchmarkCase,
        out: Path,
        videos: bool = True,
    ) -> CaseResult:
        case_id = case.case_id(state_id=state.state_id)
        base = CaseResult(
            case_id=case_id,
            state_id=state.state_id,
            family=state.family,
            skill=case.skill,
            cube=case.cube,
            params=list(case.params),
            label="invalid",
        )
        StateCapture.restore(env=env, state=state)
        if not StateCapture.applicable(case=case, atoms=state.atoms):
            return base.model_copy(update={"error": "preconditions do not hold"})
        directory = out / "runs" / case_id
        directory.mkdir(parents=True, exist_ok=True)
        controllers = env.controllers()
        steps = StepLog(env=env, frames=videos)
        shelf, kinder = controllers._skills, controllers._kinder  # noqa: SLF001
        assert shelf is not None and kinder is not None
        shelf.recorder = steps
        plans = PlanLog(kinder=kinder, steps=steps)
        started = time.monotonic()
        try:
            env.take_action(action=StateCapture.action(case=case))
        finally:
            shelf.recorder = None
        wall = time.monotonic() - started
        trace = SkillTraces.last(env=env)
        assert trace is not None
        label, pybullet = Verdicts.label(trace=trace)
        (directory / "trace.json").write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "label": label,
                    "pybullet": pybullet,
                    "trace": trace.model_dump(mode="json"),
                    "plans": plans.plans,
                    "final_atoms": StateCapture.atoms(env=env),
                },
                indent=1,
            )
        )
        steps.save(path=directory / "steps.npz")
        video = None
        if label != "success" and videos and steps.frames:
            video = CaseRunner.encode(
                frames=steps.frames, path=directory / "video.mp4", lock_dir=out / "locks"
            )
        return base.model_copy(
            update={
                "label": label,
                "trace_outcome": trace.outcome,
                "error": trace.error,
                "pybullet": pybullet,
                "controller_steps": trace.controller_steps,
                "wall_seconds": round(wall, 2),
                "video": None if video is None else str(video.relative_to(out)),
            }
        )

    @staticmethod
    def encode(*, frames: list[np.ndarray], path: Path, lock_dir: Path) -> Path:
        """libx264, holding one of `MAX_CONCURRENT_ENCODES` slots across processes."""
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


class ShardWorker:
    """`python -m hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.runner <states>
    <out> <shard> <shards> [--no-videos]`: run every case whose index % shards == shard,
    skipping those already in this shard's results file."""

    @staticmethod
    def cases(*, states: list[BenchmarkState]) -> list[tuple[BenchmarkState, BenchmarkCase]]:
        return [(state, case) for state in states for case in state.cases]

    @staticmethod
    def results_path(*, out: Path, shard: int) -> Path:
        return out / f"results-{shard:02d}.jsonl"

    @staticmethod
    def done(*, out: Path) -> set[str]:
        ids = set()
        for path in out.glob("results-*.jsonl"):
            for line in path.read_text().splitlines():
                if line.strip():
                    ids.add(json.loads(line)["case_id"])
        return ids

    @staticmethod
    def main(*, argv: list[str]) -> int:
        states_path, out, shard, shards = Path(argv[0]), Path(argv[1]), int(argv[2]), int(argv[3])
        videos = "--no-videos" not in argv
        states = StateSet.load(path=states_path)
        done = ShardWorker.done(out=out)
        env = SweepOffShelf3DEnvironment(scene_bg=True)
        try:
            for index, (state, case) in enumerate(ShardWorker.cases(states=states)):
                case_id = case.case_id(state_id=state.state_id)
                if index % shards != shard or case_id in done:
                    continue
                try:
                    result = CaseRunner.run(env=env, state=state, case=case, out=out, videos=videos)
                except Exception as exc:  # noqa: BLE001  (one broken case must not end the shard)
                    result = CaseResult(
                        case_id=case_id,
                        state_id=state.state_id,
                        family=state.family,
                        skill=case.skill,
                        cube=case.cube,
                        params=list(case.params),
                        label="harness_error",
                        error=f"{type(exc).__name__}: {exc}"[:300],
                    )
                with ShardWorker.results_path(out=out, shard=shard).open("a") as stream:
                    stream.write(result.model_dump_json() + "\n")
                print(f"[{shard}] {case_id}: {result.label} ({result.wall_seconds}s)", flush=True)
        finally:
            env.close()
        return 0

    @staticmethod
    def launch(*, states_path: Path, out: Path, shards: int, videos: bool) -> list[int]:
        """Run `shards` workers to completion; their exit codes."""
        out.mkdir(parents=True, exist_ok=True)
        workers = []
        for shard in range(shards):
            log = (out / f"worker-{shard:02d}.log").open("a")
            command = [
                sys.executable,
                "-m",
                "hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.runner",
                str(states_path),
                str(out),
                str(shard),
                str(shards),
                *([] if videos else ["--no-videos"]),
            ]
            workers.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        return [worker.wait() for worker in workers]

    @staticmethod
    def results(*, out: Path) -> list[CaseResult]:
        return [
            CaseResult.model_validate_json(line)
            for path in sorted(out.glob("results-*.jsonl"))
            for line in path.read_text().splitlines()
            if line.strip()
        ]


if __name__ == "__main__":
    sys.exit(ShardWorker.main(argv=sys.argv[1:]))
