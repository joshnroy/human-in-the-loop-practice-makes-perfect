"""Fixed-seed native controller parameter-existence diagnostics.

Prepare physically picks and transports once. Search restores that full simulator
checkpoint between trials, outside skills. Final proofs replay from episode start.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.pouring3d.potsticker_controller import (
    PotstickerController,
    PotstickerParameters,
)
from hitl_pmp.environments.pouring3d.potsticker_native import make_potsticker_env
from scripts.run_sweep import SweepRun, SweepRunner

FIELDS = (
    "last_q",
    "base_target",
    "phase",
    "held",
    "steps",
    "grip",
    "hand_offset",
    "lost_contacts",
    "posture",
    "posture_goal",
    "home",
    "tray_home",
    "base_home",
    "return_speed_deg_s",
)
SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def json_default(x):  # noqa: PLR0917 -- json.dumps callback protocol
    return x.tolist() if isinstance(x, np.ndarray) else float(x)


def hashes():
    root = Path("src/hitl_pmp/environments/pouring3d")
    paths = [*root.rglob("*.py"), *root.glob("assets/potsticker*"), Path(__file__)]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def save_checkpoint(*, c, folder):
    state = np.empty(mujoco.mj_stateSize(c.model, SPEC))
    mujoco.mj_getState(c.model, c.data, state, SPEC)
    np.save(folder / "state.npy", state)
    d = {name: getattr(c, name) for name in FIELDS}
    d["rotation_offset"] = (
        None if c.rotation_offset is None else c.rotation_offset.as_quat().tolist()
    )
    (folder / "controller.json").write_text(json.dumps(d, default=json_default, indent=2))


def restore(*, c, folder):
    mujoco.mj_setState(c.model, c.data, np.load(folder / "state.npy"), SPEC)
    mujoco.mj_forward(c.model, c.data)
    d = json.loads((folder / "controller.json").read_text())
    for k in FIELDS:
        value = d[k]
        setattr(c, k, np.array(value) if isinstance(value, list) else value)
    c.rotation_offset = (
        None if d["rotation_offset"] is None else Rotation.from_quat(d["rotation_offset"])
    )
    c.events = []


def prepare(*, root, count, seed):
    source_hashes = hashes()
    root.mkdir(parents=True, exist_ok=True)
    env = make_potsticker_env(count=count, requested=1, seed=seed)
    c = PotstickerController.create(env=env)
    frames = []
    phases = []
    times = []

    def capture(**_):
        frames.append(c.data.qpos.copy())
        phases.append(c.phase)
        times.append(float(c.data.time))
        if len(frames) % 50 == 0:
            (root / "progress.json").write_text(
                json.dumps({"phase": c.phase, "frames": len(frames), "time": float(c.data.time)})
            )

    c.callback = capture
    start = time.monotonic()
    error = None
    try:
        capture()
        assert c.ready(), c.scene.cube_locations()
        c.diagnostic_deadline = start + 1800
        c.pick_pitcher()
        c.prepare_to_pour()
        if c.scene.cube_locations() != ["pitcher"] * count:
            raise ValueError("Payload lost during pickup/transport")
        save_checkpoint(c=c, folder=root)
    except (ValueError, AssertionError) as exc:
        error = str(exc)
    finally:
        np.savez_compressed(root / "motion.npz", qpos=frames, phases=phases, times=times)
        r = {
            "count": count,
            "seed": seed,
            "error": error,
            "held": c.held,
            "phase": c.phase,
            "snapshot": c.scene.physical_snapshot(),
            "seconds_wall": time.monotonic() - start,
            "source_sha256": source_hashes,
        }
        (root / "result.json").write_text(json.dumps(r, indent=2, default=json_default))
        print(
            json.dumps({k: r[k] for k in ("count", "seed", "error", "phase", "seconds_wall")}),
            flush=True,
        )
        env.close()


def evaluate(*, root, count, seed, checkpoint, candidates, replay=False, stop_on_success=False):
    source_hashes = hashes()
    root.mkdir(parents=True, exist_ok=True)
    env = make_potsticker_env(count=count, requested=1, seed=seed)
    c = PotstickerController.create(env=env)
    try:
        for i, candidate in enumerate(candidates):
            out = root / str(i)
            if (out / "result.json").exists():
                continue
            out.mkdir(exist_ok=True)
            frames = []
            phases = []
            times = []

            def capture(*, frames=frames, phases=phases, times=times, out=out, **_):
                frames.append(c.data.qpos.copy())
                phases.append(c.phase)
                times.append(float(c.data.time))
                if len(frames) % 50 == 0:
                    (out / "progress.json").write_text(
                        json.dumps({
                            "phase": c.phase,
                            "frames": len(frames),
                            "time": float(c.data.time),
                            "locations": c.scene.cube_locations(),
                        })
                    )

            c.callback = capture
            c.diagnostic_deadline = time.monotonic() + (4200 if count == 10 else 1200)
            if replay:
                c.pick_pitcher()
                c.prepare_to_pour()
            else:
                restore(c=c, folder=checkpoint)
            capture()
            c.mode = candidate["mode"]
            c.return_speed_deg_s = candidate.get("return_speed", 10.0)
            c.continuous_return = candidate.get("continuous_return", False)
            c.anticipation_seconds = candidate.get("anticipation_seconds", 0.0)
            c.bounded_ik_recoveries = 0
            start = time.monotonic()
            result = c.pour_prepared(
                parameters=PotstickerParameters(**candidate["params"]),
                requested=candidate["requested"],
            )
            if replay and result["error"] is None:
                try:
                    c.place_pitcher()
                    result["placement_success"] = c.held is None and c.vessel_home(name="pitcher")
                    result["after_place"] = c.scene.physical_snapshot()
                    result["full_cycle_success"] = (
                        result["success"]
                        and result["placement_success"]
                        and result["after_place"]["goal"]
                    )
                except ValueError as exc:
                    result["placement_success"] = False
                    result["placement_error"] = str(exc)
                    result["full_cycle_success"] = False
            result.update(
                count=count,
                seed=seed,
                seconds_wall=time.monotonic() - start,
                checkpoint=str(checkpoint),
                full_replay=replay,
                source_sha256=source_hashes,
            )
            (out / "result.json").write_text(json.dumps(result, indent=2, default=json_default))
            np.savez_compressed(out / "motion.npz", qpos=frames, phases=phases, times=times)
            save_checkpoint(c=c, folder=out)
            print(
                json.dumps({
                    k: result[k]
                    for k in (
                        "count",
                        "mode",
                        "requested",
                        "parameters",
                        "delivered",
                        "retained",
                        "success",
                        "error",
                        "seconds_wall",
                    )
                }),
                flush=True,
            )
            if replay or (stop_on_success and result["success"]):
                break
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--candidates", type=Path)
    p.add_argument("--replay", action="store_true")
    p.add_argument("--launch", type=Path)
    p.add_argument("--wait-pid", type=int)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--stop-on-success", action="store_true")
    args = p.parse_args()
    if args.wait_pid:
        while True:
            try:
                os.kill(args.wait_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(10)
    if args.launch:
        spec = json.loads(args.launch.read_text())
        runs = []
        for i, job in enumerate(spec):
            command = [
                sys.executable,
                "-m",
                "scripts.potsticker_feasibility",
                "--root",
                job["root"],
                "--count",
                str(job["count"]),
                "--seed",
                str(job.get("seed", 0)),
            ]
            if job.get("prepare"):
                command += ["--prepare"]
            else:
                command += ["--checkpoint", job["checkpoint"], "--candidates", job["candidates"]]
                if job.get("replay"):
                    command += ["--replay"]
                if job.get("stop_on_success"):
                    command += ["--stop-on-success"]
            runs.append(
                SweepRun(
                    method=f"job{i}",
                    seed=job.get("seed", 0),
                    output_dir=Path(job["root"]),
                    command=command,
                )
            )
        outcomes = SweepRunner.execute(runs=runs, max_workers=args.workers)
        if not all(o.succeeded for o in outcomes):
            raise SystemExit(1)
    elif args.prepare:
        prepare(root=args.root, count=args.count, seed=args.seed)
    else:
        evaluate(
            root=args.root,
            count=args.count,
            seed=args.seed,
            checkpoint=args.checkpoint,
            candidates=json.loads(args.candidates.read_text()),
            replay=args.replay,
            stop_on_success=args.stop_on_success,
        )


if __name__ == "__main__":
    main()
