"""Record one Tossing3D throw per `(release_speed, gripper_release_ms)` cell, densely enough
to *see* it.

Nine cells of the `20 x 20 x 5` grid `scripts/tossing3d_toss_parameter_grid.py` measures: a
3x3 spanning both dials at their axis endpoints and midpoints, standoff at
`ORACLE_THROW_STANDOFF`, one fixed seed. 3x3 rather than 5x1 because a family that varies
only the speed cannot show that the second dial moves the throw. Whether a throw scores is
not the criterion, and no goal box is drawn.

Rendered from inside `sim.step` every `RENDER_EVERY_N_TICKS` ticks -- 200 frames per
simulated second, at 1.7 ms per render. `KinderBackend.set_substep_recording` instead gives
one frame per `env.step()`, measured at 100 MuJoCo ticks x 0.0005 s = 0.05 s, so a 0.4-0.9 s
flight gets 8-18 frames and reads as a teleport. Position and contact state are recorded on
every tick regardless, which is what the ballistic fit reads.

Distance is the ballistic ground-crossing, not first contact: a cube whose crossing lies
past the bin's far wall still hits that wall, so first contact is a function of the
independent variable. Resting x is recorded too, since the two differ.

Frames go out as one `clip_<speed>_<ms>.mp4` per cell plus a `tosses.json`;
`analysis/tossing3d_release_speed_video.py` composes them.

Wrap in a memory-capped scope, and use `scripts/with_kinder_env.sh` so `kinder_models`
resolves to this worktree rather than the main checkout:

    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        scripts/with_kinder_env.sh python scripts/tossing3d_release_speed_clips.py \
        --output-dir /tmp/toss-clips --seed 0 --max-workers 4
"""

import argparse
import inspect
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Both KINDER packages are installed editable against the *main checkout's* absolute paths,
# so without this a worktree imports whatever commit that checkout is sitting on.
for _path in (
    _REPO_ROOT / "src",
    _REPO_ROOT / "reference" / "kindergarden" / "src",
    _REPO_ROOT / "reference" / "kinder-baselines" / "kinder-models" / "src",
    _REPO_ROOT,
):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from hitl_pmp.environments.tossing3d.predicates import (  # noqa: E402
    TOSS_RELEASE_MS_BOUNDS,
    TOSS_SPEED_BOUNDS,
)
from hitl_pmp.environments.tossing3d.skill_oracle_policy import (  # noqa: E402
    ORACLE_PICK_DISTANCE,
    ORACLE_PICK_ROTATION,
    ORACLE_THROW_STANDOFF,
)
from scripts.tossing3d_release_angle_probe import (  # noqa: E402
    assert_kinder_pins,
    longest_contact_free_window,
)

# Half the 0.05 m cube. #227's distance grid solves for this same height.
CUBE_RESTING_HALF_HEIGHT = 0.025

# Fallback for the pure fit when a caller supplies no model (i.e. in tests); the probe reads
# gravity off the live model.
NOMINAL_GRAVITY = 9.81

# One rendered frame per this many MuJoCo ticks: 200 fps at the scene's 0.0005 s timestep.
RENDER_EVERY_N_TICKS = 10

# `env.step` calls to hold after the swing, so the cube is at rest before the clip ends.
# #227's grids used 30; kept identical so `cube_x_final` means the same thing.
DEFAULT_SETTLE_STEPS = 30

DEFAULT_SEED = 0

# Imported rather than retyped: #239 corrected a retyped `240 deg/s` ceiling that was never
# a command range.
DEFAULT_SPEED_START, DEFAULT_SPEED_STOP = TOSS_SPEED_BOUNDS
DEFAULT_RELEASE_MS_START, DEFAULT_RELEASE_MS_STOP = TOSS_RELEASE_MS_BOUNDS

# Nine arcs is about the most a viewer reads as a family at a glance. Evenly spaced, so
# these do *not* land on the committed grid's axis points -- pass `--speeds`/`--release-ms`
# for that.
DEFAULT_SPEED_COUNT = 3
DEFAULT_RELEASE_MS_COUNT = 3


class GroundCrossing(BaseModel):
    """Where a free-flying cube's parabola meets a resting cube's centre height.

    The velocities are at release, not at the crossing."""

    x: float
    t: float
    launch_vx: float
    launch_vz: float
    residual_m: float


class TossClip(BaseModel):
    """One commanded speed: what the throw did, and where its frames were written."""

    seed: int
    commanded_speed_deg: float
    commanded_release_ms: float
    standoff: float

    clip_filename: str
    frame_hz: float
    physics_timestep: float

    # Per rendered frame, so the arc can be drawn in lockstep with the footage.
    frame_times: list[float] = []
    frame_cube_x: list[float] = []
    frame_cube_z: list[float] = []
    frame_contact: list[bool] = []

    release_t: float | None = None
    land_t: float | None = None
    free_flight_seconds: float | None = None
    free_flight_ticks: int | None = None
    free_flight_frames: int | None = None

    ballistic_impact_x: float | None = None
    ballistic_impact_t: float | None = None
    ballistic_fit_residual_m: float | None = None
    launch_elevation_deg: float | None = None
    launch_speed_mps: float | None = None

    base_x_before_toss: float | None = None
    cube_x_final: float | None = None
    cube_z_final: float | None = None
    solved: bool | None = None

    bin_x: float | None = None
    bin_z: float | None = None

    pick_error: str | None = None
    move_error: str | None = None
    toss_error: str | None = None

    # Provenance: the tree that actually ran, not the tree that was read.
    kinder_models_file: str | None = None
    kindergarden_file: str | None = None


def evenly_spaced_speeds(*, start: float, stop: float, count: int) -> list[float]:
    """`count` speeds from `start` to `stop` inclusive, both endpoints hit exactly."""
    if count < 2:
        raise ValueError(f"a speed sweep needs at least 2 speeds to span a range, got {count}")
    step = (stop - start) / (count - 1)
    return [float(start + i * step) for i in range(count)]


def ballistic_ground_crossing(
    *, times: np.ndarray, xs: np.ndarray, zs: np.ndarray, gravity: float = NOMINAL_GRAVITY
) -> GroundCrossing:
    """Fit the free-flight parabola and solve it for a resting cube's centre height.

    `times`/`xs`/`zs` must already be restricted to the contact-free window. The
    **descending** root is taken: both are real and positive when the cube is released below
    the resting height, and the ascending one is it first passing 0.025 m going up.
    """
    if len(times) < 3:
        raise ValueError(f"a parabola needs at least 3 samples to fit, got {len(times)}")
    rel = np.asarray(times, dtype=float) - float(times[0])
    z_coeffs = np.polyfit(rel, np.asarray(zs, dtype=float), 2)
    x_coeffs = np.polyfit(rel, np.asarray(xs, dtype=float), 1)
    residual = float(np.max(np.abs(np.polyval(z_coeffs, rel) - np.asarray(zs, dtype=float))))

    # z(t) = a t^2 + b t + c, solved for z = h. `a` is -g/2, fitted rather than assumed.
    a, b, c = (float(v) for v in z_coeffs)
    discriminant = b * b - 4.0 * a * (c - CUBE_RESTING_HALF_HEIGHT)
    if discriminant < 0.0 or a == 0.0:
        raise ValueError(
            f"this flight never reaches z={CUBE_RESTING_HALF_HEIGHT} "
            f"(fit z(t) = {a:.4f} t^2 + {b:.4f} t + {c:.4f})"
        )
    roots = sorted(
        ((-b + sign * float(np.sqrt(discriminant))) / (2.0 * a) for sign in (1.0, -1.0)),
    )
    t_cross = roots[-1]
    return GroundCrossing(
        x=float(np.polyval(x_coeffs, t_cross)),
        t=float(times[0]) + t_cross,
        launch_vx=float(x_coeffs[0]),
        launch_vz=float(b),
        residual_m=residual,
    )


def record_one_toss(  # noqa: PLR0917
    *,
    env: Any,
    env_cls: Any,
    mujoco: Any,
    writer: Any,
    seed: int,
    speed: float,
    release_ms: float,
    standoff: float,
    settle_steps: int,
    clip_filename: str,
) -> TossClip:
    """`reset -> Pick -> MoveToThrowPose -> Toss -> settle`, filmed from inside the physics.

    The whole sequence, because `Tossing3DEnvironment.set_state` restores only
    episode-initial states. Only the toss and the settle are filmed; neither dial reaches
    the pick or the base motion.
    """
    env.reset_to_seed(seed=seed)
    backend = env.backend()
    backend.api()
    raw = backend._raw_env  # noqa: SLF001
    robot_env = raw.unwrapped._object_centric_env._robot_env  # noqa: SLF001
    sim = robot_env.sim
    model = sim.model.mj_model
    data = sim.data.mj_data

    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, backend.cube_name)
    cube_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == cube_body}

    clip = TossClip(
        seed=seed,
        commanded_speed_deg=speed,
        commanded_release_ms=release_ms,
        standoff=standoff,
        clip_filename=clip_filename,
        physics_timestep=float(model.opt.timestep),
        frame_hz=1.0 / (float(model.opt.timestep) * RENDER_EVERY_N_TICKS),
    )

    env.take_action(action=np.array([env_cls.pick_id, ORACLE_PICK_DISTANCE, ORACLE_PICK_ROTATION]))
    clip.pick_error = env.last_skill_error()
    env.take_action(action=np.array([env_cls.move_to_throw_pose_id, standoff, 0.0]))
    clip.move_error = env.last_skill_error()

    times: list[float] = []
    pos_x: list[float] = []
    pos_z: list[float] = []
    contacted: list[bool] = []
    frame_of_tick: list[int] = []

    original_step = sim.step
    ticks = {"n": 0}

    def recording_step(*a: Any, **k: Any) -> Any:
        out = original_step(*a, **k)
        partner = None
        for c in range(data.ncon):
            con = data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 in cube_geoms:
                partner = g2
                break
            if g2 in cube_geoms:
                partner = g1
                break
        centre = data.xpos[cube_body]
        times.append(float(data.time))
        pos_x.append(float(centre[0]))
        pos_z.append(float(centre[2]))
        contacted.append(partner is not None)
        if ticks["n"] % RENDER_EVERY_N_TICKS == 0:
            writer.append_data(backend.render())
            frame_of_tick.append(len(times) - 1)
        ticks["n"] += 1
        return out

    observation = backend.observe()
    # `pos_base_x`, not `x`: the robot carries a joint-space feature schema
    # (`pos_base_*` / `pos_arm_joint*` / `vel_*`) while movable objects carry a pose.
    clip.base_x_before_toss = observation.get(name=backend.robot_name, feature="pos_base_x")
    clip.bin_x = observation.get(name=backend.bin_name, feature="x")
    clip.bin_z = observation.get(name=backend.bin_name, feature="z")

    sim.step = recording_step
    try:
        # Slot two is `gripper_release_ms`; a literal `0.0` here is a release at time zero.
        env.take_action(action=np.array([env_cls.toss_id, speed, release_ms]))
        clip.toss_error = env.last_skill_error()
        hold = np.zeros(11, dtype=np.float32)
        gym_env = backend._env  # noqa: SLF001
        for _ in range(settle_steps):
            observation_vector, _, _, _, _ = gym_env.step(hold)
            backend._state = gym_env.observation_space.devectorize(  # noqa: SLF001
                observation_vector
            )
    finally:
        sim.step = original_step

    final = backend.observe()
    clip.solved = final.solved
    clip.cube_x_final = final.get(name=backend.cube_name, feature="x")
    clip.cube_z_final = final.get(name=backend.cube_name, feature="z")

    clip.frame_times = [times[i] for i in frame_of_tick]
    clip.frame_cube_x = [pos_x[i] for i in frame_of_tick]
    clip.frame_cube_z = [pos_z[i] for i in frame_of_tick]
    clip.frame_contact = [contacted[i] for i in frame_of_tick]

    start, stop = longest_contact_free_window(contacted=contacted)
    clip.free_flight_ticks = stop - start
    if stop - start >= 3:
        clip.release_t = times[start]
        clip.land_t = times[stop - 1]
        clip.free_flight_seconds = times[stop - 1] - times[start]
        clip.free_flight_frames = sum(
            1 for t in clip.frame_times if times[start] <= t <= times[stop - 1]
        )
        crossing = ballistic_ground_crossing(
            times=np.array(times[start:stop]),
            xs=np.array(pos_x[start:stop]),
            zs=np.array(pos_z[start:stop]),
        )
        clip.ballistic_impact_x = crossing.x
        clip.ballistic_impact_t = crossing.t
        clip.ballistic_fit_residual_m = crossing.residual_m
        clip.launch_elevation_deg = float(
            np.degrees(np.arctan2(crossing.launch_vz, crossing.launch_vx))
        )
        clip.launch_speed_mps = float(np.hypot(crossing.launch_vx, crossing.launch_vz))
    return clip


def _worker(job: tuple[int, list[tuple[float, float]], float, int, str]) -> list[dict[str, Any]]:  # noqa: PLR0917
    """One process, one live simulator, however many `(speed, release_ms)` cells it was handed.

    One positional tuple because `Pool.map` passes exactly one positional argument.
    """
    import imageio.v2 as imageio
    import kinder
    import kinder_models
    import mujoco
    from kinder_models.dynamic3d.tossing.parameterized_skills import TossController

    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment

    assert_kinder_pins(kinder_models=kinder_models, toss_controller=TossController)

    seed, cells, standoff, settle_steps, output_dir = job
    out = Path(output_dir)
    env = Tossing3DEnvironment()
    clips: list[TossClip] = []
    try:
        for speed, release_ms in cells:
            filename = f"clip_{int(round(speed)):03d}_{int(round(release_ms)):04d}.mp4"
            fps = 1.0 / (0.0005 * RENDER_EVERY_N_TICKS)
            writer = imageio.get_writer(
                out / filename,
                fps=fps,
                codec="libx264",
                macro_block_size=1,
                # Near-lossless: the analysis step re-encodes these, so compression here
                # would be paid for twice.
                output_params=["-crf", "12", "-preset", "medium", "-pix_fmt", "yuv420p"],
            )
            try:
                clip = record_one_toss(
                    env=env,
                    env_cls=Tossing3DEnvironment,
                    mujoco=mujoco,
                    writer=writer,
                    seed=seed,
                    speed=speed,
                    release_ms=release_ms,
                    standoff=standoff,
                    settle_steps=settle_steps,
                    clip_filename=filename,
                )
            finally:
                writer.close()
            clip.kinder_models_file = kinder_models.__file__
            clip.kindergarden_file = kinder.__file__
            clips.append(clip)
            print(
                f"  speed={speed:6.1f} ms={release_ms:7.1f} "
                f"ballistic={clip.ballistic_impact_x} resting={clip.cube_x_final} "
                f"solved={clip.solved} flight={clip.free_flight_seconds}s "
                f"frames={clip.free_flight_frames}",
                flush=True,
            )
    finally:
        env.close()
    return [clip.model_dump() for clip in clips]


def _parse_args(*, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--speed-start", type=float, default=DEFAULT_SPEED_START)
    parser.add_argument("--speed-stop", type=float, default=DEFAULT_SPEED_STOP)
    parser.add_argument("--speed-count", type=int, default=DEFAULT_SPEED_COUNT)
    parser.add_argument("--release-ms-start", type=float, default=DEFAULT_RELEASE_MS_START)
    parser.add_argument("--release-ms-stop", type=float, default=DEFAULT_RELEASE_MS_STOP)
    parser.add_argument("--release-ms-count", type=int, default=DEFAULT_RELEASE_MS_COUNT)
    # Explicit axis values, so a clip lands on a cell the committed grid measured. The
    # grid's 20-point axis over `(60, 140)` steps by `80/19`, whose interior points are
    # 64.21, 68.42, ... and never the 100 an evenly-spaced 3-point axis gives.
    parser.add_argument("--speeds", type=float, nargs="+", default=None)
    parser.add_argument("--release-ms", type=float, nargs="+", default=None)
    parser.add_argument("--standoff", type=float, default=ORACLE_THROW_STANDOFF)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    speeds = args.speeds or evenly_spaced_speeds(
        start=args.speed_start, stop=args.speed_stop, count=args.speed_count
    )
    release_ms_values = args.release_ms or evenly_spaced_speeds(
        start=args.release_ms_start, stop=args.release_ms_stop, count=args.release_ms_count
    )
    cells = [(speed, release_ms) for speed in speeds for release_ms in release_ms_values]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(args.max_workers, len(cells)))
    # A worker holds one simulator and reuses it, so fewer workers means fewer scene builds.
    shares: list[list[tuple[float, float]]] = [cells[i::workers] for i in range(workers)]
    jobs = [
        (args.seed, share, args.standoff, args.settle_steps, str(args.output_dir))
        for share in shares
        if share
    ]

    started = time.monotonic()
    with mp.get_context("spawn").Pool(processes=len(jobs)) as pool:
        chunks = pool.map(_worker, jobs)
    rows = sorted(
        (row for chunk in chunks for row in chunk),
        key=lambda r: (r["commanded_speed_deg"], r["commanded_release_ms"]),
    )
    elapsed = time.monotonic() - started

    payload = {
        "seed": args.seed,
        "speeds": speeds,
        "release_ms": release_ms_values,
        "standoff": args.standoff,
        "settle_steps": args.settle_steps,
        "render_every_n_ticks": RENDER_EVERY_N_TICKS,
        "cube_resting_half_height": CUBE_RESTING_HALF_HEIGHT,
        "max_workers": workers,
        "elapsed_seconds": elapsed,
        "release_speed_signature": str(
            inspect.signature(
                __import__(
                    "kinder_models.dynamic3d.tossing.parameterized_skills",
                    fromlist=["TossController"],
                ).TossController.reset
            )
        ),
        "clips": rows,
    }
    output = args.output_dir / "tosses.json"
    output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {output}: {len(rows)} clips in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
