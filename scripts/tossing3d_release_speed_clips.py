"""Record one Tossing3D throw per commanded release speed, densely enough to *see* it.

PR #227 charted where the cube lands across 60-240 deg/s. This records the throws that
chart is made of: the standoff held at `ORACLE_THROW_STANDOFF`, one fixed seed, and five
speeds spanning the dial, so "how far the cube goes" stops being a curve and becomes five
arcs. Whether a throw scores is deliberately not the criterion -- one that sails past the
bin is exactly as interesting as one that drops in.

## Why this records at 200 Hz rather than using the domain's existing substep recording

`KinderBackend.set_substep_recording` wraps the env in `gymnasium.wrappers.
RenderCollection`, which collects one frame per `env.step()`. Measured on this scene, one
`env.step` is **100 MuJoCo ticks at a 0.0005 s timestep = 0.05 s**, and the environment's
own `render_fps` metadata is 20 -- consistent. That is the right rate for a whole episode
and the wrong rate for a throw: the cube's free flight here lasts 0.4-0.9 s, so the entire
parabola gets 8-18 frames and reads on screen as a teleport rather than a throw.

So this probe renders from *inside* `sim.step`, every `RENDER_EVERY_N_TICKS` ticks -- 200
frames per simulated second. A render measured 1.7 ms on this machine, so a whole throw's
worth costs under two seconds of wall clock; the rate is a choice about legibility, not a
budget. The cube's position and contact state are recorded on **every** tick regardless,
since that costs nothing and is what the ballistic fit reads.

Frames go out as one `clip_<speed>.mp4` per speed plus a `tosses.json` of measurements;
`analysis/tossing3d_release_speed_video.py` reads both back and composes the annotated,
arc-accumulating video. The split is deliberate -- the simulation is minutes and the
visual design wants iterating, so the two must not be one script.

## How far the cube goes

The same instrument PR #227's distance grid used, and for the same reason: **the ballistic
ground-crossing**, not first contact. On the coincident config the bin sits on the goal
region and is a catcher, so a cube whose ground-crossing lies past the bin's far wall still
hits that wall and drops in -- making "where it first touched something" a function of the
independent variable. Between release and first contact the cube is a free body that MuJoCo
integrates without drag, so its flight is an exact parabola; fitting it and solving for the
height a resting cube's centre sits at gives a bin-independent distance. The resting x is
recorded too, since the two genuinely differ.

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
# `src/` and the two `reference/` source roots ahead of everything else -- see
# `tossing3d_release_angle_probe.py`'s own note. The KINDER venv installs both submodules
# editable against the *main checkout's* absolute paths, so without this a worktree
# silently imports whatever commit that checkout is sitting on. Here that would be fatal
# and invisible at once: at a pin without `release_speed` every cell runs the default
# 140 deg/s toss, and the video would be ten identical throws looking entirely normal.
for _path in (
    _REPO_ROOT / "src",
    _REPO_ROOT / "reference" / "kindergarden" / "src",
    _REPO_ROOT / "reference" / "kinder-baselines" / "kinder-models" / "src",
    _REPO_ROOT,
):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from hitl_pmp.environments.tossing3d.skill_oracle_policy import (  # noqa: E402
    ORACLE_PICK_DISTANCE,
    ORACLE_PICK_ROTATION,
    ORACLE_THROW_STANDOFF,
)
from scripts.tossing3d_release_angle_probe import (  # noqa: E402
    assert_kinder_pins,
    longest_contact_free_window,
)

# The z a resting cube's centre sits at: half the 0.05 m cube. PR #227's distance grid
# solves the flight parabola for this same height, so the two are directly comparable.
CUBE_RESTING_HALF_HEIGHT = 0.025

# Gravity is read from the live model rather than assumed; this is only the fallback used
# by the pure fit when a caller supplies no model (i.e. in tests).
NOMINAL_GRAVITY = 9.81

# One rendered frame per this many MuJoCo ticks. At the scene's 0.0005 s timestep that is
# 200 frames per simulated second -- see the module docstring for why 20 is not enough.
RENDER_EVERY_N_TICKS = 10

# `env.step` calls to hold after the swing, so the cube is at rest before the clip ends.
# PR #227's grids used 30; kept identical so `cube_x_final` means the same thing.
DEFAULT_SETTLE_STEPS = 30

DEFAULT_SEED = 0
DEFAULT_SPEED_START = 60.0
DEFAULT_SPEED_STOP = 240.0
# Five, not ten: a viewer reads a family of five arcs at a glance and has to work at ten,
# and five still lands on 60/105/150/195/240 -- every one of them a multiple of 5 and so
# a speed PR #227's 5 deg/s grid also measured, which makes each clip independently
# checkable against that grid rather than merely consistent with it.
DEFAULT_SPEED_COUNT = 5


class GroundCrossing(BaseModel):
    """Where a free-flying cube's parabola meets a resting cube's centre height.

    The two velocities are the ones at **release**, not at the crossing: they are the
    fitted parabola's derivatives at its first sample, and what they describe is the
    launch the release speed produced. The arrival velocity is a different quantity that
    nothing here reports -- see PR #227's entry on why the two are easy to conflate.
    """

    x: float
    t: float
    launch_vx: float
    launch_vz: float
    residual_m: float


class TossClip(BaseModel):
    """One commanded speed: what the throw did, and where its frames were written."""

    seed: int
    commanded_speed_deg: float
    standoff: float

    clip_filename: str
    frame_hz: float
    physics_timestep: float

    # Every rendered frame's simulated time and the cube's position at it, so the arc can
    # be drawn progressively in lockstep with the footage rather than approximately.
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

    goal_region: tuple[float, float, float, float, float, float] | None = None
    bin_x: float | None = None
    bin_z: float | None = None

    pick_error: str | None = None
    move_error: str | None = None
    toss_error: str | None = None

    # Provenance: the tree that actually ran, not the tree that was read.
    kinder_models_file: str | None = None
    kindergarden_file: str | None = None


def evenly_spaced_speeds(*, start: float, stop: float, count: int) -> list[float]:
    """`count` speeds from `start` to `stop` inclusive, both endpoints hit exactly.

    Endpoints matter here in a way they would not for a grid: the video's whole claim is
    that it spans the dial, so the first and last clips must be the dial's own limits
    rather than something a step size happened to land on.
    """
    if count < 2:
        raise ValueError(f"a speed sweep needs at least 2 speeds to span a range, got {count}")
    step = (stop - start) / (count - 1)
    return [float(start + i * step) for i in range(count)]


def ballistic_ground_crossing(
    *, times: np.ndarray, xs: np.ndarray, zs: np.ndarray, gravity: float = NOMINAL_GRAVITY
) -> GroundCrossing:
    """Fit the free-flight parabola and solve it for a resting cube's centre height.

    Pure, so the arithmetic the video's distance labels rest on is testable without
    MuJoCo. `times`/`xs`/`zs` must already be restricted to the contact-free window --
    picking that window is `longest_contact_free_window`'s job, and mixing a post-landing
    sample in would bend the parabola.

    The **descending** root is taken. Both roots are real and positive whenever the cube is
    released below the resting height, and the ascending one is the instant it first passes
    0.025 m on the way *up* -- a plausible-looking number that is most of a flight short.
    """
    if len(times) < 3:
        raise ValueError(f"a parabola needs at least 3 samples to fit, got {len(times)}")
    rel = np.asarray(times, dtype=float) - float(times[0])
    z_coeffs = np.polyfit(rel, np.asarray(zs, dtype=float), 2)
    x_coeffs = np.polyfit(rel, np.asarray(xs, dtype=float), 1)
    residual = float(np.max(np.abs(np.polyval(z_coeffs, rel) - np.asarray(zs, dtype=float))))

    # z(t) = a t^2 + b t + c, solved for z = h. `a` is -g/2 and is fitted, not assumed, so
    # a scene whose gravity differed would still be read correctly.
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
    standoff: float,
    settle_steps: int,
    clip_filename: str,
) -> TossClip:
    """`reset -> Pick -> MoveToThrowPose -> Toss -> settle`, filmed from inside the physics.

    A whole sequence rather than a jump into a throw-ready state, because there is no
    shortcut: `Tossing3DEnvironment.set_state` restores only episode-initial states, so a
    throw has to be arrived at.

    Only the toss skill and the settle are filmed. The pick and the base motion are
    identical across every speed by construction -- same seed, same parameters, and
    `release_speed` reaches the swing only -- so filming them would put the same 20 seconds
    in front of all ten clips.
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
    # `pos_base_x`, not `x`: the robot carries a joint-space feature schema of its own
    # (`pos_base_*` / `pos_arm_joint*` / `vel_*`) while every movable object carries a
    # pose. `KinderObservation.get` raises rather than returning 0.0 for a wrong name,
    # which is how this one was found.
    clip.base_x_before_toss = observation.get(name=backend.robot_name, feature="pos_base_x")
    clip.goal_region = observation.goal_region
    clip.bin_x = observation.get(name=backend.bin_name, feature="x")
    clip.bin_z = observation.get(name=backend.bin_name, feature="z")

    sim.step = recording_step
    try:
        env.take_action(action=np.array([env_cls.toss_id, speed, 0.0]))
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


def _worker(job: tuple[int, list[float], float, int, str]) -> list[dict[str, Any]]:  # noqa: PLR0917
    """One process, one live simulator, however many speeds it was handed.

    Takes one positional tuple because `Pool.map` passes exactly one positional argument;
    the project's keyword-only rule cannot apply to a callable multiprocessing owns the
    calling convention for.
    """
    import imageio.v2 as imageio
    import kinder
    import kinder_models
    import mujoco
    from kinder_models.dynamic3d.tossing.parameterized_skills import TossController

    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment

    assert_kinder_pins(kinder_models=kinder_models, toss_controller=TossController)

    seed, speeds, standoff, settle_steps, output_dir = job
    out = Path(output_dir)
    env = Tossing3DEnvironment()
    clips: list[TossClip] = []
    try:
        for speed in speeds:
            filename = f"clip_{int(round(speed)):03d}.mp4"
            fps = 1.0 / (0.0005 * RENDER_EVERY_N_TICKS)
            writer = imageio.get_writer(
                out / filename,
                fps=fps,
                codec="libx264",
                macro_block_size=1,
                # Near-visually-lossless: these clips are re-read and re-encoded by the
                # analysis step, so compression here would be paid for twice.
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
                f"  speed={speed:6.1f} range={clip.ballistic_impact_x} "
                f"resting={clip.cube_x_final} solved={clip.solved} "
                f"flight={clip.free_flight_seconds}s frames={clip.free_flight_frames}",
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
    parser.add_argument("--standoff", type=float, default=ORACLE_THROW_STANDOFF)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    speeds = evenly_spaced_speeds(
        start=args.speed_start, stop=args.speed_stop, count=args.speed_count
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(args.max_workers, len(speeds)))
    # Contiguous shares rather than round-robin: a worker holds one simulator and reuses
    # it across its speeds, so fewer workers means fewer scene builds, not fewer throws.
    shares: list[list[float]] = [speeds[i::workers] for i in range(workers)]
    jobs = [
        (args.seed, share, args.standoff, args.settle_steps, str(args.output_dir))
        for share in shares
        if share
    ]

    started = time.monotonic()
    with mp.get_context("spawn").Pool(processes=len(jobs)) as pool:
        chunks = pool.map(_worker, jobs)
    rows = sorted(
        (row for chunk in chunks for row in chunk), key=lambda r: r["commanded_speed_deg"]
    )
    elapsed = time.monotonic() - started

    payload = {
        "seed": args.seed,
        "speeds": speeds,
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
