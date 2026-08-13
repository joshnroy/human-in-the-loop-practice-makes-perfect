"""Measure Tossing3D's **`Toss` parameter surface** over any of its three dials.

`Toss` has two parameters -- `release_speed` (joint-path deg/s) and `gripper_release_ms`
(the wall-clock millisecond, from the start of the swing, at which the gripper opens) -- and
the throw is aimed by a third, `MoveToThrowPose`'s `standoff`. This one driver sweeps a
Cartesian grid over all three, holding any of them fixed at `points=1`. Which axes are live
is a command-line choice, not a fork of this file: it started life as PR #234's
`(standoff x speed)` probe and was generalised rather than copied, because two divergent
copies of a 500-line sweep driver is a worse outcome than any figure it could produce.

## What one cell is

One `(standoff, speed, release_ms, seed)` cell is a full `Pick -> MoveToThrowPose(standoff)
-> Toss(speed, release_ms)` sequence in the real simulator, with `Pick` held at the oracle's
point for the same reason `tossing3d_skill_parameter_sweep.py` holds it there: letting the
grasp's own variance in would confound "do these parameters work" with "did the grasp land".
`solved` is `KinderBackend.check_goals()` -- the environment's own verdict, not a
re-derivation of it.

This is **not** re-timeable the way `tossing3d_release_angle_probe.py`'s angle grid was.
That probe exploited the toss path's geometry being speed-independent, so one execution
re-timed to every speed. Nothing here is independent of any of the three: where the cube
actually lands is the measurement, and it needs the throw to actually happen.

## Ballistic distance is the primary criterion; resting position rides along

`fit_free_flight` fits the free-flight parabola and solves it for the crossing of
`z = cube_half_height` -- the height the cube's centre sits at when resting on the floor --
**whether or not the cube got that far**. A cube that hits the bin's near wall stops being
recorded well above the floor; its ballistic distance is still the distance it would have
flown. Taking the last recorded sample instead would put wall cells and floor cells on two
different scales.

The **resting** x is recorded beside it (`cube_x_final`) because it is free -- `take_action`
runs the toss to completion anyway -- and because a downstream predicate needs it. It is
deliberately *not* the primary criterion: it is contaminated by bin contact, and the
contamination is a step rather than a drift. Scanning the millisecond axis at 140 deg/s,
resting x went `690 -> 1.7175`, `705 -> 1.7428`, then **jumped 244 mm to `710 -> 1.9870`**.
That step is the bin catching the cube versus not, and it is exactly the artifact the
extrapolated ballistic crossing avoids. `first_contact_body` is recorded separately and is
what distinguishes "flew the predicted distance and was stopped by the wall" from "flew the
wrong distance".

## Cells where nothing is thrown

`gripper_release_ms` is **not clamped** upstream, so a value at or past the end of the swing
means the gripper never opens and the cube is never thrown. That is a real corner of the
space rather than an error, and this driver has to be able to land in it: such a cell gets
`threw=False` and leaves every ballistic field unset, so a reader can never mistake it for a
flat measurement of zero distance.

It should not arise inside `TOSS_RELEASE_MS_BOUNDS`. The swing runs 3100 ms at 60 deg/s down
to 1700 ms at 140 deg/s, and the bounds stop at 1400, so every cell of a default-bounds grid
is still swinging when the gripper opens. A dead cell inside those bounds is therefore a
finding about the bounds, not a quirk of the sweep.

## Axes are `start/stop/points`, and the defaults are read from `predicates`

A 20-point axis over `(60, 140)` has step `80/19`, which `start/stop/step` cannot express
without the endpoint drifting off the bound the sampler actually draws from -- so
`linear_axis` takes a point count and pins both endpoints exactly.

The defaults are **imported from `hitl_pmp.environments.tossing3d.predicates`** rather than
retyped, so this sweep cannot silently disagree with the interval the sampler draws from.
That is not a stylistic preference: this task has already shipped two measurements against
wrong constants -- a `240 deg/s` speed ceiling that was a measurement range rather than a
command range, and a `723 ms` default that was the nominal arithmetic rather than the
motion-planned one, worth 52 mm of landing distance.

## Row order: a half-finished sweep is still a whole-range sweep

Jobs are `(standoff, seed, speed)` triples -- one persistent environment runs every
millisecond for one triple, since constructing the simulator is what costs. They are
submitted in `refinement_order` over the speed axis: both endpoints, then the midpoint, then
the quarter points, and so on. A sweep stopped or inspected halfway is then a *coarse* grid
of the full speed range rather than a fine grid of its bottom third, so a partial figure
shows the shape and can be early-stopped on. The parent rewrites its checkpoint after every
completed job.

Run it under a memory cap. **From a worktree, `PYTHONPATH` must shadow the shared checkout's
editable KINDER with this worktree's own `reference/` submodules** -- since PR #232 the
`tossing3d` extra is installed editable by absolute path to the shared checkout, so a
worktree otherwise imports whatever pin that tree happens to sit on. `assert_kinder_pins`
(reused from the release-angle probe) refuses to start when it does, because a pin without
`gripper_release_ms` turns the millisecond axis into N copies of one column while looking
entirely normal:

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env PYTHONPATH=<worktree>/reference/kinder-baselines/kinder-models/src:\
    <worktree>/reference/kindergarden/src:<worktree>/src \
        scripts/with_kinder_env.sh python scripts/tossing3d_toss_parameter_grid.py \
        --output grid.json --speed-points 20 --release-ms-points 20 \
        --seeds 0 1 2 3 4 --max-workers 20

`analysis/tossing3d_toss_parameter_surface.py` reads the JSON back and draws the figures;
this script never plots.
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

_REPO_ROOT = Path(__file__).resolve().parent.parent
# See `tossing3d_release_angle_probe`'s own note: the KINDER venv installs both packages
# editable, so their `.pth` files carry the *main checkout's* absolute paths and a worktree
# otherwise imports whatever commit that checkout is sitting on.
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
)
from scripts.tossing3d_release_angle_probe import (  # noqa: E402
    assert_kinder_pins,
    longest_contact_free_window,
)

# Five fixed scene seeds, never randomly drawn. Seed 0 is the one #226/#227 and PR #234's
# surface used, so a column here is the *same* scene as theirs rather than merely a
# comparable one; the other four are what make a cell an `x/5` rather than a coin flip.
DEFAULT_SEEDS = (0, 1, 2, 3, 4)

# Both `Toss` axes span exactly the interval the sampler draws from, **imported rather than
# retyped**. See the module docstring: a sweep whose bounds are a copy of the sampler's can
# silently drift off them, and this task has already published one measurement over a
# `240 deg/s` ceiling that was never a command range.
DEFAULT_SPEED_START, DEFAULT_SPEED_STOP = TOSS_SPEED_BOUNDS
DEFAULT_SPEED_POINTS = 20

DEFAULT_RELEASE_MS_START, DEFAULT_RELEASE_MS_STOP = TOSS_RELEASE_MS_BOUNDS
DEFAULT_RELEASE_MS_POINTS = 20

# The standoff is **held fixed** by default -- `points=1` -- because the throw's ballistic
# distance is a property of the swing and the standoff only translates where it lands. 1.35
# is the value #226/#227 and PR #234's surface fixed it at, so distances are directly
# comparable with those grids; it sits inside `THROW_STANDOFF_BOUNDS = (1.10, 1.75)`.
# A follow-up that wants the standoff dependence back sets `--standoff-points` and needs no
# change to this file.
DEFAULT_STANDOFF_START = 1.35
DEFAULT_STANDOFF_STOP = 1.35
DEFAULT_STANDOFF_POINTS = 1

# Physics substeps to hold after the toss so the cube comes to rest before `check_goals`.
DEFAULT_SETTLE_STEPS = 30

# Fewer free-flight samples than this cannot determine a parabola, which is what
# `fit_free_flight` requires. A cell below it is a cell where the gripper never opened.
MIN_FREE_FLIGHT_SAMPLES = 3


class BallisticFit(BaseModel):
    """The free-flight parabola, and the ground crossing extrapolated from it."""

    samples: int
    launch_vx: float
    launch_vz: float
    launch_speed_mps: float
    launch_elevation_deg: float
    residual_m: float
    impact_t: float | None = None
    impact_x: float | None = None
    impact_vx: float | None = None
    impact_vz: float | None = None


class GridCell(BaseModel):
    """One `(standoff, speed, release_ms, seed)` cell of the surface."""

    standoff: float
    commanded_speed_deg: float
    commanded_release_ms: float
    seed: int

    solved: bool | None = None

    # Did the gripper ever open? `False` marks a cell where no throw happened at all, which
    # is a third state rather than a throw of zero distance -- see the module docstring.
    threw: bool | None = None

    # Where the base actually ended up, against the standoff it was commanded. Every
    # distance here is reported *from the base*, so this is what makes a cell's ballistic
    # distance a property of the throw rather than of where the robot happened to park.
    base_x_before_toss: float | None = None
    bin_x: float | None = None

    # The throw itself.
    free_flight_samples: int | None = None
    ballistic_residual_m: float | None = None
    ballistic_impact_x: float | None = None
    ballistic_impact_t: float | None = None
    ballistic_impact_vx: float | None = None
    ballistic_impact_vz: float | None = None
    launch_vx: float | None = None
    launch_vz: float | None = None
    launch_speed_mps: float | None = None
    launch_elevation_deg: float | None = None
    cube_half_height: float | None = None

    # What actually stopped it, which is how a wall failure is told from a short throw.
    first_contact_x: float | None = None
    first_contact_z: float | None = None
    first_contact_body: str | None = None
    cube_x_final: float | None = None
    cube_y_final: float | None = None
    cube_z_final: float | None = None

    pick_error: str | None = None
    move_error: str | None = None
    toss_error: str | None = None

    # Provenance: the tree that actually ran, not the tree that was read.
    kinder_models_file: str | None = None
    kindergarden_file: str | None = None


def linear_axis(*, start: float, stop: float, points: int) -> list[float]:
    """An inclusive `start..stop` axis of exactly `points` values, both endpoints pinned.

    A point count rather than a step, because the interesting axes here do not divide
    evenly: 20 points over `TOSS_SPEED_BOUNDS = (60, 140)` steps by `80/19`, and expressing
    that as a step drifts the top of the axis off 140.0 -- which would make the sweep a
    measurement of a range slightly different from the one the sampler draws from.

    Values are rounded to 6 decimals so that a float written into the JSON and the same
    float recomputed by the analysis compare equal as dict keys. Without that, a column
    silently vanishes from the surface rather than raising.
    """
    if points <= 1:
        return [round(start, 6)]
    span = stop - start
    return [round(start + i * span / (points - 1), 6) for i in range(points)]


def refinement_order(*, n: int) -> list[int]:
    """`range(n)` reordered endpoints-first, then repeatedly bisecting.

    The point is that *any prefix* of the result is a roughly-even coarse sample of the
    whole range, so a sweep inspected or stopped halfway shows the shape of the surface
    rather than one edge of it. Returns a permutation: every row runs exactly once.
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]
    order = [0, n - 1]
    seen = {0, n - 1}
    frontier = [(0, n - 1)]
    while frontier:
        nxt: list[tuple[int, int]] = []
        for lo, hi in frontier:
            mid = (lo + hi) // 2
            if lo < mid < hi and mid not in seen:
                seen.add(mid)
                order.append(mid)
                nxt.append((lo, mid))
                nxt.append((mid, hi))
        frontier = nxt
    return order


def fit_free_flight(
    *, times: np.ndarray, pos_x: np.ndarray, pos_z: np.ndarray, ground_z: float
) -> BallisticFit | None:
    """Fit the cube's free flight and solve it for its crossing of `z = ground_z`.

    `times` are absolute; the fit is done against `times - times[0]` so `launch_*` are the
    velocities at the window's first sample -- the first substep after the gripper let go.

    The crossing is **extrapolated**: it is where the parabola would reach `ground_z`, not
    where the recording stopped. That is what makes a bin-wall cell's range comparable with
    an open-floor cell's. Returns `None` when the window is too short to determine a
    parabola at all, and leaves the `impact_*` fields unset when the parabola has no real
    descending crossing.
    """
    if len(times) < 3:
        return None
    rel = np.asarray(times, dtype=float) - float(times[0])
    z_coeffs = np.polyfit(rel, np.asarray(pos_z, dtype=float), 2)
    x_coeffs = np.polyfit(rel, np.asarray(pos_x, dtype=float), 1)
    residual = float(np.max(np.abs(np.polyval(z_coeffs, rel) - np.asarray(pos_z, dtype=float))))
    quad, launch_vz, z0 = (float(c) for c in z_coeffs)
    launch_vx, x0 = (float(c) for c in x_coeffs)

    fit = BallisticFit(
        samples=len(rel),
        launch_vx=launch_vx,
        launch_vz=launch_vz,
        launch_speed_mps=float(np.hypot(launch_vx, launch_vz)),
        launch_elevation_deg=float(np.degrees(np.arctan2(launch_vz, launch_vx))),
        residual_m=residual,
    )

    # The later root of `quad t^2 + launch_vz t + (z0 - ground_z) = 0`: the descending
    # crossing. `quad` is about -g/2 for a real throw, so `roots` is well conditioned.
    roots = np.roots([quad, launch_vz, z0 - ground_z]) if quad != 0.0 else np.array([])
    real_roots = [float(r.real) for r in np.atleast_1d(roots) if abs(r.imag) < 1e-9 and r.real > 0]
    if real_roots:
        impact_t = max(real_roots)
        fit.impact_t = impact_t
        fit.impact_x = x0 + launch_vx * impact_t
        fit.impact_vx = launch_vx
        fit.impact_vz = 2.0 * quad * impact_t + launch_vz
    return fit


def run_cell(  # noqa: PLR0917
    *,
    env: Any,
    env_cls: Any,
    mujoco: Any,
    seed: int,
    standoff: float,
    speed: float,
    release_ms: float,
    settle_steps: int,
) -> GridCell:
    """One `Pick -> MoveToThrowPose -> Toss`, with the cube recorded at every substep."""
    cell = GridCell(
        standoff=standoff,
        commanded_speed_deg=speed,
        commanded_release_ms=release_ms,
        seed=seed,
    )
    state0 = env.reset_to_seed(seed=seed)
    backend = env.backend()
    backend.api()
    raw = backend._raw_env  # noqa: SLF001
    robot_env = raw.unwrapped._object_centric_env._robot_env  # noqa: SLF001
    sim = robot_env.sim
    model = sim.model.mj_model
    data = sim.data.mj_data

    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, backend.cube_name)
    cube_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == cube_body}
    # The cube's centre height when it rests on the floor, so the ground crossing is solved
    # for where the *centre* arrives rather than where a corner would.
    cell.cube_half_height = float(min(model.geom_size[g][2] for g in cube_geoms))
    # No goal-region object is read: PR #228 dropped it, and the bin's interior *is* the
    # goal region now. This grid does not need one either way -- its criterion is ballistic
    # distance, not whether a cell scored.
    cell.bin_x = float(state0.get(obj=env.bin, feature_name="x"))

    env.take_action(action=np.array([env_cls.pick_id, ORACLE_PICK_DISTANCE, ORACLE_PICK_ROTATION]))
    cell.pick_error = env.last_skill_error()
    state2 = env.take_action(action=np.array([env_cls.move_to_throw_pose_id, standoff, 0.0]))
    cell.move_error = env.last_skill_error()
    cell.base_x_before_toss = float(state2.get(obj=env.robot, feature_name="pos_base_x"))

    times: list[float] = []
    pos_x: list[float] = []
    pos_z: list[float] = []
    contacted: list[bool] = []
    partners: list[int | None] = []
    original_step = sim.step
    recording = {"on": False}

    def recording_step(*a: Any, **k: Any) -> Any:
        out = original_step(*a, **k)
        if recording["on"]:
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
            partners.append(partner)
        return out

    sim.step = recording_step
    try:
        recording["on"] = True
        # Slot two is `gripper_release_ms`. PR #234's version of this line passed a literal
        # `0.0` there, which was harmless at a pin where the slot was unread and is a
        # release at time zero at this one -- exactly the kind of silent axis collapse
        # `assert_kinder_pins` now refuses to start without.
        env.take_action(action=np.array([env_cls.toss_id, speed, release_ms]))
        cell.toss_error = env.last_skill_error()
        hold = np.zeros(11, dtype=np.float32)
        gym_env = backend._env  # noqa: SLF001
        for _ in range(settle_steps):
            observation, _, _, _, _ = gym_env.step(hold)
            backend._state = gym_env.observation_space.devectorize(observation)  # noqa: SLF001
    finally:
        recording["on"] = False
        sim.step = original_step

    cell.solved = bool(backend.check_goals())
    # Straight off the simulator rather than through a devectorized observation: this is
    # the same body the flight was recorded from, so a resting position and a flight path
    # can never disagree about which object they describe.
    cell.cube_x_final = float(data.xpos[cube_body][0])
    cell.cube_y_final = float(data.xpos[cube_body][1])
    cell.cube_z_final = float(data.xpos[cube_body][2])

    start, stop = longest_contact_free_window(contacted=contacted)
    cell.free_flight_samples = stop - start
    if stop < len(contacted):
        cell.first_contact_x = pos_x[stop]
        cell.first_contact_z = pos_z[stop]
        partner_geom = partners[stop]
        if partner_geom is not None:
            body = int(model.geom_bodyid[partner_geom])
            cell.first_contact_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
    # A gripper that never opened leaves the cube in contact for the whole recording, so
    # there is no free-flight window to fit. That is a real corner of the parameter space
    # (`gripper_release_ms` is not clamped to the swing), and it is recorded as its own
    # state rather than as a throw of zero distance.
    cell.threw = (stop - start) >= MIN_FREE_FLIGHT_SAMPLES
    fit = fit_free_flight(
        times=np.array(times[start:stop]),
        pos_x=np.array(pos_x[start:stop]),
        pos_z=np.array(pos_z[start:stop]),
        ground_z=cell.cube_half_height,
    )
    if fit is not None:
        cell.ballistic_residual_m = fit.residual_m
        cell.ballistic_impact_x = fit.impact_x
        cell.ballistic_impact_t = fit.impact_t
        cell.ballistic_impact_vx = fit.impact_vx
        cell.ballistic_impact_vz = fit.impact_vz
        cell.launch_vx = fit.launch_vx
        cell.launch_vz = fit.launch_vz
        cell.launch_speed_mps = fit.launch_speed_mps
        cell.launch_elevation_deg = fit.launch_elevation_deg
    return cell


def _worker(job: tuple[float, int, float, list[float], int]) -> list[dict[str, Any]]:  # noqa: PLR0917
    """One `(standoff, seed, speed)` triple, every release millisecond, in its own process.

    The millisecond is the inner axis because constructing the simulator is what costs, not
    switching a parameter -- so the job is sized to amortise one `Tossing3DEnvironment` over
    a whole column while still leaving `standoffs x seeds x speeds` jobs to spread across
    workers. At this grid's shape that is `1 x 5 x 20 = 100` jobs of 20 cells each.

    One positional tuple because `Pool.imap_unordered` passes exactly one positional
    argument; the project's keyword-only rule cannot apply to a callable whose calling
    convention the multiprocessing API owns.
    """
    import kinder
    import kinder_models
    import mujoco
    from kinder_models.dynamic3d.tossing.parameterized_skills import TossController

    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment

    assert_kinder_pins(kinder_models=kinder_models, toss_controller=TossController)

    standoff, seed, speed, release_ms_values, settle_steps = job
    env = Tossing3DEnvironment()
    rows: list[GridCell] = []
    try:
        for release_ms in release_ms_values:
            cell = run_cell(
                env=env,
                env_cls=Tossing3DEnvironment,
                mujoco=mujoco,
                seed=seed,
                standoff=standoff,
                speed=speed,
                release_ms=release_ms,
                settle_steps=settle_steps,
            )
            cell.kinder_models_file = kinder_models.__file__
            cell.kindergarden_file = kinder.__file__
            rows.append(cell)
    finally:
        env.close()
    return [r.model_dump() for r in rows]


def ensure_assets_once() -> None:
    """Fetch the MimicLabs scene assets in the parent, before any worker is forked.

    `kinder.make` auto-downloads a ~2 GB archive into the checkout on first use, and the
    guard it skips on is "does `mimiclabs_scenes/meshes` exist". In a *fresh worktree* that
    is false for every worker at once, so N workers start N downloads into the same
    `assets.zip`, each overwrites the others, and every one of them then fails to unpack
    with `assets.zip is not a zip file`. That is not hypothetical -- it is how this probe's
    first launch died, at `--max-workers 14`.

    Doing it once here serialises the download by construction. It is a no-op once the
    assets are present, so the cost on every later run is one import.
    """
    import kinder
    import kinder.envs.dynamic3d.envs  # noqa: F401  # the *module*, so mujoco is imported

    kinder._ensure_assets_for_env("kinder/Tossing3D-o1-v0")  # noqa: SLF001


def _parse_args(*, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--speed-start", type=float, default=DEFAULT_SPEED_START)
    parser.add_argument("--speed-stop", type=float, default=DEFAULT_SPEED_STOP)
    parser.add_argument("--speed-points", type=int, default=DEFAULT_SPEED_POINTS)
    parser.add_argument("--release-ms-start", type=float, default=DEFAULT_RELEASE_MS_START)
    parser.add_argument("--release-ms-stop", type=float, default=DEFAULT_RELEASE_MS_STOP)
    parser.add_argument("--release-ms-points", type=int, default=DEFAULT_RELEASE_MS_POINTS)
    parser.add_argument("--standoff-start", type=float, default=DEFAULT_STANDOFF_START)
    parser.add_argument("--standoff-stop", type=float, default=DEFAULT_STANDOFF_STOP)
    parser.add_argument("--standoff-points", type=int, default=DEFAULT_STANDOFF_POINTS)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--max-workers", type=int, default=14)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    speeds = linear_axis(start=args.speed_start, stop=args.speed_stop, points=args.speed_points)
    release_ms_values = linear_axis(
        start=args.release_ms_start, stop=args.release_ms_stop, points=args.release_ms_points
    )
    standoffs = linear_axis(
        start=args.standoff_start, stop=args.standoff_stop, points=args.standoff_points
    )
    # Refined over the *speed* axis, since that is the outer axis a partial sweep most needs
    # to span: a half-finished grid should be a coarse read of the whole speed range.
    ordered_speeds = [speeds[i] for i in refinement_order(n=len(speeds))]
    jobs = [
        (standoff, seed, speed, release_ms_values, args.settle_steps)
        for speed in ordered_speeds
        for standoff in standoffs
        for seed in args.seeds
    ]
    total_cells = len(jobs) * len(release_ms_values)
    print(
        f"{len(standoffs)} standoffs x {len(speeds)} speeds x "
        f"{len(release_ms_values)} release-ms x {len(args.seeds)} seeds "
        f"= {total_cells} cells, in {len(jobs)} jobs on {args.max_workers} workers",
        flush=True,
    )

    ensure_assets_once()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    started = time.monotonic()
    rows: list[dict[str, Any]] = []

    def flush(*, done: int) -> None:
        payload = {
            "standoffs": standoffs,
            "speeds": speeds,
            "release_ms": release_ms_values,
            "seeds": list(args.seeds),
            "settle_steps": args.settle_steps,
            "max_workers": args.max_workers,
            "jobs_total": len(jobs),
            "jobs_done": done,
            "cells_total": total_cells,
            "elapsed_seconds": time.monotonic() - started,
            "rows": rows,
        }
        tmp_path.write_text(json.dumps(payload, indent=1))
        tmp_path.replace(args.output)

    with mp.get_context("spawn").Pool(processes=min(args.max_workers, len(jobs))) as pool:
        for done, chunk in enumerate(pool.imap_unordered(_worker, jobs), start=1):
            rows.extend(chunk)
            flush(done=done)
            elapsed = time.monotonic() - started
            print(
                f"[{done}/{len(jobs)} jobs] {len(rows)}/{total_cells} cells, "
                f"{elapsed:.0f}s elapsed, {elapsed / len(rows):.2f}s/cell, "
                f"eta {elapsed / done * (len(jobs) - done) / 60:.0f} min",
                flush=True,
            )
    flush(done=len(jobs))
    print(f"wrote {args.output}: {len(rows)} cells in {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
