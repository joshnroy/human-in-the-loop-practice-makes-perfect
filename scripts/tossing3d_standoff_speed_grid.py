"""Measure Tossing3D's **2-D (standoff x commanded release speed) success surface**.

Every grid this domain has on record is a 1-D slice of this one. PR #221 swept 11 standoffs
at 6 low-end speeds; PR #226/#227 swept 37 speeds at a single fixed standoff (1.35). Neither
can answer the question both of them raise, because the two parameters are coupled by one
piece of arithmetic:

    base_x    = bin_x - standoff          (`MoveToThrowPose` parks the base a standoff back)
    landing_x = base_x + range(speed)     (the toss is a ballistic throw of some range)

so a throw solves when `landing_x` lands in the goal box, i.e. when

    range(speed) - standoff  in  [-0.15, +0.15]                (bin_x = 2.0, a 0.300 m box)

The prediction that follows is that the solving window in **speed** slides one-for-one with
**standoff**, at whatever `d range / d speed` is -- about 0.0049 m per deg/s from #226's
grid, i.e. one 5 deg/s step per 0.025 m of standoff. This probe measures the surface so that
prediction can be checked rather than assumed. Where the two disagree is the finding.

## What one cell is

One `(standoff, speed, seed)` cell is a full `Pick -> MoveToThrowPose(standoff) ->
Toss(speed)` sequence in the real simulator, with `Pick` held at the oracle's point for the
same reason `tossing3d_skill_parameter_sweep.py` holds it there: letting the grasp's own
variance in would confound "does this standoff and speed work" with "did the grasp land".
`solved` is `KinderBackend.check_goals()` -- the environment's own verdict, not a
re-derivation of it.

This is **not** re-timeable the way `tossing3d_release_angle_probe.py`'s angle grid was.
That probe exploited the toss path's geometry being speed-independent, so one execution
re-timed to every speed. Nothing here is speed- or standoff-independent: where the cube
actually lands is the measurement, and it needs the throw to actually happen.

## The ground crossing is extrapolated to a fixed height, deliberately

`range(speed)` above has to be a property of the throw, so `fit_free_flight` fits the
free-flight parabola and solves it for the crossing of `z = cube_half_height` -- the height
the cube's centre sits at when resting on the floor -- **whether or not the cube got that
far**. A cube that hits the bin's near wall stops being recorded well above the floor; its
range is still the range it would have flown. Taking the last recorded sample instead would
put wall cells and floor cells on two different scales, and the geometry overlay would
compare against neither. `first_contact_body` is recorded separately, and is what
distinguishes "flew the predicted distance and was stopped by the wall" from "flew the
wrong distance".

## Resolution, and the one seed

The grid is **10 standoffs x 10 speeds x 1 seed = 100 cells**. Both axes are therefore
coarser than the 1-D slices this generalises, and the seed axis is as thin as it can be.
What that buys and costs:

  * every cell is `0/1` or `1/1` -- **binary, never partial**. The structure that mattered
    most in the 1-D grids lived exactly in the partial cells (band edges at `4/10` and
    `9/10`, and the `185: 0/10 -> 190: 10/10 -> 195: 1/10` reversal). With one seed a
    marginal cell is a coin flip, so this surface supports claims about the **shape and
    rough location** of the solving region and **not** about where its edges fall;
  * the speed axis is 20 deg/s rather than 5, so it cannot see the release sawtooth. That is
    fine here and deliberate -- #227 already resolves it at one standoff -- but it means a
    single anomalous speed can land between two columns and simply not appear.

## Row order: a half-finished sweep is still a whole-range sweep

Jobs are `(standoff, seed)` pairs -- one persistent environment runs all speeds for one
pair, since `reset_to_seed` rebuilds only the scene. They are submitted in
`refinement_order`: both endpoints, then the midpoint, then the quarter points, and so on.
A sweep stopped or inspected halfway is then a *coarse* grid of the full standoff range
rather than a fine grid of its bottom third, so a partial heatmap shows the shape and can be
early-stopped on. The parent rewrites its checkpoint after every completed job.

Run it under a memory cap, with `scripts/with_kinder_env.sh` so `kinder_models` resolves to
this worktree's pin rather than the main checkout's -- `assert_kinder_pins` (reused from the
release-angle probe) refuses to start otherwise, because a stale pin has no `release_speed`
and would turn the speed axis into N copies of one throw while looking entirely normal:

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        scripts/with_kinder_env.sh python scripts/tossing3d_standoff_speed_grid.py \
        --output grid.json --max-workers 14

`analysis/tossing3d_standoff_speed_surface.py` reads the JSON back and draws the heatmap;
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

from hitl_pmp.environments.tossing3d.skill_oracle_policy import (  # noqa: E402
    ORACLE_PICK_DISTANCE,
    ORACLE_PICK_ROTATION,
)
from scripts.tossing3d_release_angle_probe import (  # noqa: E402
    assert_kinder_pins,
    longest_contact_free_window,
)

# One fixed scene seed. It is seed 0 of the same set #226/#227 used, so the standoff=1.35
# column of this surface is the *same* scene as that grid's seed-0 column rather than
# merely a comparable one. Never randomly drawn. One seed is a deliberate budget choice and
# a real limitation -- see the resolution note above.
DEFAULT_SEEDS = (0,)

# 60-240 deg/s in 10 steps of 20. Deliberately *coarser* than #226/#227's 5 deg/s, which
# was chosen to avoid aliasing the release sawtooth (the realised release fraction resets
# at roughly 5 deg/s intervals, and #213's 3-point grid missed it entirely). That fine
# structure is already characterised at one standoff; this grid's job is the *standoff*
# dependence, so the budget goes to covering the plane rather than to re-resolving a
# sawtooth. A reader wanting the fine speed structure should read #227, not this.
DEFAULT_SPEED_START = 60.0
DEFAULT_SPEED_STOP = 240.0
DEFAULT_SPEED_STEP = 20.0

# Standoff wide enough to enclose the solving region with empty margin on both sides,
# checked against #226's committed grid rather than assumed: the ballistic range there runs
# 0.964 m (at 60 deg/s) to 1.869 m (at 240), so `range +/- 0.15` confines the solving region
# to [0.814, 2.019] and no wider. 0.75 -> 2.10 clears both ends. A narrower axis -- 1.05 to
# 1.75 was proposed -- would have truncated the region at *both* ends and made an edge of
# the grid look like a boundary of the domain.
#
# 0.15 is also exactly the goal box's half-width, so the predicted band is about one cell
# thick, and the predicted 1:1 slide (0.0049 m/deg/s x 20 = 0.098 m per speed step) moves it
# about two thirds of a cell per column -- both resolvable at this resolution.
DEFAULT_STANDOFF_START = 0.75
DEFAULT_STANDOFF_STOP = 2.10
DEFAULT_STANDOFF_STEP = 0.15

# Physics substeps to hold after the toss so the cube comes to rest before `check_goals`.
DEFAULT_SETTLE_STEPS = 30


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
    """One `(standoff, speed, seed)` cell of the surface."""

    standoff: float
    commanded_speed_deg: float
    seed: int

    solved: bool | None = None

    # Where the base actually ended up, against the standoff it was commanded. The
    # geometry above assumes `base_x = bin_x - standoff`; this is what checks it.
    base_x_before_toss: float | None = None
    bin_x: float | None = None
    goal_region: list[float] | None = None

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
    settle_steps: int,
) -> GridCell:
    """One `Pick -> MoveToThrowPose -> Toss`, with the cube recorded at every substep."""
    cell = GridCell(standoff=standoff, commanded_speed_deg=speed, seed=seed)
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
    cell.bin_x = float(state0.get(obj=env.bin, feature_name="x"))
    cell.goal_region = [
        float(state0.get(obj=env.goal_region, feature_name=name))
        for name in ("x_min", "y_min", "z_min", "x_max", "y_max", "z_max")
    ]

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
        env.take_action(action=np.array([env_cls.toss_id, speed, 0.0]))
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


def _worker(job: tuple[float, int, list[float], int]) -> list[dict[str, Any]]:  # noqa: PLR0917
    """One `(standoff, seed)` pair, every speed, in its own process.

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

    standoff, seed, speeds, settle_steps = job
    env = Tossing3DEnvironment()
    rows: list[GridCell] = []
    try:
        for speed in speeds:
            cell = run_cell(
                env=env,
                env_cls=Tossing3DEnvironment,
                mujoco=mujoco,
                seed=seed,
                standoff=standoff,
                speed=speed,
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
    parser.add_argument("--speed-step", type=float, default=DEFAULT_SPEED_STEP)
    parser.add_argument("--standoff-start", type=float, default=DEFAULT_STANDOFF_START)
    parser.add_argument("--standoff-stop", type=float, default=DEFAULT_STANDOFF_STOP)
    parser.add_argument("--standoff-step", type=float, default=DEFAULT_STANDOFF_STEP)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--max-workers", type=int, default=14)
    return parser.parse_args(argv)


def _axis(*, start: float, stop: float, step: float) -> list[float]:
    """An inclusive `start..stop` axis, rounded so JSON keys compare exactly."""
    n = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 6) for i in range(n)]


def main() -> None:
    args = _parse_args()
    speeds = _axis(start=args.speed_start, stop=args.speed_stop, step=args.speed_step)
    standoffs = _axis(start=args.standoff_start, stop=args.standoff_stop, step=args.standoff_step)
    ordered = [standoffs[i] for i in refinement_order(n=len(standoffs))]
    jobs = [
        (standoff, seed, speeds, args.settle_steps) for standoff in ordered for seed in args.seeds
    ]
    total_cells = len(jobs) * len(speeds)
    print(
        f"{len(standoffs)} standoffs x {len(speeds)} speeds x {len(args.seeds)} seeds "
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
