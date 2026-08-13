"""Measure Tossing3D's `Toss` parameter surface over any of its three dials.

Axes are `(standoff, release_speed, gripper_release_ms)`, any held fixed at `points=1`. One
cell is a full `Pick -> MoveToThrowPose -> Toss`, `Pick` pinned at the oracle's point;
`solved` is the environment's own `check_goals()`.

The primary criterion is the ballistic distance -- the free-flight parabola extrapolated to
`z = cube_half_height` whether or not the cube got that far -- so a cell stopped by the
bin's wall stays comparable with one that reached open floor. Resting x is contaminated by
bin contact as a step: at 140 deg/s, `690 -> 1.7175`, `705 -> 1.7428`, `710 -> 1.9870`.

`gripper_release_ms` is not clamped upstream: at or past the end of the swing the gripper
never opens, and that cell gets `threw=False` and no ballistic fields. The swing runs
3100 ms at 60 deg/s down to 1700 ms at 140; `TOSS_RELEASE_MS_BOUNDS` stops at 1400.

Jobs are `(standoff, seed, speed)` triples, one persistent environment each, submitted in
`refinement_order` over the speed axis so a partial sweep spans the whole range.

Run under a memory cap, with `PYTHONPATH` shadowing the shared checkout's editable KINDER
with this worktree's own `reference/`; `assert_kinder_pins` refuses to start otherwise.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env PYTHONPATH=<worktree>/reference/kinder-baselines/kinder-models/src:\
    <worktree>/reference/kindergarden/src:<worktree>/src \
        scripts/with_kinder_env.sh python scripts/tossing3d_toss_parameter_grid.py \
        --output grid.json --speed-points 20 --release-ms-points 20 \
        --seeds 0 1 2 3 4 --max-workers 20

`analysis/tossing3d_toss_parameter_surface.py` draws the figures; this never plots.
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
)
from scripts.tossing3d_release_angle_probe import (  # noqa: E402
    assert_kinder_pins,
    longest_contact_free_window,
)

# Fixed, never randomly drawn. Seed 0 is the scene #226/#227 measured.
DEFAULT_SEEDS = (0, 1, 2, 3, 4)

# Imported rather than retyped: #239 corrected a retyped `240 deg/s` ceiling that was never
# a command range.
DEFAULT_SPEED_START, DEFAULT_SPEED_STOP = TOSS_SPEED_BOUNDS
DEFAULT_SPEED_POINTS = 20

DEFAULT_RELEASE_MS_START, DEFAULT_RELEASE_MS_STOP = TOSS_RELEASE_MS_BOUNDS
DEFAULT_RELEASE_MS_POINTS = 20

# 1.35 is what #226/#227 fixed it at, so distances are comparable with those grids. Held at
# `points=1`: the standoff only translates where the throw lands.
DEFAULT_STANDOFF_START = 1.35
DEFAULT_STANDOFF_STOP = 1.35
DEFAULT_STANDOFF_POINTS = 1

# Physics substeps to hold after the toss so the cube comes to rest before `check_goals`.
DEFAULT_SETTLE_STEPS = 30

# Below this a parabola is underdetermined: the gripper never opened.
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

    # `False` is a third state, not a throw of zero distance.
    threw: bool | None = None

    # Distances are reported from the base, not in world x.
    base_x_before_toss: float | None = None
    bin_x: float | None = None

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

    # What stopped it, which is how a wall failure is told from a short throw.
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

    A point count, not a step: 20 points over `(60, 140)` steps by `80/19`. Rounded to 6
    decimals so JSON round-tripped values still compare equal as dict keys.
    """
    if points <= 1:
        return [round(start, 6)]
    span = stop - start
    return [round(start + i * span / (points - 1), 6) for i in range(points)]


def refinement_order(*, n: int) -> list[int]:
    """`range(n)` reordered endpoints-first, then repeatedly bisecting.

    Any prefix is a coarse sample of the whole range. A permutation: every row runs once.
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

    `launch_*` are the velocities at the first substep after the gripper let go. The
    crossing is extrapolated, not where the recording stopped. `None` when the window is too
    short to fit; `impact_*` unset when there is no real descending crossing.
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
    # crossing. The ascending one is the cube passing `ground_z` on the way up.
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
    # The cube's centre height at rest: the crossing is solved for the centre, not a corner.
    cell.cube_half_height = float(min(model.geom_size[g][2] for g in cube_geoms))
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
        # Slot two is `gripper_release_ms`; a literal `0.0` here is a release at time zero.
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
    # Straight off the simulator: the same body the flight was recorded from.
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
    # A gripper that never opened leaves the cube in contact throughout: no window to fit.
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

    The millisecond is the inner axis so one `Tossing3DEnvironment` amortises over a column.
    One positional tuple because `Pool.imap_unordered` passes exactly one argument.
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

    `kinder.make` auto-downloads ~2 GB, guarded only on whether `mimiclabs_scenes/meshes`
    exists, so N workers otherwise download into the same `assets.zip` and all fail to
    unpack. A no-op once the assets are present.
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
    # Refined over the speed axis, so a half-finished grid still spans it.
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
