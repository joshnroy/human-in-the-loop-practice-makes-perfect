"""Measure Tossing3D's **impact** range as a function of the toss's release speed.

`predicates.THROW_RANGE` is the one calibrated constant the success band is derived from,
and `RobotAtSuccessfulThrowPose` currently answers "is this a pose a *140 deg/s* throw
scores from" while `Toss` may be executed anywhere in `TOSS_SPEED_BOUNDS`. Reformulating
that predicate as a union over the speed range needs the range as a *function* of speed,
which is what this probe measures.

## Why "first ground contact" is not directly measurable in this scene, and what is

The obvious instrument -- step the physics and record the first substep at which the cube
touches something -- does not measure what it appears to. On the coincident config the bin
sits **on** the goal region, and the bin is a catcher with 0.09 m half-height walls
(`bin_0` geoms, x half-extent 0.15, walls spanning z 0.02-0.20). A cube on a descending
parabola whose ground-crossing lies *beyond* the bin's far wall still hits that wall and
drops in. So:

  * first contact is with the **bin**, not the ground, for exactly the throws that score;
  * the contaminated cells are speed-dependent, because which throws reach the bin is what
    the speed dial changes. That is the worst possible confound for a speed sweep -- the
    measurement error would be a function of the independent variable.

Measured directly at 140 deg/s, seed 0, standoff 1.35: first contact is with a `bin_0`
geom at x = 1.9818, z = 0.0549, and the cube comes to rest at x = 2.0125 **inside the
bin**, never touching the floor at all.

What this probe measures instead is the **ballistic ground-crossing**: the cube is a free
body between release and its first contact, MuJoCo integrates it without drag, so its
flight is an exact parabola. Fitting that parabola over the contact-free window and
solving for the height at which a resting cube's centre sits (`GROUND_CUBE_CENTRE_Z`,
half the cube's 0.05 m edge above the floor) gives where the cube *would* first touch open
floor. That quantity is bin-independent by construction, which is the entire point: it is
the same number whether or not something catches the cube on the way down.

The parabola is fit over ~1000 substeps (0.5 s of flight at the 0.0005 s
`SIMULATION_TIMESTEP`), so it is heavily over-determined; the residual is reported per
cell so a bad fit is visible rather than silent.

## What each cell records, and why all four

`ballistic_impact_x` is the quantity above. `first_contact_x` is what actually happened,
with the body it hit, so the bin-interference story is evidence in the results rather than
an assertion in this docstring. `cube_x_final` is where it came to rest, so the
impact-to-rest offset -- the "roll" -- is a measured per-cell difference rather than an
assumed constant. `solved` is the ground truth the predicate ultimately answers to.

Ranges are reported **relative to the base's x**, not the world origin: the base sits at
x ~ 0.65 at the throw pose, so an origin-relative "range" would overstate every throw by
that offset and wreck any fit against release speed. This follows PR #213's convention so
the two grids are directly comparable.

Needs the KINDER venv and this worktree's pins -- use `scripts/with_kinder_env.sh`, which
exists precisely because the shared venv otherwise resolves `kinder_models` to the main
checkout. Every row records the `kinder_models.__file__` it actually ran against, so a
skew is visible in the results rather than trusted away. Wrap in a memory-capped scope:

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        scripts/with_kinder_env.sh python scripts/tossing3d_toss_impact_probe.py \
        --output results.json --max-workers 8

`analysis/tossing3d_toss_impact_probe.py` reads the JSON back and draws the figure; this
script never plots.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

# `scripts/` is not an installed package under the KINDER venv, so make `import hitl_pmp`
# resolve the same way the sibling probes do.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hitl_pmp.environments.tossing3d.skill_oracle_policy import (  # noqa: E402
    ORACLE_PICK_DISTANCE,
    ORACLE_PICK_ROTATION,
    ORACLE_THROW_STANDOFF,
)

# Half the cube's edge length. `cube_0`'s single box geom has `geom_size` (0.025, 0.025,
# 0.025), read from the live model, and the floor's top surface is z = 0: a cube resting
# on open floor therefore has its centre at exactly this height. This is the height the
# ballistic fit is solved for. Verified against a cube at rest in the bin, whose centre
# sits at 0.0444 = 0.0194 (bin floor top) + 0.025.
GROUND_CUBE_CENTRE_Z = 0.025

# `SIMULATION_TIMESTEP` in `kinder.envs.dynamic3d.mujoco_utils`. Copied rather than
# imported so this module's arithmetic can be tested without KINDER installed; the probe
# asserts the live model agrees before it trusts a single fit.
SIMULATION_TIMESTEP = 0.0005

# Ten fixed scene seeds, shared across every speed so comparisons across speeds are
# **paired on scene**. Never randomly drawn.
DEFAULT_SEEDS = tuple(range(10))

# The full shipped `TOSS_SPEED_BOUNDS`, stepped at 5 deg/s.
#
# **The step is 5 deg/s deliberately, and 3 points would be a repeat of a known mistake.**
# PR #213 fit `range = 0.004872*v + 0.6614` at R^2 = 0.99997 from a *three-point* grid
# (60/100/140) and concluded the dial was linear. PR #221 then stepped 5 deg/s over
# 60-83.34 and found the dial **significantly non-monotone**: 65 -> 70 deg/s lands 0.0329 m
# shorter, paired t = +4.51, p = 1.6e-05. A three-point grid cannot see that. The mechanism
# is release quantisation -- the gripper opens on the first control step past
# `fraction_covered >= 0.46`, once per 0.1 s -- and the swing gets *coarser* as speed
# rises (32 control steps at 60, 18 at 140, 14 at 240), so there is more reason to expect
# the artifact worse at the top of this range than milder.
DEFAULT_SPEED_START = 60.0
DEFAULT_SPEED_STOP = 240.0
DEFAULT_SPEED_STEP = 5.0


class ImpactCellResult(BaseModel):
    """One (speed, seed) cell: where the cube would land, where it hit, where it stopped."""

    seed: int
    commanded_speed_deg: float
    standoff: float

    pick_error: str | None = None
    move_error: str | None = None
    toss_error: str | None = None

    # Provenance: the tree that actually ran, not the tree that was read.
    kinder_models_file: str | None = None
    kindergarden_file: str | None = None

    base_x_before_toss: float | None = None
    base_x_after_toss: float | None = None

    # The contact-free window the parabola was fit over.
    free_flight_samples: int | None = None
    ballistic_fit_residual_m: float | None = None
    ballistic_impact_x: float | None = None
    ballistic_impact_t: float | None = None
    ballistic_impact_vx: float | None = None
    ballistic_impact_vz: float | None = None

    # What actually happened first, and to what.
    first_contact_x: float | None = None
    first_contact_z: float | None = None
    first_contact_body: str | None = None

    cube_x_final: float | None = None
    cube_y_final: float | None = None
    cube_z_final: float | None = None
    solved: bool | None = None
    goal_region: tuple[float, float, float, float, float, float] | None = None

    @property
    def ballistic_impact_range_m(self) -> float | None:
        """Ballistic ground-crossing x minus the base's x, in metres."""
        if self.ballistic_impact_x is None or self.base_x_before_toss is None:
            return None
        return self.ballistic_impact_x - self.base_x_before_toss

    @property
    def resting_range_m(self) -> float | None:
        """Cube resting x minus the base's x, in metres -- PR #213's quantity."""
        if self.cube_x_final is None or self.base_x_before_toss is None:
            return None
        return self.cube_x_final - self.base_x_before_toss


def ballistic_ground_crossing(
    *,
    times: np.ndarray,
    xs: np.ndarray,
    zs: np.ndarray,
    ground_z: float = GROUND_CUBE_CENTRE_Z,
) -> tuple[float, float, float, float, float]:
    """Fit the free-flight parabola and solve for where it descends through `ground_z`.

    Returns `(impact_x, impact_t, residual, impact_vx, impact_vz)`. `residual` is the max
    absolute error of the quadratic fit to `zs`, in metres -- a drag-free MuJoCo free body
    is an exact parabola, so anything above ~1e-6 means the window was not actually free
    flight and the caller should discard the cell rather than trust the number.

    The two velocity components are returned because the landing *point* turns out not to
    determine whether the bin catches the cube: the bin is a 0.09 m half-height catcher, so
    how steeply the cube arrives decides whether it clears the near wall and whether it is
    intercepted by the far one. A cell's arrival angle is therefore evidence, not colour.

    Kept pure and free of any KINDER import so it can be tested without a simulator.

    The descending root is taken specifically: a parabola crosses any height below its
    apex twice, and the ascending crossing is behind the throw rather than in front of it.
    """
    if times.size < 3:
        raise ValueError(f"need at least 3 free-flight samples to fit a parabola, got {times.size}")

    # Shift time to the window's start for conditioning; the fit is over t' = t - t0.
    t0 = float(times[0])
    rel = times - t0

    z_coeffs = np.polyfit(rel, zs, 2)
    x_coeffs = np.polyfit(rel, xs, 1)
    residual = float(np.max(np.abs(np.polyval(z_coeffs, rel) - zs)))

    a, b, c = (float(v) for v in z_coeffs)
    if a >= 0:
        raise ValueError(f"free-flight z is not a downward parabola (a={a}); not ballistic")

    disc = b * b - 4 * a * (c - ground_z)
    if disc < 0:
        raise ValueError("parabola never reaches the ground height")
    sqrt_disc = float(np.sqrt(disc))
    # `a < 0`, so the *larger* root is `(-b - sqrt)/2a`; that is the descending crossing.
    t_impact = (-b - sqrt_disc) / (2 * a)

    impact_x = float(np.polyval(x_coeffs, t_impact))
    impact_vx = float(x_coeffs[0])
    impact_vz = float(2 * a * t_impact + b)
    return impact_x, float(t_impact + t0), residual, impact_vx, impact_vz


def longest_contact_free_window(*, contacted: list[bool]) -> tuple[int, int]:
    """The `[start, stop)` index range of the longest run of contact-free samples.

    The cube is in contact with the gripper before release and with the bin or the floor
    after landing, so its free flight is the longest contact-free run in the recording.
    Taking the *longest* run rather than the first is what makes this robust to the brief
    contact-free moments that occur while the gripper is opening.
    """
    best_start, best_len = 0, 0
    run_start: int | None = None
    for i, touching in enumerate([*contacted, True]):
        if not touching:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_len > best_len:
                    best_start, best_len = run_start, run_len
                run_start = None
    return best_start, best_start + best_len


def _parse_args(*, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--speed-start", type=float, default=DEFAULT_SPEED_START)
    parser.add_argument("--speed-stop", type=float, default=DEFAULT_SPEED_STOP)
    parser.add_argument("--speed-step", type=float, default=DEFAULT_SPEED_STEP)
    parser.add_argument("--speeds", type=float, nargs="+", default=None)
    parser.add_argument("--standoff", type=float, default=ORACLE_THROW_STANDOFF)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args(argv)


def _speeds_from(*, args: argparse.Namespace) -> list[float]:
    if args.speeds is not None:
        return [float(v) for v in args.speeds]
    n = int(round((args.speed_stop - args.speed_start) / args.speed_step)) + 1
    return [float(args.speed_start + i * args.speed_step) for i in range(n)]


def _worker(chunk: list[tuple[int, float, float, int]]) -> list[dict[str, Any]]:  # noqa: PLR0917
    """Run a chunk of cells in one process, reusing one environment across all of them."""
    import kinder
    import kinder_models
    import mujoco

    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment

    env = Tossing3DEnvironment()
    rows: list[dict[str, Any]] = []
    for seed, speed, standoff, settle_steps in chunk:
        row = run_cell(
            env=env,
            seed=seed,
            speed_deg=speed,
            standoff=standoff,
            settle_steps=settle_steps,
            mujoco=mujoco,
            env_cls=Tossing3DEnvironment,
        )
        row.kinder_models_file = kinder_models.__file__
        row.kindergarden_file = kinder.__file__
        rows.append(row.model_dump())
        print(
            f"  seed={seed} speed={speed:.1f} "
            f"impact={row.ballistic_impact_range_m} rest={row.resting_range_m} "
            f"hit={row.first_contact_body} solved={row.solved}",
            flush=True,
        )
    env.close()
    return rows


def run_cell(
    *,
    env: Any,
    seed: int,
    speed_deg: float,
    standoff: float,
    settle_steps: int,
    mujoco: Any,
    env_cls: Any,
) -> ImpactCellResult:
    """Run one cell end to end, recording the cube at every physics substep of the throw."""
    row = ImpactCellResult(seed=seed, commanded_speed_deg=speed_deg, standoff=standoff)

    env.reset_to_seed(seed=seed)
    backend = env.backend()
    backend.api()

    raw = backend._raw_env  # noqa: SLF001
    robot_env = raw.unwrapped._object_centric_env._robot_env  # noqa: SLF001
    sim = robot_env.sim
    model = sim.model.mj_model
    mj_data = sim.data.mj_data

    if abs(float(model.opt.timestep) - SIMULATION_TIMESTEP) > 1e-12:
        raise RuntimeError(
            f"physics timestep is {model.opt.timestep}, not the {SIMULATION_TIMESTEP} this "
            "probe's flight arithmetic assumes"
        )

    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, backend.cube_name)
    cube_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == cube_body}

    env.take_action(action=np.array([env_cls.pick_id, ORACLE_PICK_DISTANCE, ORACLE_PICK_ROTATION]))
    row.pick_error = env.last_skill_error()
    env.take_action(action=np.array([env_cls.move_to_throw_pose_id, standoff, 0.0]))
    row.move_error = env.last_skill_error()

    state = env.get_current_state()
    row.base_x_before_toss = float(state.get(obj=env_cls.robot, feature_name="pos_base_x"))

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
            partner: int | None = None
            for c in range(mj_data.ncon):
                con = mj_data.contact[c]
                g1, g2 = int(con.geom1), int(con.geom2)
                if g1 in cube_geoms:
                    partner = g2
                    break
                if g2 in cube_geoms:
                    partner = g1
                    break
            centre = mj_data.xpos[cube_body]
            times.append(float(mj_data.time))
            pos_x.append(float(centre[0]))
            pos_z.append(float(centre[2]))
            contacted.append(partner is not None)
            partners.append(partner)
        return out

    sim.step = recording_step
    try:
        recording["on"] = True
        env.take_action(action=np.array([env_cls.toss_id, speed_deg, 0.0]))
        row.toss_error = env.last_skill_error()
        # Let the cube settle, still recording, so first contact and rest are one series.
        hold = np.zeros(11, dtype=np.float32)
        gym_env = backend._env  # noqa: SLF001
        for _ in range(settle_steps):
            observation, _, _, _, _ = gym_env.step(hold)
            backend._state = gym_env.observation_space.devectorize(observation)  # noqa: SLF001
    finally:
        recording["on"] = False
        sim.step = original_step

    state = env.get_current_state()
    row.base_x_after_toss = float(state.get(obj=env_cls.robot, feature_name="pos_base_x"))

    if not times:
        return row

    start, stop = longest_contact_free_window(contacted=contacted)
    row.free_flight_samples = stop - start
    if stop - start >= 3:
        try:
            impact_x, impact_t, residual, impact_vx, impact_vz = ballistic_ground_crossing(
                times=np.array(times[start:stop]),
                xs=np.array(pos_x[start:stop]),
                zs=np.array(pos_z[start:stop]),
            )
            row.ballistic_impact_x = impact_x
            row.ballistic_impact_t = impact_t
            row.ballistic_fit_residual_m = residual
            row.ballistic_impact_vx = impact_vx
            row.ballistic_impact_vz = impact_vz
        except ValueError as exc:
            row.toss_error = f"{row.toss_error or ''} ballistic-fit: {exc}".strip()

    for i in range(stop, len(contacted)):
        if contacted[i]:
            row.first_contact_x = pos_x[i]
            row.first_contact_z = pos_z[i]
            geom = partners[i]
            if geom is not None:
                body_id = int(model.geom_bodyid[geom])
                row.first_contact_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            break

    cube_state = env.get_current_state()
    cube = env_cls.cube
    row.cube_x_final = float(cube_state.get(obj=cube, feature_name="x"))
    row.cube_y_final = float(cube_state.get(obj=cube, feature_name="y"))
    row.cube_z_final = float(cube_state.get(obj=cube, feature_name="z"))
    row.solved = bool(backend.check_goals())
    row.goal_region = backend.goal_region_bbox()
    return row


def main() -> None:
    args = _parse_args()
    speeds = _speeds_from(args=args)
    cells = [
        (seed, speed, args.standoff, args.settle_steps) for speed in speeds for seed in args.seeds
    ]
    print(
        f"{len(cells)} cells: {len(speeds)} speeds x {len(args.seeds)} seeds "
        f"at standoff {args.standoff}, {args.max_workers} workers",
        flush=True,
    )

    workers = max(1, min(args.max_workers, len(cells)))
    chunks: list[list[tuple[int, float, float, int]]] = [[] for _ in range(workers)]
    for i, cell in enumerate(cells):
        chunks[i % workers].append(cell)

    started = time.monotonic()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        results = pool.map(_worker, chunks)
    rows = [row for chunk in results for row in chunk]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "standoff": args.standoff,
                "speeds": speeds,
                "seeds": list(args.seeds),
                "settle_steps": args.settle_steps,
                "max_workers": args.max_workers,
                "cpu_count": os.cpu_count(),
                "elapsed_seconds": time.monotonic() - started,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"wrote {len(rows)} rows to {args.output} in {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
