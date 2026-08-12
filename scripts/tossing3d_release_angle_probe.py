"""Measure the toss's **release angle** as a function of commanded release speed.

PR #226 measured where the cube lands across 60-240 deg/s and found the dial significantly
non-monotone -- `4/5` tested steps reverse. The mechanism it names is **release
quantisation**: `TossController` opens the gripper on the first control step at which
`fraction_covered >= self._release_fraction` (0.46), and control steps come once per
`_CONTROL_DT = 0.1` s. Raising the commanded speed shortens the swing, so the profile is
sampled at *coarser* fractions of the path and the step the release actually lands on
jumps. This probe measures the angle that quantisation produces.

## What "release angle" means here, and what it deliberately is not

The quantity is the **launch elevation of the gripper**: `atan2(v_z, v_x)` of `J(q_r) . d`,
where `d` is the toss path's unit direction in joint space, `q_r` is the arm configuration
at the control step the gripper actually opened on, and `J` is the world-frame translational
Jacobian at `robot_pinch_site` -- MuJoCo's own site for the point between the fingers.

Three things that choice pins down, each of which is a way to get this wrong:

  * **the realised release configuration, not the nominal one.** `_release_fraction` is
    0.46, but the fraction the controller actually releases at is whatever the first
    profile sample past 0.46 happens to be, and that is what determines the arm's pose.
    Evaluating at 0.46 would measure a throw the robot never makes;
  * **the pinch site, not a link frame.** The site is where the cube is held, and it is
    ~0.18 m past the last arm link. A Jacobian at `robot_bracelet_link` answers a
    different question with a plausible-looking number;
  * **the arm's 7 columns only.** The base is stationary through the swing, so the base
    dofs contribute nothing and including them would let base drift leak in.

This is a **kinematic** quantity -- where the gripper is heading at release. It is *not*
the cube's actual launch velocity: the controller tracks its profile through a kp/kv servo
with real lag, and the cube leaves with whatever the gripper hands it. Those two can differ,
so the probe measures the cube's launch velocity as well, at a few speeds, rather than
asserting they agree. See `--validation-speeds`.

It is also *not* the **arrival** angle PR #226 reports (74.1 deg at 185, 64.7 deg at 190).
That is the angle the cube comes *down* at, after a whole parabola; this is the angle it
goes *up* at. Conflating them is easy and they behave differently.

## Why this is not a re-run of PR #226's grid

The toss path's geometry -- the windup configuration it starts from, the joint-space
direction `d`, and the path length `s_total` -- is set in `TossController.reset` **before**
`release_speed` is used for anything. `release_speed` enters only through
`toss_profile_limits`, which scales the trapezoidal profile's three limits. So the geometry
is *speed-independent*, and one execution per seed yields the release configuration at
every speed on the dial by re-timing the same path. 370 cells of angle come out of 10 seeds
of simulation, not 370.

Wrap in a memory-capped scope, and use `scripts/with_kinder_env.sh` so `kinder_models`
resolves to this worktree rather than the main checkout:

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        scripts/with_kinder_env.sh python scripts/tossing3d_release_angle_probe.py \
        --output results.json --max-workers 10

`analysis/tossing3d_range_and_release_angle.py` reads this JSON back alongside PR #226's
range grid and draws the figure; this script never plots.
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
# `src/` and the two `reference/` source roots, ahead of everything else. The KINDER venv
# installs `kindergarden` and `kinder-models` **editable**, so their `.pth` files carry the
# *main checkout's* absolute paths -- from a worktree, `import kinder_models` otherwise
# resolves to whatever commit that checkout happens to be sitting on rather than to this
# branch's pin, and nothing errors, because both trees export the same module names.
#
# That is not a hypothetical: on 2026-08-12 the main checkout was on `3524010` while this
# branch pins `1b564a1`, and the difference is exactly the release-speed parameter this
# probe sweeps (`grep -c release_speed`: 6 in one tree, 0 in the other). A run that picked
# up the main checkout's copy would have measured the *unparameterised* toss at a single
# speed and looked entirely normal doing it. `assert_kinder_pins` below turns that from a
# silent wrong answer into a refusal to start.
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

# MuJoCo's own site for the point between the fingers -- where the cube is actually held.
PINCH_SITE = "robot_pinch_site"

# `robot_joint_1` .. `robot_joint_7`. Resolved by name at run time rather than hardcoded as
# addresses; the probe asserts it found all seven.
ARM_JOINT_NAMES = tuple(f"robot_joint_{i}" for i in range(1, 8))

# Ten fixed scene seeds. Never randomly drawn.
DEFAULT_SEEDS = tuple(range(10))

# The same 60-240 deg/s at 5 deg/s that PR #226 stepped, so the two figures line up.
DEFAULT_SPEED_START = 60.0
DEFAULT_SPEED_STOP = 240.0
DEFAULT_SPEED_STEP = 5.0

# Speeds at which the toss is *actually executed* so the kinematic angle can be checked
# against the cube's measured launch velocity. Chosen as the three PR #226 makes the
# non-monotonicity argument from: 140 is the default, and 185/190 is the reversal where
# 0/10 becomes 10/10.
DEFAULT_VALIDATION_SPEEDS = (140.0, 185.0, 190.0)

# `_release_fraction` on `TossController`, and `_CONTROL_DT` in kinder-models' utils.
# Copied so this module's arithmetic is testable without KINDER; the probe asserts the
# live class agrees before trusting a number.
RELEASE_FRACTION = 0.46
CONTROL_DT = 0.1


class ReleaseAngleCell(BaseModel):
    """One (seed, speed) cell: where the gripper was heading when it let go."""

    seed: int
    commanded_speed_deg: float
    standoff: float

    # Geometry of the toss path, captured once per seed and identical across speeds.
    s_total: float | None = None
    trajectory_end: float | None = None
    n_control_steps: int | None = None

    # The quantisation itself.
    release_step_index: int | None = None
    realised_release_fraction: float | None = None

    # The measurement.
    release_elevation_deg: float | None = None
    release_speed_mps: float | None = None

    # Physics cross-check, present only at --validation-speeds.
    executed: bool = False
    cube_launch_elevation_deg: float | None = None
    cube_launch_speed_mps: float | None = None
    cube_launch_vx: float | None = None
    cube_launch_vz: float | None = None
    free_flight_samples: int | None = None
    ballistic_fit_residual_m: float | None = None
    solved: bool | None = None

    pick_error: str | None = None
    move_error: str | None = None
    toss_error: str | None = None

    # Provenance: the tree that actually ran, not the tree that was read.
    kinder_models_file: str | None = None
    kindergarden_file: str | None = None


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


def assert_kinder_pins(*, kinder_models: Any, toss_controller: Any) -> None:
    """Refuse to run unless KINDER resolved inside this checkout, at a pin that has the dial.

    Two independent failure modes, both of which otherwise produce a full grid of
    plausible numbers rather than an error:

      * `kinder_models` resolving to a *different* checkout (see the `sys.path` note at the
        top of this module);
      * that checkout being on a pin where `TossController.reset` has no `release_speed`
        parameter, so every cell silently runs the default 140 deg/s toss and the "speed
        sweep" is 37 copies of one throw.

    The second check is on the signature rather than the pin SHA on purpose: it tests the
    capability actually depended on, so it keeps working when the pin moves.
    """
    resolved = Path(kinder_models.__file__).resolve()
    if _REPO_ROOT not in resolved.parents:
        raise RuntimeError(
            f"kinder_models resolved to {resolved}, which is outside this checkout "
            f"({_REPO_ROOT}). Populate reference/kinder-baselines here, or the grid would "
            "be measured against another checkout's pin. See the sys.path note in this file."
        )
    if "release_speed" not in inspect.signature(toss_controller.reset).parameters:
        raise RuntimeError(
            "TossController.reset has no release_speed parameter, so every cell would run "
            "the default toss. reference/kinder-baselines is on a pin older than 1b564a1."
        )


def release_index_and_fraction(*, trajectory: np.ndarray) -> tuple[int, float, float]:
    """The control step the gripper opens on, and the path fraction it opens at.

    Returns `(release_index, realised_fraction, trajectory_end)`. Pure, so it is testable
    without KINDER, and it is the whole quantisation rule in one place.

    Reproduces `TossController.step` exactly, including the detail that is easy to get
    wrong: the denominator is the profile's **last sample** (`self._trajectory[-1]`), not
    the `total_dist` handed to the profile. The two differ because the profile's time grid
    is `arange(0, duration + step, step)` and so overshoots the motion's duration -- and
    using `total_dist` instead shifts the realised fraction enough to move which step the
    release lands on.
    """
    end = float(trajectory[-1])
    fractions = trajectory / end if end > 0 else np.ones_like(trajectory)
    index = int(np.argmax(fractions >= RELEASE_FRACTION))
    return index, float(fractions[index]), end


def trapezoidal_release(
    *, s_total: float, release_speed_rad: float
) -> tuple[int, float, float, int]:
    """`release_index_and_fraction` for the profile a toss at `release_speed_rad` is timed by."""
    from kinder_models.dynamic3d.tossing.parameterized_skills import toss_profile_limits
    from kinder_models.dynamic3d.utils import _trapezoidal_motion_profile

    max_vel, max_accel, max_decel = toss_profile_limits(release_speed_rad)
    trajectory = _trapezoidal_motion_profile(
        s_total,
        max_vel=max_vel,
        max_accel=max_accel,
        max_decel=max_decel,
        step_size=CONTROL_DT,
    )
    index, fraction, end = release_index_and_fraction(trajectory=trajectory)
    return index, fraction, end, len(trajectory)


def elevation_from_jacobian(
    *,
    model: Any,
    data: Any,
    mujoco: Any,
    qpos_at_release: np.ndarray,
    arm_qpos_adr: np.ndarray,
    arm_dof_adr: np.ndarray,
    site_id: int,
    toss_dir: np.ndarray,
) -> tuple[float, float]:
    """`atan2(v_z, v_x)` in degrees, and `|v|`, for the pinch site moving along `toss_dir`.

    `qpos_at_release` is the full configuration -- base included, at the throw pose -- with
    the seven arm joints already set to the release configuration. The Jacobian is taken in
    the world frame, so the base's yaw matters and is therefore the real one rather than a
    canonical pose.
    """
    saved = data.qpos.copy()
    saved_vel = data.qvel.copy()
    try:
        data.qpos[:] = qpos_at_release
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        velocity = jacp[:, arm_dof_adr] @ toss_dir
    finally:
        data.qpos[:] = saved
        data.qvel[:] = saved_vel
        mujoco.mj_forward(model, data)
    elevation = float(np.degrees(np.arctan2(velocity[2], velocity[0])))
    return elevation, float(np.linalg.norm(velocity))


def _parse_args(*, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--speed-start", type=float, default=DEFAULT_SPEED_START)
    parser.add_argument("--speed-stop", type=float, default=DEFAULT_SPEED_STOP)
    parser.add_argument("--speed-step", type=float, default=DEFAULT_SPEED_STEP)
    parser.add_argument(
        "--validation-speeds", type=float, nargs="+", default=list(DEFAULT_VALIDATION_SPEEDS)
    )
    parser.add_argument("--standoff", type=float, default=ORACLE_THROW_STANDOFF)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--max-workers", type=int, default=10)
    return parser.parse_args(argv)


def _speeds_from(*, args: argparse.Namespace) -> list[float]:
    n = int(round((args.speed_stop - args.speed_start) / args.speed_step)) + 1
    return [float(args.speed_start + i * args.speed_step) for i in range(n)]


def _worker(job: tuple[int, list[float], list[float], float, int]) -> list[dict[str, Any]]:  # noqa: PLR0917
    """One seed, start to finish, in its own process.

    Takes its arguments as one positional tuple because `Pool.map` passes exactly one
    positional argument; the project's keyword-only rule cannot apply to a callable the
    multiprocessing API owns the calling convention for.
    """
    import kinder
    import kinder_models
    import mujoco
    from kinder_models.dynamic3d.tossing.parameterized_skills import TossController

    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment

    assert_kinder_pins(kinder_models=kinder_models, toss_controller=TossController)

    seed, speeds, validation_speeds, standoff, settle_steps = job
    env = Tossing3DEnvironment()
    try:
        rows = run_seed(
            env=env,
            env_cls=Tossing3DEnvironment,
            mujoco=mujoco,
            seed=seed,
            speeds=speeds,
            validation_speeds=validation_speeds,
            standoff=standoff,
            settle_steps=settle_steps,
        )
    finally:
        env.close()
    for row in rows:
        row.kinder_models_file = kinder_models.__file__
        row.kindergarden_file = kinder.__file__
    nominal = next(
        (r.release_elevation_deg for r in rows if r.commanded_speed_deg == 140.0), float("nan")
    )
    print(f"  seed={seed} s_total={rows[0].s_total:.6f} elev@140={nominal:.2f}", flush=True)
    return [r.model_dump() for r in rows]


def execute_toss(  # noqa: PLR0917
    *,
    env: Any,
    env_cls: Any,
    mujoco: Any,
    seed: int,
    speed: float,
    standoff: float,
    settle_steps: int,
    captured: dict[str, Any],
    toss_controller: Any,
    recording_reset: Any,
    original_reset: Any,
) -> tuple[dict[str, Any], Any, Any]:
    """Run pick -> move -> toss once, recording the cube at every physics substep.

    A function rather than a loop body so the two closures below capture *parameters*
    rather than a caller's loop variables -- the same code inlined in a `for` loop is a
    live `B023`, and a closure that reads the wrong iteration's simulator handle would
    record one throw's contacts against another throw's geometry.

    Returns the cell's measurements plus the live `(model, data)`, which the caller needs
    afterwards to evaluate the Jacobian.
    """
    env.reset_to_seed(seed=seed)
    backend = env.backend()
    backend.api()
    raw = backend._raw_env  # noqa: SLF001
    robot_env = raw.unwrapped._object_centric_env._robot_env  # noqa: SLF001
    sim = robot_env.sim
    model = sim.model.mj_model
    data = sim.data.mj_data
    captured["_live_qpos"] = lambda: data.qpos

    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, backend.cube_name)
    cube_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == cube_body}

    entry: dict[str, Any] = {}
    env.take_action(action=np.array([env_cls.pick_id, ORACLE_PICK_DISTANCE, ORACLE_PICK_ROTATION]))
    entry["pick_error"] = env.last_skill_error()
    env.take_action(action=np.array([env_cls.move_to_throw_pose_id, standoff, 0.0]))
    entry["move_error"] = env.last_skill_error()

    times: list[float] = []
    pos_x: list[float] = []
    pos_z: list[float] = []
    contacted: list[bool] = []

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
        return out

    toss_controller.reset = recording_reset
    sim.step = recording_step
    try:
        recording["on"] = True
        env.take_action(action=np.array([env_cls.toss_id, speed, 0.0]))
        entry["toss_error"] = env.last_skill_error()
        hold = np.zeros(11, dtype=np.float32)
        gym_env = backend._env  # noqa: SLF001
        for _ in range(settle_steps):
            observation, _, _, _, _ = gym_env.step(hold)
            backend._state = gym_env.observation_space.devectorize(observation)  # noqa: SLF001
    finally:
        recording["on"] = False
        sim.step = original_step
        toss_controller.reset = original_reset

    entry["solved"] = bool(backend.check_goals())
    # The cube's launch velocity: the free-flight parabola's velocity at the window's first
    # sample, which is the first substep after the gripper let go.
    start, stop = longest_contact_free_window(contacted=contacted)
    entry["free_flight_samples"] = stop - start
    if stop - start >= 3:
        rel = np.array(times[start:stop]) - times[start]
        z_coeffs = np.polyfit(rel, np.array(pos_z[start:stop]), 2)
        x_coeffs = np.polyfit(rel, np.array(pos_x[start:stop]), 1)
        entry["ballistic_fit_residual_m"] = float(
            np.max(np.abs(np.polyval(z_coeffs, rel) - np.array(pos_z[start:stop])))
        )
        launch_vx = float(x_coeffs[0])
        launch_vz = float(z_coeffs[1])
        entry["cube_launch_vx"] = launch_vx
        entry["cube_launch_vz"] = launch_vz
        entry["cube_launch_elevation_deg"] = float(np.degrees(np.arctan2(launch_vz, launch_vx)))
        entry["cube_launch_speed_mps"] = float(np.hypot(launch_vx, launch_vz))
    return entry, model, data


def run_seed(  # noqa: PLR0917
    *,
    env: Any,
    env_cls: Any,
    mujoco: Any,
    seed: int,
    speeds: list[float],
    validation_speeds: list[float],
    standoff: float,
    settle_steps: int,
) -> list[ReleaseAngleCell]:
    """Execute the toss once per validation speed, then re-time the path at every speed."""
    from kinder_models.dynamic3d.tossing.parameterized_skills import TossController

    captured: dict[str, Any] = {}
    original_reset = TossController.reset

    # Positional, because it replaces `TossController.reset` and must match the signature
    # upstream's own callers use.
    def recording_reset(self: Any, x: Any, params: Any, **kwargs: Any) -> None:  # noqa: PLR0917
        original_reset(self, x, params, **kwargs)
        # Assert the live class agrees with the constant this probe re-derives the
        # release step from. A drift here would silently move every reported angle.
        if abs(float(self._release_fraction) - RELEASE_FRACTION) > 1e-12:  # noqa: SLF001
            raise RuntimeError(
                f"TossController._release_fraction is {self._release_fraction}, not the "  # noqa: SLF001
                f"{RELEASE_FRACTION} this probe's arithmetic assumes"
            )
        captured["start_conf"] = np.array(self._start_joint_angles, dtype=float).copy()  # noqa: SLF001
        captured["toss_dir"] = np.array(self._toss_dir, dtype=float).copy()  # noqa: SLF001
        captured["target"] = np.array(  # noqa: SLF001
            self._current_arm_joint_plan[-1][:7], dtype=float
        ).copy()
        captured["qpos_at_reset"] = captured["_live_qpos"]().copy()

    rows: list[ReleaseAngleCell] = []
    physics: dict[float, dict[str, Any]] = {}

    for speed in validation_speeds:
        entry, model, data = execute_toss(
            env=env,
            env_cls=env_cls,
            mujoco=mujoco,
            seed=seed,
            speed=speed,
            standoff=standoff,
            settle_steps=settle_steps,
            captured=captured,
            toss_controller=TossController,
            recording_reset=recording_reset,
            original_reset=original_reset,
        )
        physics[speed] = entry

    # The path geometry is speed-independent, so one capture re-times to every speed.
    s_total = float(np.linalg.norm(captured["target"] - captured["start_conf"]))
    toss_dir = captured["toss_dir"]
    qpos_at_reset = captured["qpos_at_reset"]

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, PINCH_SITE)
    if site_id < 0:
        raise RuntimeError(f"the model has no site named {PINCH_SITE!r}")
    arm_qpos_adr = []
    arm_dof_adr = []
    for name in ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"the model has no joint named {name!r}")
        arm_qpos_adr.append(int(model.jnt_qposadr[jid]))
        arm_dof_adr.append(int(model.jnt_dofadr[jid]))
    arm_qpos_adr = np.array(arm_qpos_adr)
    arm_dof_adr = np.array(arm_dof_adr)

    for speed in speeds:
        row = ReleaseAngleCell(seed=seed, commanded_speed_deg=speed, standoff=standoff)
        index, fraction, end, n_steps = trapezoidal_release(
            s_total=s_total, release_speed_rad=float(np.deg2rad(speed))
        )
        row.s_total = s_total
        row.trajectory_end = end
        row.n_control_steps = n_steps
        row.release_step_index = index
        row.realised_release_fraction = fraction

        q_release = captured["start_conf"] + toss_dir * (fraction * end)
        qpos = np.array(qpos_at_reset, dtype=float).copy()
        qpos[arm_qpos_adr] = q_release
        elevation, magnitude = elevation_from_jacobian(
            model=model,
            data=data,
            mujoco=mujoco,
            qpos_at_release=qpos,
            arm_qpos_adr=arm_qpos_adr,
            arm_dof_adr=arm_dof_adr,
            site_id=site_id,
            toss_dir=toss_dir,
        )
        row.release_elevation_deg = elevation
        # Scale the unit-direction Jacobian velocity by the profile's speed at release.
        row.release_speed_mps = magnitude * float(np.deg2rad(speed))

        if speed in physics:
            entry = physics[speed]
            row.executed = True
            for key, value in entry.items():
                setattr(row, key, value)
        rows.append(row)
    return rows


def main() -> None:
    args = _parse_args()
    speeds = _speeds_from(args=args)
    validation = [s for s in args.validation_speeds if s in speeds]
    jobs = [(seed, speeds, validation, args.standoff, args.settle_steps) for seed in args.seeds]
    started = time.monotonic()
    with mp.get_context("spawn").Pool(processes=min(args.max_workers, len(jobs))) as pool:
        chunks = pool.map(_worker, jobs)
    rows = [row for chunk in chunks for row in chunk]
    elapsed = time.monotonic() - started

    payload = {
        "standoff": args.standoff,
        "speeds": speeds,
        "seeds": list(args.seeds),
        "validation_speeds": validation,
        "settle_steps": args.settle_steps,
        "release_fraction": RELEASE_FRACTION,
        "control_dt": CONTROL_DT,
        "pinch_site": PINCH_SITE,
        "max_workers": args.max_workers,
        "elapsed_seconds": elapsed,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output}: {len(rows)} cells in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
