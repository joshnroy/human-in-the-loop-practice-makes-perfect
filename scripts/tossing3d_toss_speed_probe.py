"""Drive `Toss` at a grid of commanded release speeds and record what the arm and the
cube actually did -- the rung-0 feasibility probe for a speed-parameterised toss.

## Why this script exists

`Toss` has `param_dim=0`, so there is nothing about the throw for a practice-based
method to improve; the domain compensates by back-calculating a "successful throw pose"
band from a single measured constant, `predicates.py:THROW_RANGE = 1.275`. The proposed
fix is to make the throw's own energy a continuous parameter. `TossController.reset`
sets that energy in one place -- three literals passed inline to
`_trapezoidal_motion_profile`:

    max_vel=np.deg2rad(140), max_accel=np.deg2rad(300), max_decel=np.deg2rad(200)

and the motion planner is *not* involved: `TossController` keeps only `plan[-1]` and
executes a straight line in joint space timed by that scalar profile.

Before any of that is built, three things have to be measured rather than derived, and
this script measures them:

  * **the ceiling** -- how fast a release the dial can actually command. Above some
    speed the release point moves out of the profile's cruise phase, and the commanded
    release speed stops tracking `max_vel`.
  * **the dial's authority in metres** -- the span of cube resting positions the whole
    parameter range reaches. If that span does not straddle the goal box from a
    realistic standoff, the parameter is unlearnable no matter how it is wired.
  * **release-pose invariance** -- whether the arm is in the same configuration when the
    gripper opens at every speed. The design argument for the parameter is that only the
    *magnitude* of the release velocity moves; if the release *pose* drifts, the release
    direction drifts with it and that argument fails.

## The three scaling modes, and why there are three

`--mode` picks which of the profile's limits the dial scales:

  * `vel` -- `max_vel` only. The design doc's option A.
  * `vel-accel` -- `max_vel` and `max_accel`, the design doc's option D as worded.
  * `vel-accel-decel` -- all three limits by the same factor.

They are not three flavours of the same thing. `_trapezoidal_motion_profile` goes
triangular once `0.5*v^2/a + 0.5*v^2/d` exceeds the path length, and the release fires at
a fixed *fraction of distance* (`_release_fraction = 0.46`). So a mode that scales `v`
faster than it scales `a` and `d` walks the release point backwards out of the cruise
phase and into the acceleration phase, where the speed is set by the acceleration limit
rather than by `max_vel`. Only scaling all three together keeps the profile's shape, and
therefore keeps the release point in the same phase. `commanded_release_speed_deg` below
answers this analytically, with no simulator, and the probe then checks the analytic
answer against a real arm.

In every mode the dial defaults to upstream's own literals, so `--speeds-deg 140` is
upstream's toss unchanged. That is what makes the grid comparable to every committed
Tossing3D number.

## Grid design

One cell is `Pick(oracle)` -> `MoveToThrowPose(standoff)` -> `move_arm_to_conf(windup)`
-> instrumented toss -> settle. `Pick` and `MoveToThrowPose` are both held at fixed
points -- the oracle grasp and `ORACLE_THROW_STANDOFF` -- for the reason
`tossing3d_skill_parameter_sweep.py` gives for pinning `Pick`: letting their variance in
would confound "did this speed work" with "did the grasp even land". The same scene
seeds are used at every speed, so a comparison across speeds is paired on scene.

The cube is given a **fixed settle budget**, identical across speeds, and read twice.
A faster throw has a shorter profile, so reading the cube the instant the controller
terminates would give a fast throw less time to land than a slow one and would show up
as a spurious range effect. The second, longer reading is what says whether the first one
had settled at all.

## How the dial is applied without an upstream change

`_trapezoidal_motion_profile` is monkeypatched in `parameterized_skills`'s *own module
namespace*. That is narrower than it sounds and it matters: `MoveArmToConfController`
reaches the same function through `utils._compute_per_joint_profile`, i.e. through
`utils`'s namespace, so the windup runs on upstream's unmodified per-joint profile and
only the swing is scaled. The probe therefore measures the speed dial and nothing else.

This is a probe, not the mechanism. The real change is an optional argument on
`TossController.reset` upstream; patching here is what lets the measurement happen
*before* that argument is designed.

## Running it

Needs the KINDER venv, this worktree's `src` on `PYTHONPATH`, and a memory cap -- a
planner grounds a fresh controller, hence a fresh PyBullet sim, per attempt:

    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \\
        -p OOMPolicy=continue -- \\
        scripts/with_kinder_env.sh python scripts/tossing3d_toss_speed_probe.py \\
        --mode vel-accel-decel --output docs/experiment-logs/<name>.json

`analysis/tossing3d_toss_speed_probe.py` reads the JSON back and draws the figure; this
script never plots.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, computed_field

# `scripts/` is not an installed package under the KINDER venv, so make `import hitl_pmp`
# resolve the same way `tossing3d_skill_parameter_sweep.py` does.
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

# The three literals `TossController.reset` passes inline to
# `_trapezoidal_motion_profile`. Copied from upstream rather than imported, because the
# point of every "the default is unchanged" assertion is to compare against a number that
# does not move when upstream's does -- a stale copy is a loud test failure, an import
# would silently track the drift it exists to catch.
UPSTREAM_TOSS_MAX_VEL_DEG = 140.0
UPSTREAM_TOSS_MAX_ACCEL_DEG = 300.0
UPSTREAM_TOSS_MAX_DECEL_DEG = 200.0

SCALING_MODES = ("vel", "vel-accel", "vel-accel-decel")

# `in-spec` is deliberately *not* a scaling mode. The three above all multiply upstream's
# literals, so each reproduces them exactly at 140 deg/s. This one replaces them with
# limits derived from KINDER's own declared per-joint ceilings, under which 140 is 1.68x
# too fast and gets clamped -- so it cannot satisfy that invariant and must not be
# parametrized alongside modes that do.
IN_SPEC_MODE = "in-spec"
ALL_MODES = (*SCALING_MODES, IN_SPEC_MODE)

# `_CONTROL_DT` in `kinder_models.dynamic3d.utils`, and `_release_fraction` in
# `TossController.__init__`. Both are needed by the analytic helper below, which runs
# without importing KINDER's heavier dependencies where it can.
CONTROL_DT = 0.1
RELEASE_FRACTION = 0.46

DEFAULT_SEEDS = tuple(range(10))
DEFAULT_SPEEDS_DEG = (60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 240.0)


def toss_profile_ceilings_deg() -> tuple[float, float]:
    """The `(max_path_rate, max_path_accel)` the toss may use, in deg/s and deg/s^2.

    Derived, never hardcoded, so it tracks `_ARM_MAX_VEL`/`_ARM_MAX_ACCEL` rather than
    going stale beside them. The toss is a straight line in joint space, so every joint
    moves `|toss_dir_j|` of one path rate and the admissible rate is
    `min_j(limit_j / |toss_dir_j|)` -- exactly what `_compute_per_joint_profile` computes
    for `MoveArmToConfController`, which `TossController` bypasses.

    Joint 6 binds: it carries 0.8399 of the toss direction against the arm's smallest
    velocity limit, giving 83.34 deg/s against the 140 that ships.
    """
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        TOSS_RELEASE_ARM_CONF,
        TOSS_WINDUP_ARM_CONF,
    )
    from kinder_models.dynamic3d.utils import _ARM_MAX_ACCEL, _ARM_MAX_VEL  # noqa: PLC2701

    direction = TOSS_RELEASE_ARM_CONF - TOSS_WINDUP_ARM_CONF
    direction = direction / float(np.linalg.norm(direction))
    moving = np.abs(direction) > 1e-6
    max_rate = float(np.rad2deg(np.min(_ARM_MAX_VEL[moving] / np.abs(direction)[moving])))
    max_accel = float(np.rad2deg(np.min(_ARM_MAX_ACCEL[moving] / np.abs(direction)[moving])))
    return (max_rate, max_accel)


def profile_limits_deg(*, release_speed_deg: float, mode: str) -> tuple[float, float, float]:
    """The `(max_vel, max_accel, max_decel)` triple, in deg/s and deg/s^2, that `mode`
    hands `_trapezoidal_motion_profile` for a commanded `release_speed_deg`.

    At `release_speed_deg == UPSTREAM_TOSS_MAX_VEL_DEG` every *scaling* mode returns
    upstream's own literals exactly, so the dial's default is upstream's toss in all
    three. `in-spec` is the exception and the point: it clamps to the derived ceiling,
    where 140 deg/s is 1.68x too fast.
    """
    if mode == IN_SPEC_MODE:
        max_rate, max_accel = toss_profile_ceilings_deg()
        # Deceleration equals acceleration, adopting `_compute_per_joint_profile`'s own
        # convention. That repairs the shipped profile's 1.120x decel asymmetry as a side
        # effect rather than as a separate change.
        return (min(release_speed_deg, max_rate), max_accel, max_accel)
    if mode not in SCALING_MODES:
        raise ValueError(f"unknown scaling mode {mode!r}; expected one of {ALL_MODES}")
    scale = release_speed_deg / UPSTREAM_TOSS_MAX_VEL_DEG
    accel_scale = scale if mode in ("vel-accel", "vel-accel-decel") else 1.0
    decel_scale = scale if mode == "vel-accel-decel" else 1.0
    return (
        release_speed_deg,
        UPSTREAM_TOSS_MAX_ACCEL_DEG * accel_scale,
        UPSTREAM_TOSS_MAX_DECEL_DEG * decel_scale,
    )


def commanded_release_speed_deg(*, release_speed_deg: float, mode: str) -> float:
    """The path speed the profile *commands* at the release step, in deg/s.

    Pure profile arithmetic against upstream's own `_trapezoidal_motion_profile` -- no
    physics, no PD controller, no simulator. This is the ceiling question answered
    analytically: it is the quantity that stops tracking `release_speed_deg` once the
    release point leaves the cruise phase.

    The path length is taken from the nominal windup and release configurations. The real
    controller measures it from the arm's *achieved* windup pose, so this is close to but
    not identical with what a run sees -- which is exactly why the probe measures rather
    than trusting it.
    """
    from kinder_models.dynamic3d.tossing.parameterized_skills import (
        TOSS_RELEASE_ARM_CONF,
        TOSS_WINDUP_ARM_CONF,
    )
    from kinder_models.dynamic3d.utils import _trapezoidal_motion_profile  # noqa: PLC2701

    max_vel, max_accel, max_decel = profile_limits_deg(
        release_speed_deg=release_speed_deg, mode=mode
    )
    total_dist = float(np.linalg.norm(TOSS_RELEASE_ARM_CONF - TOSS_WINDUP_ARM_CONF))
    trajectory = _trapezoidal_motion_profile(
        total_dist,
        max_vel=float(np.deg2rad(max_vel)),
        max_accel=float(np.deg2rad(max_accel)),
        max_decel=float(np.deg2rad(max_decel)),
        step_size=CONTROL_DT,
    )
    final = float(trajectory[-1])
    for idx in range(len(trajectory)):
        if final > 0 and trajectory[idx] / final >= RELEASE_FRACTION:
            if idx == 0:
                return 0.0
            return float(np.rad2deg((trajectory[idx] - trajectory[idx - 1]) / CONTROL_DT))
    return float("nan")


class ProbeCellResult(BaseModel):
    """One `(scene seed, commanded speed)` cell.

    Every field that a cell can genuinely fail to produce is `None` rather than zero. A
    cell whose grasp failed never released, so its release speed is *unknown*; recording
    it as `0.0` would put a fabricated point on the curve.
    """

    seed: int
    commanded_speed_deg: float
    mode: str
    standoff: float

    pick_error: str | None = None
    move_error: str | None = None
    windup_error: str | None = None
    windup_terminated: bool | None = None
    toss_error: str | None = None
    toss_terminated: bool | None = None
    toss_steps: int | None = None

    # The profile the controller actually built, read off the grounded controller.
    profile_steps: int | None = None
    s_total_rad: float | None = None

    # The release instant.
    release_step: int | None = None
    release_fraction_covered: float | None = None
    commanded_release_speed_deg: float | None = None
    # Two readings that bracket the launch, because the cube separates *during* the
    # release step rather than at either end of it. `..._deg` is the arm's speed when the
    # controller decides to open the gripper; `..._after_deg` is its speed once that step
    # has been simulated, by which time the step's own velocity feedforward
    # (`action[11:18] = toss_dir * ds * kv`) has been applied. Two scaling modes can share
    # the first reading exactly and still throw the cube different distances, because
    # their profiles differ in `ds` at that step -- so quoting only the pre-step reading
    # would make the range curve look inexplicable.
    achieved_release_speed_deg: float | None = None
    achieved_release_speed_after_deg: float | None = None
    commanded_release_conf_deg: list[float] | None = None
    achieved_release_conf_deg: list[float] | None = None
    base_x_at_release: float | None = None

    # Torque headroom over the whole swing -- the design doc's R4.
    max_torque_fraction: float | None = None
    torque_saturated_steps: int | None = None
    toss_control_steps: int | None = None

    # Where the cube came to rest, at two settle budgets.
    cube_x_at_first_settle: float | None = None
    solved_at_first_settle: bool | None = None
    cube_x_final: float | None = None
    cube_y_final: float | None = None
    cube_z_final: float | None = None
    base_x_final: float | None = None
    solved: bool | None = None
    goal_region: tuple[float, float, float, float, float, float] | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def range_m(self) -> float | None:
        """Cube resting x minus the base's x at release.

        Measured from the base rather than from the world origin: the base sits at
        x ~ 0.64 at the throw pose, so an origin-relative "range" would overstate every
        throw by that offset and wreck any fit against release speed.
        """
        if self.cube_x_final is None or self.base_x_at_release is None:
            return None
        return self.cube_x_final - self.base_x_at_release


def _parse_args(*, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=ALL_MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--speeds-deg", type=float, nargs="+", default=list(DEFAULT_SPEEDS_DEG))
    parser.add_argument("--standoff", type=float, default=ORACLE_THROW_STANDOFF)
    parser.add_argument("--settle-steps", type=int, default=25)
    parser.add_argument("--extra-settle-steps", type=int, default=35)
    parser.add_argument(
        "--record-video-dir",
        type=Path,
        default=None,
        help=(
            "record one captioned .mp4 per cell here instead of instrumenting the swing. "
            "See `record_cell` for why the two paths cannot be the same one."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment

    env = Tossing3DEnvironment()
    backend = env.backend()
    backend.api()  # forces the KINDER import and its EGL configuration

    import kinder_models.dynamic3d.tossing.parameterized_skills as ps

    print(f"kinder_models: {ps.__file__}", flush=True)

    original_profile = ps._trapezoidal_motion_profile  # noqa: SLF001
    commanded: dict[str, tuple[float, float, float]] = {
        "limits": (
            float(np.deg2rad(UPSTREAM_TOSS_MAX_VEL_DEG)),
            float(np.deg2rad(UPSTREAM_TOSS_MAX_ACCEL_DEG)),
            float(np.deg2rad(UPSTREAM_TOSS_MAX_DECEL_DEG)),
        )
    }

    # Positional, because it replaces a function upstream calls positionally --
    # the one place this project's keyword-only rule cannot reach.
    def patched_profile(  # noqa: PLR0917
        total_dist: float,
        max_vel: float,
        max_accel: float,
        max_decel: float,
        step_size: float,
    ) -> Any:
        del max_vel, max_accel, max_decel  # the whole point of the probe
        vel, accel, decel = commanded["limits"]
        return original_profile(
            total_dist,
            max_vel=vel,
            max_accel=accel,
            max_decel=decel,
            step_size=step_size,
        )

    ps._trapezoidal_motion_profile = patched_profile  # noqa: SLF001

    rows: list[ProbeCellResult] = []
    started = time.monotonic()
    total = len(args.speeds_deg) * len(args.seeds)
    for speed_deg in args.speeds_deg:
        vel, accel, decel = profile_limits_deg(release_speed_deg=speed_deg, mode=args.mode)
        commanded["limits"] = (
            float(np.deg2rad(vel)),
            float(np.deg2rad(accel)),
            float(np.deg2rad(decel)),
        )
        for seed in args.seeds:
            cell = run_cell if args.record_video_dir is None else record_cell
            row = cell(
                env=env,
                backend=backend,
                seed=int(seed),
                speed_deg=float(speed_deg),
                mode=str(args.mode),
                standoff=float(args.standoff),
                settle_steps=int(args.settle_steps),
                extra_settle_steps=int(args.extra_settle_steps),
                video_dir=args.record_video_dir,
            )
            rows.append(row)
            print(
                f"[{len(rows)}/{total}] mode={args.mode} speed={speed_deg:.0f} "
                f"seed={seed} release_step={row.release_step} "
                f"achieved={row.achieved_release_speed_deg} range={row.range_m} "
                f"solved={row.solved} ({time.monotonic() - started:.0f}s)",
                flush=True,
            )
            _write_checkpoint(rows=rows, output=args.output, mode=str(args.mode), args=args)

    print(f"wrote {args.output}", flush=True)


def _write_checkpoint(
    *, rows: list[ProbeCellResult], output: Path, mode: str, args: argparse.Namespace
) -> None:
    """Rewrite the whole JSON after every cell, so an interrupted run is still readable.

    The metadata block records what the run was measured *against*, not just what it
    measured. `kinder_models.__file__` is in there deliberately: this repo has already
    had a read-vs-run skew where the tree that was read and the tree that ran sat at
    different commits, and the resulting SHA was stated as fact.
    """
    import kinder_models

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "standoff": args.standoff,
        "seeds": list(args.seeds),
        "speeds_deg": list(args.speeds_deg),
        "settle_steps": args.settle_steps,
        "extra_settle_steps": args.extra_settle_steps,
        "upstream_toss_limits_deg": [
            UPSTREAM_TOSS_MAX_VEL_DEG,
            UPSTREAM_TOSS_MAX_ACCEL_DEG,
            UPSTREAM_TOSS_MAX_DECEL_DEG,
        ],
        "kinder_models_file": kinder_models.__file__,
        "pid": os.getpid(),
        "cells": [row.model_dump() for row in rows],
    }
    output.write_text(json.dumps(payload, indent=2))


def run_cell(
    *,
    env: Any,
    backend: Any,
    seed: int,
    speed_deg: float,
    mode: str,
    standoff: float,
    settle_steps: int,
    extra_settle_steps: int,
    video_dir: Path | None = None,
) -> ProbeCellResult:
    """Run one cell end to end and return what it measured.

    `video_dir` is accepted and ignored so that this and `record_cell` share one call
    shape; recording happens in `record_cell`.
    """
    del video_dir
    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment

    row = ProbeCellResult(seed=seed, commanded_speed_deg=speed_deg, mode=mode, standoff=standoff)

    env.reset_to_seed(seed=seed)
    env.take_action(
        action=np.array([Tossing3DEnvironment.pick_id, ORACLE_PICK_DISTANCE, ORACLE_PICK_ROTATION])
    )
    row.pick_error = env.last_skill_error()
    env.take_action(action=np.array([Tossing3DEnvironment.move_to_throw_pose_id, standoff, 0.0]))
    row.move_error = env.last_skill_error()

    windup = backend.run_controller(
        module="tossing",
        key="move_arm_to_conf",
        object_names=(backend.robot_name,),
        params=np.deg2rad(backend.windup_conf_deg),
        limit=backend.arm_step_limit,
    )
    row.windup_terminated = windup.terminated
    row.windup_error = windup.error
    if not windup.terminated:
        return row

    _instrumented_toss(backend=backend, row=row, limit=backend.arm_step_limit)

    gym_env = backend._env  # noqa: SLF001
    hold = np.zeros(11, dtype=np.float32)
    for budget, is_first in ((settle_steps, True), (extra_settle_steps, False)):
        for _ in range(budget):
            observation, _, _, _, _ = gym_env.step(hold)
            backend._state = gym_env.observation_space.devectorize(observation)  # noqa: SLF001
        state = backend._state  # noqa: SLF001
        cube = state.get_object_from_name(backend.cube_name)
        if is_first:
            row.cube_x_at_first_settle = float(state.get(cube, "x"))
            row.solved_at_first_settle = bool(backend.check_goals())

    state = backend._state  # noqa: SLF001
    cube = state.get_object_from_name(backend.cube_name)
    robot = state.get_object_from_name(backend.robot_name)
    row.cube_x_final = float(state.get(cube, "x"))
    row.cube_y_final = float(state.get(cube, "y"))
    row.cube_z_final = float(state.get(cube, "z"))
    row.base_x_final = float(state.get(robot, "pos_base_x"))
    row.solved = bool(backend.check_goals())
    row.goal_region = backend.goal_region_bbox()
    return row


def record_cell(
    *,
    env: Any,
    backend: Any,
    seed: int,
    speed_deg: float,
    mode: str,
    standoff: float,
    settle_steps: int,
    extra_settle_steps: int,
    video_dir: Path | None,
) -> ProbeCellResult:
    """Run one cell for the camera and write a captioned clip of the throw.

    **Why this is a second path rather than a flag inside `run_cell`.** `run_cell` drives
    the swing one control step at a time so it can read the arm's velocity and
    configuration at the exact step the gripper opens. That hand-driven loop is the
    measurement, and it must not acquire a rendering branch that could change what it
    measures. This path instead runs the same three skills through the ordinary public
    `env.take_action` route -- the same two controllers, the same monkeypatched profile --
    and records. It reports `cube_x_final` and `solved` so a clip can be checked against
    the grid cell it is supposed to illustrate; if those disagree, the clip is not showing
    what the table says and the disagreement is the finding.

    **Only the throw is recorded.** `Pick` and `MoveToThrowPose` are identical in every
    clip by construction -- both are held at fixed parameters -- so including them would
    add several seconds of identical driving to every file and bury the one second that
    differs. The clip therefore starts at the windup.
    """
    import imageio.v2 as imageio

    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
    from hitl_pmp.environments.tossing3d.renderer import Tossing3DRenderer

    assert video_dir is not None
    row = ProbeCellResult(seed=seed, commanded_speed_deg=speed_deg, mode=mode, standoff=standoff)

    backend.set_substep_recording(enabled=False)
    env.reset_to_seed(seed=seed)
    env.take_action(
        action=np.array([Tossing3DEnvironment.pick_id, ORACLE_PICK_DISTANCE, ORACLE_PICK_ROTATION])
    )
    row.pick_error = env.last_skill_error()
    env.take_action(action=np.array([Tossing3DEnvironment.move_to_throw_pose_id, standoff, 0.0]))
    row.move_error = env.last_skill_error()

    # Recording starts here, so the clip is the throw and not the approach.
    backend.set_substep_recording(enabled=True)
    backend.drain_substep_frames()  # discard anything buffered by the wrapper going on
    state = env.take_action(action=np.array([Tossing3DEnvironment.toss_id, 0.0, 0.0]))
    row.toss_error = env.last_skill_error()
    label = f"{mode} @ {speed_deg:.0f} deg/s | seed {seed} | standoff {standoff:.2f} m"
    frames = Tossing3DRenderer.render_substep_frames(
        frames=backend.drain_substep_frames(), state=state, env=env, label=f"{label} | throw"
    )

    # Settle on the same budget the measurement grid uses, so the clip ends where the
    # table's `range_m` was read rather than mid-bounce.
    gym_env = backend._env  # noqa: SLF001
    hold = np.zeros(11, dtype=np.float32)
    for _ in range(settle_steps + extra_settle_steps):
        observation, _, _, _, _ = gym_env.step(hold)
        backend._state = gym_env.observation_space.devectorize(observation)  # noqa: SLF001
    settled = env.build_state(
        observation=backend.observe(),
        seed=seed,
        steps_taken=int(round(state.get(obj=env.scene, feature_name="steps_taken"))),
    )
    row.solved = bool(backend.check_goals())
    verdict = "IN the goal box" if row.solved else "MISSED the goal box"
    frames += Tossing3DRenderer.render_substep_frames(
        frames=backend.drain_substep_frames(),
        state=settled,
        env=env,
        label=f"{label} | settling -- {verdict}",
    )
    backend.set_substep_recording(enabled=False)

    kinder_state = backend._state  # noqa: SLF001
    cube = kinder_state.get_object_from_name(backend.cube_name)
    robot = kinder_state.get_object_from_name(backend.robot_name)
    row.cube_x_final = float(kinder_state.get(cube, "x"))
    row.cube_y_final = float(kinder_state.get(cube, "y"))
    row.cube_z_final = float(kinder_state.get(cube, "z"))
    row.base_x_final = float(kinder_state.get(robot, "pos_base_x"))
    # No release instant is observed on this path, so `range_m` has no base-at-release to
    # subtract. The base does not move during a throw, so its final x is the same number.
    row.base_x_at_release = row.base_x_final
    row.goal_region = backend.goal_region_bbox()

    video_dir.mkdir(parents=True, exist_ok=True)
    path = video_dir / f"{mode}-{speed_deg:03.0f}deg-seed{seed}.mp4"
    imageio.mimsave(path, frames, fps=backend.render_fps(), macro_block_size=16)
    print(f"  wrote {path} ({len(frames)} frames)", flush=True)
    return row


def _instrumented_toss(*, backend: Any, row: ProbeCellResult, limit: int) -> None:
    """Drive `toss` one control step at a time, recording the release instant.

    `backend.run_toss()` cannot be used here: it runs the controller to completion, and
    the quantities this probe exists to measure -- the arm's achieved joint velocity and
    configuration *at the step the gripper opens* -- are gone by the time it returns.
    """
    api = backend.api()
    state = backend._state  # noqa: SLF001
    lifted = api.tossing_controllers(backend._env.action_space)  # noqa: SLF001
    controller = lifted["toss"].ground((state.get_object_from_name(backend.robot_name),))
    try:
        controller.reset(state, np.deg2rad(backend.toss_conf_deg))
    except Exception as exc:  # noqa: BLE001
        row.toss_error = f"{type(exc).__name__}: {exc}"
        return

    # Live views into MuJoCo's qvel/ctrl for the seven arm joints, set up by
    # `_setup_robot_references`. `sim.data` is a binding wrapper, not `mujoco.MjData`, so
    # `sim.data.qvel` does not exist -- these views are the supported path.
    robot_env = backend._raw_env.unwrapped._object_centric_env._robot_env  # noqa: SLF001
    arm_qvel = robot_env.qvel["arm"]
    arm_ctrl = robot_env.ctrl["arm"]
    torque_limits = np.array(type(robot_env).ARM_TORQUE_LIMITS, dtype=float)

    toss_dir = np.array(controller._toss_dir, dtype=float)  # noqa: SLF001
    trajectory = np.array(controller._trajectory, dtype=float)  # noqa: SLF001
    row.profile_steps = int(trajectory.size)
    row.s_total_rad = float(trajectory[-1]) if trajectory.size else 0.0

    max_torque_fraction = 0.0
    saturated_steps = 0
    control_steps = 0
    previously_released = False
    for step in range(limit):
        # Read the pre-step state: the controller decides on it, so this is the arm
        # configuration and velocity at the moment the gripper command is issued.
        conf_before = _arm_conf(state=backend._state, backend=backend)  # noqa: SLF001
        qvel_before = np.array(arm_qvel, dtype=float)
        idx = min(controller._step_idx, len(trajectory) - 1)  # noqa: SLF001
        s = float(trajectory[idx])
        ds = float((trajectory[idx] - trajectory[idx - 1]) / CONTROL_DT) if idx > 0 else 0.0

        try:
            action = controller.step()
            observation, _, _, _, _ = backend._env.step(action)  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            row.toss_error = f"{type(exc).__name__}: {exc}"
            break
        backend._state = backend._env.observation_space.devectorize(observation)  # noqa: SLF001
        controller.observe(backend._state)  # noqa: SLF001
        control_steps += 1

        qvel_after = np.array(arm_qvel, dtype=float)
        torque = np.array(arm_ctrl, dtype=float)
        max_torque_fraction = max(
            max_torque_fraction, float(np.max(np.abs(torque) / torque_limits))
        )
        saturated_steps += int(np.any(np.abs(torque) >= torque_limits - 1e-6))

        released_now = bool(controller._has_released) and not previously_released  # noqa: SLF001
        if released_now and row.release_step is None:
            row.release_step = step
            row.commanded_release_speed_deg = float(np.rad2deg(ds))
            row.achieved_release_speed_deg = float(np.rad2deg(np.dot(qvel_before, toss_dir)))
            row.achieved_release_speed_after_deg = float(np.rad2deg(np.dot(qvel_after, toss_dir)))
            row.achieved_release_conf_deg = np.rad2deg(conf_before).tolist()
            row.commanded_release_conf_deg = np.rad2deg(
                np.array(controller._start_joint_angles, dtype=float) + toss_dir * s  # noqa: SLF001
            ).tolist()
            row.release_fraction_covered = float(s / trajectory[-1]) if trajectory[-1] > 0 else 1.0
            robot = backend._state.get_object_from_name(backend.robot_name)  # noqa: SLF001
            row.base_x_at_release = float(backend._state.get(robot, "pos_base_x"))  # noqa: SLF001
        previously_released = bool(controller._has_released)  # noqa: SLF001
        if controller.terminated():
            row.toss_terminated = True
            break
    row.toss_steps = control_steps
    row.toss_control_steps = control_steps
    row.max_torque_fraction = max_torque_fraction
    row.torque_saturated_steps = saturated_steps
    if row.toss_terminated is None:
        row.toss_terminated = False


def _arm_conf(*, state: Any, backend: Any) -> Any:
    robot = state.get_object_from_name(backend.robot_name)
    return np.array([state.get(robot, f"pos_arm_joint{i}") for i in range(1, 8)], dtype=float)


if __name__ == "__main__":
    main()
