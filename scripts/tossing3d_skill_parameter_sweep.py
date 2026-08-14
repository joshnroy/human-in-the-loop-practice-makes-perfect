"""Grid `Pick` and `MoveToThrowPose` over their *entire* sampling parameter space and
label every cell with the real classifier EES trains against, not a hand-rolled check.

## Why this script exists, and what it is not

Every number this session had on record before this script ran was one of: a coarse
bin, a handful of spot-checks, or pooled EES-practice data -- which mixes parameters
non-uniformly, since it is whatever the sampler happened to draw rather than a
controlled grid. `Pick`'s own non-stationarity (`rotation=0.65` fixed measured 5/30
across 30 scene seeds) and `MoveToThrowPose`'s edge effects around the tightened
`RobotAtSuccessfulThrowPose` band (`predicates.py`, `[1.150, 1.375]`) were both found
that way. This runs the dense, controlled sweep instead: every `(param, seed)` cell is
visited once, labelled by the same classifier `SkillOraclePolicy`/EES read, and written
to JSON so `analysis/practice_makes_perfect/tossing3d_skill_parameter_sweep.py` can
regenerate the figures without touching the simulator again.

## The one genuinely new thing this script does: imports `hitl_pmp` while driving KINDER

Every other simulator-driving script here (`tossing3d_oracle_demo.py`) deliberately
stays self-contained -- it re-derives its own success check
(`_check_goals()`/`RestingPlace`) rather than importing this project's package. That
habit dates from when KINDER lived in a *separate venv* from `hitl-pmp`, so importing
this package under the simulator's interpreter was not something to rely on. The two
environments are now unified (KINDER installs into `hitl-pmp` as the `tossing3d` extra),
so the constraint is gone -- but the self-containment is still the right default for a
script whose question does not need this package's own classifiers.

This script needs the real thing instead: the task this sweep exists for is "does a
cell's outcome match the labels `Holding`/`RobotAtSuccessfulThrowPose` tag EES's
training data with", so re-deriving an equivalent check would
not answer that question, it would answer a different one. `KinderBackend` (the only
module in `hitl_pmp` that imports KINDER) imports it lazily, so nothing MuJoCo-shaped
loads until the first `env.reset_to_seed(...)`, which happens after this module's own
imports -- so importing `hitl_pmp.environments.tossing3d.*` costs nothing beyond
`pydantic`/`numpy`/`gymnasium`. The `sys.path` bootstrap below is retained for the case
where this runs under an interpreter that never `pip install -e`'d this package.

Run with this worktree's `src` on `PYTHONPATH` (`with_kinder_env.sh` sets it; the
bootstrap below also does, if the environment variable is not already set), and under a
memory cap -- this is thousands of skill executions in one process:

    systemd-run --user --unit=t3d-skill-param-sweep -p MemoryMax=8G \\
        -p MemorySwapMax=0 -p OOMPolicy=continue -- \\
        scripts/with_kinder_env.sh python scripts/tossing3d_skill_parameter_sweep.py \\
        --which both --output-dir docs/experiment-logs

## Grid design

**`Pick`** (`param_dim=2`: distance, rotation) is gridded over its own sampling bounds
exactly, `PICK_DISTANCE_BOUNDS=(0.5, 0.6)` x `PICK_ROTATION_BOUNDS=(-pi/4, pi/4)` --
there is nothing outside those bounds to sweep, since nothing ever draws a `Pick`
parameter from outside them. `_PICK_ROTATION_STEPS` is odd (13) so rotation=0.0 sits
exactly on the grid, next to the oracle's own point
(`ORACLE_PICK_ROTATION=-0.7008563...`), which is marked on the figure but is not itself
a grid point.

**`MoveToThrowPose`** (`param_dim=1`: standoff) is gridded *wider* than either the old
or the (in-flight, separate-PR) new `THROW_STANDOFF_BOUNDS`, `[0.35, 1.85]` at 0.025 m
resolution -- fine enough to resolve the tightened classifier band's own edges
(`[1.150, 1.375]`, a 0.225 m band) to about nine grid points, and wide enough to run
well past both ends of every bound this domain has used. `Pick`'s own params are held
**fixed at the oracle's point** for this sweep specifically
(`ORACLE_PICK_DISTANCE`/`ORACLE_PICK_ROTATION`) -- a deliberate methodological choice,
not an oversight: `MoveToThrowPose` needs a post-`Pick` state to run from at all, and
letting `Pick`'s own variance leak into this sweep would confound "does this standoff
work" with "did the grasp even land this attempt". A `pick_success`/`pick_error` field
is still recorded per cell so a reader can check how often the fixed point itself held.

**The classifier is arithmetic in the commanded standoff, and the pilot for this script
already showed it**: `RobotAtSuccessfulThrowPose` reads `pos_base_x` after
`move_to_target` terminates, and that controller lands the base within `WAYPOINT_TOL`
of its commanded pose, so the *labelled* outcome is close to a step function of
`standoff` alone. That is a real, reportable finding -- it is exactly why the classifier
is a trainable label at all (`predicates.py`'s own docstring: the old `NearBin` was
untrainable precisely because it accepted *every* standoff). It does **not** on its own
show the barrier/bin-collision zone at low standoffs or the no-motion-plan zone at high
ones, because the classifier looks only at where the base ended up, not at what it hit
on the way. So every cell also records the physical diagnostics that *do* show those
zones: the bin's own position before and after the skill sequence (a shove shows up as
a nonzero delta), the barrier's position, the achieved base pose against the commanded
standoff, and whichever controller reported an error or failed to terminate.

## Seeds are scene seeds, shared across every grid cell (paired, not independent)

The same `SEEDS = tuple(range(_NUM_SEEDS))` scene seeds are used at *every* grid cell
in a sweep, rather than drawing fresh seeds per cell. That is deliberate: it isolates
the parameter's effect from scene-to-scene variance (a comparison across cells is then
paired on scene, not just parameter), matching `tossing3d_throw_band_sweep`'s own
methodology. The cost is that a single pathological seed shifts every cell by the same
`1/_NUM_SEEDS`, which would show up as correlated structure in a heatmap that is not
about the parameter at all -- `print_report`'s per-seed breakdown exists so that is
checkable rather than assumed away.
"""

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

# Bootstrap `hitl_pmp` onto the path before anything imports it -- see the module
# docstring. A worktree's own `src/` is what must resolve, matching `scripts/
# with_env.sh`'s reasoning for the hitl-pmp-conda side of this same trap; here there is
# no wrapper, since this script runs under a different interpreter entirely.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment  # noqa: E402
from hitl_pmp.environments.tossing3d.predicates import (  # noqa: E402
    HOLDING,
    ROBOT_AT_SUCCESSFUL_THROW_POSE,
)
from hitl_pmp.environments.tossing3d.skill_oracle_policy import (  # noqa: E402
    ORACLE_PICK_DISTANCE,
    ORACLE_PICK_ROTATION,
)
from hitl_pmp.environments.tossing3d.skills import (  # noqa: E402
    PICK_DISTANCE_BOUNDS,
    PICK_ROTATION_BOUNDS,
)

# Same 12 scene seeds at every grid cell -- see the module docstring on why paired
# rather than fresh-per-cell. 12 sits inside the task's own 10-15 target.
_NUM_SEEDS = 12
SEEDS: tuple[int, ...] = tuple(range(_NUM_SEEDS))

# `Pick`'s grid is exactly its own sampling bounds -- nothing is ever drawn outside them.
PICK_DISTANCE_STEPS = 12
PICK_ROTATION_STEPS = 13  # odd, so rotation = 0.0 lands exactly on the grid

# `MoveToThrowPose`'s grid is deliberately wider than any `THROW_STANDOFF_BOUNDS` this
# domain has used (old (0.45, 1.75); in-flight (1.10, 1.75)), so this sweep is not
# retired the next time that constant moves.
MOVE_STANDOFF_MIN = 0.35
MOVE_STANDOFF_MAX = 1.85
MOVE_STANDOFF_STEP = 0.025

# How often to flush a checkpoint of whatever has been measured so far, and how often to
# print a progress line. Small enough that a service killed mid-run still leaves most of
# its data on disk; large enough not to dominate wall-clock with disk I/O.
_CHECKPOINT_EVERY = 25

_PICK_JSON_NAME = "2026-08-10-tossing3d-skill-parameter-sweep-pick.json"
_MOVE_JSON_NAME = "2026-08-10-tossing3d-skill-parameter-sweep-movetothrowpose.json"


class PickCellResult(BaseModel):
    """One `(distance, rotation, seed)` cell of the `Pick` grid.

    `success` is `Holding` on the state right after `Pick` -- the same label
    EES's own practice data is tagged with. The rest are diagnostics: `pick_error`/
    `pick_terminated` distinguish "the grasp planned and executed but did not hold" from
    "the controller never terminated/raised", and the position fields let a reader
    reconstruct where the grasp actually placed the base and the cube without re-running
    the simulator.
    """

    distance: float
    rotation: float
    seed: int
    success: bool
    pick_error: str | None
    pick_terminated: bool
    pick_steps: int
    robot_base_x: float
    robot_base_y: float
    cube_x: float
    cube_y: float
    cube_z: float


class MoveCellResult(BaseModel):
    """One `(standoff, seed)` cell of the `MoveToThrowPose` grid, `Pick` fixed at the
    oracle's point.

    `success` is `RobotAtSuccessfulThrowPose` on the state right after
    `MoveToThrowPose` -- the label EES trains against. `pick_success` records whether
    the fixed-oracle-point `Pick` itself held in this cell (it is not the thing being
    swept, but a `MoveToThrowPose` run from a dropped cube is not a real measurement of
    this skill, and confounded cells are kept rather than silently dropped so the
    analysis module can filter them explicitly). The `bin_*`/`barrier_*` fields are what
    make the barrier/bin-collision zone at low standoffs visible: a genuine shove shows
    up as `bin_x != bin_x_initial` (or `barrier_x != barrier_x_initial`), which the
    classifier itself never looks at. The pilot for this script found the barrier moving
    too, from a seed-dependent rest position up toward ~1.5 m, at the lowest standoffs
    tried -- consistent with the background finding that a small enough standoff drives
    the base straight into it (upstream's base motion planner has collision-checking
    hardcoded off).
    """

    standoff: float
    seed: int
    success: bool
    pick_success: bool
    pick_error: str | None
    move_error: str | None
    move_terminated: bool
    robot_base_x: float
    robot_base_y: float
    lateral_offset: float
    bin_x_initial: float
    bin_y_initial: float
    bin_x: float
    bin_y: float
    barrier_x_initial: float
    barrier_x: float


def _rss_mb() -> float:
    """This process's peak resident set size, in MiB.

    `ru_maxrss` is KiB on Linux (unlike macOS's bytes), so `/1024` is correct here and
    would be wrong ported elsewhere -- this script only ever runs on the machine
    `CLAUDE.md`'s memory-cap guidance is written for.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _write_checkpoint(*, path: Path, records: list[BaseModel]) -> None:
    """Write `records` as a JSON list, atomically -- a reader (a progress check, or this
    process being killed mid-write) never sees a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps([record.model_dump() for record in records], indent=2))
    os.replace(tmp_path, path)


def run_pick_grid(
    *,
    env: Tossing3DEnvironment,
    distances: np.ndarray,
    rotations: np.ndarray,
    seeds: tuple[int, ...],
    checkpoint_path: Path,
) -> list[PickCellResult]:
    """`Pick(distance, rotation)` once per `(distance, rotation, seed)` cell, in place
    on one persistent `env` -- `KinderBackend.reset(seed=...)` rebuilds only the scene,
    not the whole gym env, which is what makes a dense grid tractable at all."""
    results: list[PickCellResult] = []
    total = len(distances) * len(rotations) * len(seeds)
    started = time.monotonic()
    for distance in distances:
        for rotation in rotations:
            for seed in seeds:
                env.reset_to_seed(seed=int(seed))
                action = np.array([Tossing3DEnvironment.pick_id, float(distance), float(rotation)])
                state = env.take_action(action=action)
                success = HOLDING.holds(state, (env.robot, env.cube))
                steps = env.last_controller_steps()
                results.append(
                    PickCellResult(
                        distance=float(distance),
                        rotation=float(rotation),
                        seed=int(seed),
                        success=success,
                        pick_error=env.last_skill_error(),
                        pick_terminated=env.last_skill_error() is None,
                        pick_steps=steps[0] if steps else 0,
                        robot_base_x=float(state.get(obj=env.robot, feature_name="pos_base_x")),
                        robot_base_y=float(state.get(obj=env.robot, feature_name="pos_base_y")),
                        cube_x=float(state.get(obj=env.cube, feature_name="x")),
                        cube_y=float(state.get(obj=env.cube, feature_name="y")),
                        cube_z=float(state.get(obj=env.cube, feature_name="z")),
                    )
                )
                if len(results) % _CHECKPOINT_EVERY == 0:
                    _write_checkpoint(path=checkpoint_path, records=list(results))
                    elapsed = time.monotonic() - started
                    print(
                        f"[pick] {len(results)}/{total} cells, {elapsed:.0f}s elapsed, "
                        f"{elapsed / len(results):.2f}s/cell, RSS {_rss_mb():.0f} MiB",
                        flush=True,
                    )
    _write_checkpoint(path=checkpoint_path, records=list(results))
    print(f"[pick] done: {len(results)}/{total} cells written to {checkpoint_path}")
    return results


def run_move_grid(
    *,
    env: Tossing3DEnvironment,
    standoffs: np.ndarray,
    seeds: tuple[int, ...],
    checkpoint_path: Path,
) -> list[MoveCellResult]:
    """`Pick(oracle) -> MoveToThrowPose(standoff)` once per `(standoff, seed)` cell.

    `Pick`'s own params are fixed at the oracle's point throughout -- see the module
    docstring for why. Every cell still runs a fresh `Pick`, since `MoveToThrowPose`
    needs a post-`Pick` state to move from and there is no cheaper way to get one.
    """
    results: list[MoveCellResult] = []
    total = len(standoffs) * len(seeds)
    started = time.monotonic()
    for standoff in standoffs:
        for seed in seeds:
            state0 = env.reset_to_seed(seed=int(seed))
            bin_x_initial = float(state0.get(obj=env.bin, feature_name="x"))
            bin_y_initial = float(state0.get(obj=env.bin, feature_name="y"))
            barrier_x_initial = float(state0.get(obj=env.barrier, feature_name="x"))

            pick_action = np.array([
                Tossing3DEnvironment.pick_id,
                ORACLE_PICK_DISTANCE,
                ORACLE_PICK_ROTATION,
            ])
            state1 = env.take_action(action=pick_action)
            pick_success = HOLDING.holds(state1, (env.robot, env.cube))
            pick_error = env.last_skill_error()

            move_action = np.array([
                Tossing3DEnvironment.move_to_throw_pose_id,
                float(standoff),
                0.0,
            ])
            state2 = env.take_action(action=move_action)
            move_error = env.last_skill_error()
            success = ROBOT_AT_SUCCESSFUL_THROW_POSE.holds(state2, (env.robot, env.bin))
            robot_base_y = float(state2.get(obj=env.robot, feature_name="pos_base_y"))
            target_y = float(state2.get(obj=env.bin, feature_name="y"))
            results.append(
                MoveCellResult(
                    standoff=float(standoff),
                    seed=int(seed),
                    success=success,
                    pick_success=pick_success,
                    pick_error=pick_error,
                    move_error=move_error,
                    move_terminated=move_error is None,
                    robot_base_x=float(state2.get(obj=env.robot, feature_name="pos_base_x")),
                    robot_base_y=robot_base_y,
                    lateral_offset=abs(robot_base_y - target_y),
                    bin_x_initial=bin_x_initial,
                    bin_y_initial=bin_y_initial,
                    bin_x=float(state2.get(obj=env.bin, feature_name="x")),
                    bin_y=float(state2.get(obj=env.bin, feature_name="y")),
                    barrier_x_initial=barrier_x_initial,
                    barrier_x=float(state2.get(obj=env.barrier, feature_name="x")),
                )
            )
            if len(results) % _CHECKPOINT_EVERY == 0:
                _write_checkpoint(path=checkpoint_path, records=list(results))
                elapsed = time.monotonic() - started
                print(
                    f"[move] {len(results)}/{total} cells, {elapsed:.0f}s elapsed, "
                    f"{elapsed / len(results):.2f}s/cell, RSS {_rss_mb():.0f} MiB",
                    flush=True,
                )
    _write_checkpoint(path=checkpoint_path, records=list(results))
    print(f"[move] done: {len(results)}/{total} cells written to {checkpoint_path}")
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--which", choices=("pick", "move", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pick-distance-steps", type=int, default=PICK_DISTANCE_STEPS)
    parser.add_argument("--pick-rotation-steps", type=int, default=PICK_ROTATION_STEPS)
    parser.add_argument("--move-min", type=float, default=MOVE_STANDOFF_MIN)
    parser.add_argument("--move-max", type=float, default=MOVE_STANDOFF_MAX)
    parser.add_argument("--move-step", type=float, default=MOVE_STANDOFF_STEP)
    parser.add_argument("--num-seeds", type=int, default=_NUM_SEEDS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = tuple(range(args.num_seeds))
    env = Tossing3DEnvironment()
    try:
        if args.which in ("pick", "both"):
            distances = np.linspace(*PICK_DISTANCE_BOUNDS, args.pick_distance_steps)
            rotations = np.linspace(*PICK_ROTATION_BOUNDS, args.pick_rotation_steps)
            run_pick_grid(
                env=env,
                distances=distances,
                rotations=rotations,
                seeds=seeds,
                checkpoint_path=args.output_dir / _PICK_JSON_NAME,
            )
        if args.which in ("move", "both"):
            num_steps = round((args.move_max - args.move_min) / args.move_step) + 1
            standoffs = np.linspace(args.move_min, args.move_max, num_steps)
            run_move_grid(
                env=env,
                standoffs=standoffs,
                seeds=seeds,
                checkpoint_path=args.output_dir / _MOVE_JSON_NAME,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
