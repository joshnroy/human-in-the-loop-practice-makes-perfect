"""Tossing3D pick benchmark states: fixed KINDER values, captured once, restored exactly.

A state is the scene's whole object-centric state at MuJoCo's float64 precision
(`KinderBackend.snapshot_to_plain` of `live_state`), restored with
`Tossing3DEnvironment.restore_plain_snapshot` -- no RNG and no re-run of the setup.
The set lives in one append-only JSONL file (`PickStateSet.append`).

Every placement is read off the live scene: the bin's pose and its own dimensions,
the barrier's pose, the room's wall colliders. Nothing hard-codes a bin coordinate.
"""

import hashlib
import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend
from hitl_pmp.environments.tossing3d.pick_plannable import PickPlannableGate

from .harvest import Harvest
from .types import PickState

SETTLE_STEPS = 20
# Upstream's own name for "in the bin": the scored region rides on the bin.
IN_BIN_ATOM = "MovableInGoalRegion(cube_0)"
GAPS_IN_BIN = (0.0, 0.015, 0.03)
GAPS_IN_CORNER = (0.0, 0.015)
GAPS_OUTSIDE = (0.005, 0.02)
GAPS_BARRIER = (0.005, 0.02, 0.05)
GAPS_WALL = (0.005, 0.02)
BARRIER_YS = (-1.5, 0.0, 1.5)
ROOM_WALL_ALONG = 0.5
# Bins brought to the robot's side, by `reset_movables("robot_side")` at these seeds.
BIN_POSE_SEEDS = (0, 1)
NUM_START_SEEDS = 20
PUSHED_BIN_YAWS_DEG = (12.0, 30.0)


class PickStateSet:
    @staticmethod
    def load(*, path: Path) -> list[PickState]:
        if not path.exists():
            return []
        return [
            PickState.model_validate_json(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    @staticmethod
    def append(*, path: Path, states: list[PickState]) -> list[str]:
        known = {state.state_id for state in PickStateSet.load(path=path)}
        added = [state for state in states if state.state_id not in known]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            for state in added:
                stream.write(state.model_dump_json() + "\n")
        return [state.state_id for state in added]

    @staticmethod
    def sha256(*, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


class PickPlacements:
    """Pure geometry: labelled (x, y) cube centres. Every generated state is classified
    after it settles and kept as whatever it classifies as, so these are proposals."""

    @staticmethod
    def to_world(
        *, bin_xy: tuple[float, float], bin_yaw: float, lx: float, ly: float
    ) -> tuple[float, float]:
        c, s = math.cos(bin_yaw), math.sin(bin_yaw)
        return (bin_xy[0] + c * lx - s * ly, bin_xy[1] + s * lx + c * ly)

    @staticmethod
    def in_bin(
        *,
        bin_xy: tuple[float, float],
        bin_yaw: float,
        inner_half: float,
        cube_half: float,
        gaps: tuple[float, ...] = GAPS_IN_BIN,
        corner_gaps: tuple[float, ...] = GAPS_IN_CORNER,
    ) -> list[tuple[str, float, float]]:
        out = []
        for gap in gaps:
            reach = inner_half - cube_half - gap
            for wall, (lx, ly) in {
                "px": (reach, 0.0),
                "nx": (-reach, 0.0),
                "py": (0.0, reach),
                "ny": (0.0, -reach),
            }.items():
                out.append((
                    f"bin-wall-{wall}-g{gap:g}",
                    *PickPlacements.to_world(bin_xy=bin_xy, bin_yaw=bin_yaw, lx=lx, ly=ly),
                ))
        for gap in corner_gaps:
            reach = inner_half - cube_half - gap
            for corner, (sx, sy) in {
                "pp": (1, 1),
                "pn": (1, -1),
                "np": (-1, 1),
                "nn": (-1, -1),
            }.items():
                out.append((
                    f"bin-corner-{corner}-g{gap:g}",
                    *PickPlacements.to_world(
                        bin_xy=bin_xy, bin_yaw=bin_yaw, lx=sx * reach, ly=sy * reach
                    ),
                ))
        return out

    @staticmethod
    def around_bin(
        *,
        bin_xy: tuple[float, float],
        bin_yaw: float,
        outer_half: float,
        cube_half: float,
        gaps: tuple[float, ...] = GAPS_OUTSIDE,
    ) -> list[tuple[str, float, float]]:
        out = []
        for gap in gaps:
            reach = outer_half + cube_half + gap
            for side, (lx, ly) in {
                "px": (reach, 0.0),
                "nx": (-reach, 0.0),
                "py": (0.0, reach),
                "ny": (0.0, -reach),
            }.items():
                out.append((
                    f"outside-{side}-g{gap:g}",
                    *PickPlacements.to_world(bin_xy=bin_xy, bin_yaw=bin_yaw, lx=lx, ly=ly),
                ))
        return out

    @staticmethod
    def beside_barrier(
        *,
        barrier_x: float,
        barrier_half: float,
        robot_x: float,
        cube_half: float,
        gaps: tuple[float, ...] = GAPS_BARRIER,
        ys: tuple[float, ...] = BARRIER_YS,
    ) -> list[tuple[str, float, float]]:
        toward_robot = 1.0 if robot_x > barrier_x else -1.0
        return [
            (
                f"barrier-y{y:g}-g{gap:g}",
                barrier_x + toward_robot * (barrier_half + gap + cube_half),
                y,
            )
            for gap in gaps
            for y in ys
        ]

    @staticmethod
    def near_room_walls(
        *,
        walls: dict[str, tuple[str, float]],
        cube_half: float,
        gaps: tuple[float, ...] = GAPS_WALL,
        along: float = ROOM_WALL_ALONG,
    ) -> list[tuple[str, float, float]]:
        out = []
        for name, (axis, value) in walls.items():
            for gap in gaps:
                inward = -1.0 if value > 0 else 1.0
                coordinate = value + inward * (gap + cube_half)
                x, y = (coordinate, along) if axis == "x" else (along, coordinate)
                out.append((f"wall-{name}-g{gap:g}", x, y))
        return out


class PickCapture:
    """Capture and restore on a live environment."""

    @staticmethod
    def fingerprint(*, env: Tossing3DEnvironment) -> str:
        snapshot = env.backend().live_state()
        names = ",".join(sorted(obj.name for obj in snapshot))
        bin_object = env.backend()._object_centric()._objects_dict[KinderBackend.bin_name]  # noqa: SLF001
        dims = (bin_object.length, bin_object.width, bin_object.height, bin_object.wall_thickness)
        return f"{env.backend().env_id}|{env.layout.value}|{names}|bin={dims}"

    @staticmethod
    def atoms(*, env: Tossing3DEnvironment) -> list[str]:
        return sorted(
            f"{name}({','.join(objects)})" for name, objects in env.backend().abstract_atoms()
        )

    @staticmethod
    def plain(*, env: Tossing3DEnvironment) -> dict[str, list[float]]:
        backend = env.backend()
        return {
            name: values
            for name, values in backend.snapshot_to_plain(snapshot=backend.live_state()).items()
            if not name.startswith("collider:")
        }

    @staticmethod
    def restore(*, env: Tossing3DEnvironment, state: PickState) -> None:
        env.reset_to_seed(seed=state.seed)
        fingerprint = PickCapture.fingerprint(env=env)
        if fingerprint != state.scene_fingerprint:
            raise ValueError(
                f"state {state.state_id} is from scene {state.scene_fingerprint}, not this one "
                f"({fingerprint}); regenerate the state set for this scene"
            )
        env.restore_plain_snapshot(plain=state.plain, dtype=np.float64)
        atoms = PickCapture.atoms(env=env)
        if atoms != state.atoms:
            raise ValueError(
                f"state {state.state_id} restored with different atoms: "
                f"{sorted(set(atoms) ^ set(state.atoms))}"
            )

    @staticmethod
    def in_target(*, env: Tossing3DEnvironment, atoms: list[str]) -> bool:
        """The pick operator's preconditions, `PickPlannable` aside: hand empty, bin
        upright on the floor, cube on the robot's side of the barrier."""
        features = env.backend().observe().features
        across = PickPlannableGate.cube_across_barrier(
            robot_x=features[env.backend().robot_name]["pos_base_x"],
            cube=features[KinderBackend.cube_name],
            barrier=features[KinderBackend.barrier_name],
        )
        return "HandEmpty(robot)" in atoms and "BinOnGround(bin_0)" in atoms and not across

    @staticmethod
    def capture(
        *, env: Tossing3DEnvironment, state_id: str, family: str, seed: int, provenance: str
    ) -> PickState:
        atoms = PickCapture.atoms(env=env)
        return PickState(
            state_id=state_id,
            family=family,
            seed=seed,
            layout=env.layout.value,
            scene_fingerprint=PickCapture.fingerprint(env=env),
            plain=PickCapture.plain(env=env),
            atoms=atoms,
            skill="PickCubeFromBin" if IN_BIN_ATOM in atoms else "PickCube",
            in_target=PickCapture.in_target(env=env, atoms=atoms),
            provenance=provenance,
        )


class PickSetups:
    """Edits of the live scene, settled with the arm holding still."""

    @staticmethod
    def pose(*, env: Tossing3DEnvironment, name: str, position: Any, quat_wxyz: Any) -> None:
        backend = env.backend()
        snapshot = backend.live_state()
        plain = backend.snapshot_to_plain(snapshot=snapshot)
        names = list(snapshot.type_features[snapshot.get_object_from_name(name).type])
        values = plain[name]
        for key, value in zip(
            ("x", "y", "z", "qw", "qx", "qy", "qz"), (*position, *quat_wxyz), strict=True
        ):
            values[names.index(key)] = float(value)
        for key in ("vx", "vy", "vz", "wx", "wy", "wz"):
            if key in names:
                values[names.index(key)] = 0.0
        plain[name] = values
        kept = {k: v for k, v in plain.items() if not k.startswith("collider:")}
        env.restore_plain_snapshot(plain=kept, dtype=np.float64)

    @staticmethod
    def settle(*, env: Tossing3DEnvironment, steps: int = SETTLE_STEPS) -> None:
        backend = env.backend()
        raw = backend._env  # noqa: SLF001
        hold = np.zeros(raw.action_space.shape, dtype=np.float32)
        for _ in range(steps):
            observation, *_ = raw.step(hold)
            backend._state = raw.observation_space.devectorize(observation)  # noqa: SLF001
        backend.drain_substep_states()
        env.restore_plain_snapshot(plain=PickCapture.plain(env=env), dtype=np.float64)

    @staticmethod
    def bin_pose(*, env: Tossing3DEnvironment) -> tuple[tuple[float, float], float, list[float]]:
        f = env.backend().observe().features[KinderBackend.bin_name]
        qw, qx, qy, qz = f["qw"], f["qx"], f["qy"], f["qz"]
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return (f["x"], f["y"]), yaw, [f["x"], f["y"], f["z"]]


class PickSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    state_id: str
    family: str
    seed: int
    provenance: str
    setup: Callable[..., None] | None = None


class PickFamilies:
    """Josh's pick families (2026-10-15): start states over seeds; the cube in the bin
    0-3 cm from each wall and in the corners; on the floor against the bin's outside
    wall; beside the barrier; near a room wall; with the bin tipped; with the bin pushed
    by a pick (turned and moved); and the harvested PickPlannable-false resets."""

    @staticmethod
    def quat_yaw(*, yaw: float) -> list[float]:
        return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]

    @staticmethod
    def specs(*, env: Tossing3DEnvironment, harvests: list[Harvest]) -> Iterator[PickSpec]:
        backend = env.backend()
        env.reset_to_seed(seed=0)
        bin_object = backend._object_centric()._objects_dict[KinderBackend.bin_name]  # noqa: SLF001
        cube_f = backend.observe().features[KinderBackend.cube_name]
        half = float(cube_f["bb_z"]) / 2
        inner = bin_object.length / 2 - bin_object.wall_thickness
        outer = bin_object.length / 2
        floor_in_bin = bin_object.wall_thickness
        barrier = backend.observe().features[KinderBackend.barrier_name]
        robot_x = backend.observe().features[backend.robot_name]["pos_base_x"]
        walls = PickFamilies.room_walls(env=env)

        for seed in [*range(NUM_START_SEEDS), env.canonical_seed]:
            yield PickSpec(
                state_id=f"start-s{seed}",
                family="start",
                seed=seed,
                provenance=f"reset_to_seed({seed})",
            )

        def cube_at(*, x: float, y: float, z: float) -> Callable[..., None]:
            def setup(*, env: Tossing3DEnvironment) -> None:
                PickSetups.pose(
                    env=env,
                    name=KinderBackend.cube_name,
                    position=[x, y, z],
                    quat_wxyz=[1, 0, 0, 0],
                )

            return setup

        def robot_side_bin(
            *, then: Callable[..., None] | None = None, bin_edit: Callable[..., None] | None = None
        ) -> Callable[..., None]:
            def setup(*, env: Tossing3DEnvironment) -> None:
                env.reset_movables(destination="robot_side")
                if bin_edit is not None:
                    bin_edit(env=env)
                    PickSetups.settle(env=env)
                if then is not None:
                    then(env=env)

            return setup

        for seed in BIN_POSE_SEEDS:
            env.reset_to_seed(seed=seed)
            env.reset_movables(destination="robot_side")
            bin_xy, bin_yaw, bin_pos = PickSetups.bin_pose(env=env)
            for label, x, y in PickPlacements.in_bin(
                bin_xy=bin_xy, bin_yaw=bin_yaw, inner_half=inner, cube_half=half
            ):
                yield PickSpec(
                    state_id=f"{label}-s{seed}",
                    family="in_bin_near_wall",
                    seed=seed,
                    provenance=f"bin brought to the robot's side (seed {seed}); cube at {label}",
                    setup=robot_side_bin(
                        then=cube_at(x=x, y=y, z=bin_pos[2] + floor_in_bin + half + 0.001)
                    ),
                )
            for label, x, y in PickPlacements.around_bin(
                bin_xy=bin_xy, bin_yaw=bin_yaw, outer_half=outer, cube_half=half
            ):
                yield PickSpec(
                    state_id=f"{label}-s{seed}",
                    family="outside_bin_wall",
                    seed=seed,
                    provenance=f"robot-side bin (seed {seed}); floor cube at {label}",
                    setup=robot_side_bin(then=cube_at(x=x, y=y, z=half + 0.001)),
                )
        for label, x, y in PickPlacements.beside_barrier(
            barrier_x=barrier["x"],
            barrier_half=float(barrier.get("bb_x", 0.06)) / 2,
            robot_x=robot_x,
            cube_half=half,
        ):
            yield PickSpec(
                state_id=label,
                family="beside_barrier",
                seed=0,
                provenance=f"cube on the floor at {label}",
                setup=cube_at(x=x, y=y, z=half + 0.001),
            )
        for label, x, y in PickPlacements.near_room_walls(walls=walls, cube_half=half):
            yield PickSpec(
                state_id=label,
                family="near_room_wall",
                seed=0,
                provenance=f"cube on the floor at {label}",
                setup=cube_at(x=x, y=y, z=half + 0.001),
            )
        env.reset_to_seed(seed=BIN_POSE_SEEDS[0])
        env.reset_movables(destination="robot_side")
        bin_xy, bin_yaw, bin_pos = PickSetups.bin_pose(env=env)
        for k, (roll_axis, label) in enumerate(((0, "tipped-x"), (1, "tipped-y"))):
            tilt = [math.cos(math.pi / 4), 0.0, 0.0, 0.0]
            tilt[1 + roll_axis] = math.sin(math.pi / 4)

            def tip(*, env: Tossing3DEnvironment, tilt: list[float] = tilt) -> None:
                PickSetups.pose(
                    env=env,
                    name=KinderBackend.bin_name,
                    position=[bin_xy[0], bin_xy[1], outer + 0.01],
                    quat_wxyz=tilt,
                )

            near = PickPlacements.to_world(bin_xy=bin_xy, bin_yaw=0.0, lx=0.0, ly=-(outer + 0.25))
            yield PickSpec(
                state_id=f"{label}-cube-beside",
                family="tipped_bin",
                seed=BIN_POSE_SEEDS[0],
                provenance=f"bin tipped on its side ({label}); cube on the floor beside it",
                setup=robot_side_bin(
                    bin_edit=tip, then=cube_at(x=near[0], y=near[1], z=half + 0.001)
                ),
            )
            yield PickSpec(
                state_id=f"{label}-cube-open-floor-{k}",
                family="tipped_bin",
                seed=BIN_POSE_SEEDS[0],
                provenance=f"bin tipped on its side ({label}); cube on the open floor",
                setup=robot_side_bin(
                    bin_edit=tip, then=cube_at(x=0.6, y=-0.6 if k else 0.6, z=half + 0.001)
                ),
            )
        for yaw_deg in PUSHED_BIN_YAWS_DEG:
            yaw = bin_yaw + math.radians(yaw_deg)
            shifted = (bin_xy[0] + 0.06, bin_xy[1] - 0.04)

            def push(
                *,
                env: Tossing3DEnvironment,
                yaw: float = yaw,
                shifted: tuple[float, float] = shifted,
            ) -> None:
                PickSetups.pose(
                    env=env,
                    name=KinderBackend.bin_name,
                    position=[shifted[0], shifted[1], bin_pos[2]],
                    quat_wxyz=PickFamilies.quat_yaw(yaw=yaw),
                )

            for label, x, y in [
                ("in-centre", *shifted),
                (
                    "in-wall",
                    *PickPlacements.to_world(
                        bin_xy=shifted, bin_yaw=yaw, lx=inner - half - 0.01, ly=0.0
                    ),
                ),
                (
                    "outside",
                    *PickPlacements.to_world(
                        bin_xy=shifted, bin_yaw=yaw, lx=-(outer + half + 0.01), ly=0.0
                    ),
                ),
            ]:
                z = (
                    bin_pos[2] + floor_in_bin + half + 0.001
                    if label.startswith("in")
                    else half + 0.001
                )
                yield PickSpec(
                    state_id=f"pushed-bin-{yaw_deg:g}deg-{label}",
                    family="pushed_bin",
                    seed=BIN_POSE_SEEDS[0],
                    provenance=f"bin pushed {yaw_deg:g} deg and 7 cm; cube {label}",
                    setup=robot_side_bin(bin_edit=push, then=cube_at(x=x, y=y, z=z)),
                )
        for index, harvest in enumerate(harvests):

            def restore(
                *, env: Tossing3DEnvironment, plain: dict[str, list[float]] = harvest.state
            ) -> None:
                kept = {k: v for k, v in plain.items() if not k.startswith("collider:")}
                env.restore_plain_snapshot(plain=kept, dtype=np.float64)

            yield PickSpec(
                state_id=f"harvest-{index:03d}",
                family=f"harvested: {harvest.d4}",
                seed=0,
                provenance=f"{harvest.source}: last logged tick before a PickPlannable-false reset",
                setup=restore,
            )

    @staticmethod
    def room_walls(*, env: Tossing3DEnvironment) -> dict[str, tuple[str, float]]:
        """The room's four vertical walls, from its colliders: name -> (axis, coordinate)."""
        snapshot = env.backend().live_state()
        walls: dict[str, tuple[str, float]] = {}
        barrier_x = float(env.backend().observe().features[KinderBackend.barrier_name]["x"])
        robot_x = float(env.backend().observe().features[env.backend().robot_name]["pos_base_x"])
        for obj in snapshot:
            if not obj.name.startswith("collider:tossing_room"):
                continue
            x, y = float(snapshot.get(obj, "x")), float(snapshot.get(obj, "y"))
            if abs(abs(y) - 3.0) < 0.05:
                walls["north" if y > 0 else "south"] = ("y", y)
            elif abs(y) < 1e-6 and (x - barrier_x) * (robot_x - barrier_x) > 0:
                # A wall across the barrier has no reachable cube beside it.
                walls["west" if x < barrier_x else "east"] = ("x", x)
        return walls

    @staticmethod
    def build(*, env: Tossing3DEnvironment, spec: PickSpec) -> tuple[PickState | None, str]:
        env.reset_to_seed(seed=spec.seed)
        if spec.setup is not None:
            spec.setup(env=env)
        PickSetups.settle(env=env)
        state = PickCapture.capture(
            env=env,
            state_id=spec.state_id,
            family=spec.family,
            seed=spec.seed,
            provenance=spec.provenance,
        )
        if "Holding(robot,cube_0)" in state.atoms:
            return None, f"{spec.state_id}: the cube is held; not a pick state"
        plannable = "PickPlannable(robot,cube_0)" in state.atoms
        note = (
            f"{spec.state_id}: ok ({state.skill}, in_target={state.in_target}, "
            f"PickPlannable={plannable})"
        )
        return state, note
