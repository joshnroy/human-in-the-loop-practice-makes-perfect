"""Benchmark start states: captured once as fixed MuJoCo values, restored exactly.

A state is the scene's whole `qpos`/`qvel`/`ctrl` after `reset_to_seed(seed)` and a
deterministic setup (teleports, a human reset, a prefix skill), so restoring it needs
no RNG and no re-run of the setup. The set lives in one JSONL file; new states, such as
failures harvested from a run, are appended and never rewritten (`StateSet.append`).

The scene the states were captured in is fingerprinted (`StateCapture.fingerprint`):
the KINDER pin moves the geometry, and a state set from another scene is refused rather
than restored into the wrong one.

Everything container-specific is read from the live scene through `Container` -- the
interior bounds and the wall thickness -- so the same families regenerate on a scene
with a different container; nothing here hard-codes the box's coordinates.
"""

import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from hitl_pmp.environments.sweepoffshelf3d.backend import CUBE_SHELF, CUPBOARD, TASK_ID
from hitl_pmp.environments.sweepoffshelf3d.environment import SKILL_IDS, SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects as O
from hitl_pmp.environments.sweepoffshelf3d.skill_provider import ORACLE_SWEEP_PARAMS
from hitl_pmp.environments.sweepoffshelf3d.skill_trace import SkillTraces

from .types import BenchmarkCase, BenchmarkState

SETTLE_STEPS = 30
# Clearances for generated placements: the gap between a cube and the wall it is put
# against, inside and outside the container, beside a cupboard panel and on the shelf.
INSIDE_MARGIN = 0.004
OUTSIDE_GAP = 0.02
PANEL_GAP = 0.02
SHELF_MARGIN = 0.006
STATIC_PREDICATES = frozenset({"CubeOrder"})


class StateSet:
    """The JSONL file of `BenchmarkState`s."""

    @staticmethod
    def load(*, path: Path) -> list[BenchmarkState]:
        if not path.exists():
            return []
        return [
            BenchmarkState.model_validate_json(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    @staticmethod
    def append(*, path: Path, states: list[BenchmarkState]) -> list[str]:
        """Add the states whose ids are new; never rewrite an existing one. Returns the
        ids added."""
        known = {state.state_id for state in StateSet.load(path=path)}
        added = [state for state in states if state.state_id not in known]
        if len({s.state_id for s in added}) != len(added):
            raise ValueError("duplicate state ids in one append")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            for state in added:
                stream.write(state.model_dump_json() + "\n")
        return [state.state_id for state in added]

    @staticmethod
    def sha256(*, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


class StateCapture:
    """Capture and restore on a live environment."""

    @staticmethod
    def fingerprint(*, env: SweepOffShelf3DEnvironment) -> str:
        backend = env.backend()
        model = backend.model()
        geoms = ",".join(model.geom(g).name or "" for g in range(model.ngeom))
        digest = hashlib.sha256(geoms.encode()).hexdigest()[:12]
        return f"{TASK_ID}|nq={model.nq}|nv={model.nv}|nu={model.nu}|geoms={digest}"

    @staticmethod
    def atoms(*, env: SweepOffShelf3DEnvironment) -> list[str]:
        return SkillTraces.names(atoms=env.backend().abstract_atoms())

    @staticmethod
    def capture(
        *,
        env: SweepOffShelf3DEnvironment,
        state_id: str,
        family: str,
        seed: int,
        cases: list[BenchmarkCase],
        provenance: str,
    ) -> BenchmarkState:
        data = env.backend().data()
        return BenchmarkState(
            state_id=state_id,
            family=family,
            seed=seed,
            scene_fingerprint=StateCapture.fingerprint(env=env),
            qpos=[float(v) for v in data.qpos],
            qvel=[float(v) for v in data.qvel],
            ctrl=[float(v) for v in data.ctrl],
            grip=float(env.backend().grip),
            atoms=StateCapture.atoms(env=env),
            cases=cases,
            provenance=provenance,
        )

    @staticmethod
    def restore(*, env: SweepOffShelf3DEnvironment, state: BenchmarkState) -> None:
        """Put `env` exactly in `state`; raise if it is from another scene or does not
        classify as it did when captured."""
        env.reset_to_seed(seed=state.seed)
        fingerprint = StateCapture.fingerprint(env=env)
        if fingerprint != state.scene_fingerprint:
            raise ValueError(
                f"state {state.state_id} was captured in scene {state.scene_fingerprint}, "
                f"not this one ({fingerprint}); regenerate the state set for this scene"
            )
        backend = env.backend()
        data = backend.data()
        data.qpos[:] = state.qpos
        data.qvel[:] = state.qvel
        data.ctrl[:] = state.ctrl
        backend.sim().forward()
        backend.grip = state.grip
        env.controllers().on_objects_moved()
        env.current_state = env._observed_state(seed=state.seed, steps_taken=0)  # noqa: SLF001
        atoms = StateCapture.atoms(env=env)
        if atoms != state.atoms:
            raise ValueError(
                f"state {state.state_id} restored with different atoms: "
                f"{sorted(set(atoms) ^ set(state.atoms))}"
            )

    @staticmethod
    def action(*, case: BenchmarkCase) -> np.ndarray:
        names = [cube.name for cube in O.cubes]
        cube = 0.0 if case.cube is None else float(names.index(case.cube))
        height, speed = case.params if case.params else (0.0, 0.0)
        return np.array([float(SKILL_IDS[case.skill]), cube, height, speed])

    @staticmethod
    def applicable(*, case: BenchmarkCase, atoms: list[str]) -> bool:
        ground = SkillTraces.ground(action=StateCapture.action(case=case))
        if ground is None:
            return False
        required = {
            SkillTraces.names(atoms=[SkillTraces.key(atom=atom)])[0]
            for atom in ground.preconditions
            if atom.predicate.name not in STATIC_PREDICATES
        }
        return required <= set(atoms)


class Container:
    """The container the cubes go into, read from the live scene."""

    @staticmethod
    def name(*, env: SweepOffShelf3DEnvironment) -> str:
        return env.backend().box_name

    @staticmethod
    def interior(*, env: SweepOffShelf3DEnvironment) -> tuple[float, float, float, float]:
        (x0, y0, x1, y1), _ = env.backend().core().box_interior()
        return (float(x0), float(y0), float(x1), float(y1))

    @staticmethod
    def floor_z(*, env: SweepOffShelf3DEnvironment) -> float:
        core = env.backend().core()
        obj = core._objects_dict[Container.name(env=env)]  # noqa: SLF001
        position, _ = obj.get_pose()
        return float(position[2]) + float(getattr(obj, "wall_thickness", 0.01))

    @staticmethod
    def wall(*, env: SweepOffShelf3DEnvironment) -> float:
        core = env.backend().core()
        obj = core._objects_dict[Container.name(env=env)]  # noqa: SLF001
        return float(getattr(obj, "wall_thickness", 0.01))


class CupboardGeometry(BaseModel):
    """The cupboard's panels and cube shelf, read from its MuJoCo geoms."""

    model_config = ConfigDict(frozen=True)

    front_x: float
    back_inner_x: float
    panel_inner_y: float
    panel_outer_y: float
    board_top_z: float

    @staticmethod
    def read(*, env: SweepOffShelf3DEnvironment) -> "CupboardGeometry":
        backend = env.backend()
        model, data, core = backend.model(), backend.data(), backend.core()

        def geom(*, suffix: str) -> tuple[np.ndarray, np.ndarray]:
            index = model.geom(f"{CUPBOARD}_{suffix}").id
            return np.asarray(data.geom_xpos[index]), np.asarray(model.geom_size[index])

        back, back_size = geom(suffix="back_panel")
        right, right_size = geom(suffix="right_panel")
        return CupboardGeometry(
            front_x=float(core.shelf_front_x()),
            back_inner_x=float(back[0] - min(back_size[:2])),
            panel_inner_y=float(abs(right[1]) - min(right_size[:2])),
            panel_outer_y=float(abs(right[1]) + min(right_size[:2])),
            board_top_z=float(core.shelf_top_z(shelf=CUBE_SHELF)),
        )


class Placements:
    """Pure geometry: labelled (x, y) cube centres against each wall that runs reach.
    Every generated state is classified after it settles, and one that does not
    classify as intended is dropped (`StateFamilies.build`), so a placement here is a
    proposal, not a promise."""

    @staticmethod
    def inside(
        *, bounds: tuple[float, float, float, float], half: float
    ) -> list[tuple[str, float, float]]:
        x0, y0, x1, y1 = bounds
        reach = half + INSIDE_MARGIN
        xs = {"near": x0 + reach, "mid": (x0 + x1) / 2, "far": x1 - reach}
        ys = {"right": y0 + reach, "centre": (y0 + y1) / 2, "left": y1 - reach}
        return [(f"in-{xn}-{yn}", x, y) for xn, x in xs.items() for yn, y in ys.items()]

    @staticmethod
    def around(
        *, bounds: tuple[float, float, float, float], wall: float, half: float
    ) -> list[tuple[str, float, float]]:
        x0, y0, x1, y1 = bounds
        out = wall + OUTSIDE_GAP + half
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        return [
            ("out-near-centre", x0 - out, ym),
            ("out-near-left", x0 - out, y1 - half),
            ("out-near-right", x0 - out, y0 + half),
            ("out-left-mid", xm, y1 + out),
            ("out-right-mid", xm, y0 - out),
            ("out-left-near", x0 + half, y1 + out),
            ("out-right-near", x0 + half, y0 - out),
            # Over the far rim: the gap between the container and the cupboard.
            ("out-far-centre", x1 + out, ym),
        ]

    @staticmethod
    def beside_panels(*, cupboard: CupboardGeometry, half: float) -> list[tuple[str, float, float]]:
        y = cupboard.panel_outer_y + PANEL_GAP + half
        xs = {
            "front": cupboard.front_x + 0.04,
            "mid": (cupboard.front_x + cupboard.back_inner_x) / 2,
        }
        return [
            (f"panel-{side}-{xn}", x, sign * y)
            for side, sign in (("left", 1.0), ("right", -1.0))
            for xn, x in xs.items()
        ]

    @staticmethod
    def shelf_edges(*, cupboard: CupboardGeometry, half: float) -> list[tuple[str, float, float]]:
        y = cupboard.panel_inner_y - half - SHELF_MARGIN
        back = cupboard.back_inner_x - half - SHELF_MARGIN
        front = cupboard.front_x + half + 0.03
        return [
            ("shelf-left-front", front, y),
            ("shelf-left-back", back, y),
            ("shelf-right-front", front, -y),
            ("shelf-right-back", back, -y),
            ("shelf-back-centre", back, 0.0),
            ("shelf-front-centre", front, 0.0),
        ]


class StateSpec(BaseModel):
    """How to make one state: a seed and a setup run on the freshly reset scene."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    state_id: str
    family: str
    seed: int
    cases: list[BenchmarkCase]
    provenance: str
    # Returns why the setup could not be made, or None when it was.
    setup: Callable[..., str | None] | None = None


class Setups:
    """The setup steps the families compose. Each returns a reason on failure."""

    @staticmethod
    def teleport(*, env: SweepOffShelf3DEnvironment, name: str, position: Any, quat: Any) -> None:
        backend = env.backend()
        obj = backend.core().get_object(name)
        obj.set_pose(list(position), list(quat))
        obj.set_velocity([0.0] * 3, [0.0] * 3)
        backend.sim().forward()

    @staticmethod
    def settle(*, env: SweepOffShelf3DEnvironment, steps: int = SETTLE_STEPS) -> None:
        backend = env.backend()
        hold = np.zeros(11)
        hold[-1] = backend.grip
        for _ in range(steps):
            backend.env().step(hold)
        env.controllers().on_objects_moved()

    @staticmethod
    def place(*, name: str, position: Any, quat: Any) -> Callable[..., str | None]:
        def setup(*, env: SweepOffShelf3DEnvironment) -> str | None:
            Setups.teleport(env=env, name=name, position=position, quat=quat)
            return None

        return setup

    @staticmethod
    def cube_at(*, x: float, y: float, z: float, cube: str) -> Callable[..., str | None]:
        def setup(*, env: SweepOffShelf3DEnvironment) -> str | None:
            Setups.teleport(env=env, name=cube, position=[x, y, z], quat=[1.0, 0.0, 0.0, 0.0])
            return None

        return setup

    @staticmethod
    def skill(*, action: np.ndarray) -> Callable[..., str | None]:
        """Run one robot skill as a prefix; the state exists only if it succeeded."""

        def setup(*, env: SweepOffShelf3DEnvironment) -> str | None:
            seed, _ = env._bookkeeping()  # noqa: SLF001
            env.current_state = env._observed_state(seed=seed, steps_taken=0)  # noqa: SLF001
            env.take_action(action=action)
            trace = SkillTraces.last(env=env)
            if trace is None or trace.outcome != "success":
                outcome = "no trace" if trace is None else f"{trace.skill} {trace.outcome}"
                return f"prefix {outcome}: {env.last_skill_error()}"
            return None

        return setup

    @staticmethod
    def reset_cubes(*, rng_seed: int) -> Callable[..., str | None]:
        def setup(*, env: SweepOffShelf3DEnvironment) -> str | None:
            backend = env.backend()
            backend._human_rng = np.random.default_rng(rng_seed)  # noqa: SLF001
            moved = backend.reset_cubes()
            return None if moved else "reset_cubes moved nothing"

        return setup

    @staticmethod
    def chain(*steps: Callable[..., str | None]) -> Callable[..., str | None]:
        def setup(*, env: SweepOffShelf3DEnvironment) -> str | None:
            for step in steps:
                reason = step(env=env)
                if reason is not None:
                    return reason
                Setups.settle(env=env)
            return None

        return setup


class StateFamilies:
    """The benchmark's families of start states (Josh, 2026-10-15): start states over
    many seeds; `reset_cubes` layouts; wiper poses `reset_wiper` produces; a dropped
    wiper (floor, container); cubes against the container's walls and the cupboard's
    side panels; a cube or the wiper in hand. `HarvestedStates` adds real run states."""

    NUM_START_SEEDS = 20
    NUM_LAYOUTS = 60
    WIPER_SHIFT_SEEDS = (0, 3)
    WIPER_FLOOR_POSES = (
        (0.90, 0.45, 0.0),
        (0.65, -0.50, 1.5708),
        (1.00, -0.45, 0.7854),
        (0.75, 0.0, 2.0944),
    )
    FLOOR_PICK_POSITIONS = ((0.95, 0.45), (0.60, -0.55), (0.70, 0.05), (0.85, -0.30))
    WIPER_IN_HAND_SEEDS = (0, 1)
    WIPER_IN_HAND_LAYOUTS = 3

    @staticmethod
    def specs(*, env: SweepOffShelf3DEnvironment) -> Iterator[StateSpec]:
        """Every spec; reading container and cupboard geometry needs a reset scene."""
        from .verdicts import SWEEP_GRID

        env.reset_to_seed(seed=0)
        half = O.cube_half_size
        bounds = Container.interior(env=env)
        wall = Container.wall(env=env)
        floor_z = Container.floor_z(env=env)
        cupboard = CupboardGeometry.read(env=env)
        wiper_start = env.backend()._wiper_start  # noqa: SLF001
        lift = 0.0005

        def case(
            *, skill: str, cube: int | None = None, params: list[float] | None = None
        ) -> BenchmarkCase:
            return BenchmarkCase(
                skill=skill, cube=None if cube is None else f"cube_{cube}", params=params or []
            )

        for seed in range(StateFamilies.NUM_START_SEEDS):
            yield StateSpec(
                state_id=f"start-s{seed}",
                family="start",
                seed=seed,
                cases=[case(skill="PickWiper"), case(skill="PickCubeFromShelf", cube=seed % 5)],
                provenance=f"reset_to_seed({seed})",
            )
        for k in range(StateFamilies.NUM_LAYOUTS):
            seed = k % StateFamilies.NUM_START_SEEDS
            cases = [case(skill="PickCubeFromShelf", cube=k % 5)]
            if k % 3 == 0:
                cases.append(case(skill="PickWiper"))
            yield StateSpec(
                state_id=f"layout-{k}",
                family="reset_cubes",
                seed=seed,
                cases=cases,
                provenance=f"reset_to_seed({seed}); reset_cubes with rng {10_000 + k}",
                setup=Setups.chain(Setups.reset_cubes(rng_seed=10_000 + k)),
            )
        from hitl_pmp.environments.sweepoffshelf3d.backend import WIPER_SHIFTS

        start_position, start_quat = wiper_start
        for seed in StateFamilies.WIPER_SHIFT_SEEDS:
            for shift in WIPER_SHIFTS[1:]:
                position = np.asarray(start_position, dtype=float).copy()
                position[1] += shift
                yield StateSpec(
                    state_id=f"wiper-shift-s{seed}-{shift:+.2f}",
                    family="reset_wiper",
                    seed=seed,
                    cases=[case(skill="PickWiper")],
                    provenance=f"reset_to_seed({seed}); wiper start shifted {shift:+.2f} m in y",
                    setup=Setups.chain(
                        Setups.place(name=O.wiper.name, position=position, quat=start_quat)
                    ),
                )
        for j, (x, y, yaw) in enumerate(StateFamilies.WIPER_FLOOR_POSES):
            quat = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            yield StateSpec(
                state_id=f"wiper-floor-{j}",
                family="dropped_wiper",
                seed=j,
                cases=[case(skill="PickCubeFromShelf", cube=j % 5)],
                provenance=f"wiper dropped on the floor at ({x}, {y}), yaw {yaw:.2f}",
                setup=Setups.chain(
                    Setups.place(name=O.wiper.name, position=[x, y, 0.003], quat=quat)
                ),
            )
        cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
        for j, yaw in enumerate((0.0, 1.5708)):
            quat = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            yield StateSpec(
                state_id=f"wiper-container-{j}",
                family="dropped_wiper",
                seed=10 + j,
                cases=[case(skill="PickCubeFromShelf", cube=(10 + j) % 5)],
                provenance=f"wiper dropped into the container, yaw {yaw:.2f}",
                setup=Setups.chain(
                    Setups.place(name=O.wiper.name, position=[cx, cy, floor_z + 0.01], quat=quat)
                ),
            )
        for i, (label, x, y) in enumerate(Placements.inside(bounds=bounds, half=half)):
            cube = i % 5
            yield StateSpec(
                state_id=f"container-{label}",
                family="near_walls",
                seed=i,
                cases=[case(skill="ReturnCubeToShelfFromBox", cube=cube)],
                provenance=f"cube_{cube} in the container at {label} ({x:.3f}, {y:.3f})",
                setup=Setups.chain(
                    Setups.cube_at(x=x, y=y, z=floor_z + half + lift, cube=f"cube_{cube}")
                ),
            )
        x_far = bounds[2] - half - INSIDE_MARGIN
        for j, ys in enumerate(((-0.06, 0.0, 0.06), (-0.18, 0.12, 0.2))):
            steps = [
                Setups.cube_at(x=x_far, y=y, z=floor_z + half + lift, cube=f"cube_{c}")
                for c, y in zip((0, 1, 2), ys, strict=True)
            ]
            yield StateSpec(
                state_id=f"container-cluster-{j}",
                family="near_walls",
                seed=j,
                cases=[
                    case(skill="ReturnCubeToShelfFromBox", cube=1),
                    case(skill="ReturnCubeToShelfFromBox", cube=2),
                ],
                provenance=f"cubes 0-2 against the far wall at y={ys}",
                setup=Setups.chain(*steps),
            )
        floor_places = [
            *Placements.around(bounds=bounds, wall=wall, half=half),
            *Placements.beside_panels(cupboard=cupboard, half=half),
        ]
        for i, (label, x, y) in enumerate(floor_places):
            cube = i % 5
            yield StateSpec(
                state_id=f"floor-{label}",
                family="near_walls",
                seed=i,
                cases=[
                    case(skill="PickCubeFromFloor", cube=cube),
                    case(skill="ReturnCubeToShelfFromFloor", cube=cube),
                ],
                provenance=f"cube_{cube} on the floor at {label} ({x:.3f}, {y:.3f})",
                setup=Setups.chain(Setups.cube_at(x=x, y=y, z=half + lift, cube=f"cube_{cube}")),
            )
        for i, (label, x, y) in enumerate(Placements.shelf_edges(cupboard=cupboard, half=half)):
            cube = i % 5
            yield StateSpec(
                state_id=f"shelf-{label}",
                family="near_walls",
                seed=i,
                cases=[case(skill="PickCubeFromShelf", cube=cube)],
                provenance=f"cube_{cube} on the cube shelf at {label} ({x:.3f}, {y:.3f})",
                setup=Setups.chain(
                    Setups.cube_at(
                        x=x, y=y, z=cupboard.board_top_z + half + lift, cube=f"cube_{cube}"
                    )
                ),
            )
        for i, (x, y) in enumerate(StateFamilies.FLOOR_PICK_POSITIONS):
            cube = (i + 1) % 5
            pick = np.array([float(SKILL_IDS["PickCubeFromFloor"]), float(cube), 0.0, 0.0])
            yield StateSpec(
                state_id=f"cube-in-hand-floor-{i}",
                family="cube_in_hand",
                seed=i,
                cases=[case(skill="PlaceCubeInBox", cube=cube)],
                provenance=f"cube_{cube} picked off the floor at ({x}, {y}) by PickCubeFromFloor",
                setup=Setups.chain(
                    Setups.cube_at(x=x, y=y, z=half + lift, cube=f"cube_{cube}"),
                    Setups.skill(action=pick),
                ),
            )
        for i in range(2):
            pick = np.array([float(SKILL_IDS["PickCubeFromShelf"]), float(i), 0.0, 0.0])
            yield StateSpec(
                state_id=f"cube-in-hand-shelf-{i}",
                family="cube_in_hand",
                seed=i,
                cases=[case(skill="PlaceCubeInBox", cube=i)],
                provenance=f"cube_{i} picked off the shelf by PickCubeFromShelf (seed {i})",
                setup=Setups.chain(Setups.skill(action=pick)),
            )
        pick_wiper = np.array([float(SKILL_IDS["PickWiper"]), 0.0, 0.0, 0.0])
        sweep = np.array([float(SKILL_IDS["SweepShelf"]), 0.0, *ORACLE_SWEEP_PARAMS])
        for seed in StateFamilies.WIPER_IN_HAND_SEEDS:
            for layout in range(StateFamilies.WIPER_IN_HAND_LAYOUTS + 1):
                steps = [Setups.skill(action=pick_wiper)]
                if layout:
                    steps.append(Setups.reset_cubes(rng_seed=20_000 + 10 * seed + layout))
                yield StateSpec(
                    state_id=f"wiper-in-hand-s{seed}-l{layout}",
                    family="wiper_in_hand",
                    seed=seed,
                    cases=[
                        *(case(skill="SweepShelf", params=[h, v]) for h, v in SWEEP_GRID),
                        case(skill="ReturnWiper"),
                    ],
                    provenance=(
                        f"reset_to_seed({seed}); PickWiper"
                        + (
                            f"; reset_cubes with rng {20_000 + 10 * seed + layout}"
                            if layout
                            else ""
                        )
                    ),
                    setup=Setups.chain(*steps),
                )
            yield StateSpec(
                state_id=f"after-sweep-s{seed}",
                family="wiper_in_hand",
                seed=seed,
                cases=[case(skill="ReturnWiper")],
                provenance=f"reset_to_seed({seed}); PickWiper; SweepShelf{ORACLE_SWEEP_PARAMS}",
                setup=Setups.chain(Setups.skill(action=pick_wiper), Setups.skill(action=sweep)),
            )

    @staticmethod
    def build(
        *, env: SweepOffShelf3DEnvironment, spec: StateSpec
    ) -> tuple[BenchmarkState | None, str]:
        """Make the state; keep only the cases its classification makes applicable.
        Returns (state, note); the state is None when it could not be made."""
        env.reset_to_seed(seed=spec.seed)
        if spec.setup is not None:
            reason = spec.setup(env=env)
            if reason is not None:
                return None, f"{spec.state_id}: setup failed: {reason}"
        Setups.settle(env=env)
        env.current_state = env._observed_state(seed=spec.seed, steps_taken=0)  # noqa: SLF001
        atoms = StateCapture.atoms(env=env)
        cases = [c for c in spec.cases if StateCapture.applicable(case=c, atoms=atoms)]
        dropped = len(spec.cases) - len(cases)
        if not cases:
            return None, f"{spec.state_id}: no case applicable; atoms {atoms}"
        state = StateCapture.capture(
            env=env,
            state_id=spec.state_id,
            family=spec.family,
            seed=spec.seed,
            cases=cases,
            provenance=spec.provenance,
        )
        note = f"{spec.state_id}: ok" + (f" ({dropped} case(s) inapplicable)" if dropped else "")
        return state, note
