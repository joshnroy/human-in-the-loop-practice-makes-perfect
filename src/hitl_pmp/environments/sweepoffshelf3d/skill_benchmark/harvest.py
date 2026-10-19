"""Benchmark states harvested from real runs: the scene just before a robot skill that
failed, taken from a run's `recording.jsonl`.

A recording keeps, at each action boundary, the domain's flat features
(`step_protocol.live_state`): the base pose and gripper command, each cube's position,
the wiper's position and yaw, and the container's interior bounds. That is not MuJoCo's
whole state, so a harvested state is a reconstruction, and it says so: the arm is put
at the skill-start pose every skill retracts to, each cube upright at yaw 0 (a
recording does not keep cube orientation), the wiper flat at its recorded yaw. A
moment with something in the hand cannot be rebuilt from these features and is
skipped. After the scene settles, the state is kept only if the failed skill's
preconditions hold in it (`StateFamilies.build`).
"""

import json
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects as O

from .states import Container, Setups, StateSpec
from .types import BenchmarkCase

RANDOM_PREFIX = "random: "
HELD_GRIP = 0.5
# Two harvested moments closer than this (m, every object) are the same state.
DEDUPE_RESOLUTION = 0.02
BOUNDS_TOLERANCE = 0.01
PER_SKILL = 8


class Harvest(BaseModel):
    source: str
    scene_seed: int
    skill: str
    cube: str | None
    outcome: str
    features: dict[str, list[float]]

    def key(self) -> tuple[Any, ...]:
        rounded = tuple(
            (name, tuple(np.round(np.asarray(values[:3]) / DEDUPE_RESOLUTION).astype(int)))
            for name, values in sorted(self.features.items())
            if name != O.box.name
        )
        return (self.skill, self.cube, rounded)


class HarvestedStates:
    """A static-method container, never instantiated."""

    @staticmethod
    def recordings(*, paths: list[Path]) -> Iterator[Path]:
        for path in paths:
            if path.is_file():
                yield path
            else:
                yield from sorted(path.rglob("recording.jsonl"))

    @staticmethod
    def candidates(*, paths: list[Path], per_skill: int = PER_SKILL) -> list[Harvest]:
        """Failed robot skills from empty-handed states, deduplicated, at most
        `per_skill` per skill, in file order."""
        kept: dict[str, list[Harvest]] = defaultdict(list)
        seen: set[tuple[Any, ...]] = set()
        for recording in HarvestedStates.recordings(paths=paths):
            for harvest in HarvestedStates.failures(path=recording):
                key = harvest.key()
                if key in seen or len(kept[harvest.skill]) >= per_skill:
                    continue
                seen.add(key)
                kept[harvest.skill].append(harvest)
        return [h for skill in sorted(kept) for h in kept[skill]]

    @staticmethod
    def failures(*, path: Path) -> Iterator[Harvest]:
        scene_seed = 0
        features: dict[str, list[float]] | None = None
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                kind = record.get("kind")
                if kind == "header" and record.get("environment") != "sweepoffshelf3d":
                    return
                if kind == "segment":
                    scene_seed = int(record.get("scene_seed") or record.get("task_seed") or 0)
                elif kind == "tick" and record.get("state"):
                    features = record["state"]
                elif kind == "action":
                    state, features = features, None
                    if state is None or record.get("actor") != "robot" or record.get("success"):
                        continue
                    skill = str(record.get("skill", "")).removeprefix(RANDOM_PREFIX)
                    if skill in {"", "STOP", "no-op"}:
                        continue
                    if state.get(O.robot.name, [0, 0, 0, 0])[3] > HELD_GRIP:
                        continue
                    objects = record.get("objects") or []
                    cubes = [o for o in objects if o.startswith("cube_")]
                    yield Harvest(
                        source=f"{path}#{record.get('seq')}",
                        scene_seed=scene_seed,
                        skill=skill,
                        cube=cubes[0] if len(cubes) == 1 else None,
                        outcome=str(record.get("outcome"))[:200],
                        features=state,
                    )

    @staticmethod
    def specs(*, harvests: list[Harvest]) -> list[StateSpec]:
        specs = []
        for index, harvest in enumerate(harvests):
            specs.append(
                StateSpec(
                    state_id=f"harvest-{harvest.skill}-{index}",
                    family="harvested",
                    seed=harvest.scene_seed,
                    cases=[BenchmarkCase(skill=harvest.skill, cube=harvest.cube)],
                    provenance=f"{harvest.source}: failed with {harvest.outcome}",
                    setup=HarvestedStates.setup(features=harvest.features),
                )
            )
        return specs

    @staticmethod
    def setup(*, features: dict[str, list[float]]) -> Callable[..., str | None]:
        def apply(*, env: SweepOffShelf3DEnvironment) -> str | None:
            backend = env.backend()
            model, data = backend.model(), backend.data()
            base = features[O.robot.name][:3]
            for axis, value in zip(("x", "y", "th"), base, strict=True):
                data.qpos[model.jnt_qposadr[model.joint(f"robot_joint_{axis}").id]] = value
            backend.sim().forward()
            for cube in O.cubes:
                if cube.name in features:
                    x, y, z = features[cube.name][:3]
                    Setups.teleport(
                        env=env, name=cube.name, position=[x, y, z + 0.0005], quat=[1, 0, 0, 0]
                    )
            if O.wiper.name in features:
                x, y, z, yaw = features[O.wiper.name][:4]
                quat = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
                Setups.teleport(env=env, name=O.wiper.name, position=[x, y, z + 0.001], quat=quat)
            recorded = features.get(Container.name(env=env))
            if recorded is not None:
                reason = HarvestedStates.move_container(env=env, bounds=recorded[:4])
                if reason is not None:
                    return reason
            return None

        return apply

    @staticmethod
    def move_container(*, env: SweepOffShelf3DEnvironment, bounds: list[float]) -> str | None:
        """Translate the container to the recorded interior bounds; refuse when their
        size says it was turned, which the bounds alone cannot undo."""
        x0, y0, x1, y1 = Container.interior(env=env)
        rx0, ry0, rx1, ry1 = bounds
        if (
            abs((rx1 - rx0) - (x1 - x0)) > BOUNDS_TOLERANCE
            or abs((ry1 - ry0) - (y1 - y0)) > BOUNDS_TOLERANCE
        ):
            return "container was turned in the recording; its pose cannot be rebuilt"
        core = env.backend().core()
        obj = core._objects_dict[Container.name(env=env)]  # noqa: SLF001
        position, quat = obj.get_pose()
        shifted = np.asarray(position, dtype=float).copy()
        shifted[0] += (rx0 + rx1) / 2 - (x0 + x1) / 2
        shifted[1] += (ry0 + ry1) / 2 - (y0 + y1) / 2
        Setups.teleport(env=env, name=Container.name(env=env), position=shifted, quat=quat)
        return None
