"""Continuous native floor-sweeping state; only explicit human actions restore it."""

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from gymnasium.spaces import Box
from pydantic import PrivateAttr

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State
from hitl_pmp.environments.sweep_drawer3d.motion import ExecutionError

from .controllers import FloorPrimitives
from .regions import SimpleRegions
from .session import SweepSimpleSession
from .symbolic import SimpleSymbols


class SweepSimpleEnvironment(Environment):
    canonical_seed: int = 0
    evaluation: bool = False
    output_dir: Path | None = None
    action_space: ClassVar[Box] = Box(
        low=np.array([-1.0, -1.0, -100.0, -100.0]),
        high=np.array([5.0, 4.0, 100.0, 100.0]),
        dtype=np.float64,
    )
    ACTION_NAMES: ClassVar[tuple[str, ...]] = tuple(s.name for s in SimpleSymbols.skills())
    _session: SweepSimpleSession | None = PrivateAttr(default=None)
    _primitive: FloorPrimitives | None = PrivateAttr(default=None)
    _initial_state: Any = PrivateAttr(default=None)
    _action_count: int = PrivateAttr(default=0)
    _hard_reset_count: int = PrivateAttr(default=0)
    _human_reset_count: int = PrivateAttr(default=0)

    def session(self) -> SweepSimpleSession:
        assert self._session is not None
        return self._session

    def primitive(self) -> FloorPrimitives:
        if self._primitive is None:
            self._primitive = FloorPrimitives.create(
                session=self.session(), distance=0.7, heading_offset=0.0
            )
        return self._primitive

    def hard_reset(self) -> None:
        self.reset_to_seed(seed=self.canonical_seed)

    def reset_to_seed(self, *, seed: int) -> State:
        if self._action_count and not self.evaluation:
            raise RuntimeError("Automatic reset of continuous Simple practice is forbidden")
        self.close()
        directory = (
            None
            if self.output_dir is None
            else self.output_dir / "episodes" / f"{self._hard_reset_count:06d}-seed{seed}"
        )
        self._session = SweepSimpleSession(
            seed=seed,
            log_path=None if directory is None else directory / "state.jsonl",
            replay_path=None if directory is None else directory / "replay.jsonl",
        )
        validation = SimpleRegions.validate(session=self.session())
        self._write_event(
            event={
                "kind": "episode_start",
                "seed": seed,
                "evaluation": self.evaluation,
                "validation": validation.model_dump(),
            }
        )
        if not validation.valid:
            raise ValueError(f"Invalid native Simple start {seed}: {validation.reasons}")
        self._initial_state = self.session().state.copy()
        self._hard_reset_count += 1
        self.current_state = self.observe()
        return self.current_state

    def set_state(self, *, state: State) -> None:
        self.reset_to_seed(seed=int(state.get(obj=SimpleSymbols.SCENE, feature_name="seed")))

    def noop_action(self) -> Action:
        return np.array([-1.0, -1.0, 0.0, 0.0])

    def get_valid_actions(self) -> list[Action]:
        return [
            np.array([float(i), float(c), 0.0, 0.0])
            for i, skill in enumerate(SimpleSymbols.skills())
            for c in (range(5) if len(skill.parameters) == 2 else [-1])
        ]

    def observe(self) -> State:
        session = self.session()
        primitive = self.primitive()
        held = primitive.wiper_in_hand(
            gripper=np.asarray(primitive.scene.ee_now().position),
            wiper=session.position(name="wiper_0"),
        )
        validation = SimpleRegions.validate(session=session)
        wiper_home = all(
            validation.checks[f"wiper_0:{key}"] for key in ("region", "yaw", "upright")
        )
        robot_home = all(validation.checks[f"robot:{key}"] for key in ("region", "yaw"))
        facts = {
            "HandEmpty": not held and session.gripper() < 0.2,
            "HoldingWiper": held,
            "WiperAvailable": not held and session.position(name="wiper_0")[2] < 0.08,
            "WiperHome": wiper_home,
            "RobotHome": robot_home,
            "RobotAway": not robot_home,
            "ClosedEmpty": not held and session.gripper() >= 0.2,
            "seed": session.seed,
        }
        data = {
            SimpleSymbols.SCENE: np.array([
                float(facts[n]) for n in SimpleSymbols.SCENE_TYPE.feature_names
            ])
        }
        for cube in SimpleSymbols.CUBES:
            position = session.position(name=cube.name)
            at_start = SimpleRegions.contains(
                session=session, name=cube.name, region="blocks_init_region"
            )
            in_goal = SimpleRegions.contains(session=session, name=cube.name, region="sweep_region")
            values = {
                "AtStart": at_start,
                "NotAtStart": not at_start,
                "InGoal": in_goal,
                "NotInGoal": not in_goal,
                "OnFloor": -0.01 <= position[2] < 0.08,
                "x": position[0],
                "y": position[1],
                "z": position[2],
            }
            data[cube] = np.array([float(values[n]) for n in SimpleSymbols.CUBE_TYPE.feature_names])
        return State(data=data)

    def take_action(self, *, action: Action) -> State:
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("Simple action must contain skill,cube,distance,heading")
        index, cube_index = int(action[0]), int(action[1])
        if index != action[0] or not -1 <= index < len(self.ACTION_NAMES):
            raise ValueError("Invalid Simple action id")
        self._action_count += 1
        if index == -1:
            self._write_event(
                event={"kind": "action", "name": "NoOp", "index": self._action_count, "ticks": 0}
            )
            return self.get_current_state()
        name = self.ACTION_NAMES[index]
        skill = SimpleSymbols.skills()[index]
        if len(skill.parameters) == 2 and (cube_index != action[1] or not 0 <= cube_index < 5):
            raise ValueError("Invalid cube binding")
        objects: tuple[Object, ...] = (
            (SimpleSymbols.SCENE, SimpleSymbols.CUBES[cube_index])
            if len(skill.parameters) == 2
            else (SimpleSymbols.SCENE,)
        )
        grounded = GroundSkill(skill=skill, objects=objects)
        before = self.session().ticks
        self.session().begin(
            name=name, kind=name, phase="evaluation" if self.evaluation else "practice"
        )
        error = ""
        try:
            if not all(
                a.predicate.holds(self.get_current_state(), a.objects)
                for a in grounded.preconditions
            ):
                raise ExecutionError("Observed action preconditions do not hold")
            primitive = self.primitive()
            if name == "PickFloorWiper":
                primitive.distance = float(action[2])
                primitive.heading_offset = float(action[3])
                primitive.recover_wiper()
            elif name in ("SweepCubeToGoal", "SweepCubeToStart"):
                primitive.sweep_cube(
                    cube=objects[1].name,
                    region="sweep_region" if name == "SweepCubeToGoal" else "blocks_init_region",
                    distance=float(action[2]) if skill.param_dim else 0.70,
                    heading_offset=float(action[3]) if skill.param_dim else 0.0,
                )
            elif name == "PlaceWiperAtStart":
                # This fixed skill must not inherit the previous learned pickup.
                primitive.distance = 0.70
                primitive.heading_offset = 0.0
                primitive.place_wiper_at_start()
            elif name == "OpenGripper":
                primitive.motion.set_gripper(command=0.0)
            elif name == "ReturnRobotToStart":
                robot = self._initial_state.get_object_from_name("robot")
                initial_arm = np.array([
                    self._initial_state.get(robot, f"pos_arm_joint{i}") for i in range(1, 8)
                ])
                path = primitive.scene.plan_arm(
                    goal=initial_arm, bodies=primitive.scene.bodies() | {primitive.scene.wiper_body}
                )
                if path is None or not primitive.motion.follow(path=path, grip=0.0):
                    raise ExecutionError("No collision-free empty-hand return posture")
                target = tuple(
                    float(self._initial_state.get(robot, key))
                    for key in ("pos_base_x", "pos_base_y", "pos_base_rot")
                )
                primitive.motion.drive_to(target=(target[0], target[1], target[2]), grip=0.0)
        except ExecutionError as exc:
            error = str(exc)
        self.current_state = self.observe()
        success = all(
            a.predicate.holds(self.current_state, a.objects) for a in grounded.add_effects
        )
        self.session().end(success=success, note=error)
        self._write_event(
            event={
                "kind": "action",
                "name": name,
                "cube": cube_index,
                "params": action[2:].tolist(),
                "index": self._action_count,
                "ticks": self.session().ticks - before,
                "symbolic_success": success,
                "error": error,
            }
        )
        return self.current_state

    def reset_movables(self, *, destination: str | None = None) -> bool:
        if self.evaluation or destination is not None or self._initial_state is None:
            raise RuntimeError("Human reset requires the original validated practice sample")
        self.session().begin(name="HumanReset", kind="HumanReset", phase="practice")
        self.session().restore(state=self._initial_state.copy())
        self.current_state = self.observe()
        validation = SimpleRegions.validate(session=self.session())
        success = validation.valid and all(
            a.predicate.holds(self.current_state, a.objects) for a in SimpleSymbols.initial_atoms()
        )
        self._human_reset_count += 1
        self.session().end(success=success)
        self._write_event(
            event={
                "kind": "human_reset",
                "index": self._human_reset_count,
                "success": success,
                "validation": validation.model_dump(),
            }
        )
        if not success:
            raise RuntimeError("Human restoration failed the shared start contract")
        return True

    def _write_event(self, *, event: dict[str, Any]) -> None:
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / "sweep_events.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")

    def close(self) -> None:
        if self._primitive is not None:
            self._primitive.scene._sim.close()
            self._primitive = None
        if self._session is not None:
            self._session.close()
            self._session = None
