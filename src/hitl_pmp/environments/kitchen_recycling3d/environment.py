"""Experiment-harness boundary for the observed multi-object recycling simulator."""

from collections.abc import Callable
from typing import Any, ClassVar, Literal, cast

import numpy as np
from gymnasium.spaces import Box
from pydantic import ConfigDict, Field, PrivateAttr
from scipy.spatial.transform import Rotation

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State

from .domain import KitchenConfig, classify_zone, projected_bounds
from .scene import initial_positions
from .types import NEAR_BIN_POSITION, KitchenObjects, KitchenState


class KitchenRecyclingEnvironment(Environment):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    action_space: ClassVar[Box] = Box(-np.inf, np.inf, (6,), dtype=np.float64)
    config: KitchenConfig = Field(default_factory=KitchenConfig)
    phase: Literal["training", "deployment"] = "training"
    seed: int = 0
    scene_bg: bool = False

    _simulator: Any = PrivateAttr(default=None)
    _skills_taken: int = PrivateAttr(default=0)
    _blocked: dict[str, tuple[float, ...]] = PrivateAttr(default_factory=dict)
    _error: str | None = PrivateAttr(default=None)
    _controller_steps: tuple[int, ...] = PrivateAttr(default=())
    _before: Any = PrivateAttr(default=None)
    _after: Any = PrivateAttr(default=None)
    _frames: list = PrivateAttr(default_factory=list)
    _record_frames: bool = PrivateAttr(default=False)
    _state_key: tuple | None = PrivateAttr(default=None)

    @property
    def objects(self) -> KitchenObjects:
        return KitchenObjects(config=self.config)

    @property
    def robot(self) -> Object:
        return self.objects.robot

    @property
    def bin(self) -> Object:
        return self.objects.bin

    @property
    def items(self) -> tuple[Object, ...]:
        return self.objects.items

    def simulator(self) -> Any:
        if self._simulator is None:
            self.hard_reset()
        return self._simulator

    def hard_reset(self) -> None:
        self.reset_to_seed(seed=self.seed)

    def reset_to_seed(self, *, seed: int) -> KitchenState:
        from .simulator import RecyclingSimulator

        self.close()
        self.seed = seed
        self._skills_taken = 0
        self._blocked = {}
        self._error = None
        self._state_key = None
        self._simulator = RecyclingSimulator(
            config=self.config, phase=self.phase, seed=seed, pretty=self.scene_bg
        )
        self._simulator.before_step = self._before
        self._simulator.after_step = self._after
        self._simulator.frame_callback = self._capture_frame if self._record_frames else None
        return self.get_current_state()

    def get_current_state(self) -> KitchenState:
        simulator = self.simulator()
        episode = simulator.episode
        objects = self.objects
        raw = simulator.backend._require_state()
        robot = raw.get_object_from_name(simulator.backend.robot_name)
        opened = raw.get(robot, "pos_gripper") < 0.2
        data = {
            objects.robot: np.array(
                [
                    *(raw.get(robot, key) for key in ("pos_base_x", "pos_base_y", "pos_base_rot")),
                    opened and not episode.holding,
                    self.phase == "training",
                    self.seed,
                    self._skills_taken,
                    opened,
                ],
                dtype=float,
            )
        }
        for item in objects.items:
            observation = episode.observations[item.name]
            spec = self.config.object(name=item.name)
            lower, _ = projected_bounds(
                spec=spec, rotation=Rotation.from_quat(observation.quaternion)
            )
            on_floor = classify_zone(config=self.config, observation=observation) == "kitchen" and (
                abs(observation.position[2] + lower[2]) < 0.035
            )
            nearest = np.clip(
                observation.position[:2],
                np.array(self.config.kitchen_bounds[:2]) + self.config.robot_radius,
                np.array(self.config.kitchen_bounds[2:]) - self.config.robot_radius,
            )
            reachable = (
                np.linalg.norm(nearest - observation.position[:2])
                <= self.config.direct_placement_reach
            )
            last = self._blocked.get(item.name)
            signature = tuple((
                *observation.position,
                *observation.quaternion,
                *data[objects.robot][:3],
            ))
            blocked = last is not None and np.allclose(last, signature, atol=0.01)
            data[item] = np.array(
                [
                    *observation.position,
                    *observation.quaternion,
                    observation.linear_speed,
                    observation.angular_speed,
                    observation.finger_contacts,
                    item.name in episode.holding,
                    item.name in episode.settled,
                    reachable,
                    on_floor,
                    blocked,
                ],
                dtype=float,
            )
        position, quaternion = episode.bin_pose
        at_target = np.linalg.norm(np.array(position) - self.config.bin_position) < 0.025 and (
            Rotation.from_quat(quaternion).magnitude() < 0.05
        )
        data[objects.bin] = np.array([
            *position,
            *quaternion,
            episode.context() == "near",
            at_target,
        ])
        key = (simulator.steps, self._skills_taken, self.seed, self.phase)
        current = cast(KitchenState, self.current_state)
        if self._state_key == key and all(
            np.array_equal(values, current.data[obj]) for obj, values in data.items()
        ):
            return current
        self.current_state = KitchenState(data=data)
        self._state_key = key
        return self.current_state

    def set_state(self, *, state: State) -> None:
        if isinstance(state, KitchenState) and state.snapshot is not None:
            self.restore(snapshot=state.snapshot)
            return
        if state.get(obj=self.robot, feature_name="skills_taken") != 0:
            raise ValueError("Mid-episode restore requires a complete simulator snapshot")
        self.reset_to_seed(seed=int(state.get(obj=self.robot, feature_name="seed")))

    def reset_to_state(self, *, state: State) -> State:
        self.set_state(state=state)
        return self.get_current_state()

    def take_action(self, *, action: Action) -> KitchenState:
        action = np.asarray(action, dtype=float)
        simulator = self.simulator()
        self._error = None
        start = simulator.steps
        opcode = int(action[0]) if action.shape == (6,) and np.isfinite(action).all() else -99
        if opcode in (3, 4, 5):
            raise PermissionError("Human interventions require the costed human-request protocol")
        try:
            if opcode == 0 or opcode == 1:
                index = int(action[1])
                if not 0 <= index < len(self.config.objects):
                    raise ValueError("Unknown recycling object index")
                name = self.config.objects[index].name
                calibration = self.config.calibration(
                    name=name, context=simulator.episode.context()
                )
                if opcode == 0:
                    calibration = type(calibration).model_validate({
                        **calibration.model_dump(),
                        "grasp_offset": action[2],
                    })
                    result = simulator.pick(name=name, calibration=calibration)
                else:
                    values = dict(
                        zip(
                            ("toss_standoff", "toss_azimuth", "toss_speed_deg_s", "release_ms"),
                            action[2:],
                            strict=True,
                        )
                    )
                    calibration = type(calibration).model_validate({
                        **calibration.model_dump(),
                        **values,
                    })
                    result = simulator.toss(name=name, calibration=calibration)
                if opcode == 0 and result.get("error") and result["steps"] == 0:
                    state = self.get_current_state()
                    observation = simulator.episode.observations[name]
                    self._blocked[name] = tuple((
                        *observation.position,
                        *observation.quaternion,
                        *state.data[self.robot][:3],
                    ))
                self._error = result.get("error")
            elif opcode == 2:
                result = simulator.open_gripper()
                self._error = result.get("error")
            elif opcode != -1:
                self._error = "Invalid recycling action"
        except (ValueError, RuntimeError, PermissionError) as error:
            self._error = f"{type(error).__name__}: {error}"
        self._skills_taken += 1
        self._controller_steps = (simulator.steps - start,)
        return self.get_current_state()

    def get_valid_actions(self) -> list[Action]:
        return []

    def noop_action(self) -> Action:
        return np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def reset_movables(self, *, destination: str | None = None) -> bool:
        if self.phase != "training":
            raise PermissionError("Human recovery and bin relocation are disabled in deployment")
        simulator = self.simulator()
        before, after = simulator.before_step, simulator.after_step
        simulator.before_step = simulator.after_step = None
        try:
            if destination and destination.startswith("recover:"):
                name = destination.split(":", 1)[1]
                spec = self.config.object(name=name)
                requested = len([
                    event for event in simulator.episode.events if event["kind"] == "human"
                ])
                robot_position = self.get_current_state().data[self.robot][:2]
                for offset in range(1000):
                    positions = initial_positions(
                        config=self.config.model_copy(update={"scatter_objects": True}),
                        seed=self.seed + 100000 + requested * 1000 + offset,
                    )
                    horizontal, lateral = positions[0]
                    position = np.array((horizontal, lateral))
                    if np.linalg.norm(position - robot_position) < self.config.robot_radius + 0.2:
                        continue
                    if any(
                        np.linalg.norm(position - observation.position[:2]) < 0.3
                        for other, observation in simulator.episode.observations.items()
                        if other != name
                    ):
                        continue
                    break
                else:
                    raise ValueError("No clear reachable recovery position")
                event = simulator.human(
                    action="recover_object",
                    name=name,
                    destination=(horizontal, lateral, spec.dimensions[2] / 2 + 0.002),
                )
                self._blocked.pop(name, None)
            elif destination == "relocate_bin_near":
                event = simulator.human(action="relocate_bin", destination=NEAR_BIN_POSITION)
                self._blocked.clear()
            elif destination == "restore_bin_target":
                event = simulator.human(action="restore_bin")
                self._blocked.clear()
            else:
                raise ValueError(f"Unknown recycling human destination: {destination}")
        finally:
            simulator.before_step, simulator.after_step = before, after
            self.get_current_state()
        return bool(event["completed"])

    def cost_observation(self) -> dict[str, Any]:
        return {"human_cost": self.simulator().episode.human_cost, "phase": self.phase}

    def is_solved(self) -> bool:
        return self.simulator().episode.solved

    def snapshot(self) -> dict[str, Any]:
        return {
            "simulator": self.simulator().snapshot(),
            "phase": self.phase,
            "config": self.config.model_dump(),
            "skills_taken": self._skills_taken,
            "blocked": dict(self._blocked),
            "error": self._error,
        }

    def restore(self, *, snapshot: Any) -> KitchenState:
        if snapshot["phase"] != self.phase or snapshot["config"] != self.config.model_dump():
            raise ValueError("Snapshot phase/config differs from this environment")
        self.simulator().restore(snapshot=snapshot["simulator"])
        self.seed = snapshot["simulator"]["seed"]
        self._skills_taken = snapshot["skills_taken"]
        self._blocked = dict(snapshot["blocked"])
        self._error = snapshot["error"]
        self._state_key = None
        return self.get_current_state()

    def set_control_step_observers(
        self, *, before: Callable[[], None] | None, after: Callable[[], None] | None
    ) -> None:
        self._before, self._after = before, after
        if self._simulator is not None:
            self._simulator.before_step, self._simulator.after_step = before, after

    def adopt_interrupted_skill(self, *, action: Action, reason: str) -> State:
        del action
        self._skills_taken += 1
        self._error = reason
        return self.get_current_state()

    def last_skill_error(self) -> str | None:
        return self._error

    def last_controller_steps(self) -> tuple[int, ...]:
        return self._controller_steps

    def drain_execution_timing(self) -> None:
        return None

    def set_substep_recording(self, *, enabled: bool) -> None:
        self._record_frames = enabled
        self._frames.clear()
        if self._simulator is not None:
            self._simulator.frame_callback = self._capture_frame if enabled else None

    def _capture_frame(self, *, frame: Any, caption: Any, episode: Any) -> None:
        del caption, episode
        self._frames.append(frame)

    def drain_substep_frames(self) -> list[np.ndarray]:
        frames, self._frames = self._frames, []
        return frames

    def close(self) -> None:
        if self._simulator is not None:
            self._simulator.close()
            self._simulator = None
