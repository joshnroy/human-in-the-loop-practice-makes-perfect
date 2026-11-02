"""MuJoCo recycling adapter with explicit action authorization and physical scoring.

Only initialization and logged human interventions may set object poses. Pick and
toss always execute motor commands; failed throws remain in the observed scene.
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend

from .domain import (
    KitchenConfig,
    ObjectObservation,
    RecyclingEpisode,
    classify_zone,
    projected_bounds,
)
from .drive_zone import KitchenDriveZone
from .scene import internal_name, register_simulation_objects, scene_config


class RecyclingSimulator:
    """An opt-in skill-level environment, not a replacement benchmark registration."""

    def __init__(
        self,
        *,
        config: KitchenConfig | None = None,
        phase: Any = "training",
        seed: int = 0,
        frame_callback: Any = None,
        pretty: bool = True,
    ) -> None:
        self.config = config or KitchenConfig()
        self.episode = RecyclingEpisode(config=self.config, phase=phase)
        self.frame_callback = frame_callback
        self.caption = "Initial scene"
        self.steps = 0
        self.before_step: Any = None
        self.after_step: Any = None
        self.seed = seed
        self.backend = KinderBackend(scene_bg=pretty, allow_state_access=True)
        api = self.backend.api()
        register_simulation_objects()
        from gymnasium.envs.registration import EnvSpec
        from kinder.envs.dynamic3d.envs import TidyBot3DConfig

        with tempfile.TemporaryDirectory(prefix="kitchen-recycling-") as temporary:
            path = Path(temporary) / "scene.json"
            path.write_text(json.dumps(scene_config(config=self.config, seed=seed)))
            specification = EnvSpec(
                id="KitchenRecycling3D-v0",
                entry_point="hitl_pmp.environments.kitchen_recycling3d.furniture:RecyclingEnv",
                kwargs={
                    "task_config_path": str(path),
                    "scene_render_camera": "task_view",
                    "config": TidyBot3DConfig(use_arm_velocities=True),
                },
            )
            self.backend._raw_env = api.kinder.make(
                specification, render_mode="rgb_array", scene_bg=pretty, allow_state_access=True
            )
        self.backend._env = self.backend._raw_env
        vector, _ = self.backend._env.reset(seed=seed)
        self.backend._state = self.backend._env.observation_space.devectorize(vector)
        self._remove_visuals_from_planning()
        self.backend._robot_name = next(iter(self.backend._state.get_objects(api.robot_type))).name
        self.scene = self.backend._object_centric()
        if not self.scene._robot_env.act_delta:
            raise ValueError("The recycling motor guard requires delta-position control")
        self.scene.set_render_camera("task_view")
        physics = self.scene._robot_env.sim
        self.model, self.data = physics.model.mj_model, physics.data.mj_data
        self.names = {
            spec.name: internal_name(index=index) for index, spec in enumerate(self.config.objects)
        }
        self._initialize_sideways_bottle()
        self.observe()
        for observation in self.episode.observations.values():
            if classify_zone(config=self.config, observation=observation) != "kitchen":
                raise ValueError("Initial objects must fit entirely on the kitchen side")
        self.idle(seconds=0.8)

    def _initialize_sideways_bottle(self) -> None:
        state = self.backend._require_state().copy()
        rng = np.random.default_rng(self.seed + 81000)
        for spec in self.config.objects:
            if spec.kind != "bottle":
                continue
            obj = state.get_object_from_name(self.names[spec.name])
            yaw = rng.uniform(-np.pi, np.pi) if self.config.scatter_objects else 0.0
            quaternion = (
                Rotation.from_euler("z", yaw) * Rotation.from_euler("y", np.pi / 2)
            ).as_quat()
            for feature, value in zip(("qx", "qy", "qz", "qw"), quaternion, strict=True):
                state.set(obj, feature, float(value))
            state.set(obj, "z", spec.dimensions[0] / 2 + 0.002)
        self.scene.set_state(state)
        self.backend._state = state

    def _remove_visuals_from_planning(self) -> None:
        state = self.backend._state
        for obj in list(state):
            if obj.name == "sprout_mats":
                del state.data[obj]

    def observe(self) -> RecyclingEpisode:
        import mujoco

        observations = []
        for spec in self.config.objects:
            body = self.model.body(self.names[spec.name]).id
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0
            )
            contacts = set()
            for contact in self.data.contact:
                bodies = [
                    int(self.model.geom_bodyid[geom]) for geom in (contact.geom1, contact.geom2)
                ]
                if body not in bodies:
                    continue
                other = bodies[1] if bodies[0] == body else bodies[0]
                body_name = self.model.body(other).name
                if "pad" in body_name or "finger" in body_name:
                    for side in ("left", "right"):
                        if side in body_name:
                            contacts.add(side)
            observations.append(
                ObjectObservation(
                    name=spec.name,
                    position=tuple(self.data.xpos[body]),
                    quaternion=tuple(self.data.xquat[body][[1, 2, 3, 0]]),
                    angular_speed=float(np.linalg.norm(velocity[:3])),
                    linear_speed=float(np.linalg.norm(velocity[3:])),
                    finger_contacts=len(contacts),
                )
            )
        body = self.model.body("bin_0").id
        bin_pose = (tuple(self.data.xpos[body]), tuple(self.data.xquat[body][[1, 2, 3, 0]]))
        self.episode.update(
            observations=observations, bin_pose=bin_pose, time=float(self.data.time)
        )
        return self.episode

    def step(self, *, action: Any) -> None:
        action = np.asarray(action)
        if (
            action.ndim not in (1, 2)
            or action.shape[-1] != self.backend._env.action_space.shape[-1]
        ):
            raise ValueError("Unexpected motor action shape")
        if not np.isfinite(action).all():
            raise ValueError("Nonfinite motor command")
        state = self.backend._require_state()
        robot = state.get_object_from_name(self.backend.robot_name)
        start = np.array([state.get(robot, key) for key in ("pos_base_x", "pos_base_y")])
        for row in np.atleast_2d(action):
            self.episode.check_drive(start=start, end=start + row[:2])
        if self.before_step is not None:
            self.before_step()
        vector, _, _, _, _ = self.backend._env.step(action)
        self.backend._state = self.backend._env.observation_space.devectorize(vector)
        self._remove_visuals_from_planning()
        self.steps += 1
        self.observe()
        current = self.backend._state.get_object_from_name(self.backend.robot_name)
        actual = [self.backend._state.get(current, key) for key in ("pos_base_x", "pos_base_y")]
        self.episode.check_drive(start=actual, end=actual)
        if self.after_step is not None:
            self.after_step()
        if self.frame_callback is not None:
            self.frame_callback(
                frame=self.backend.render(), caption=self.caption, episode=self.episode
            )

    def idle(self, *, seconds: float, closed: bool = False) -> None:
        action = np.zeros(self.backend._env.action_space.shape, dtype=np.float32)
        action[10] = float(closed)
        for _ in range(max(1, int(np.ceil(seconds * self.scene._robot_env.control_frequency)))):
            self.step(action=action)

    def _execute(self, *, controller: Any, params: Any, limit: int) -> dict[str, Any]:
        from bilevel_planning.trajectory_samplers.trajectory_sampler import (
            TrajectorySamplingFailure,
        )

        start = self.steps
        error: str | None = None
        try:
            controller.reset(self.backend._require_state(), params)
            for _ in range(limit):
                self.step(action=controller.step())
                controller.observe(self.backend._require_state())
                if controller.terminated():
                    break
            else:
                error = "control-step limit"
            error = getattr(controller, "failure", None) or error
        except (ValueError, RuntimeError, PermissionError, TrajectorySamplingFailure) as failure:
            error = f"{type(failure).__name__}: {failure}"
        return {"steps": self.steps - start, "error": error}

    def pick(self, *, name: str, calibration: Any = None) -> dict[str, Any]:
        from .controllers import RecyclingPickController

        spec = self.config.object(name=name)
        if self.episode.holding:
            raise ValueError("Gripper must be empty before picking another object")
        context = self.episode.context()
        observation = self.episode.observations[name]
        self.episode.start_pick(name=name)
        if classify_zone(
            config=self.config, observation=observation
        ) != "kitchen" and not KitchenDriveZone.pick_reachable(
            config=self.config, position=observation.position
        ):
            result: dict[str, Any] = {
                "steps": 0,
                "error": "Object outside permitted robot reach",
                "success": False,
            }
            self.episode.record_practice(name=name, skill="pick", success=False, context=context)
            return result
        state = self.backend._require_state()
        objects = [
            state.get_object_from_name(self.backend.robot_name),
            state.get_object_from_name(self.names[name]),
        ]
        objects.extend(
            obj
            for obj in state
            if obj not in objects
            and (
                obj.name == "cuboid_barrier"
                or obj.name == "bin_0"
                or obj.name in self.names.values()
            )
        )
        collision = self.backend._pick_collision_sim(state=state)
        collision._bounding_boxes["cuboid_barrier"] = (0.016, 5.0, 0.001)
        controller = RecyclingPickController(objects, pybullet_sim=collision)
        vars(controller)["ACTION_DIM"] = self.backend._env.action_space.shape[0]
        controller.configure(
            episode=self.episode,
            spec=spec,
            calibration=calibration or self.config.calibration(name=name, context=context),
        )
        self.caption = f"PICK {name} | {context} context"
        try:
            result = self._execute(controller=controller, params=None, limit=550)
            self.idle(seconds=0.3, closed=True)
            result["success"] = name in self.episode.holding
            self.episode.record_practice(
                name=name, skill="pick", success=result["success"], context=context
            )
            return result
        finally:
            collision.close()

    def toss(self, *, name: str, calibration: Any = None) -> dict[str, Any]:
        if name not in self.episode.holding:
            raise ValueError("Toss requires sustained two-finger contact and observed lift")
        context = self.episode.context()
        calibration = calibration or self.config.calibration(name=name, context=context)
        state = self.backend._require_state()
        collision = self.backend._pick_collision_sim(state=state)
        collision._bounding_boxes["cuboid_barrier"] = (0.016, 5.0, 0.001)
        from .controllers import RecyclingTossController

        controller = RecyclingTossController(
            tuple(
                state.get_object_from_name(key)
                for key in (self.backend.robot_name, self.names[name], "cuboid_barrier")
            ),
            pybullet_sim=collision,
        )
        vars(controller)["ACTION_DIM"] = self.backend._env.action_space.shape[0]
        controller.configure(config=self.config)
        self.caption = f"TOSS {name} | {context} context | uncalibrated simulation proposal"
        try:
            result = self._execute(
                controller=controller,
                params=np.array([
                    calibration.toss_standoff,
                    calibration.toss_azimuth,
                    np.deg2rad(calibration.toss_speed_deg_s),
                    calibration.release_ms,
                ]),
                limit=550,
            )
            if not getattr(controller, "_has_released", False):
                self.idle(seconds=0.3, closed=True)
                result["outcome"] = "not_released"
            else:
                quiet_since = None
                for _ in range(
                    int(calibration.settle_timeout * self.scene._robot_env.control_frequency)
                ):
                    self.idle(seconds=1 / self.scene._robot_env.control_frequency)
                    observation = self.episode.observations[name]
                    quiet = (
                        observation.linear_speed <= calibration.maximum_linear_speed
                        and observation.angular_speed <= calibration.maximum_angular_speed
                    )
                    quiet_since = (
                        (float(self.data.time) if quiet_since is None else quiet_since)
                        if quiet
                        else None
                    )
                    if name in self.episode.settled or (
                        quiet_since is not None
                        and float(self.data.time) - quiet_since >= calibration.settle_duration
                    ):
                        result["outcome"] = "settled"
                        break
                else:
                    result["outcome"] = "unsettled_at_timeout"
            result["success"] = name in self.episode.settled
            self.episode.record_practice(
                name=name, skill="toss", success=result["success"], context=context
            )
            return result
        finally:
            collision.close()

    def human(self, *, action: str, name: Any = None, destination: Any = None) -> dict[str, Any]:
        event = self.episode.request_human(action=action, object_name=name, destination=destination)
        if action == "restore_bin":
            destination = self.config.bin_position
        target = self.names[name] if action == "recover_object" else "bin_0"
        state = self.backend._require_state().copy()
        obj = state.get_object_from_name(target)
        for feature, value in zip("xyz", destination, strict=True):
            state.set(obj, feature, float(value))
        for feature, value in zip(("qx", "qy", "qz", "qw"), (0, 0, 0, 1), strict=True):
            state.set(obj, feature, value)
        for feature in state.type_features[obj.type]:
            if feature in ("vx", "vy", "vz", "wx", "wy", "wz"):
                state.set(obj, feature, 0.0)
        self.scene.set_state(state)
        self.backend._state = state
        self.episode._settle_since.clear()
        self.episode._hold_since.clear()
        self.episode._initial_height.pop(name, None)
        self.caption = f"HUMAN {action} | interruption + handling cost = {event['cost']:g}"
        self.idle(seconds=0.8)
        actual = self.episode.observations[name].position if name else self.episode.bin_pose[0]
        event["observed_position"] = list(actual)
        event["completed"] = bool(np.linalg.norm(np.array(actual) - destination) < 0.025)
        return event

    def begin_deployment(self) -> None:
        self.episode.begin_deployment()
        self.caption = "DEPLOYMENT: human at gym | no help, no bin relocation"

    def open_gripper(self) -> dict[str, Any]:
        from hitl_pmp.environments.tossing3d.kinder_backend import (
            ClearReleaseController,
            OpenGripperSettlingController,
        )

        robot = self.backend._state.get_object_from_name(self.backend.robot_name)
        if not self.episode.holding and self.backend._state.get(robot, "pos_gripper") < 0.2:
            return {"steps": 0, "error": None, "success": True}
        released_name = next(iter(self.episode.holding), None)
        name = released_name or self.config.objects[0].name
        view = cast(
            KinderBackend, RecyclingBackendView(backend=self.backend, cube_name=self.names[name])
        )
        state = self.backend._require_state()
        controller = (
            self.backend
            .api()
            .tossing_controllers(self.backend._env.action_space, init_constant_state=state)[
                "open_gripper"
            ]
            .ground((state.get_object_from_name(self.backend.robot_name),))
        )
        controller = ClearReleaseController(
            backend=view,
            controller=OpenGripperSettlingController(backend=view, controller=controller),
        )
        try:
            result = self._execute(controller=controller, params=None, limit=150)
            robot = self.backend._state.get_object_from_name(self.backend.robot_name)
            result["success"] = (
                not self.episode.holding and self.backend._state.get(robot, "pos_gripper") < 0.2
            )
            if released_name is not None:
                observation = self.episode.observations[released_name]
                lower, _ = projected_bounds(
                    spec=self.config.object(name=released_name),
                    rotation=Rotation.from_quat(observation.quaternion),
                )
                nearest = np.clip(
                    observation.position[:2],
                    np.array(self.config.kitchen_bounds[:2]) + self.config.robot_radius,
                    np.array(self.config.kitchen_bounds[2:]) - self.config.robot_radius,
                )
                result["success"] = bool(
                    result["success"]
                    and classify_zone(config=self.config, observation=observation) == "kitchen"
                    and abs(observation.position[2] + lower[2]) < 0.035
                    and np.linalg.norm(nearest - observation.position[:2])
                    <= self.config.direct_placement_reach
                )
            return result
        finally:
            planning = getattr(controller, "planning", None)
            if planning is not None:
                planning.close()

    def snapshot(self) -> dict[str, Any]:
        import mujoco

        specification = mujoco.mjtState.mjSTATE_INTEGRATION
        vector = np.empty(mujoco.mj_stateSize(self.model, specification))
        mujoco.mj_getState(self.model, self.data, vector, specification)
        return {
            "physics": vector,
            "episode": self.episode.model_copy(deep=True),
            "state": self.backend._require_state().copy(),
            "steps": self.steps,
            "robot_qpos": copy.deepcopy(self.scene._robot_env.qpos),
            "seed": self.seed,
        }

    def restore(self, *, snapshot: Any) -> None:
        import mujoco

        self.scene.set_state(snapshot["state"].copy())
        mujoco.mj_setState(
            self.model, self.data, snapshot["physics"], mujoco.mjtState.mjSTATE_INTEGRATION
        )
        mujoco.mj_forward(self.model, self.data)
        self.scene._robot_env.qpos = copy.deepcopy(snapshot["robot_qpos"])
        self.backend._state = snapshot["state"].copy()
        self.episode = snapshot["episode"].model_copy(deep=True)
        self.steps = snapshot["steps"]
        self.seed = snapshot["seed"]

    def close(self) -> None:
        self.backend.close()


class RecyclingBackendView:
    def __init__(self, *, backend: Any, cube_name: Any) -> None:
        self.backend = backend
        self.cube_name = cube_name

    def __getattr__(self, *args: Any) -> Any:
        (name,) = args
        return getattr(self.backend, name)
