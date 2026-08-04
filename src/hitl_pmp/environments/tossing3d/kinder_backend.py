"""The ONE module that talks to KINDER/MuJoCo.

Everything else under `environments/tossing3d/` -- the `State` layout, the predicates,
the lifted skill models, the goal test, the action encoding -- is pure arithmetic over
feature vectors and is exercised by ordinary CI tests. This module is the narrow seam
where the real simulator lives, so a machine without `kindergarden` installed can still
import, typecheck and test the rest of the domain.

To keep that true, **nothing here imports `kinder` at module scope**: `_ensure_env`
does the imports on first use. Importing this file is therefore always safe; only
actually driving the simulator needs the optional dependency (see
`environments/tossing3d/README.md` for the install, which is deliberately not in
`pyproject.toml` -- it pulls MuJoCo, PyBullet and a pinned numpy).

The skills are KINDER's own parameterized controllers, not reimplementations:
`pick_shelf` from `kinder_models.dynamic3d.shelf`, and `move_to_target` /
`move_arm_to_conf` / `toss` from `kinder_models.dynamic3d.tossing`. Each `execute_*`
below runs one controller to termination (a few hundred MuJoCo steps), which is what
makes one `hitl_pmp` transition equal one *skill* execution rather than one 20 ms
control tick -- the same convention every other domain in this repo uses.
"""

import os
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr


class KinderBackend(BaseModel):
    """A live `kinder/Tossing3D-<variant>-v0` simulator plus its controllers.

    Real per-run state (an open MuJoCo model, an open PyBullet client, the controller
    objects), so a pydantic instance rather than a static-method container. Held by
    exactly one `Tossing3DEnvironment`; constructing two means two MuJoCo scenes.

    The controllers are built once and reused for the whole run. That is not just
    thrift: each `PickShelfController`/`TossController` lazily opens its own PyBullet
    client on first `reset`, so rebuilding them per skill execution leaks one client
    per call over the tens of thousands of executions a sweep performs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Windup and full-power toss arm configurations, in radians. These are the values
    # KINDER's own Tossing3D demo drives `move_arm_to_conf`/`toss` with; `swing` (see
    # execute_toss) interpolates between them, so swing=1.0 reproduces that demo toss
    # exactly and smaller values release earlier in the arc, landing the cube shorter.
    WINDUP_CONF: ClassVar[tuple[float, ...]] = tuple(
        np.deg2rad([0.0, 50.0, 180.0, -110.0, 0.0, -100.0, 90.0]).tolist()
    )
    FULL_TOSS_CONF: ClassVar[tuple[float, ...]] = tuple(
        np.deg2rad([0.0, 20.0, 180.0, -35.0, 0.0, 25.0, 90.0]).tolist()
    )

    # Per-controller caps on how many MuJoCo steps one skill execution may take before
    # being abandoned. Generous relative to what the controllers actually use (~70
    # steps for a pick, ~20 for a toss) -- they exist so a controller that never
    # reports `terminated()` cannot hang a sweep.
    MAX_PICK_STEPS: ClassVar[int] = 400
    MAX_MOVE_STEPS: ClassVar[int] = 300
    MAX_ARM_STEPS: ClassVar[int] = 300
    # Free-fall settling after a toss: the controller terminates the instant the arm
    # finishes its swing, with the cube still in the air. Without this the state read
    # back would be mid-flight and no goal test could mean anything.
    SETTLE_STEPS: ClassVar[int] = 150

    variant: str = "o1"
    cube_name: str = "cube_0"
    bin_name: str = "bin_0"
    barrier_name: str = "cuboid_barrier"
    # Cube height above which it counts as held. The cube rests at z=0.025 on the
    # floor and z=0.044 inside the bin, and is carried at z~0.59 in the retract pose,
    # so anything in between separates the two cleanly.
    held_height: float = 0.2
    render_camera: str = "task_view"

    _env: Any = PrivateAttr(default=None)
    _tossing_controllers: Any = PrivateAttr(default=None)
    _shelf_controllers: Any = PrivateAttr(default=None)
    _object_type: Any = PrivateAttr(default=None)
    # The most recent observation, already devectorized into KINDER's own
    # ObjectCentricState. Kept here rather than re-read on demand because the
    # simulator exposes no "give me the current object-centric state" call that the
    # gym wrapper agrees with -- every read goes through an observation.
    _last_state: Any = PrivateAttr(default=None)

    def _ensure_env(self) -> Any:
        """Open the MuJoCo scene and build the controllers, once.

        Two pieces of KINDER-specific setup are load-bearing on a headless box:

        * `kinder.register_all_environments()` overwrites `MUJOCO_GL` and
          `PYOPENGL_PLATFORM` to `osmesa` whenever `DISPLAY` is unset, which is exactly
          the case under `scripts/run_sweep.py`. On a machine with EGL but no OSMesa
          that then dies with `'EGLPlatform' object has no attribute 'OSMesa'`. So the
          caller's choice is snapshotted and put back afterwards: whatever `MUJOCO_GL`
          the run was launched with wins.
        * `scene_bg=False` keeps KINDER off its background-scene render path, which
          reaches for the OSMesa context regardless of the variables above.
        """
        if self._env is not None:
            return self._env
        import mujoco  # noqa: F401, PLC0415 (import order matters -- see below)

        chosen_gl = {name: os.environ.get(name) for name in ("MUJOCO_GL", "PYOPENGL_PLATFORM")}

        import kinder  # noqa: PLC0415 (optional dependency -- see module docstring)
        from kinder.envs.dynamic3d.object_types import (  # noqa: PLC0415
            MujocoTidyBotRobotObjectType,
        )
        from kinder_models.dynamic3d.shelf.parameterized_skills import (  # noqa: PLC0415
            create_lifted_controllers as create_shelf_controllers,
        )
        from kinder_models.dynamic3d.tossing.parameterized_skills import (  # noqa: PLC0415
            create_lifted_controllers as create_tossing_controllers,
        )

        kinder.register_all_environments()
        for name, value in chosen_gl.items():
            if value is not None:
                os.environ[name] = value
        env = kinder.make(
            f"kinder/Tossing3D-{self.variant}-v0", render_mode="rgb_array", scene_bg=False
        )
        env.unwrapped._object_centric_env.set_render_camera(self.render_camera)
        self._env = env
        self._object_type = MujocoTidyBotRobotObjectType
        self._tossing_controllers = create_tossing_controllers(env.action_space)
        self._shelf_controllers = create_shelf_controllers(env.action_space)
        return env

    def goal_region_bounds(self) -> tuple[float, ...]:
        """KINDER's own `blocks_goal_region` box, `(x_min, y_min, z_min, x_max, y_max,
        z_max)`, read straight out of the variant's task JSON rather than hardcoded --
        this is the region `_check_goals` tests `cube_0` against, so reading it is what
        lets this port's `InGoalRegion` predicate be the benchmark's own success
        criterion instead of a lookalike."""
        env = self._ensure_env()
        ranges = env.unwrapped._object_centric_env.task_config["regions"]["blocks_goal_region"][
            "ranges"
        ]
        return tuple(float(value) for value in ranges[0])

    def reset(self, *, seed: int) -> dict[str, tuple[float, ...]]:
        """Reset the scene to `seed`'s initial state and read the features back.

        Verified deterministic: resetting to the same seed reproduces the robot
        configuration, base pose and cube pose bit-for-bit, including after the
        simulator has been driven far away in between.
        """
        env = self._ensure_env()
        observation, _ = env.reset(seed=seed)
        self._last_state = env.observation_space.devectorize(observation)
        return self.read_features()

    def read_features(self) -> dict[str, tuple[float, ...]]:
        """The current scene as this domain's flat feature vectors. `holding` is
        derived from the cube's height rather than tracked separately, so it stays
        honest when a grasp silently slips."""
        state = self._current_state()
        robot = list(state.get_objects(self._object_type))[0]
        cube = state.get_object_from_name(self.cube_name)
        cube_position = tuple(float(state.get(cube, axis)) for axis in "xyz")
        return {
            "robot": (
                float(state.get(robot, "pos_base_x")),
                float(state.get(robot, "pos_base_y")),
                float(state.get(robot, "pos_base_rot")),
                float(cube_position[2] > self.held_height),
            ),
            "cube": cube_position,
            "bin": self._object_position(state=state, name=self.bin_name),
            "barrier": self._object_position(state=state, name=self.barrier_name),
        }

    def _object_position(self, *, state: Any, name: str) -> tuple[float, ...]:
        obj = state.get_object_from_name(name)
        return tuple(float(state.get(obj, axis)) for axis in "xyz")

    def _current_state(self) -> Any:
        self._ensure_env()
        assert self._last_state is not None, "reset(seed=...) must run before any read."
        return self._last_state

    def _step(self, *, action: np.ndarray) -> None:
        env = self._ensure_env()
        observation, _, _, _, _ = env.step(action)
        self._last_state = env.observation_space.devectorize(observation)

    def execute_pick(self, *, distance: float, rot: float) -> bool:
        """KINDER's `pick_shelf`: drive the base to `(distance, rot)` relative to the
        cube, plan a grasp there, close the gripper and retract to the carry pose.

        `(distance, rot)` are exactly the two continuous parameters
        `PickShelfController.sample_parameters` draws, so a learned sampler here is
        learning the same quantity KINDER's own sampler randomizes. Returns whether
        the controller ran to completion -- inverse kinematics genuinely has no
        solution for some base poses, which is a *failure of this skill*, not an
        error: the caller leaves the world unchanged and carries on.
        """
        controller = self._ground_shelf(name="pick_shelf", with_target=True)
        return self._run(
            controller=controller,
            max_steps=self.MAX_PICK_STEPS,
            params=np.array([distance, rot], dtype=np.float64),
        )

    def execute_move_to_throw_pose(self, *, distance: float) -> bool:
        """KINDER's `move_to_target` grounded on the bin: base motion planning to a
        pose `distance` metres from the bin, facing it. Collision checking against the
        cube is disabled because the robot is carrying it."""
        controller = self._ground_tossing(name="move_to_target", with_target=True)
        return self._run(
            controller=controller,
            max_steps=self.MAX_MOVE_STEPS,
            params=np.array([distance, 0.0], dtype=np.float64),
            disable_collision_objects=[self.cube_name],
        )

    def execute_toss(self, *, swing: float) -> bool:
        """Wind the arm up and swing, releasing at KINDER's fixed 46% of the arc.

        `swing` interpolates the swing's end configuration between the windup pose
        (swing=0) and KINDER's demo toss pose (swing=1), so it is a single dial on how
        far the cube flies -- the continuous parameter this domain's learned sampler
        exists to tune. Windup and swing are one skill rather than two because the
        windup is a fixed, parameterless prelude: splitting them would add a
        transition that carries no decision.
        """
        windup = np.array(self.WINDUP_CONF, dtype=np.float64)
        full = np.array(self.FULL_TOSS_CONF, dtype=np.float64)
        if not self._run(
            controller=self._ground_tossing(name="move_arm_to_conf", with_target=False),
            max_steps=self.MAX_ARM_STEPS,
            params=windup,
        ):
            return False
        completed = self._run(
            controller=self._ground_tossing(name="toss", with_target=False),
            max_steps=self.MAX_ARM_STEPS,
            params=windup + swing * (full - windup),
        )
        self._settle()
        return completed

    def _settle(self) -> None:
        env = self._ensure_env()
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(self.SETTLE_STEPS):
            self._step(action=zero_action)

    def _ground_shelf(self, *, name: str, with_target: bool) -> Any:
        return self._ground(controllers=self._shelf_controllers, name=name, with_target=with_target)

    def _ground_tossing(self, *, name: str, with_target: bool) -> Any:
        return self._ground(
            controllers=self._tossing_controllers, name=name, with_target=with_target
        )

    def _ground(self, *, controllers: Any, name: str, with_target: bool) -> Any:
        state = self._current_state()
        robot = list(state.get_objects(self._object_type))[0]
        if not with_target:
            return controllers[name].ground((robot,))
        target_name = self.bin_name if name == "move_to_target" else self.cube_name
        return controllers[name].ground((robot, state.get_object_from_name(target_name)))

    def _run(self, *, controller: Any, max_steps: int, params: np.ndarray, **kwargs: Any) -> bool:
        """Drive one controller to termination. Every KINDER-side failure mode --
        no inverse-kinematics solution, motion planning returning None, a controller
        that never terminates -- is funnelled into a `False` return so that
        `Environment.take_action` stays total over the whole action space, the same
        contract every other domain here holds to."""
        try:
            controller.reset(self._current_state(), params=params, **kwargs)
        except Exception:  # noqa: BLE001 -- see docstring: a planning failure is a skill failure
            return False
        for _ in range(max_steps):
            try:
                action = controller.step()
            except Exception:  # noqa: BLE001
                return False
            self._step(action=action)
            controller.observe(self._current_state())
            if controller.terminated():
                return True
        return False

    def render(self) -> np.ndarray:
        env = self._ensure_env()
        return np.asarray(env.render(), dtype=np.uint8)

    def check_goals(self) -> bool:
        """KINDER's own `_check_goals`, for the fidelity test that pins this port's
        `InGoalRegion` predicate against it. Nothing in the running domain calls
        this -- the predicate does the work, so that the goal test stays a pure
        function of `State` like every other predicate in the repo."""
        env = self._ensure_env()
        return bool(env.unwrapped._object_centric_env._check_goals())
