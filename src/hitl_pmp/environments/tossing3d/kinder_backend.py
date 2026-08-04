"""The ONE module that talks to KINDER/MuJoCo.

Everything else under `environments/tossing3d/` -- the `State` layout, the predicates,
the lifted skill models, the goal test, the action encoding -- is pure arithmetic over
feature vectors and is exercised by ordinary CI tests. This module is the narrow seam
where the real simulator lives, so a machine without `kindergarden` installed can still
import, typecheck and test the rest of the domain.

To keep that true, **nothing here imports `kinder` at module scope**: `_ensure_env`
does the imports on first use. Importing this file is therefore always safe; only
actually driving the simulator needs the optional dependency -- `pyproject.toml`'s
`tossing3d` extra, which pins `kindergarden` and `kinder_models` to exact upstream
commits and is deliberately kept out of the core dependencies, since it pulls MuJoCo,
PyBullet and OpenCV. See `environments/tossing3d/README.md` for the install.

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
    # KINDER's own background scene. False selects its "simple" scene -- a bare white
    # void -- and True selects the MimicLabs room the `Tossing3D-o1` task JSON names,
    # which is what every clip on the benchmark's own site shows: KINDER's
    # `scripts/docs/generate_env_docs.py` passes `scene_bg=True` for every Dynamic3D
    # env, and `scripts/generate_demo_video.py` does the same. It defaults to False
    # here only because a sweep pays for it (measured: 4.18 ms vs 1.36 ms per
    # `render()`) and because every number already measured on this domain was measured
    # with the simple scene. Verified purely cosmetic -- an oracle rollout produces a
    # bit-identical cube trajectory either way -- so the demo renderer turns it on.
    scene_bg: bool = False
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
    # Set by `capture_frames_into`, drained by nobody here. When present, every MuJoCo
    # control tick appends one `env.render()` to it, which is how a *smooth* clip gets
    # made -- see `capture_frames_into` for why that is a separate mode.
    _frame_sink: list[np.ndarray] | None = PrivateAttr(default=None)

    def capture_frames_into(self, *, sink: list[np.ndarray] | None) -> None:
        """Record one KINDER frame per control tick into `sink` (None turns it off).

        This is the seam that makes a KINDER-quality clip possible. KINDER's own
        `scripts/generate_demo_video.py` renders once per `env.step()` -- one frame per
        10 Hz control tick -- which is why the clips on the benchmark's site are smooth.
        A `hitl_pmp` transition is a whole *skill*, several hundred ticks, so the
        `Renderer` interface (one frame per transition, by construction) can only ever
        produce a storyboard. Rather than widen that interface for every domain, this
        lets a demo script tap the tick loop directly.

        Off by default and never enabled during a sweep: it costs one `render()` per
        tick (~4 ms) on top of ~8 ms of physics, i.e. roughly +50% wall-clock, for
        frames a learning curve has no use for.
        """
        self._frame_sink = sink

    def render_fps(self) -> int:
        """KINDER's own `render_fps` for this env (20), so a clip written from
        `capture_frames_into` plays at the rate the benchmark intends rather than one
        picked here. Read from gym metadata, exactly as their scripts do
        (`generate_demo_video.py`: `fps = env.metadata.get("render_fps", 10)`)."""
        env = self._ensure_env()
        return int(env.metadata.get("render_fps", 10))

    def _ensure_env(self) -> Any:
        """Open the MuJoCo scene and build the controllers, once.

        Two pieces of KINDER-specific setup are load-bearing on a headless box:

        * `kinder.register_all_environments()` overwrites `MUJOCO_GL` and
          `PYOPENGL_PLATFORM` to `osmesa` whenever `DISPLAY` is unset, which is exactly
          the case under `scripts/run_sweep.py`. On a machine with EGL but no OSMesa
          that then dies with `'EGLPlatform' object has no attribute 'OSMesa'`. So the
          caller's choice is snapshotted and put back afterwards: whatever `MUJOCO_GL`
          the run was launched with wins.
        * `scene_bg` picks KINDER's background scene, and it is passed through rather
          than pinned: an earlier version hardcoded `False` as an OSMesa workaround,
          which is no longer needed now that the snapshot above keeps EGL. Measured
          under `MUJOCO_GL=egl`, `scene_bg=True` loads its MimicLabs room fine.
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
            f"kinder/Tossing3D-{self.variant}-v0",
            render_mode="rgb_array",
            scene_bg=self.scene_bg,
        )
        env.unwrapped._object_centric_env.set_render_camera(self.render_camera)
        self._env = env
        self._object_type = MujocoTidyBotRobotObjectType
        self._tossing_controllers = create_tossing_controllers(env.action_space)
        self._shelf_controllers = create_shelf_controllers(env.action_space)
        return env

    def goal_region_bounds(self) -> tuple[float, ...]:
        """KINDER's own `blocks_goal_region` box, `(x_min, y_min, z_min, x_max, y_max,
        z_max)`, read straight out of the variant's task JSON rather than hardcoded.

        KNOWN DISCREPANCY, fix pending: `_check_goals` does not compare against this
        raw range. It goes through `Region.check_in_region`, which tests `Region.bbox`
        -- and `_create_regions` builds that bbox by inflating the JSON range by
        `ground_placement_threshold = 0.05` on every axis
        (`kinder/envs/dynamic3d/objects/base.py:840, 875-880`). So the region this
        returns is 0.05 m tighter per side than the one the benchmark actually scores.
        `test_in_goal_region_agrees_with_kinders_own_goal_check` does not catch it
        because no state its random walk visits happens to land in the 0.05 m band
        where the two verdicts differ.
        """
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
        if self._frame_sink is not None:
            self._frame_sink.append(self.render())

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
        """One frame from KINDER's own renderer, on whichever camera
        `set_render_camera` selected.

        Copied, not viewed: MuJoCo renders into a buffer it reuses, so `np.asarray`
        (which does not copy an array that is already uint8) hands back a view that the
        *next* render silently overwrites. Harmless when a caller consumes one frame at
        a time, and a wrecked clip when `capture_frames_into` is accumulating hundreds.
        """
        env = self._ensure_env()
        return np.array(env.render(), dtype=np.uint8, copy=True)

    def check_goals(self) -> bool:
        """KINDER's own `_check_goals`, for the fidelity test that pins this domain's
        `InGoalRegion` predicate against it. Nothing in the running domain calls
        this -- the predicate does the work, so that the goal test stays a pure
        function of `State` like every other predicate in the repo."""
        env = self._ensure_env()
        return bool(env.unwrapped._object_centric_env._check_goals())
