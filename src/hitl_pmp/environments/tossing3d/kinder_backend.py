"""The one module in this package that touches KINDER, and the only place a KINDER
import may appear.

Everything above it (`environment.py`, `predicates.py`, `skills.py`, `tasks.py`,
`problem.py`, `cli.py`) talks to `KinderBackend` in plain Python types, so the whole
package imports, typechecks and tests on a machine with no MuJoCo -- which CI is, since
it never installs the optional `tossing3d` extra.

**Nothing here reimplements KINDER.** The simulator, the physics, all four controllers
and the success criterion are upstream's, driven exactly as upstream's own
`kinder-models/tests/dynamic3d/tossing/test_tossing_parameterized_skills.py::test_pick_ground_toss`
drives them. What this module supplies is the translation: a KINDER `ObjectCentricState`
becomes a `KinderObservation` of plain floats, and a whole *skill* becomes one call.

## The import dance, which is the trap that costs an hour

`kinder.register_all_environments()` rewrites `MUJOCO_GL` to `osmesa` whenever `DISPLAY`
is unset (`src/kinder/__init__.py:67-74`). Under `osmesa`, `import mujoco` raises on this
machine -- and `_check_deps` swallows *every* exception, so the entire `Dynamic3D`
category is skipped in silence and the failure surfaces much later as

    gymnasium.error.NameNotFound: Environment `Tossing3D-o1` doesn't exist in namespace kinder.

Three things together avoid it, and all three are load-bearing:

1. `DISPLAY` is set (to `:0`; nothing is ever drawn to it) so the rewrite never fires.
2. `MUJOCO_GL`/`PYOPENGL_PLATFORM` are forced to `egl`, overriding an inherited `osmesa`.
3. A *module* inside `kinder.envs.dynamic3d` is imported -- `kinder.envs.dynamic3d.envs`,
   not the package `kinder.envs.dynamic3d`, which does not pull in `mujoco` -- so
   `mujoco` is already in `sys.modules` before `register_all_environments()` runs.

`register_all_environments()` leaves `MUJOCO_GL` reading `osmesa` afterwards even when
it worked, so the environment is re-asserted immediately after the call rather than left
to a caller who might forget.

## The PyBullet client leak is fixed upstream, not worked around here

Grounding a KINDER controller mints a fresh instance, and `MoveArmToConfController`/
`TossController`/`PickShelfController` each open a `PyBulletSim` -- i.e. a live
`p.connect(p.DIRECT)` physics server -- inside `reset()`. Historically nothing ever
disconnected them, so an iterative run leaked ~136 MB and one client per skill
execution, without bound.

The fix is a `weakref.finalize(self, p.disconnect, ...)` in `PyBulletSim`, which landed
upstream as PR #87 (squash-merged as `9512b9e`). `reference/kinder-baselines` is a git
submodule pinned at `1b564a1`, and `9512b9e` is an ancestor of that pin, so the fix is
present and the client is released when the controller is collected.
**Do not add a `_release`-style explicit `close()` here**: with the finalizer in place
that double-disconnects.

## Physics-rate frames come from a wrapper, not from a hand-rolled buffer

One `take_action` is a whole controller execution, so a `core.Renderer` -- which emits
one frame per *transition* -- gives a four-frame clip of a domain whose entire point is a
throw. The frames exist; nothing captured them.

Capturing them is `gymnasium.wrappers.RenderCollection`'s exact job: it collects
`env.render()` on every `step()` and `reset()`, and its own `render()` hands the list
back and clears it. KINDER wraps its envs with the sibling `RecordVideo` throughout its
own tests, so wrapping is upstream's idiom too. `RecordVideo` itself is deliberately
*not* used here -- it writes its own `rl-video-episode-N.mp4` per gym reset, with no
caption, which would leave the harness's own `episode.mp4` still four frames long and add
a second, differently-named file beside it. `RenderCollection` instead feeds the frames
into the list `Problem.run_task_episode` already returns, so every existing consumer
(`--output-dir`, `--num-render-checkpoints`, `--record-full-loop`) gets the smooth clip
with no new file and no new codepath.

**Collected frames are not aliases.** `RenderCollection` stores whatever `render()`
returned, without copying, so a simulator that handed back a view into one reused buffer
would silently yield a clip of N identical frames. Measured on this scene: two successive
`env.render()` calls share no memory (`np.shares_memory` is `False`) and collected frames
are distinct arrays, so nothing needs copying on the way out. `test_kinder_fidelity.py`
asserts the collected frames genuinely differ rather than trusting that measurement.

Recording is **off by default**: it renders every physics tick, which is tens to hundreds
of MuJoCo renders per skill, and a training run that wants no video must not pay for it.
`Tossing3DProblem.run_task_episode` turns it on for exactly the episodes it was given a
renderer for.
"""

import os
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

# A DISPLAY only has to *exist* -- nothing is ever drawn to it. See the module docstring.
FALLBACK_DISPLAY = ":0"

# `task_view` is the camera this scene's own task config defines, and the only one that
# shows the throw. `agentview_1` (what upstream's demo script sets, but only
# `if "TidyBot" in env_id`, which `kinder/Tossing3D-o1-v0` is not) is absent from this
# scene's `camera_names` entirely, and `set_render_camera` does not validate the name --
# so choosing it silently renders a near-static shot of a wall.
DEFAULT_CAMERA = "task_view"

# Upstream's own `test_pick_ground_toss` value; the *only* seed any number in this
# package's docs was measured at.
DEFAULT_SCENE_SEED = 125


class KinderObservation(BaseModel):
    """One KINDER `ObjectCentricState`, flattened to plain floats.

    This is the boundary type: `KinderBackend` produces it, `Tossing3DEnvironment`
    consumes it, and nothing about it requires KINDER to construct -- which is what lets
    a test build one by hand and exercise the translation offline.

    `goal_region` is the **live** `Region.bbox` of `blocks_goal_region`, read back from
    the compiled model rather than re-derived from the task JSON. The JSON range is
    inflated by `ground_placement_threshold` (0.05 m per side, z clamped at 0) before it
    becomes a region, so the literal in the file is not the box that scores. Reading it
    back is what lets `predicates.InGoalRegionClassifier` be a pure function of `State`
    and still agree with `_check_goals()` exactly.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    features: dict[str, dict[str, float]]
    goal_region: tuple[float, float, float, float, float, float]
    solved: bool

    def get(self, *, name: str, feature: str) -> float:
        """One named feature of one named KINDER object, or a loud error.

        Loud because the alternative is a silent zero: every feature this package reads
        is named in `Tossing3DEnvironment`'s own `Type` declarations, so a miss means
        upstream renamed something and the translation is now wrong, not merely empty.
        """
        if name not in self.features:
            raise KeyError(f"no object named {name!r} in this observation: {sorted(self.features)}")
        obj = self.features[name]
        if feature not in obj:
            raise KeyError(f"object {name!r} has no feature {feature!r}: {sorted(obj)}")
        return obj[feature]


class ControllerRun(BaseModel):
    """What one upstream controller execution did.

    `steps` is how many `env.step` calls it took to terminate -- the quantity the
    validation record reports as `71 / 23 / 16 / 18` for the oracle's four controller
    executions, so it is returned rather than discarded.

    `error` is non-`None` when the controller raised. That is not exceptional: KINDER's
    motion planners `assert plan is not None`, so an unreachable grasp or an
    unplannable arm trajectory surfaces as an `AssertionError` from `reset()`. A
    `core.Environment.take_action` must be total over its action space, so the caller
    turns this into a no-op transition rather than propagating -- see
    `Tossing3DEnvironment.take_action`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    steps: int
    terminated: bool
    error: str | None = None


class KinderApi(BaseModel):
    """Handles to everything this package needs from KINDER, imported in one place.

    Held as one object so the import happens exactly once per process and so the EGL
    environment is guaranteed to have been configured first (see the module docstring).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kinder: ModuleType
    robot_type: Any
    tossing_controllers: Any
    shelf_controllers: Any
    # `gymnasium.wrappers.RenderCollection`, the class itself. Gymnasium is KINDER's own
    # dependency rather than this repo's, so it is imported here with everything else
    # instead of at module scope, which would break the offline import.
    render_collection: Any


class KinderBackend(BaseModel):
    """A live `kinder/Tossing3D-<variant>-v0` and the translation to and from it.

    Construction imports nothing: `KinderBackend()` is pure pydantic, so
    `Tossing3DEnvironment` can hold one as an ordinary field and the package stays
    importable without MuJoCo. KINDER is imported on the first `reset()`.

    One `KinderBackend` owns at most one live gym env at a time. `reset(seed=...)`
    rebuilds the scene from that seed, which is the *only* way this domain can restore a
    state: a flat `core.State` of positions cannot round-trip MuJoCo's qpos/qvel, so a
    mid-episode state is not restorable and `Tossing3DEnvironment.set_state` refuses one.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Names as they appear in `Tossing3D-o1.json`'s `objects`/`initial_state`. Fixed by
    # upstream's scene, not configuration, so ClassVars rather than fields.
    cube_name: ClassVar[str] = "cube_0"
    bin_name: ClassVar[str] = "bin_0"
    barrier_name: ClassVar[str] = "cuboid_barrier"
    goal_region_name: ClassVar[str] = "blocks_goal_region"

    # Upstream's own arm configurations, in degrees, copied verbatim from
    # `test_pick_ground_toss`. The windup is the posture upstream's toss is demonstrated
    # from; the toss target is the swing itself. Neither is interpolated or retuned here
    # -- this package invents no controller parameter (see `skills.py`).
    windup_conf_deg: ClassVar[tuple[int, ...]] = (0, 50, 180, -110, 0, -100, 90)
    toss_conf_deg: ClassVar[tuple[int, ...]] = (0, 20, 180, -35, 0, 25, 90)

    # Upstream's own per-controller step budgets, again from `test_pick_ground_toss`.
    pick_step_limit: ClassVar[int] = 400
    move_step_limit: ClassVar[int] = 200
    arm_step_limit: ClassVar[int] = 200

    env_id: str = "kinder/Tossing3D-o1-v0"
    task_config_path: Path | None = None
    scene_bg: bool = True
    camera: str = DEFAULT_CAMERA
    render_mode: str = "rgb_array"
    # Unlocks `ObjectCentricTidyBot3DEnv.set_state`, which otherwise raises
    # "State access is not allowed". A KINDER `ObjectCentricState` carries velocities as
    # well as poses (`MujocoMovableObjectType` has `vx..wz`, the robot has `vel_*`), so it
    # really is a full state -- upstream's own `tidybot3d_shelf3D.py` uses exactly this to
    # build a transition function. That is what `snapshot`/`restore` below rest on.
    # Upstream's own env models pass `allow_state_access=True` too.
    allow_state_access: bool = True
    # Collect one frame per physics tick while a controller runs. Off by default: see the
    # module docstring. `Tossing3DProblem.run_task_episode` flips it per episode, so a
    # rendered demo episode records and a training episode does not.
    record_substeps: bool = False

    _api: KinderApi | None = PrivateAttr(default=None)
    # The gym env exactly as `kinder.make` returned it. Single frames, metadata and the
    # object-centric handle are all read from *this* one, so they are unaffected by
    # whether a recording wrapper is currently in place.
    _raw_env: Any = PrivateAttr(default=None)
    # What gets stepped: `_raw_env`, or a `RenderCollection` around it while recording.
    _env: Any = PrivateAttr(default=None)
    _state: Any = PrivateAttr(default=None)
    _robot_name: str = PrivateAttr(default="")

    @staticmethod
    def configure_headless_rendering(
        *, environ: MutableMapping[str, str] | None = None
    ) -> dict[str, str]:
        """Point MuJoCo at EGL and make sure a `DISPLAY` exists, before KINDER is imported.

        `DISPLAY` is `setdefault`, so a real display wins; `MUJOCO_GL`/`PYOPENGL_PLATFORM`
        are forced, because the one inherited value known to break the import is exactly
        the one `register_all_environments()` writes.
        """
        target = os.environ if environ is None else environ
        target.setdefault("DISPLAY", FALLBACK_DISPLAY)
        target["MUJOCO_GL"] = "egl"
        target["PYOPENGL_PLATFORM"] = "egl"
        return {key: target[key] for key in ("DISPLAY", "MUJOCO_GL", "PYOPENGL_PLATFORM")}

    def api(self) -> KinderApi:
        """Import KINDER once, in the right order, and register its environments.

        Registration happens here rather than in a caller because
        `register_all_environments()` is itself the call that rewrites `MUJOCO_GL`, so
        the environment is re-asserted on the very next line.
        """
        if self._api is not None:
            return self._api
        self.configure_headless_rendering()

        import kinder
        import kinder.envs.dynamic3d.envs  # noqa: F401  (the MODULE, not the package)
        from gymnasium.wrappers import RenderCollection
        from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
        from kinder_models.dynamic3d.shelf.parameterized_skills import (
            create_lifted_controllers as shelf_create_lifted_controllers,
        )
        from kinder_models.dynamic3d.tossing.parameterized_skills import (
            create_lifted_controllers as tossing_create_lifted_controllers,
        )

        kinder.register_all_environments()
        self.configure_headless_rendering()

        self._api = KinderApi(
            kinder=kinder,
            robot_type=MujocoTidyBotRobotObjectType,
            tossing_controllers=tossing_create_lifted_controllers,
            shelf_controllers=shelf_create_lifted_controllers,
            render_collection=RenderCollection,
        )
        return self._api

    @property
    def robot_name(self) -> str:
        """The robot's name in the live scene, resolved by *type* at reset.

        By type rather than by literal because the robot is the one object whose name is
        a property of the robot config (`sim.robot_name`) rather than of the task JSON's
        `objects` block, and it carries a different feature schema from everything else.
        """
        if not self._robot_name:
            raise RuntimeError("KinderBackend.reset() has not run yet; there is no scene.")
        return self._robot_name

    def reset(self, *, seed: int) -> KinderObservation:
        """Build (or rebuild) the scene at `seed` and return the initial observation.

        The gym env is remade whenever there is not one already; `env.reset(seed=...)`
        alone is enough afterwards, and is what makes a differently-seeded initial state
        cheap relative to a fresh `make`.
        """
        api = self.api()
        if self._raw_env is None:
            overrides: dict[str, Any] = {}
            if self.task_config_path is not None:
                overrides["task_config_path"] = str(self.task_config_path)
            self._raw_env = api.kinder.make(
                self.env_id,
                render_mode=self.render_mode,
                scene_bg=self.scene_bg,
                allow_state_access=self.allow_state_access,
                **overrides,
            )
            object_centric = self._object_centric()
            available = list(getattr(object_centric, "camera_names", []))
            if available and self.camera not in available:
                raise ValueError(f"camera {self.camera!r} is not in this scene: {available}")
            # Before any wrapper goes on, as upstream's own `test_pick_ground_toss` does:
            # a recording made through the wrong camera is a silent shot of a wall.
            object_centric.set_render_camera(self.camera)
        self._sync_recording_wrapper()
        observation, _ = self._env.reset(seed=seed)
        self._state = self._env.observation_space.devectorize(observation)
        self._robot_name = next(iter(self._state.get_objects(self.api().robot_type))).name
        return self.observe()

    def close(self) -> None:
        """Release the gym env, if one was ever built. Idempotent."""
        if self._raw_env is not None:
            self._raw_env.close()
            self._raw_env = None
            self._env = None
            self._state = None

    def set_substep_recording(self, *, enabled: bool) -> None:
        """Turn per-physics-tick frame collection on or off, effective immediately.

        Imports nothing on its own: a backend with no scene yet just remembers the flag,
        and the wrapper goes on at the next `reset()`. That matters because this is
        called from `run_task_episode`, which a test may reach without ever intending to
        start MuJoCo.
        """
        self.record_substeps = enabled
        self._sync_recording_wrapper()

    def drain_substep_frames(self) -> list[np.ndarray]:
        """Every frame collected since the last drain, in order, clearing the buffer.

        This is `RenderCollection.render()` -- the wrapper's own drain -- not a buffer
        kept here. Empty whenever recording is off or no scene exists, which are both
        ordinary states rather than errors: `run_task_episode` drains unconditionally.

        No copy is taken. See the module docstring: collected frames were measured to be
        distinct arrays rather than aliases of one reused MuJoCo buffer, and copying a
        physics-rate episode would double an already-large peak for nothing.
        """
        if not self.record_substeps or self._env is None or self._env is self._raw_env:
            return []
        return [np.asarray(frame, dtype=np.uint8) for frame in self._env.render()]

    def _sync_recording_wrapper(self) -> None:
        """Put the `RenderCollection` on, or take it off, to match `record_substeps`.

        Wrapping and unwrapping are pure re-bindings around the one `_raw_env`, so
        toggling between episodes never rebuilds the scene.
        """
        if self._raw_env is None:
            return
        recording = self._env is not None and self._env is not self._raw_env
        if self.record_substeps and not recording:
            self._env = self.api().render_collection(
                self._raw_env, pop_frames=True, reset_clean=True
            )
        elif not self.record_substeps:
            self._env = self._raw_env

    def snapshot(self) -> Any:
        """An opaque handle to the live simulator state, restorable by `restore`.

        This is a *KINDER* `ObjectCentricState`, copied -- not one of this package's flat
        `core.State`s, which are a lossy projection (four of the robot's twenty-two
        features, six of the cube's sixteen). A caller that wants to rewind must hold one
        of these, which is why the handle is deliberately opaque rather than something
        that looks interchangeable with a `core.State`.
        """
        return self._require_state().copy()

    def restore(self, *, snapshot: Any) -> KinderObservation:
        """Put the simulator back to a `snapshot`, exactly, and re-observe.

        Faithful in a way `Tossing3DEnvironment.set_state` cannot be: this restores
        velocities and the full arm configuration too, because a KINDER
        `ObjectCentricState` carries them. Requires `allow_state_access`.

        **Faithful to float32, not bit-exact.** An `ObjectCentricState` is the observation
        vector, and `ObjectCentricBoxSpace` is float32, while MuJoCo integrates in
        float64. So a round-trip reintroduces ~1.2e-7 of relative error, which 200
        substeps per env step amplify: measured over one `move_to_target`, two runs from
        the same snapshot end up ~1e-7 apart on x and ~2.7e-4 on the quaternion. That is
        four orders of magnitude below any state change a symbolic predicate here cares
        about, but it does mean a rewound rollout is *not* byte-reproducible, and nothing
        should be built on the assumption that it is.
        """
        self._object_centric().set_state(snapshot)
        self._state = snapshot.copy()
        return self.observe()

    def observe(self) -> KinderObservation:
        """Flatten the live KINDER state (plus the live goal box and verdict) to floats.

        Feature names come from `ObjectCentricState.type_features`, the state's own
        type -> feature-name mapping, rather than from anything on the `Type` itself:
        KINDER's `Type` is `relational_structs.Type`, which carries only a name and a
        parent, and the schema lives beside it in `MujocoObjectTypeFeatures`. Reading it
        off the state means the names are whatever this particular state actually has.
        """
        state = self._require_state()
        features = {
            obj.name: {name: float(state.get(obj, name)) for name in state.type_features[obj.type]}
            for obj in state
        }
        return KinderObservation(
            features=features,
            goal_region=self.goal_region_bbox(),
            solved=self.check_goals(),
        )

    def check_goals(self) -> bool:
        """Upstream's own verdict -- `ObjectCentricTidyBot3DEnv._check_goals()`.

        This is the success criterion for this domain. `predicates.IN_GOAL_REGION` is
        written to agree with it and is differentially tested against it; it is never a
        second, independent definition of success.
        """
        return bool(self._object_centric()._check_goals())  # noqa: SLF001

    def goal_region_bbox(self) -> tuple[float, float, float, float, float, float]:
        """The live world-frame box `_check_goals()` scores containment in.

        `Region.bbox` only reads the site's simulated position when the region carries an
        `env`; ground regions are constructed with `env=None`, so it otherwise falls back
        to an XML/parent-frame value. Upstream's own `check_in_region` handles that by
        swapping `env` in and back out, and so does this -- leaving a sim reference behind
        on a region upstream deliberately left bare would be a side effect of taking a
        measurement.
        """
        object_centric = self._object_centric()
        ground = object_centric._ground_fixture  # noqa: SLF001
        regions = ground.region_objects
        found = regions.get(self.goal_region_name, [])
        if len(found) != 1:
            raise ValueError(
                f"expected exactly one {self.goal_region_name!r} region, found "
                f"{len(found)} (regions: {sorted(regions)})"
            )
        region = found[0]
        original = region.env
        region.env = object_centric._robot_env  # noqa: SLF001
        try:
            bbox = [float(value) for value in region.bbox]
        finally:
            region.env = original
        if len(bbox) != 6:
            raise ValueError(f"{self.goal_region_name!r} is not a single box: bbox={bbox}")
        x_min, y_min, z_min, x_max, y_max, z_max = bbox
        return (x_min, y_min, z_min, x_max, y_max, z_max)

    def render(self) -> np.ndarray:
        """One RGB frame from `task_view`, copied out of MuJoCo's buffer.

        Rendered from the *unwrapped* env, so a single frame is still a single frame
        while a `RenderCollection` is in place -- that wrapper's own `render()` returns
        the collected list and drains it, which would both break this signature and eat
        the episode's frames.

        The copy is belt-and-braces. Two successive `render()` calls on this scene were
        measured to share no memory, so this array is already the caller's own; the copy
        stays because `np.asarray` would not add one if that ever changed.
        """
        if self._raw_env is None:
            raise RuntimeError("KinderBackend.reset() has not run yet; there is nothing to render.")
        return np.asarray(self._raw_env.render(), dtype=np.uint8).copy()

    def render_fps(self) -> int:
        """Playback rate from the environment's own metadata, never hardcoded.

        Tossing3D reports 20; several other KINDER environments report 10, so a constant
        here would render some domain's clips at the wrong speed with nothing to say so.
        """
        if self._raw_env is None:
            raise RuntimeError("KinderBackend.reset() has not run yet; there is no metadata.")
        metadata: Mapping[str, Any] = self._raw_env.metadata
        if "render_fps" not in metadata:
            raise ValueError(f"environment metadata has no render_fps: {dict(metadata)}")
        return int(metadata["render_fps"])

    def run_controller(
        self,
        *,
        module: str,
        key: str,
        object_names: Sequence[str],
        params: np.ndarray | None,
        limit: int,
        disable_collision_objects: Sequence[str] | None = None,
        release_speed: float | None = None,
    ) -> ControllerRun:
        """Drive one upstream controller to termination, stepping the live simulator.

        The loop body is upstream's own, from `test_pick_ground_toss`: `controller.step()`
        into `env.step`, devectorize, `controller.observe`, break on `terminated()`.

        A controller that does not terminate within `limit` is reported as
        `terminated=False` rather than raised, and a controller that *raises* (KINDER's
        motion planners `assert plan is not None`) is reported through `error`. Both are
        ordinary outcomes of a skill whose continuous parameters do not work out, and the
        caller has to be able to keep going -- `take_action` must be total.

        `disable_collision_objects` and `release_speed` are both **per-controller**
        `reset` keywords, not universal ones: the first exists only on tossing's
        `MoveToTargetGroundController.reset` and the second only on
        `TossController.reset`, and passing either to a controller that does not declare
        it is a `TypeError`. So each is forwarded only when the caller actually supplies
        it, and it is the `run_*` wrapper below -- which knows which controller it is
        driving -- that decides to.
        """
        api = self.api()
        state = self._require_state()
        factory = {
            "tossing": api.tossing_controllers,
            "shelf": api.shelf_controllers,
        }
        if module not in factory:
            raise ValueError(f"unknown controller module {module!r}; known: {sorted(factory)}")
        lifted = factory[module](self._env.action_space)
        if key not in lifted:
            raise ValueError(f"{module} has no controller {key!r}; known: {sorted(lifted)}")
        objects = tuple(state.get_object_from_name(name) for name in object_names)
        controller = lifted[key].ground(objects)

        reset_kwargs: dict[str, Any] = {}
        if disable_collision_objects is not None:
            reset_kwargs["disable_collision_objects"] = list(disable_collision_objects)
        if release_speed is not None:
            reset_kwargs["release_speed"] = release_speed

        try:
            controller.reset(state, params, **reset_kwargs)
        except Exception as exc:  # noqa: BLE001  (any planner failure is a failed skill)
            return ControllerRun(steps=0, terminated=False, error=f"{type(exc).__name__}: {exc}")

        for step in range(limit):
            try:
                action = controller.step()
                observation, _, _, _, _ = self._env.step(action)
            except Exception as exc:  # noqa: BLE001  (same reasoning as above)
                return ControllerRun(
                    steps=step, terminated=False, error=f"{type(exc).__name__}: {exc}"
                )
            self._state = self._env.observation_space.devectorize(observation)
            controller.observe(self._state)
            if controller.terminated():
                return ControllerRun(steps=step + 1, terminated=True)
        return ControllerRun(steps=limit, terminated=False)

    def run_pick(self, *, distance: float, rotation: float) -> ControllerRun:
        """`pick_shelf` -- upstream's grasp, the same one `tidybot3d_shelf3D.py` models.

        `disable_collision_objects` is deliberately absent: it exists only on tossing's
        `MoveToTargetGroundController.reset`, and passing it here is a `TypeError`.
        """
        return self.run_controller(
            module="shelf",
            key="pick_shelf",
            object_names=(self.robot_name, self.cube_name),
            params=np.array([distance, rotation]),
            limit=self.pick_step_limit,
        )

    def run_move_to_throw_pose(self, *, standoff: float, rotation: float) -> ControllerRun:
        """`move_to_target` onto the bin, at `standoff` metres.

        `disable_collision_objects=["cube_0"]` is upstream's own argument here: the robot
        is *holding* the cube at this point, so planning the base motion against it as an
        obstacle makes every plan fail.
        """
        return self.run_controller(
            module="tossing",
            key="move_to_target",
            object_names=(self.robot_name, self.bin_name),
            params=np.array([standoff, rotation]),
            limit=self.move_step_limit,
            disable_collision_objects=[self.cube_name],
        )

    def run_toss(self, *, release_speed_deg_s: float) -> tuple[ControllerRun, ControllerRun]:
        """The windup and the swing, back to back -- upstream's `move_arm_to_conf` then `toss`.

        Two controllers, one skill. Upstream never demonstrates `toss` from anywhere but
        this windup, and `move_arm_to_conf` takes a raw 7-DoF joint vector that
        `sample_parameters` explicitly refuses to sample (`NotImplementedError`), so a
        windup is a posture the swing requires, not a skill anything could usefully
        select on its own. The swing is skipped if the windup did not land, since tossing
        from an unknown arm pose is not the thing that was measured.

        **This is the one place in the domain where degrees become radians.** The dial is
        carried in joint-path deg/s everywhere above here, because that is the unit every
        measurement of it is written in (see `predicates.TOSS_SPEED_BOUNDS`); upstream's
        `TossController.reset` takes rad/s. One conversion, here, pinned by
        `test_run_toss_converts_the_release_speed_to_radians_exactly_once` -- a second
        conversion or a missing one is a silent 57x error in either direction.

        The speed reaches the **swing only**. The windup is `move_arm_to_conf`, which is
        a posture change rather than a throw and whose `reset` takes no release speed at
        all; forwarding one there is a `TypeError`.
        """
        windup = self.run_controller(
            module="tossing",
            key="move_arm_to_conf",
            object_names=(self.robot_name,),
            params=np.deg2rad(self.windup_conf_deg),
            limit=self.arm_step_limit,
        )
        if not windup.terminated:
            skipped = ControllerRun(steps=0, terminated=False, error="windup did not terminate")
            return windup, skipped
        swing = self.run_controller(
            module="tossing",
            key="toss",
            object_names=(self.robot_name,),
            params=np.deg2rad(self.toss_conf_deg),
            limit=self.arm_step_limit,
            release_speed=float(np.deg2rad(release_speed_deg_s)),
        )
        return windup, swing

    def _object_centric(self) -> Any:
        if self._raw_env is None:
            raise RuntimeError("KinderBackend.reset() has not run yet; there is no scene.")
        return self._raw_env.unwrapped._object_centric_env  # noqa: SLF001

    def _require_state(self) -> Any:
        if self._state is None:
            raise RuntimeError("KinderBackend.reset() has not run yet; there is no state.")
        return self._state


class Tossing3DSceneFiles(BaseModel):
    """Where this repo's own task JSON lives, resolved from the package.

    `scripts/task_configs/Tossing3D-o1-coincident.json` is *not* duplicated into the
    package: it landed on `main` in #69 as the file `scripts/tossing3d_oracle_demo.py`
    drives, and two copies of a scene definition is exactly how a measurement ends up
    attributed to the wrong geometry. The path is walked from `__file__` instead, and a
    miss raises with the resolved path rather than falling back to stock -- silently
    running the stock scene under the coincident scene's name is the one failure mode
    this whole domain exists to avoid.
    """

    model_config = ConfigDict(frozen=True)

    # src/hitl_pmp/environments/tossing3d/kinder_backend.py -> repo root is 5 up.
    repo_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[4])

    def coincident_task_config(self) -> Path:
        path = self.repo_root / "scripts" / "task_configs" / "Tossing3D-o1-coincident.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"the coincident Tossing3D task config is missing: {path}. It ships in "
                "this repo at scripts/task_configs/ and is shared with "
                "scripts/tossing3d_oracle_demo.py; this domain does not keep a second copy."
            )
        return path
