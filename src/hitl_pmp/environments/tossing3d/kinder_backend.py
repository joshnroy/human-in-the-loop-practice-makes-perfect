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
is unset (`src/kinder/__init__.py:67-74`). Under a backend that is selected but not
installed, `import mujoco` raises -- and `_check_deps` swallows *every* exception, so the
entire `Dynamic3D` category is skipped in silence and the failure surfaces much later as

    gymnasium.error.NameNotFound: Environment `Tossing3D-o1` doesn't exist in namespace kinder.

Three things together avoid it, and all three are load-bearing:

1. `DISPLAY` is set (to `:0`; nothing is ever drawn to it) so the rewrite never fires.
2. `MUJOCO_GL`/`PYOPENGL_PLATFORM` are re-asserted from `REQUESTED_GL_BACKEND`, the
   snapshot this module takes at import, overriding whatever the rewrite left behind.
3. A *module* inside `kinder.envs.dynamic3d` is imported -- `kinder.envs.dynamic3d.envs`,
   not the package `kinder.envs.dynamic3d`, which does not pull in `mujoco` -- so
   `mujoco` is already in `sys.modules` before `register_all_environments()` runs.

`register_all_environments()` leaves `MUJOCO_GL` reading `osmesa` afterwards even when
it worked, so the environment is re-asserted immediately after the call rather than left
to a caller who might forget.

**Point 2 used to read "are forced to `egl`", and that was wrong in a way worth
recording.** The premise was that `osmesa` is the value that breaks the import -- but that
is a property of a workstation with an EGL driver and no OSMesa, not of `osmesa`. On a
headless `ubuntu-latest` runner it is the reverse: there is no EGL driver, `libosmesa6-dev`
is installed, and hardcoding `egl` here made every Tossing3D test fail with
`AttributeError: 'NoneType' object has no attribute 'eglQueryString'` -- unreachable from
the workflow, because this line overrode the job environment in-process. The backend is
now inheritable, and the two `osmesa`s are told apart by *when* they arrive: see
`REQUESTED_GL_BACKEND`.

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
from types import ModuleType
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from .types import AbstractAtom

# A DISPLAY only has to *exist* -- nothing is ever drawn to it. See the module docstring.
FALLBACK_DISPLAY = ":0"

# The GL backend, **snapshotted here at import time**, which is the whole design.
#
# `register_all_environments()` writes `MUJOCO_GL` itself, so after it has run an `osmesa`
# in the environment is ambiguous: it may be what an operator asked for, or it may be what
# upstream just put there. Reading the request *before* KINDER can be imported -- and this
# module is necessarily imported first, since importing KINDER is a method on the class
# below -- tells the two apart by construction rather than by heuristic. What is here at
# import is a request and is honoured; anything appearing later is a rewrite and is undone.
#
# An empty string counts as unset, because that is how CI passes `PYOPENGL_PLATFORM`:
# `mujoco/osmesa/__init__.py` fills a falsy one in itself but *raises* on any non-`osmesa`
# value, so the empty string is the one setting that asks for nothing and forbids nothing.
DEFAULT_GL_BACKEND = "egl"
REQUESTED_GL_BACKEND = os.environ.get("MUJOCO_GL") or DEFAULT_GL_BACKEND
REQUESTED_GL_PLATFORM = os.environ.get("PYOPENGL_PLATFORM") or REQUESTED_GL_BACKEND

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
    inflated by `MujocoObject`'s per-object placement threshold (1cm per side) before it
    becomes a region, so the literal in the file is not the box that scores. Reading it
    back is what lets this domain's `InBin` agree with `_check_goals()` exactly. It
    reaches the state carried on the **bin** object -- the region is attached to the
    bin's own body (`blocks_goal_region.target` is `bin_0`), under this domain's
    assumption that the bin's interior contains this region -- see `predicates.py`'s
    module docstring. Because the region moves with the bin, this box is not fixed
    across seeds once `bin_init_region` samples a position: it tracks wherever the bin
    actually landed.
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


def _build_state_collection_class() -> Any:
    """A `gymnasium.Wrapper` that snapshots `ObjectCentricState` after every internal
    `step()`, mirroring `gymnasium.wrappers.RenderCollection`'s own `step()` (append
    then return) exactly -- see that class for the pattern this copies. Defined inside
    a function, not at module scope, because `gymnasium.Wrapper` is KINDER's dependency
    and importing it eagerly would break this package's offline (no-MuJoCo) import."""
    from gymnasium import Wrapper

    class StateCollection(Wrapper):
        def __init__(self, env: Any) -> None:  # noqa: PLR0917 (gymnasium's own signature)
            super().__init__(env)
            self.state_list: list[Any] = []

        def reset(self, **kwargs: Any) -> Any:
            self.state_list = []
            return super().reset(**kwargs)

        def step(self, action: Any) -> Any:  # noqa: PLR0917 (gymnasium's own signature)
            output = super().step(action)
            # `.unwrapped` is `TidyBot3DEnv`, which delegates render() to its own
            # `_object_centric_env` (see that class's own `render()`) -- the object-
            # centric state lives one level further in than `.unwrapped` reaches.
            # Untyped past this point like the rest of this module's KINDER surface.
            unwrapped: Any = self.unwrapped
            object_centric = unwrapped._object_centric_env  # noqa: SLF001
            self.state_list.append(object_centric._get_current_state())  # noqa: SLF001
            return output

        def drain(self) -> list[Any]:
            states = self.state_list
            self.state_list = []
            return states

    return StateCollection


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
    # A locally-defined `gymnasium.Wrapper` subclass (see `_build_state_collection_class`)
    # that snapshots `ObjectCentricState` after every internal step, the state-capture
    # sibling of `render_collection` below -- same per-tick granularity, so a state log
    # recorded alongside a substep-frame recording covers exactly the ticks the frames
    # do. Built lazily for the same reason `render_collection` is: `gymnasium.Wrapper`
    # is KINDER's dependency, not this repo's.
    state_collection: Any
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
    # What *this domain's* `Object` calls the robot. KINDER's own name for it comes from
    # the robot config and is resolved at reset (`robot_name`), so `abstract_atoms`
    # translates one to the other rather than assuming they coincide.
    robot_atom_name: ClassVar[str] = "robot"

    # Upstream's arm configurations are no longer named here: the composed toss reads
    # `TOSS_WINDUP_ARM_CONFIGURATION` and `TOSS_RELEASE_ARM_CONFIGURATION` itself, so this
    # package no longer hands them in and cannot get them out of step with upstream's.

    # Per-controller step budgets. `pick_step_limit` is upstream's own from
    # `test_pick_ground_toss`; `toss_step_limit` covers base motion, windup and swing in
    # one controller, so it is the sum of what the three separate budgets used to be
    # (200 + 200 + 200) with headroom, and sits below the 3000 upstream's own
    # `test_pick_cube_then_move_and_toss_scores` allows.
    pick_step_limit: ClassVar[int] = 400
    toss_step_limit: ClassVar[int] = 1000

    env_id: str = "kinder/Tossing3D-o1-v0"
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
    # rendered demo episode records frames and a training episode does not -- video
    # capture stays real and opt-in. State capture (`StateCollection`, see
    # `drain_substep_states`) does NOT follow this flag: it is unconditional, once a
    # scene exists -- see `_sync_recording_wrapper`.
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
    # Upstream's `Tossing3DStateAbstractor`: the five predicates' actual implementation.
    # Held here for the backend's whole lifetime rather than on a `State`, because
    # `Tossing3DEnvironment` deep-copies states and deep-copying this would clone the
    # PyBullet client it does forward kinematics through.
    _abstractor: Any = PrivateAttr(default=None)

    @staticmethod
    def configure_headless_rendering(
        *,
        environ: MutableMapping[str, str] | None = None,
        backend: str | None = None,
        platform: str | None = None,
    ) -> dict[str, str]:
        """Re-assert the requested GL backend, and make sure a `DISPLAY` exists, before
        KINDER is imported.

        `DISPLAY` is `setdefault`, so a real display wins. `MUJOCO_GL`/`PYOPENGL_PLATFORM`
        are *written*, because the value in the environment after
        `register_all_environments()` is upstream's rewrite rather than anyone's choice --
        but what gets written is `REQUESTED_GL_BACKEND`, the snapshot taken at import,
        rather than a hardcoded `egl`. So a job that genuinely asks for `osmesa` -- CI,
        which has no EGL driver -- gets it, while the rewrite is still undone.

        `backend`/`platform` override the snapshot; they exist for tests, which cannot
        re-import the module per case. A `backend` alone carries `platform` with it, since
        the two are one choice and CI deliberately leaves `PYOPENGL_PLATFORM` empty.
        """
        target = os.environ if environ is None else environ
        target.setdefault("DISPLAY", FALLBACK_DISPLAY)
        target["MUJOCO_GL"] = backend or REQUESTED_GL_BACKEND
        target["PYOPENGL_PLATFORM"] = platform or backend or REQUESTED_GL_PLATFORM
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
            state_collection=_build_state_collection_class(),
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
            # No `task_config_path` override: this domain runs whatever scene the
            # installed KINDER registers for `env_id`. See
            # `Tossing3DEnvironment.backend` for why the choice was retired, and
            # `test_the_shipped_scene_still_puts_the_bin_on_the_box_that_scores` for
            # what stops a pin bump changing that scene silently.
            self._raw_env = api.kinder.make(
                self.env_id,
                render_mode=self.render_mode,
                scene_bg=self.scene_bg,
                allow_state_access=self.allow_state_access,
            )
            object_centric = self._object_centric()
            available = list(getattr(object_centric, "camera_names", []))
            if available and self.camera not in available:
                raise ValueError(f"camera {self.camera!r} is not in this scene: {available}")
            # Before any wrapper goes on, as upstream's own `test_pick_ground_toss` does:
            # a recording made through the wrong camera is a silent shot of a wall.
            object_centric.set_render_camera(self.camera)
            # Strictly before the seeded reset below, and only ever once.
            # `Tossing3DStateAbstractor.__init__` calls `sim.reset()` with **no seed**,
            # so building it after a seeded reset would silently re-randomise the scene
            # out from under the episode. Building it here means the unseeded reset it
            # performs is immediately overwritten by the seeded one.
            from kinder_models.dynamic3d.tossing.state_abstractions import (
                Tossing3DStateAbstractor,
            )

            self._abstractor = Tossing3DStateAbstractor(object_centric)
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
            # Dropping the reference is the whole release: `PyBulletSim` disconnects its
            # client from a `weakref.finalize` when it is collected. An explicit
            # disconnect here would double-disconnect against a reused client id.
            self._abstractor = None

    def set_substep_recording(self, *, enabled: bool) -> None:
        """Turn per-physics-tick FRAME collection on or off, effective immediately.
        State collection is separate and always on once a scene exists -- see
        `drain_substep_states` and `_sync_recording_wrapper`.

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
        physics-rate episode would double an already-large peak for nothing."""
        if not self.record_substeps or self._env is None or self._env is self._raw_env:
            return []
        return [np.asarray(frame, dtype=np.uint8) for frame in self._env.render()]

    def drain_substep_states(self) -> list[Any]:
        """Every `ObjectCentricState` snapshot collected since the last drain, in
        order, clearing the buffer.

        Unconditional, unlike `drain_substep_frames`: state capture is always on once a
        scene exists (see `_sync_recording_wrapper`), regardless of `record_substeps` --
        so a state log is recorded from every episode, not only rendered ones, and a run
        with no video at all can still be re-rendered later from its log. When frame
        recording IS also on, the two line up 1:1 (same per-tick granularity, same
        wrapper stack), so a state log recorded alongside a substep-frame recording
        covers exactly the ticks the frames do."""
        if self._env is None or self._env is self._raw_env:
            return []
        return self._env.drain()

    def _sync_recording_wrapper(self) -> None:
        """Put the `StateCollection` wrapper on unconditionally, once a scene exists --
        state logging is always-on (see `drain_substep_states`). `RenderCollection` is
        stacked beneath it only while `record_substeps` (frame/video capture) is
        requested; frames are never buffered when nothing will ever drain them, so
        video capture stays real and opt-in.

        Rebuilt fresh on every call rather than toggled in place: the only callers are
        `set_substep_recording` (used exclusively at episode boundaries, immediately
        before the next `reset()`, which clears the wrapper's own buffer anyway) and
        `reset()` itself, so nothing is ever lost by rebuilding. `StateCollection` is
        outermost when both are stacked: its `step()` calls `super().step()` (which is
        `RenderCollection`'s, which renders after stepping the raw env) and then reads
        the raw env's own `_current_state` via `self.unwrapped`, which is unaffected by
        wrapper order -- but keeping it outermost means `self._env` is always the one
        object whose `.drain()` this class calls."""
        if self._raw_env is None:
            return
        inner = (
            self.api().render_collection(self._raw_env, pop_frames=True, reset_clean=True)
            if self.record_substeps
            else self._raw_env
        )
        self._env = self.api().state_collection(inner)

    def snapshot(self) -> Any:
        """An opaque handle to the live simulator state, restorable by `restore`.

        This is a *KINDER* `ObjectCentricState`, copied -- not one of this package's flat
        `core.State`s, which are a lossy projection (four of the robot's twenty-two
        features, ten of the cube's sixteen). A caller that wants to rewind must hold one
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

    @staticmethod
    def snapshot_to_plain(*, snapshot: Any) -> dict[str, list[float]]:
        """A `snapshot()` (or a `drain_substep_states()` element), as plain
        `{object_name: [floats]}` -- JSON/pickle-safe, no KINDER `Object`/`Type`
        identities carried, since those are only ever compared by identity within one
        live scene and cannot be usefully serialized. `plain_to_snapshot` reconstructs
        against a *different* live scene's own identities."""
        return {obj.name: [float(v) for v in snapshot[obj]] for obj in snapshot}

    def plain_to_snapshot(self, *, plain: dict[str, list[float]]) -> Any:
        """The inverse of `snapshot_to_plain`, against THIS backend's own live scene --
        `restore(snapshot=this)` then puts that scene into the plain state's pose.
        Requires a scene already built (`reset()` run at least once), which is what
        supplies the `Object`/`Type` identities and the `type_features` schema a plain
        dict cannot carry on its own."""
        template = self._require_state()
        objects_by_name = {obj.name: obj for obj in template}
        data = {
            objects_by_name[name]: np.array(values, dtype=np.float32)
            for name, values in plain.items()
        }
        return type(template)(data, template.type_features)

    def reset_cube_and_bin(self) -> KinderObservation:
        """Reposition `cube_name`/`bin_name` to fresh ground poses in the live
        simulator, robot and everything else untouched. Backs
        `Tossing3DEnvironment.reset_movables`.

        Uses upstream's own placement sampler (`sample_collision_free_positions`
        + `mujoco_object.set_pose`), the same one `_initialize_object_poses` uses
        at `reset()`, scoped to just these two objects -- a real MuJoCo write, not
        a splice of two snapshots, so poses are as collision-free as any object
        upstream's own reset ever places. Both objects' regions are genuine
        ranges as of kindergarden#166, so both get independently randomized.
        Note `blocks_goal_region` is now parented on `bin_0`, so this also moves
        the scored window, not just the bin's visible position.

        Draws from the live scene's own `np_random` (not a fresh seed), same as
        every other in-episode source of randomness in this domain."""
        from kinder.envs.dynamic3d.placement_samplers import sample_collision_free_positions
        from kinder.envs.dynamic3d.utils import convert_yaw_to_quaternion

        object_centric = self._object_centric()
        ground_fixture = object_centric._ground_fixture  # noqa: SLF001
        assert ground_fixture is not None, (
            "reset_cube_and_bin needs a live scene (KinderBackend.reset() first)."
        )

        configs: dict[str, dict[str, dict[str, Any]]] = {}
        entity_region_names: dict[str, str] = {}
        entity_pos_yaw_samplers: dict[str, Any] = {}
        for object_name in (self.cube_name, self.bin_name):
            region_name = self._initial_state_region(
                object_centric=object_centric, object_name=object_name
            )
            obj = object_centric._objects_dict[object_name]  # noqa: SLF001
            obj_type = obj.__class__.REGISTERED_NAME
            obj_config = object_centric.task_config["objects"][obj_type][object_name]
            configs.setdefault(obj_type, {})[object_name] = obj_config
            entity_region_names[object_name] = region_name
            entity_pos_yaw_samplers[object_name] = ground_fixture.sample_pose_in_region

        object_poses = sample_collision_free_positions(
            configs,
            object_centric.np_random,
            entity_region_names=entity_region_names,
            entity_pos_yaw_samplers=entity_pos_yaw_samplers,
        )
        for obj_poses_dict in object_poses.values():
            for object_name, pose in obj_poses_dict.items():
                obj = object_centric._objects_dict[object_name]  # noqa: SLF001
                obj.set_pose(pose["position"], convert_yaw_to_quaternion(pose["yaw"]))

        assert object_centric._robot_env is not None  # noqa: SLF001
        assert object_centric._robot_env.sim is not None  # noqa: SLF001
        object_centric._robot_env.sim.forward()  # noqa: SLF001
        object_centric._current_state = (  # noqa: SLF001
            object_centric._get_object_centric_state()  # noqa: SLF001
        )
        self._state = object_centric._get_current_state()  # noqa: SLF001
        return self.observe()

    @staticmethod
    def _initial_state_region(*, object_centric: Any, object_name: str) -> str:
        """The region name `object_name` is placed in at a real `reset()`, read off
        `task_config["initial_state"]`'s own `["on"/"in", object_name, region_name]`
        predicates -- the same lookup `_initialize_object_poses` performs, so a task
        JSON's own region assignment is honoured rather than duplicated as a literal
        here."""
        for predicate in object_centric.task_config.get("initial_state", []):
            if predicate[0] in ("on", "in") and predicate[1] == object_name:
                return str(predicate[2])
        raise ValueError(
            f"no initial_state predicate places {object_name!r}; cannot sample a "
            "fresh ground pose for it."
        )

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

    def abstract_atoms(self, *, state: Any = None) -> frozenset[AbstractAtom]:
        """Upstream's own symbolic abstraction of `state` (default: the live state).

        This is where all five of this domain's predicates are actually evaluated --
        `predicates.py` only looks the answers up. Calling upstream's `state_abstractor`
        once per state, rather than once per predicate, also means the five answers are
        guaranteed mutually consistent: they describe one instant, not five.

        **Pure in its argument, which is the property the whole seam rests on.**
        `state_abstractor` opens with `self._pybullet_sim.set_state(state)`, re-pointing
        the kinematics sim at whatever it was handed before doing any forward kinematics.
        Verified rather than assumed: abstracting a captured state, resetting the live
        simulator to a different episode, and re-abstracting that same captured state
        gives an identical atom set.

        The one exception is `MovableInGoalRegion`, whose scored region comes off the
        live simulator rather than off `state` -- upstream says so in `state_abstractor`'s
        own comment. The region is attached to the bin's own body, not a fixed scene
        fixture, so it genuinely does vary: it tracks wherever the bin currently is, which
        differs across seeds once `bin_init_region` samples a position rather than naming
        one fixed point. That is exactly why this call needs a live scene rather than a
        cached box -- there no longer is one fixed box to cache.

        Object names are translated to *this domain's* on the way out. KINDER resolves
        the robot's name from the robot config at reset; this domain's `Object` is the
        literal `"robot"`. Everything else already agrees.
        """
        if self._abstractor is None:
            raise RuntimeError("KinderBackend.reset() has not run yet; there is no abstractor.")
        subject = self._require_state() if state is None else state
        abstract = self._abstractor.state_abstractor(subject)
        kinder_robot = self.robot_name
        return frozenset(
            (
                atom.predicate.name,
                tuple(
                    self.robot_atom_name if o.name == kinder_robot else o.name for o in atom.objects
                ),
            )
            for atom in abstract.atoms
        )

    def check_goals(self) -> bool:
        """Upstream's own verdict -- `ObjectCentricTidyBot3DEnv._check_goals()`.

        This is the success criterion for this domain. `predicates.IN_BIN` is
        written to agree with it and is differentially tested against it; it is never a
        second, independent definition of success.
        """
        return bool(self._object_centric()._check_goals())  # noqa: SLF001

    def goal_region_bbox(self) -> tuple[float, float, float, float, float, float]:
        """The live world-frame box `_check_goals()` scores containment in.

        The region is attached to the bin's own body (Tossing3D-o1.json's
        `blocks_goal_region.target` is `bin_name`), not to the ground, so `Region.bbox`
        moves with wherever the bin actually is rather than sitting at a scene-fixed
        point. That also means this no longer needs the env-swap dance a ground-attached
        region required: `MujocoObject._create_regions` passes the object's own `env`
        into every `Region` it builds, so `region.env` is already the live env by
        construction -- unlike a ground region, which is built with `env=None` and only
        gets one handed in per call.
        """
        object_centric = self._object_centric()
        bin_object = object_centric._objects_dict[self.bin_name]  # noqa: SLF001
        regions = bin_object.region_objects
        found = regions.get(self.goal_region_name, [])
        if len(found) != 1:
            raise ValueError(
                f"expected exactly one {self.goal_region_name!r} region, found "
                f"{len(found)} (regions: {sorted(regions)})"
            )
        bbox = [float(value) for value in found[0].bbox]
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
    ) -> ControllerRun:
        """Drive one upstream controller to termination, stepping the live simulator.

        The loop body is upstream's own, from `test_pick_ground_toss`: `controller.step()`
        into `env.step`, devectorize, `controller.observe`, break on `terminated()`.

        A controller that does not terminate within `limit` is reported as
        `terminated=False` rather than raised, and a controller that *raises* is reported
        through `error`. Both are ordinary outcomes of a skill whose continuous parameters
        do not work out, and the caller has to be able to keep going -- `take_action` must
        be total.

        **`BaseException`, not `Exception`, and that is load-bearing.** The controllers
        this drives raise `bilevel_planning`'s `TrajectorySamplingFailure`, which is
        **not** an `Exception` subclass, so `except Exception` misses every sampling
        failure and lets it escape `take_action`. KINDER's own motion planners also still
        `assert plan is not None`, which `Exception` would have caught -- so both kinds
        arrive here and both must be reported rather than propagated.
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

        try:
            controller.reset(state, params)
        except BaseException as exc:  # noqa: BLE001  (any planner failure is a failed skill)
            if isinstance(exc, KeyboardInterrupt | SystemExit):
                raise
            return ControllerRun(steps=0, terminated=False, error=f"{type(exc).__name__}: {exc}")

        for step in range(limit):
            try:
                action = controller.step()
                observation, _, _, _, _ = self._env.step(action)
            except BaseException as exc:  # noqa: BLE001  (same reasoning as above)
                if isinstance(exc, KeyboardInterrupt | SystemExit):
                    raise
                return ControllerRun(
                    steps=step, terminated=False, error=f"{type(exc).__name__}: {exc}"
                )
            self._state = self._env.observation_space.devectorize(observation)
            controller.observe(self._state)
            if controller.terminated():
                return ControllerRun(steps=step + 1, terminated=True)
        return ControllerRun(steps=limit, terminated=False)

    def run_pick_cube(self) -> ControllerRun:
        """`pick_cube` -- upstream's parameterless grasp of a cube off the ground.

        Where to stand and which grasp rotation to use are derived inside the controller
        (`PickCubeController.STANDOFF`, `upright_grasp_rotations`), so there is nothing to
        pass and nothing for a sampler to draw. `params` is `None` rather than an empty
        array because upstream's `sample_parameters` returns `tuple()` and `reset`
        immediately `del`s it.

        `disable_collision_objects` is deliberately absent: it exists on
        `MoveToTargetGroundController.reset` and on the composed toss, not here, and
        passing it is a `TypeError`.
        """
        return self.run_controller(
            module="tossing",
            key="pick_cube",
            object_names=(self.robot_name, self.cube_name, self.barrier_name),
            params=None,
            limit=self.pick_step_limit,
        )

    def run_move_to_toss_location_and_toss(
        self,
        *,
        distance: float,
        rotation: float,
        release_speed_deg_s: float,
        gripper_release_ms: float,
    ) -> ControllerRun:
        """Drive to a throw pose and throw, as one upstream controller.

        One controller, three internal phases (base motion, windup, swing). It replaced a
        `move_to_target` + `move_arm_to_conf` + `toss` sequence this package used to drive
        itself; the composition is upstream's, so the phase boundaries are no longer
        visible from here and neither is a per-phase step count.

        **The one place in the domain where degrees become radians**: the dial is carried
        in joint-path deg/s (`skills.TOSS_SPEED_BOUNDS`) and the controller's own
        `SPEED_BOUNDS` are rad/s. A second or missing conversion is a silent 57x error
        either way.

        `gripper_release_ms` stays a float here -- upstream's `reset` rounds it to an int
        itself (`int(round(...))`), unlike the old `TossController.reset`, which truncated
        and so needed the rounding done on this side.

        `disable_collision_objects` is left to upstream's own default, which is the held
        cube's name: the robot's own cargo would otherwise reject every base plan.
        """
        return self.run_controller(
            module="tossing",
            key="move_to_toss_location_and_toss",
            object_names=(self.robot_name, self.cube_name, self.barrier_name),
            params=np.array([
                distance,
                rotation,
                float(np.deg2rad(release_speed_deg_s)),
                gripper_release_ms,
            ]),
            limit=self.toss_step_limit,
        )

    def _object_centric(self) -> Any:
        if self._raw_env is None:
            raise RuntimeError("KinderBackend.reset() has not run yet; there is no scene.")
        return self._raw_env.unwrapped._object_centric_env  # noqa: SLF001

    def _require_state(self) -> Any:
        if self._state is None:
            raise RuntimeError("KinderBackend.reset() has not run yet; there is no state.")
        return self._state
