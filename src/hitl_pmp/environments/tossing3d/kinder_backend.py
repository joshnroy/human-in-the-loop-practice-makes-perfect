"""The one module in this package that touches KINDER, and the only place a KINDER
import may appear.

What it owns is now narrow: the **live gym environment** -- building it, seeding it,
choosing its camera, collecting frames, closing it. Everything that is true of KINDER in
general rather than of Tossing3D in particular has moved to `adapters/kinder/`:

| concern | where it lives now |
| --- | --- |
| the EGL / `DISPLAY` import dance | `adapters.kinder.bootstrap` |
| `ObjectCentricState` <-> `core.State` | `adapters.kinder.state_translation` |
| the state abstractor as `core.Predicate`s | `adapters.kinder.abstraction` |
| controllers as parameters, draws, executions | `adapters.kinder.controllers` |

This module's job is to construct those three from a live scene and hand them out.

## What this package no longer reimplements

It used to carry six hand-written predicate classifiers "ported from upstream's own
`state_abstractions`, thresholds included", plus its own sampling bounds for every skill
parameter. Both were copies, and a copy has no mechanism for noticing that the original
moved. The measured consequence: hitl's `TOSS_RELEASE_MS_BOUNDS` was `(300, 1400)` while
the controller's own band was `(700, 840)`, so it drew from a window about nine times too
wide and the large majority of its draws could not score.

Now `Tossing3DStateAbstractor` classifies and `create_lifted_controllers`' own samplers
draw. There is nothing left here to keep in agreement with upstream.

## Constructing the abstractor resets the scene, so ordering matters

`Tossing3DStateAbstractor.__init__` calls `sim.reset()` -- it needs an initial state to
build its `PyBulletSim` from. So it is constructed once, immediately after the gym
environment is made and *before* the episode's own `reset(seed=...)`, and never again.
Building it after seeding would silently discard the seed.

The abstractor and the controllers share one `PyBulletSim`, which is upstream's own
arrangement in `tidybot3d_tossing3D.py`. That is not only tidy: grounding a controller
mints an instance, and a per-grounding sim is one live `p.connect(p.DIRECT)` server each.

## The PyBullet client leak is fixed upstream, not worked around here

`PyBulletSim` disconnects from a `weakref.finalize` when it is collected (upstream PR #87,
`9512b9e`, an ancestor of the pin). **Do not add a `_release`-style explicit `close()`**:
with the finalizer in place that double-disconnects.

## Physics-rate frames come from a wrapper, not a hand-rolled buffer

One `take_action` is a whole controller execution, so a `core.Renderer` -- one frame per
*transition* -- would give a two-frame clip of a domain whose entire point is a throw.
`gymnasium.wrappers.RenderCollection` collects `env.render()` on every `step()`, and its
own `render()` hands the list back and clears it. `RecordVideo` is deliberately not used:
it writes its own uncaptioned file per gym reset, which would leave the harness's
`episode.mp4` still two frames long and add a second file beside it.

Collected frames were measured not to alias one reused MuJoCo buffer
(`np.shares_memory` is `False` between successive renders), so nothing is copied on the
way out. Recording is **off by default**: it renders every physics tick, and a training
run that wants no video must not pay for it.
"""

from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from hitl_pmp.adapters.kinder.abstraction import KinderAbstraction
from hitl_pmp.adapters.kinder.bootstrap import KinderBootstrap
from hitl_pmp.adapters.kinder.controllers import KinderControllers
from hitl_pmp.adapters.kinder.state_translation import KinderStateTranslator
from hitl_pmp.adapters.kinder.types import ControllerRun

# `task_view` is the camera this scene's own task config defines, and the only one that
# shows the throw. `set_render_camera` does not validate the name, so choosing a camera
# this scene does not define silently renders a near-static shot of a wall.
DEFAULT_CAMERA = "task_view"

# Upstream's own `test_pick_ground_toss` value; the seed most numbers in this package's
# docs were measured at.
DEFAULT_SCENE_SEED = 125


class KinderApi(BaseModel):
    """Handles to everything this package needs from KINDER, imported in one place.

    Held as one object so the import happens exactly once per process and so the EGL
    environment is guaranteed to have been configured first (see
    `adapters.kinder.bootstrap`).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kinder: ModuleType
    state_abstractor_cls: Any
    # The five `relational_structs.Predicate`s upstream's abstractor may report, in the
    # order this domain declares them.
    predicates: tuple[Any, ...]
    create_lifted_controllers: Any
    pybullet_sim_cls: Any
    # `gymnasium.wrappers.RenderCollection`, the class itself. Gymnasium is KINDER's own
    # dependency rather than this repo's, so it is imported here with everything else
    # instead of at module scope, which would break the offline import.
    render_collection: Any


class KinderBackend(BaseModel):
    """A live `kinder/Tossing3D-<variant>-v0`, and the three bridge objects over it.

    Construction imports nothing: `KinderBackend()` is pure pydantic, so
    `Tossing3DEnvironment` can hold one as an ordinary field and this module stays
    importable without MuJoCo. KINDER is imported on the first `reset()`.

    One `KinderBackend` owns at most one live gym env at a time. `reset(seed=...)`
    rebuilds the scene from that seed, which is the *only* way this domain can restore a
    state: a flat `core.State` of poses cannot round-trip MuJoCo's contact state, so a
    mid-episode state is not restorable and `Tossing3DEnvironment.set_state` refuses one.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Names as they appear in `Tossing3D-o1.json`'s `objects` block. Fixed by upstream's
    # scene, not configuration, so ClassVars rather than fields.
    cube_name: ClassVar[str] = "cube_0"
    bin_name: ClassVar[str] = "bin_0"
    barrier_name: ClassVar[str] = "cuboid_barrier"

    # Upstream's own per-controller step budgets. `pick_cube` drives a base motion and
    # four arm phases; the composed move-and-toss drives a base motion, a windup and a
    # swing, so both are generous.
    pick_step_limit: ClassVar[int] = 400
    toss_step_limit: ClassVar[int] = 400

    env_id: str = "kinder/Tossing3D-o1-v0"
    scene_bg: bool = True
    camera: str = DEFAULT_CAMERA
    render_mode: str = "rgb_array"
    # Unlocks `ObjectCentricTidyBot3DEnv.set_state`, which otherwise raises "State access
    # is not allowed". Upstream's own env models pass this too.
    allow_state_access: bool = True
    # Collect one frame per physics tick while a controller runs. Off by default; see the
    # module docstring.
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
    _translator: KinderStateTranslator | None = PrivateAttr(default=None)
    _abstraction: KinderAbstraction | None = PrivateAttr(default=None)
    _controllers: KinderControllers | None = PrivateAttr(default=None)

    def api(self) -> KinderApi:
        """Import KINDER once, in the right order, and register its environments."""
        if self._api is not None:
            return self._api
        kinder = KinderBootstrap.register_environments()

        from kinder_models.dynamic3d.tossing.parameterized_skills import (
            PyBulletSim,
            create_lifted_controllers,
        )
        from kinder_models.dynamic3d.tossing.state_abstractions import (
            HandEmpty,
            Holding,
            MovableInGoalRegion,
            MovableIsDownX,
            OnGround,
            Tossing3DStateAbstractor,
        )

        from gymnasium.wrappers import RenderCollection  # isort: skip

        self._api = KinderApi(
            kinder=kinder,
            state_abstractor_cls=Tossing3DStateAbstractor,
            predicates=(HandEmpty, OnGround, Holding, MovableInGoalRegion, MovableIsDownX),
            create_lifted_controllers=create_lifted_controllers,
            pybullet_sim_cls=PyBulletSim,
            render_collection=RenderCollection,
        )
        return self._api

    @property
    def robot_name(self) -> str:
        """The robot's name in the live scene, resolved at reset.

        A property of the robot config (`sim.robot_name`) rather than of the task JSON's
        `objects` block, which is why it is read rather than written down.
        """
        if not self._robot_name:
            raise RuntimeError("KinderBackend.reset() has not run yet; there is no scene.")
        return self._robot_name

    def translator(self) -> KinderStateTranslator:
        if self._translator is None:
            raise RuntimeError("KinderBackend.reset() has not run yet; there is no scene.")
        return self._translator

    def abstraction(self) -> KinderAbstraction:
        if self._abstraction is None:
            raise RuntimeError("KinderBackend.reset() has not run yet; there is no scene.")
        return self._abstraction

    def controllers(self) -> KinderControllers:
        if self._controllers is None:
            raise RuntimeError("KinderBackend.reset() has not run yet; there is no scene.")
        return self._controllers

    def reset(self, *, seed: int) -> Any:
        """Build (or rebuild) the scene at `seed` and return the resulting KINDER state.

        The gym env is remade only when there is not one already; `env.reset(seed=...)`
        alone is enough afterwards, which is what makes a differently-seeded initial state
        cheap relative to a fresh `make`.

        **`abstraction().invalidate()` on every reset, unconditionally.** The abstractor
        reads the goal region off the live simulator rather than off the state, so every
        cached answer is about a scene that no longer exists. Over-invalidating costs one
        abstractor call; under-invalidating is a wrong answer that never expires.
        """
        api = self.api()
        if self._raw_env is None:
            # No `task_config_path` override: this domain runs whatever scene the
            # installed KINDER registers for `env_id`. See `Tossing3DEnvironment.backend`
            # for why the choice was retired, and `test_kinder_fidelity.py`'s goal-box
            # tests for what stops a pin bump changing that scene silently.
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
            # Before any wrapper goes on, as upstream's own tests do: a recording made
            # through the wrong camera is a silent shot of a wall.
            object_centric.set_render_camera(self.camera)
            self._build_bridge()
        self._sync_recording_wrapper()
        observation, _ = self._env.reset(seed=seed)
        self._state = self._env.observation_space.devectorize(observation)
        self._robot_name = self._object_centric().robot_name
        self.abstraction().invalidate()
        return self._state

    def _build_bridge(self) -> None:
        """Construct the translator, abstraction and controllers for the live scene.

        Called once per gym environment, right after `make` and **before** any seeded
        reset -- `Tossing3DStateAbstractor.__init__` resets the simulator to build its own
        `PyBulletSim`, so doing this later would throw away the seed.
        """
        api = self.api()
        object_centric = self._object_centric()
        abstractor = api.state_abstractor_cls(object_centric)
        initial_state, _ = object_centric.reset()
        translator = KinderStateTranslator.from_kinder_state(kinder_state=initial_state)
        pybullet_sim = api.pybullet_sim_cls(initial_state, rendering=False)
        self._translator = translator
        self._abstraction = KinderAbstraction.build(
            translator=translator,
            state_abstractor=abstractor.state_abstractor,
            kinder_predicates=api.predicates,
        )
        self._controllers = KinderControllers(
            translator=translator,
            lifted_controllers=api.create_lifted_controllers(
                object_centric.action_space,
                object_centric.initial_constant_state,
                pybullet_sim=pybullet_sim,
            ),
        )

    def run_skill(
        self,
        *,
        key: str,
        object_names: Sequence[str],
        params: np.ndarray,
        state: Any,
        limit: int,
    ) -> ControllerRun:
        """Drive one upstream controller to termination against the live simulator.

        The loop itself is `KinderControllers.run`'s -- generic, and upstream's own body.
        What this supplies is the simulator step, which is the part that is not generic.
        """

        # Positional because `KinderControllers.run` calls it that way -- it stands in for
        # `gym.Env.step`, whose signature is not ours.
        def step(action: np.ndarray) -> Any:  # noqa: PLR0917
            observation, _, _, _, _ = self._env.step(action)
            self._state = self._env.observation_space.devectorize(observation)
            return self._state

        return self.controllers().run(
            key=key,
            object_names=object_names,
            params=params,
            state=state,
            step=step,
            limit=limit,
        )

    def kinder_state(self) -> Any:
        """The live KINDER state, as the simulator last reported it."""
        if self._state is None:
            raise RuntimeError("KinderBackend.reset() has not run yet; there is no state.")
        return self._state

    def close(self) -> None:
        """Release the gym env, if one was ever built. Idempotent."""
        if self._raw_env is not None:
            self._raw_env.close()
            self._raw_env = None
            self._env = None
            self._state = None
            self._translator = None
            self._abstraction = None
            self._controllers = None

    def set_substep_recording(self, *, enabled: bool) -> None:
        """Turn per-physics-tick frame collection on or off, effective immediately.

        Imports nothing on its own: a backend with no scene yet just remembers the flag,
        and the wrapper goes on at the next `reset()`.
        """
        self.record_substeps = enabled
        self._sync_recording_wrapper()

    def drain_substep_frames(self) -> list[np.ndarray]:
        """Every frame collected since the last drain, in order, clearing the buffer.

        This is `RenderCollection.render()` -- the wrapper's own drain -- not a buffer
        kept here. Empty whenever recording is off or no scene exists, which are both
        ordinary states rather than errors: `run_task_episode` drains unconditionally.
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
        """An opaque handle to the live simulator state, restorable by `restore`."""
        return self.kinder_state().copy()

    def restore(self, *, snapshot: Any) -> Any:
        """Put the simulator back to a `snapshot`, exactly, and return it.

        **Faithful to float32, not bit-exact.** An `ObjectCentricState` is the observation
        vector and `ObjectCentricBoxSpace` is float32, while MuJoCo integrates in float64,
        so a round-trip reintroduces ~1.2e-7 of relative error that 200 substeps per env
        step amplify. Four orders of magnitude below any state change a symbolic predicate
        cares about, but a rewound rollout is *not* byte-reproducible and nothing should
        be built on the assumption that it is.
        """
        self._object_centric().set_state(snapshot)
        self._state = snapshot.copy()
        # A restore moves the simulator without changing any `core.State` this process
        # already holds, which is exactly the case the cache's generation key exists for.
        self.abstraction().invalidate()
        return self._state

    def check_goals(self) -> bool:
        """Upstream's own verdict -- `ObjectCentricTidyBot3DEnv._check_goals()`.

        Kept as the independent reference the symbolic layer is checked against. It is no
        longer reimplemented anywhere: `MovableInGoalRegion` comes from upstream's own
        abstractor, so the fidelity tests compare two upstream computations of the same
        thing rather than upstream's against a port of it.
        """
        return bool(self._object_centric()._check_goals())  # noqa: SLF001

    def render(self) -> np.ndarray:
        """One RGB frame from `task_view`, copied out of MuJoCo's buffer.

        Rendered from the *unwrapped* env, so a single frame is still a single frame while
        a `RenderCollection` is in place -- that wrapper's own `render()` returns the
        collected list and drains it, which would both break this signature and eat the
        episode's frames.
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

    def _object_centric(self) -> Any:
        if self._raw_env is None:
            raise RuntimeError("KinderBackend.reset() has not run yet; there is no scene.")
        return self._raw_env.unwrapped._object_centric_env  # noqa: SLF001
