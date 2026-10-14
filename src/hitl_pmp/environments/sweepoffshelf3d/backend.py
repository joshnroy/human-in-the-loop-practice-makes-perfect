"""The live KINDER `SweepOffShelf3D-o5` scene: build, observe, classify, human resets.

It and `kinder_skills.py` (KINDER's controllers) are the only modules in this package
that import KINDER or MuJoCo, both only lazily, so the symbolic layer and its tests
need neither.

## Classification

Every predicate is decided here, once per state, from the live scene:

- a cube or the wiper is *held* when both gripper pads touch it;
- `OnShelf`: not held, behind the cube shelf's open front, resting on that board;
- `InBox`: not held and wholly inside the box -- the env's own `cube_in_box`, which is
  exactly what the task scores (measured in the box's frame: see `OrientedBox`);
- `OnFloor`: any other unheld cube -- including two places no robot skill reaches.
  A cube that goes over the box's far rim passes through the 67 mm opening between
  that rim and the cube shelf's underside and comes to rest on the cupboard's bottom
  board; and a cube can balance on any of the box's 1 cm rims. Both are "OnFloor" to
  the planner, so a floor pick or a return from the floor is attempted and fails;
  only the human's `reset_cubes`, which pours back every cube from wherever it is,
  recovers them (`test_backend.py` reproduces both);
- `WiperHome`: unheld and lying on the wiper shelf; `WiperOnFloor`: unheld anywhere else;
- `HandEmpty`: nothing held;
- the summary atoms `AnyCubeOnShelf`, `AnyCubeOffShelf` and `WiperAway`, derived from
  the above by `SweepOffShelf3DAtoms.with_summaries`.

## Human resets

The human's two skills move only what they name, exactly as in the Full-Agentic runs,
and like every skill each is all-or-nothing. `reset_cubes` succeeds with probability
`reset_cubes_success_probability` (p): a success lifts the box and pours all five cubes
back onto the cube shelf (from wherever they are, the hand included); a failure moves
nothing. `reset_wiper` always returns the wiper to its starting pose on its shelf.
Every placement is searched on a scratch copy of the model and committed only when it
clears the robot, the other objects and the cupboard.
"""

import copy
import os
import types
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .objects import SweepOffShelf3DObjects as O
from .predicates import SweepOffShelf3DAtoms

TASK_ID = "kinder/SweepOffShelf3D-o5-v0"
CUPBOARD = "cupboard_1"
CUBE_SHELF = 1
WIPER_SHELF = 2
CUBE_REGION = "cubes_init_region"
SUPPORT_PREFIXES = ("floor", f"{CUPBOARD}_shelf")
WIPER_SHIFTS = (0.0, 0.05, -0.05, 0.1, -0.1, 0.15, -0.15)
MAX_CUBE_SAMPLES = 400


class SweepOffShelf3DBackend(BaseModel):
    """Owns one KINDER env. Pure pydantic until the first `reset`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scene_bg: bool = True
    # p: the probability one `reset_cubes` succeeds (all five cubes) rather than
    # fails (nothing moves).
    reset_cubes_success_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    human_seed: int = 0
    # The gripper command the controllers last sent (0 open .. 1 closed).
    grip: float = 0.0

    cube_names: ClassVar[tuple[str, ...]] = tuple(cube.name for cube in O.cubes)
    wiper_name: ClassVar[str] = O.wiper.name
    box_name: ClassVar[str] = O.box.name

    _env: Any = PrivateAttr(default=None)
    _human_rng: np.random.Generator = PrivateAttr()
    _pads: tuple[frozenset[int], frozenset[int]] = PrivateAttr(default=(frozenset(), frozenset()))
    _wiper_start: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: object) -> None:
        self._human_rng = np.random.default_rng(self.human_seed)

    # --- lifecycle --------------------------------------------------------

    def env(self) -> Any:
        if self._env is None:
            import kinder
            import kinder.envs.dynamic3d.envs  # noqa: F401  (registers Dynamic3D)
            from kinder.envs.dynamic3d.envs import TidyBot3DConfig

            rendering = {
                key: os.environ.get(key, "egl") for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM")
            }
            kinder.register_all_environments()
            os.environ.update(rendering)
            self._env = kinder.make(
                TASK_ID,
                render_mode="rgb_array",
                allow_state_access=True,
                scene_bg=self.scene_bg,
                scene_render_camera="task_view",
                config=TidyBot3DConfig(use_arm_velocities=False),
            )
            OrientedBox.install(core=self.core())
        return self._env

    def reset(self, *, seed: int) -> None:
        self.env().reset(seed=seed)
        self.grip = 0.0
        model = self.model()
        pads: dict[int, set[int]] = {}
        for geom in range(model.ngeom):
            if "pad" in (model.geom(geom).name or ""):
                pads.setdefault(int(model.geom_bodyid[geom]), set()).add(geom)
        if len(pads) != 2:
            raise RuntimeError(f"expected two gripper pad bodies, found {len(pads)}")
        first, second = (frozenset(geoms) for geoms in pads.values())
        self._pads = (first, second)
        self._wiper_start = tuple(
            np.asarray(part, dtype=float).copy()
            for part in self.core().get_object(self.wiper_name).get_pose()
        )

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    # --- handles ----------------------------------------------------------

    def core(self) -> Any:
        return self.env().unwrapped._object_centric_env  # noqa: SLF001

    def sim(self) -> Any:
        return self.core()._robot_env.sim  # noqa: SLF001

    def model(self) -> Any:
        return self.sim().model.mj_model

    def data(self) -> Any:
        return self.sim().data.mj_data

    def render_fps(self) -> int:
        return int(self.env().metadata.get("render_fps", 10))

    # --- observation ------------------------------------------------------

    def held(self, *, name: str) -> bool:
        """Both gripper pads touch the named body."""
        model, data = self.model(), self.data()
        body = model.body(name).id
        touched = [False, False]
        for contact in data.contact[: data.ncon]:
            for mine, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                if model.geom_bodyid[other] != body:
                    continue
                for side, pad in enumerate(self._pads):
                    touched[side] |= int(mine) in pad
        return all(touched)

    def features(self) -> dict[str, np.ndarray]:
        core, data, model = self.core(), self.data(), self.model()
        base = [
            float(data.qpos[model.jnt_qposadr[model.joint(f"robot_joint_{axis}").id]])
            for axis in ("x", "y", "th")
        ]
        out = {O.robot.name: np.array([*base, self.grip])}
        for name in self.cube_names:
            out[name] = np.asarray(core.get_object(name).get_pose()[0], dtype=float).copy()
        position, quaternion = core.get_object(self.wiper_name).get_pose()
        w, x, y, z = (float(v) for v in quaternion)
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        out[self.wiper_name] = np.array([*np.asarray(position, dtype=float), yaw])
        (x0, y0, x1, y1), rim = core.box_interior()
        out[self.box_name] = np.array([x0, y0, x1, y1, rim])
        return out

    def abstract_atoms(self) -> frozenset[tuple[str, tuple[str, ...]]]:
        core = self.core()
        cube_board = core.shelf_top_z(shelf=CUBE_SHELF)
        wiper_board = core.shelf_top_z(shelf=WIPER_SHELF)
        front = core.shelf_front_x()
        atoms: set[tuple[str, tuple[str, ...]]] = set()
        holding = False
        for name in self.cube_names:
            if self.held(name=name):
                atoms.add(("HoldingCube", (O.robot.name, name)))
                holding = True
                continue
            x, _, z = core.get_object(name).get_pose()[0]
            if core.cube_in_box(name):
                atoms.add(("InBox", (name, self.box_name)))
            elif x > front and abs(z - cube_board - O.cube_half_size) < 0.01:
                atoms.add(("OnShelf", (name,)))
            else:
                atoms.add(("OnFloor", (name,)))
        if self.held(name=self.wiper_name):
            atoms.add(("HoldingWiper", (O.robot.name, self.wiper_name)))
            holding = True
        else:
            x, _, z = core.get_object(self.wiper_name).get_pose()[0]
            home = x > front and wiper_board - 0.003 < z < wiper_board + 0.06
            atoms.add(("WiperHome" if home else "WiperOnFloor", (self.wiper_name,)))
        if not holding:
            atoms.add(("HandEmpty", (O.robot.name,)))
        return SweepOffShelf3DAtoms.with_summaries(atoms=frozenset(atoms))

    # --- human resets -----------------------------------------------------

    def reset_cubes(self) -> list[str]:
        """With probability p pour all five cubes back onto the cube shelf and return
        their names; otherwise move nothing and return none."""
        if self._human_rng.random() >= self.reset_cubes_success_probability:
            return []
        moved = list(self.cube_names)
        self._commit(placements=self._cube_placements(names=moved))
        return moved

    def reset_wiper(self) -> None:
        self._commit(placements=self._wiper_placement())

    def _commit(self, *, placements: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        core = self.core()
        for name, (position, quaternion) in placements.items():
            obj = core._objects_dict[name]  # noqa: SLF001
            obj.set_pose(position, quaternion)
            obj.set_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.sim().forward()

    def _scratch(self, *, names: list[str]) -> tuple[Any, Any, dict[str, int], set[int]]:
        """A collision-checking copy where every collidable shape collides, with the
        objects to place parked far away until each is placed."""
        import mujoco

        core = self.core()
        model = copy.copy(self.model())
        data = mujoco.MjData(model)
        data.qpos[:] = self.data().qpos
        collidable = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        model.geom_contype[collidable] = 1
        model.geom_conaffinity[collidable] = 1
        model.geom_margin[collidable] = 0.003
        addresses = {}
        for name in names:
            joint = model.joint(core._objects_dict[name].joint_name).id  # noqa: SLF001
            addresses[name] = int(model.jnt_qposadr[joint])
            data.qpos[addresses[name] + 2] = 100.0 + len(addresses)
        support = {
            g for g in range(model.ngeom) if (model.geom(g).name or "").startswith(SUPPORT_PREFIXES)
        }
        return model, data, addresses, support

    @staticmethod
    def _blocked(*, model: Any, data: Any, body: int, support: set[int]) -> bool:
        for contact in data.contact[: data.ncon]:
            bodies = (int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2]))
            if body not in bodies or bodies[0] == bodies[1]:
                continue
            other = contact.geom2 if bodies[0] == body else contact.geom1
            if contact.dist < (-0.003 if other in support else 0.002):
                return True
        return False

    def _cube_placements(self, *, names: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        import mujoco

        core = self.core()
        cupboard = core._fixtures_dict[CUPBOARD]  # noqa: SLF001
        board = core.shelf_top_z(shelf=CUBE_SHELF)
        model, data, addresses, support = self._scratch(names=names)
        placements = {}
        for name in names:
            body = model.body(name).id
            address = addresses[name]
            for _ in range(MAX_CUBE_SAMPLES):
                x, y, _, yaw = cupboard.sample_pose_in_region(CUBE_REGION, self._human_rng)
                position = np.array([x, y, board + O.cube_half_size + 0.0005])
                quaternion = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
                data.qpos[address : address + 7] = np.r_[position, quaternion]
                mujoco.mj_forward(model, data)
                if not self._blocked(model=model, data=data, body=body, support=support):
                    placements[name] = (position, quaternion)
                    break
            else:
                raise RuntimeError(f"no clear shelf placement for {name}; reset not applied")
        return placements

    def _wiper_placement(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        import mujoco

        model, data, addresses, support = self._scratch(names=[self.wiper_name])
        body = model.body(self.wiper_name).id
        start, quaternion = self._wiper_start
        address = addresses[self.wiper_name]
        for shift in WIPER_SHIFTS:
            position = start.copy()
            position[1] += shift
            data.qpos[address : address + 7] = np.r_[position, quaternion]
            mujoco.mj_forward(model, data)
            if not self._blocked(model=model, data=data, body=body, support=support):
                return {self.wiper_name: (position, quaternion)}
        raise RuntimeError("no clear placement for the wiper on its shelf; reset not applied")


class OrientedBox:
    """The box's interior in its own frame, so a box the robot has turned still scores.

    KINDER's `box_interior` raises for any box not axis-aligned, and its `cube_in_box`
    and `_check_goals` -- which every `env.step` calls -- go through it, so a pick that
    nudges the box a few degrees used to crash the episode. `install` rebinds both on
    this scene's own object-centric env; nothing upstream changes. For an axis-aligned
    box both answer exactly as upstream's do.
    """

    @staticmethod
    def install(*, core: Any) -> None:
        core.box_interior = types.MethodType(OrientedBox.interior, core)
        core.cube_in_box = types.MethodType(OrientedBox.cube_in_box, core)

    @staticmethod
    def _frame(core: Any) -> tuple[Any, np.ndarray, Any, float, float]:  # noqa: PLR0917
        from scipy.spatial.transform import Rotation

        box = core._objects_dict[O.box.name]  # noqa: SLF001
        position, quaternion = box.get_pose()
        rotation = Rotation.from_quat(np.asarray(quaternion, dtype=float)[[1, 2, 3, 0]])
        half_x = box.length / 2 - box.wall_thickness
        half_y = box.width / 2 - box.wall_thickness
        return box, np.asarray(position, dtype=float), rotation, half_x, half_y

    @staticmethod
    def interior(core: Any) -> tuple[tuple[float, float, float, float], float]:  # noqa: PLR0917
        """((x0, y0, x1, y1), rim_z): the axis-aligned bound of the inner footprint
        (the footprint itself when the box is axis-aligned) and the rim height."""
        box, position, rotation, half_x, half_y = OrientedBox._frame(core)
        corners = rotation.apply([
            [sx * half_x, sy * half_y, 0.0] for sx in (-1, 1) for sy in (-1, 1)
        ])
        low = position[:2] + corners[:, :2].min(axis=0)
        high = position[:2] + corners[:, :2].max(axis=0)
        bound = (float(low[0]), float(low[1]), float(high[0]), float(high[1]))
        return bound, float(position[2]) + box.height

    @staticmethod
    def cube_in_box(core: Any, name: str) -> bool:  # noqa: PLR0917
        """All eight cube corners inside the inner walls and below the rim, both
        measured in the box's own frame."""
        from scipy.spatial.transform import Rotation

        box, position, rotation, half_x, half_y = OrientedBox._frame(core)
        cube = core._objects_dict[name]  # noqa: SLF001
        cube_position, cube_quaternion = cube.get_pose()
        cube_rotation = Rotation.from_quat(np.asarray(cube_quaternion, dtype=float)[[1, 2, 3, 0]])
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        corners = cube_rotation.apply(signs * np.asarray(cube.size)) + cube_position
        x, y, z = rotation.inv().apply(corners - position).T
        tolerance = 1e-9
        return bool(
            np.all(
                (np.abs(x) <= half_x + tolerance)
                & (np.abs(y) <= half_y + tolerance)
                & (z < box.height)
            )
        )
